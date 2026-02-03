# ultralytics/utils/gradbalance/DAGR.py
from __future__ import annotations

import math
from typing import Iterable, Literal

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class DAGR3Balancer(GradientBalancer):
    """
    DAGR: Density-Aware Gradient Rectification (robust version)
    修复版：已添加 alpha/lr/**kwargs 兼容性接口，防止 train.py 初始化报错。
    """

    def __init__(
        self,
        task_names: list[str] | None = None,
        interval: int = 1,
        warmup: int = 0,
        eps: float = 1e-8,
        # ---- 兼容性接口 (Fix: 接收 train.py 强塞的参数) ----
        alpha: float = 0.0,
        lr: float = 0.0,
        # ---- DAGR core ----
        gamma: float = 0.5,              # 密度敏感度
        beta_base: float = 0.8,          # 一步协调强度上限
        use_abs_cos: bool = True,        # |cos|
        aggregate: Literal["mean", "density_weighted"] = "mean",
        # ---- magnitude ----
        task_mag_quantile: float = 0.75, # 任务梯度范数分位数
        align_to_g0_norm: bool = True,   # v 的最终范数对齐到当前原始 shared 梯度
        # ---- robustness knobs ----
        conflict_gate: bool = True,      # 冲突门控
        gate_cos_threshold: float = 0.2, # min pairwise cos >= threshold 时认为基本不冲突
        eta_max: float = 0.6,            # 最大融合系数 eta
        eta_ramp_steps: int = 300,       # eta 线性爬坡步数
        min_cos_with_g0: float = -0.2,   # trust region
        use_ema_v: bool = True,          # 对 v_unit 做 EMA
        ema_mu: float = 0.9,
        # ---- engineering ----
        lock_shared_params: bool = True,
        ddp_sync_v: bool = True,
        **kwargs  # (Fix: 兜底其他未知参数)
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        # 虽然 DAGR3 不用 alpha 和 lr，但为了不报错，我们接收并存起来（或者直接忽略）
        self.alpha_unused = float(alpha)
        self.lr_unused = float(lr)

        self.eps = float(eps)
        self.gamma = float(gamma)
        self.beta_base = float(beta_base)
        self.use_abs_cos = bool(use_abs_cos)
        self.aggregate = str(aggregate)
        if self.aggregate not in {"mean", "density_weighted"}:
            raise ValueError(f"DAGRBalancer: unknown aggregate='{self.aggregate}'")

        self.task_mag_quantile = float(task_mag_quantile)
        if not (0.0 <= self.task_mag_quantile <= 1.0):
            raise ValueError(f"DAGRBalancer: task_mag_quantile must be in [0,1], got {self.task_mag_quantile}")
        self.align_to_g0_norm = bool(align_to_g0_norm)

        self.conflict_gate = bool(conflict_gate)
        self.gate_cos_threshold = float(gate_cos_threshold)

        self.eta_max = float(eta_max)
        if self.eta_max < 0.0:
            raise ValueError(f"DAGRBalancer: eta_max must be >= 0, got {self.eta_max}")
        self.eta_ramp_steps = int(eta_ramp_steps)
        if self.eta_ramp_steps < 0:
            raise ValueError(f"DAGRBalancer: eta_ramp_steps must be >= 0, got {self.eta_ramp_steps}")

        self.min_cos_with_g0 = float(min_cos_with_g0)

        self.use_ema_v = bool(use_ema_v)
        self.ema_mu = float(ema_mu)
        if not (0.0 <= self.ema_mu < 1.0):
            raise ValueError(f"DAGRBalancer: ema_mu must be in [0,1), got {self.ema_mu}")

        self.lock_shared_params = bool(lock_shared_params)
        self.ddp_sync_v = bool(ddp_sync_v)

        # cached shared params identity
        self._shared_params: list[torch.Tensor] | None = None
        self._shared_id: list[int] | None = None

        # caches
        self._v_unit: torch.Tensor | None = None
        self._splits: list[int] | None = None
        self._conflict: float | None = None
        self._min_pair_cos: float | None = None
        self._step_v: int = -1

        # EMA cache
        self._v_ema: torch.Tensor | None = None

    def update(self, task_losses: dict[str, torch.Tensor], shared_params: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
        self.step += 1
        if not task_losses:
            return {}

        # ---- choose task names ----
        if self.task_names is None:
            names = list(task_losses.keys())
        else:
            names = [n for n in self.task_names if n in task_losses]
        if not names:
            any_loss = next(iter(task_losses.values()))
            return {k: torch.ones((), device=any_loss.device, dtype=any_loss.dtype) for k in task_losses.keys()}

        # ---- shared params list ----
        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            any_loss = next(iter(task_losses.values()))
            return {k: torch.ones((), device=any_loss.device, dtype=any_loss.dtype) for k in task_losses.keys()}

        if self.lock_shared_params:
            if self._shared_params is None:
                self._shared_params = list(shared_in)
                self._shared_id = [int(p.data_ptr()) for p in self._shared_params]
            else:
                cur = list(shared_in)
                cur_id = [int(p.data_ptr()) for p in cur]
                # 简单校验
                if len(cur_id) != len(self._shared_id):
                     # 如果参数变了（比如模型结构变了），这里为了稳健可以重置，或者报错
                     # 这里选择重置以防万一
                     self._shared_params = list(shared_in)
                     self._shared_id = [int(p.data_ptr()) for p in self._shared_params]
            shared = self._shared_params
        else:
            shared = shared_in

        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype

        # ---- warmup / interval ----
        if self.step <= self.warmup:
            self._v_unit = None
            self._splits = None
            self._conflict = None
            self._min_pair_cos = None
            self._step_v = -1
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            return self._last_weights

        if (self.step - 1) % self.interval != 0 and self._last_weights is not None:
            return self._last_weights

        # ---- compute per-task grads on shared params ----
        g_list: list[torch.Tensor] = []
        for n in names:
            grads = torch.autograd.grad(
                task_losses[n],
                shared,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            flat = self._flatten_grads(shared, grads, device=device)
            gn = torch.linalg.norm(flat).item()
            if not math.isfinite(gn) or gn <= self.eps:
                continue
            g_list.append(flat)

        if len(g_list) < 2:
            self._v_unit = None
            self._splits = [int(p.numel()) for p in shared]
            self._conflict = None
            self._min_pair_cos = None
            self._step_v = self.step
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            return self._last_weights

        G = torch.stack(g_list, dim=0)              # [T, D]
        g_norm = torch.linalg.norm(G, dim=1).clamp_min(self.eps)
        G_unit = G / g_norm[:, None]                # [T, D]

        # ---- conflict stats (pairwise cosine) ----
        P = torch.matmul(G_unit, G_unit.t()).clamp(-1.0, 1.0)
        T = int(P.shape[0])
        mask = ~torch.eye(T, dtype=torch.bool, device=P.device)
        min_pair_cos = float(P[mask].min().item()) if T > 1 else 1.0
        conflict = float(max(0.0, -min_pair_cos))
        self._min_pair_cos = min_pair_cos
        self._conflict = conflict

        # ---- density weights (rho) ----
        rho = self._compute_density_weights(G)      # [T] sum=1

        # ---- consensus direction d* ----
        d_star = (rho[:, None] * G_unit).sum(dim=0)
        d_star = d_star / torch.linalg.norm(d_star).clamp_min(self.eps)

        # ---- one-step analytic harmonization ----
        cos_sim = torch.mv(G_unit, d_star).clamp(-1.0, 1.0)  # [T]
        score = cos_sim.abs() if self.use_abs_cos else (1.0 - cos_sim) * 0.5
        beta = (self.beta_base * (1.0 - score)).clamp(0.0, 1.0).view(-1, 1)
        G_harm = (1.0 - beta) * G + beta * d_star.view(1, -1) * g_norm.view(-1, 1)

        # ---- aggregate ----
        if self.aggregate == "mean":
            v_raw = G_harm.mean(dim=0)
        else:
            v_raw = (rho[:, None] * G_harm).sum(dim=0)

        v_norm = torch.linalg.norm(v_raw).clamp_min(self.eps)
        v_unit = v_raw / v_norm

        # ---- optional EMA ----
        if self.use_ema_v:
            if self._v_ema is None or self._v_ema.numel() != v_unit.numel() or self._v_ema.device != v_unit.device:
                self._v_ema = v_unit.detach()
            else:
                self._v_ema = (self.ema_mu * self._v_ema + (1.0 - self.ema_mu) * v_unit).detach()
                self._v_ema = self._v_ema / torch.linalg.norm(self._v_ema).clamp_min(self.eps)
            v_unit = self._v_ema

        with torch.no_grad():
            self._v_unit = v_unit.detach()
            self._splits = [int(p.numel()) for p in shared]
            self._step_v = self.step

        self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
        return self._last_weights

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        if self._v_unit is None or self._splits is None:
            return

        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return

        if self.lock_shared_params and self._shared_params is not None:
            shared = self._shared_params
            # 这里的检查如果太严格可能会在模型动态调整时报错，已在 update 中做过校验
        else:
            shared = shared_in

        splits = [int(p.numel()) for p in shared]
        if sum(splits) != int(self._v_unit.numel()):
            # 可能是参数变了，本次跳过
            return

        # ---- flatten current grads (g0) ----
        g0 = self._flatten_current_grads(shared).to(dtype=torch.float32)
        g0_norm = torch.linalg.norm(g0).clamp_min(self.eps)

        # ---- build v (aligned magnitude) ----
        v = self._v_unit.to(device=g0.device, dtype=torch.float32)
        v = v * g0_norm

        # ---- DDP sync ----
        if self.ddp_sync_v and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            world = torch.distributed.get_world_size()
            if world > 1:
                v.div_(float(world))
            v_norm = torch.linalg.norm(v).clamp_min(self.eps)
            v = v / v_norm * g0_norm

        # ---- compute eta ----
        eta = self._compute_eta(g0=g0, v=v)
        if eta <= 0.0:
            return

        g_new = (1.0 - eta) * g0 + eta * v

        # ---- write back ----
        offset = 0
        for p, n in zip(shared, splits):
            piece = g_new[offset : offset + n].view_as(p)
            offset += n

            if p.grad is None:
                p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device)

            piece_cast = piece.to(dtype=p.grad.dtype, device=p.grad.device)
            p.grad.detach_()
            p.grad.copy_(piece_cast)

    # ----------------------- helpers -----------------------

    def _flatten_grads(self, shared: list[torch.Tensor], grads: tuple[torch.Tensor | None, ...], device: torch.device) -> torch.Tensor:
        flat = []
        for p, g in zip(shared, grads):
            if g is None:
                flat.append(torch.zeros((p.numel(),), device=device, dtype=torch.float32))
            else:
                flat.append(g.reshape(-1).to(device=device, dtype=torch.float32))
        return torch.cat(flat, dim=0)

    def _flatten_current_grads(self, shared: list[torch.Tensor]) -> torch.Tensor:
        flat = []
        device = shared[0].device
        for p in shared:
            if p.grad is None:
                flat.append(torch.zeros((p.numel(),), device=device, dtype=torch.float32))
            else:
                flat.append(p.grad.detach().reshape(-1).to(device=device, dtype=torch.float32))
        return torch.cat(flat, dim=0)

    def _compute_density_weights(self, G: torch.Tensor) -> torch.Tensor:
        l2 = torch.linalg.norm(G, dim=1).clamp_min(self.eps)
        l1 = G.abs().sum(dim=1)
        D = float(G.shape[1])
        density = (l1 / (l2 * math.sqrt(D) + self.eps)).clamp_min(0.0)

        rho = 1.0 / (1.0 + self.gamma * density)
        rho_sum = rho.sum()
        if not torch.isfinite(rho_sum) or float(rho_sum.item()) <= self.eps:
            # Fallback to uniform
            return torch.ones_like(rho) / rho.numel()
        return rho / rho_sum

    def _compute_eta(self, g0: torch.Tensor, v: torch.Tensor) -> float:
        min_pair_cos = self._min_pair_cos if self._min_pair_cos is not None else 1.0
        conflict = self._conflict if self._conflict is not None else 0.0

        if self.conflict_gate and min_pair_cos >= self.gate_cos_threshold:
            return 0.0

        base = min(1.0, max(0.0, float(conflict)))
        eta = self.eta_max * base

        if self.eta_ramp_steps > 0:
            t = max(0, self.step - self.warmup)
            ramp = min(1.0, t / float(self.eta_ramp_steps))
            eta *= ramp

        if eta <= 0.0:
            return 0.0

        g0n = torch.linalg.norm(g0).clamp_min(self.eps)
        vn = torch.linalg.norm(v).clamp_min(self.eps)
        cos_v_g0 = float(torch.dot(v, g0).div(g0n * vn).clamp(-1.0, 1.0).item())

        if cos_v_g0 < self.min_cos_with_g0:
            denom = (self.min_cos_with_g0 + 1.0)
            if denom <= 1e-6:
                return 0.0
            shrink = max(0.0, (cos_v_g0 + 1.0) / denom)
            eta *= shrink

        return float(max(0.0, min(1.0, eta)))



