# ultralytics/utils/gradbanlance/DAGR.py
from __future__ import annotations

import math
from typing import Iterable, Literal

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class DAGRBalancer(GradientBalancer):
    """
    Density-Aware Analytic Gradient Harmonization (DA-AGH)
    修复版：解决了 Interval 期间旧梯度残留导致训练崩溃的问题。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # ---- legacy RGB args ----
            alpha: float = 0.5,
            lr: float = 0.1,
            mu: float = 0.9,
            steps: int = 10,
            eps: float = 1e-8,
            magnitude: Literal["rms", "mean", "max"] = "rms",
            alpha_min: float = 0.0,
            alpha_max: float = 0.5 * math.pi,
            # ---- engineering ----
            lock_shared_params: bool = True,
            ddp_sync_v: bool = True,
            # ---- DA-AGH args ----
            gamma: float = 0.5,
            beta_base: float = 0.8,
            quantile: float = 0.75,
            use_abs_cos: bool = True,
            aggregate: Literal["mean", "density_weighted"] = "mean",
            use_ema_consensus: bool = False,
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        # legacy kept
        self.lam = float(alpha)
        self.alpha_lr = float(lr)
        self.mu = float(mu)
        self.steps = int(steps)

        self.eps = float(eps)
        self.magnitude = str(magnitude)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if not (self.alpha_min <= self.alpha_max):
            raise ValueError(f"RGBBalancer: alpha_min must <= alpha_max, got {self.alpha_min} > {self.alpha_max}")

        self.lock_shared_params = bool(lock_shared_params)
        self.ddp_sync_v = bool(ddp_sync_v)

        # DA-AGH
        self.gamma = float(gamma)
        self.beta_base = float(beta_base)
        self.quantile = float(quantile)
        if not (0.0 <= self.quantile <= 1.0):
            raise ValueError(f"RGBBalancer(DA-AGH): quantile must be in [0,1], got {self.quantile}")
        if self.beta_base < 0.0:
            raise ValueError(f"RGBBalancer(DA-AGH): beta_base must be >= 0, got {self.beta_base}")

        self.use_abs_cos = bool(use_abs_cos)
        self.aggregate = str(aggregate)
        if self.aggregate not in {"mean", "density_weighted"}:
            raise ValueError(f"RGBBalancer(DA-AGH): unknown aggregate='{self.aggregate}'")
        self.use_ema_consensus = bool(use_ema_consensus)

        # caches
        self._dt: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._splits: list[int] | None = None

        # engineering safety
        self._shared_params: list[torch.Tensor] | None = None
        self._shared_id: list[int] | None = None

    def compute_density_weight(self, G: torch.Tensor) -> torch.Tensor:
        T, D = G.shape
        if D == 0:
            return torch.ones(T, device=G.device) / T

        g_l1 = torch.linalg.norm(G, ord=1, dim=1)
        g_l2 = torch.linalg.norm(G, ord=2, dim=1).clamp_min(1e-8)
        density = g_l1 / (g_l2 * math.sqrt(D))
        raw_weights = (1.0 - density).clamp_min(1e-6)
        rho_weights = raw_weights / raw_weights.sum()
        return rho_weights

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        RGB 核心更新逻辑 (已修复 Interval 缓存污染问题)
        必须在 total_loss.backward() 之前调用。
        """
        self.step += 1
        if not task_losses:
            return {}

        # ---- select task names ----
        if self.task_names is None:
            names = list(task_losses.keys())
        else:
            names = [n for n in self.task_names if n in task_losses]

        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype

        if not names:
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}

        # ---- build / lock shared param list ----
        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}

        if self.lock_shared_params:
            if self._shared_params is None:
                self._shared_params = list(shared_in)
                self._shared_id = [int(p.data_ptr()) for p in self._shared_params]
            else:
                cur = list(shared_in)
                cur_id = [int(p.data_ptr()) for p in cur]
                # 简单校验，如果变了就不做处理防止报错
                if len(cur_id) != len(self._shared_id):
                    self._shared_params = list(shared_in)
            shared = self._shared_params
        else:
            shared = shared_in

        # ================= [FIX START] =================
        # 修复逻辑：在不计算梯度的步骤（Warmup 或 Interval 跳过时），
        # 必须显式清空 _v，防止 apply_after_unscale 使用旧的梯度方向覆盖当前梯度。

        # 1. Warmup 检查
        if self.step <= self.warmup:
            self._v = None
            self._splits = None
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            # 即使在 warmup，也可以选择性更新 EMA 方向 dt，这里保持简单略过
            return self._last_weights

        # 2. Interval 检查
        if (self.step - 1) % self.interval != 0:
            # 关键修复：跳过计算时，必须清空缓存的 v
            self._v = None
            self._splits = None
            if self._last_weights is not None:
                return self._last_weights
            else:
                return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
        # ================= [FIX END] =================

        # ---- compute per-task gradients on shared params (requires graph alive) ----
        g_list: list[torch.Tensor] = []
        for n in names:
            grads = torch.autograd.grad(
                task_losses[n],
                shared,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            flat = self._flatten_grads(shared, grads, device=device)  # float32 flat
            gn = torch.linalg.norm(flat).item()
            if not math.isfinite(gn) or gn <= self.eps:
                continue
            g_list.append(flat)

        if not g_list:
            self._v = None
            self._splits = None
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            return self._last_weights

        G = torch.stack(g_list, dim=0)  # [T, D] float32
        t = int(G.shape[0])

        # 单任务旁路：无需 RGB
        if t < 2:
            self._v = G[0].detach()
            self._splits = [int(p.numel()) for p in shared]
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            return self._last_weights

        # ---- baseline sum gradient (shared params) ----
        g_sum = G.sum(dim=0)  # [D]
        base_mag = torch.linalg.norm(g_sum).clamp_min(self.eps)

        # ---- magnitude + direction weights (down-weight noisy/small task grads) ----
        g_norm = torch.linalg.norm(G, dim=1).clamp_min(self.eps)  # [T]
        w_task = (g_norm / g_norm.sum().clamp_min(self.eps)).detach()  # [T]

        # ---- normalize gradients to unit vectors ----
        Gbar = G / g_norm[:, None]  # [T, D]

        # ---- update EMA consensus direction d_t (weighted) ----
        y = (w_task[:, None] * Gbar).sum(dim=0)
        y = y / torch.linalg.norm(y).clamp_min(self.eps)
        if self._dt is None or self._dt.numel() != y.numel() or self._dt.device != y.device:
            self._dt = y.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * y).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)

        dt = self._dt

        # ---- build orthogonal directions w_i in span(g_i, d_t) ----
        W = []
        for i in range(t):
            gi = Gbar[i]
            proj = torch.dot(dt, gi)
            wi = dt - proj * gi
            wi_norm = torch.linalg.norm(wi)
            if wi_norm <= self.eps or not torch.isfinite(wi_norm):
                # fallback if parallel
                wi = torch.zeros_like(gi)
            else:
                wi = wi / wi_norm.clamp_min(self.eps)
            W.append(wi)
        W = torch.stack(W, dim=0)  # [T, D]

        # ---- optimize rotation angles alpha_i ----
        alphas = torch.zeros((t,), device=device, dtype=torch.float32, requires_grad=True)
        inner_steps = max(int(self.steps), 1)

        for _ in range(inner_steps):
            R = torch.cos(alphas)[:, None] * Gbar + torch.sin(alphas)[:, None] * W
            R = R / torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(self.eps)

            cosmat = R @ R.t()
            iu = torch.triu_indices(t, t, offset=1, device=device)
            cos_ij = cosmat[iu[0], iu[1]]
            conflict = ((1.0 - cos_ij) * 0.5).mean()

            prox = ((R - Gbar).pow(2).sum(dim=1) * 0.25).mean()
            obj = conflict + self.lam * prox

            (grad_alpha,) = torch.autograd.grad(obj, (alphas,), retain_graph=False, create_graph=False)
            with torch.no_grad():
                alphas -= self.alpha_lr * grad_alpha
                alphas.clamp_(self.alpha_min, self.alpha_max)
            alphas.requires_grad_(True)

        # ---- final shared update direction: conflict-gated blend with baseline ----
        with torch.no_grad():
            R = torch.cos(alphas)[:, None] * Gbar + torch.sin(alphas)[:, None] * W
            R = R / torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(self.eps)

            # final conflict (0..1)
            cosmat = R @ R.t()
            iu = torch.triu_indices(t, t, offset=1, device=device)
            cos_ij = cosmat[iu[0], iu[1]]
            conflict_final = ((1.0 - cos_ij) * 0.5).mean().clamp(0.0, 1.0)

            # weighted direction
            v_dir = (w_task[:, None] * R).sum(dim=0)
            v_dir = v_dir / torch.linalg.norm(v_dir).clamp_min(self.eps)

            # RGB proposal with baseline magnitude
            v_rgb = v_dir * base_mag

            # gate: only intervene when conflict is non-trivial
            tau = 0.25  # 经验阈值：冲突小于 tau 时几乎等价基线
            beta = (conflict_final / tau).clamp(0.0, 1.0)

            v_mix = (1.0 - beta) * g_sum + beta * v_rgb
            v_mix_norm = torch.linalg.norm(v_mix).clamp_min(self.eps)
            v = v_mix / v_mix_norm * base_mag  # keep step size comparable to baseline

            self._v = v.detach()
            self._splits = [int(p.numel()) for p in shared]

        self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
        return self._last_weights

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        if self._v is None or self._splits is None:
            return

        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return

        if self.lock_shared_params and self._shared_params is not None:
            shared = self._shared_params
            cur_id = [int(p.data_ptr()) for p in shared_in]
            if len(cur_id) != len(self._shared_id):
                return
        else:
            shared = shared_in

        splits = [int(p.numel()) for p in shared]
        if sum(splits) != int(self._v.numel()):
            return

        v = self._v.to(device=shared[0].device, dtype=torch.float32)

        if self.ddp_sync_v and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            world = torch.distributed.get_world_size()
            if world > 1:
                v.div_(float(world))

        offset = 0
        for p, n in zip(shared, splits):
            piece = v[offset: offset + n].view_as(p)
            offset += n
            if p.grad is None:
                p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device)
            piece_cast = piece.to(dtype=p.grad.dtype, device=p.grad.device)
            p.grad.detach_()
            p.grad.copy_(piece_cast)

    def _flatten_grads(
            self,
            shared: list[torch.Tensor],
            grads: tuple[torch.Tensor | None, ...],
            device: torch.device,
    ) -> torch.Tensor:
        flat = []
        for p, g in zip(shared, grads):
            if g is None:
                flat.append(torch.zeros((p.numel(),), device=device, dtype=torch.float32))
            else:
                flat.append(g.reshape(-1).to(device=device, dtype=torch.float32))
        return torch.cat(flat, dim=0)

    def _compute_target_magnitude(self, mags: torch.Tensor) -> torch.Tensor:
        if math.isnan(self.quantile):
            return self._legacy_target_magnitude(mags)
        q = float(self.quantile)
        if q <= 0.0:
            return mags.min().clamp_min(self.eps)
        if q >= 1.0:
            return mags.max().clamp_min(self.eps)
        target = torch.quantile(mags, q).clamp_min(self.eps)
        if not torch.isfinite(target):
            return self._legacy_target_magnitude(mags)
        return target

    def _legacy_target_magnitude(self, mags: torch.Tensor) -> torch.Tensor:
        if self.magnitude == "rms":
            return torch.sqrt((mags * mags).mean()).clamp_min(self.eps)
        if self.magnitude == "mean":
            return mags.mean().clamp_min(self.eps)
        if self.magnitude == "max":
            return mags.max().clamp_min(self.eps)
        raise ValueError(f"RGBBalancer(DA-AGH): unknown magnitude='{self.magnitude}'")

    def _update_dt_from_direction(self, d: torch.Tensor) -> None:
        if self._dt is None or self._dt.numel() != d.numel() or self._dt.device != d.device:
            self._dt = d.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * d).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)

    def _update_dt(self, names: list[str], task_losses: dict[str, torch.Tensor], shared: list[torch.Tensor]) -> None:
        g_list = []
        device = next(iter(task_losses.values())).device
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

        if not g_list:
            return

        G = torch.stack(g_list, dim=0)
        g_norm = torch.linalg.norm(G, dim=1).clamp_min(self.eps)
        G_unit = G / g_norm[:, None]
        rho = self.compute_density_weight(G)
        d_star = (rho[:, None] * G_unit).sum(dim=0)
        d_star = d_star / torch.linalg.norm(d_star).clamp_min(self.eps)
        self._update_dt_from_direction(d_star)