# ultralytics/utils/gradbalance/rgb.py
from __future__ import annotations

import math
from typing import Iterable, Literal

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class RGBBalancer(GradientBalancer):
    """
    RGB (Rotation-Based Gradient Balancing) for shared-parameter gradient surgery.

    你当前这个类是“覆盖 shared grads 的手术刀”，不是 loss-weighting。
    正确调用顺序（单卡 / DDP 都适用）：
      1) forward 得到 task_losses（每个任务一个标量 loss，保留计算图）
      2) rgb.update(task_losses, shared_params=backbone_params)   # 必须在 backward 之前
      3) total_loss.backward()                                   # 正常 backward（不需要 retain_graph）
      4) scaler.unscale_(optimizer)
      5) rgb.apply_after_unscale(shared_params=backbone_params)   # 覆盖 shared grads（DDP 下会内部 all_reduce v）
      6) clip_grad / scaler.step / scaler.update

    设计要点：
      - 对每个任务的 shared-grad 做单位化，避免幅度主导方向；
      - 用 EMA 维护共识方向 d_t；
      - 在 span(g_i, d_t) 的二维子空间内为每个任务优化旋转角 alpha_i；
      - 用旋转后的 r_i 均值作为 shared 更新方向 v；
      - 关键修复：将 v_direction 重新按梯度幅度 target_mag 做 rescale（否则训练会“没劲”或步幅怪异）；
      - 工程硬雷：锁定 shared 参数对象引用与顺序，避免张冠李戴；AMP 下按 p.grad.dtype 写回；
      - DDP 硬雷：覆盖梯度会绕开 DDP 同步，所以必须对 v 做 all_reduce 再写回。
    """

    def __init__(
        self,
        task_names: list[str] | None = None,
        interval: int = 1,
        warmup: int = 0,
        alpha: float = 0.5,  # λ: proximity 权重（越大越保守）
        lr: float = 0.1,  # 内部优化 alpha_i 的 lr（不是模型 lr）
        mu: float = 0.9,  # EMA 系数
        steps: int = 10,  # 每次更新 alpha_i 的内循环步数
        eps: float = 1e-8,
        magnitude: Literal["rms", "mean", "max"] = "rms",  # v 的尺度恢复策略
        alpha_min: float = 0.0,
        alpha_max: float = 0.5 * math.pi,
        lock_shared_params: bool = True,  # 是否在首次 update 时锁定 shared 参数引用与顺序
        ddp_sync_v: bool = True,  # DDP 下是否对 v 做 all_reduce 同步
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
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

        # caches
        self._dt: torch.Tensor | None = None  # EMA consensus direction (flat, float32)
        self._v: torch.Tensor | None = None  # latest shared update direction (flat, float32)
        self._splits: list[int] | None = None  # per-param numel splits for shared params

        # engineering safety: lock shared parameter identity + order
        self._shared_params: list[torch.Tensor] | None = None
        self._shared_id: list[int] | None = None  # data_ptr list for identity check

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        必须在 total_loss.backward() 之前调用。
        这里只计算并缓存 v（shared 更新方向），不直接写回 grads。
        写回发生在 apply_after_unscale()。
        """
        self.step += 1
        if not task_losses:
            return {}

        # ---- select task names (avoid silent mismatch) ----
        if self.task_names is None:
            names = list(task_losses.keys())
        else:
            names = [n for n in self.task_names if n in task_losses]
        if not names:
            any_loss = next(iter(task_losses.values()))
            return {k: torch.ones((), device=any_loss.device, dtype=any_loss.dtype) for k in task_losses.keys()}

        # ---- build / lock shared param list ----
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
                if len(cur_id) != len(self._shared_id) or any(a != b for a, b in zip(cur_id, self._shared_id)):
                    raise RuntimeError(
                        "RGBBalancer: shared_params identity/order mismatch. "
                        "You must pass the exact same shared parameter list in the same order every time, "
                        "or enable lock_shared_params and keep your callsite consistent."
                    )
            shared = self._shared_params
        else:
            shared = shared_in

        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype

        # ---- warmup / interval ----
        if self.step <= self.warmup:
            self._v = None
            self._splits = None
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            self._update_dt(names, task_losses, shared)
            return self._last_weights

        if (self.step - 1) % self.interval != 0 and self._last_weights is not None:
            return self._last_weights

        # ---- compute per-task gradients on shared params (requires graph alive) ----
        g_list: list[torch.Tensor] = []
        used_names: list[str] = []
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
            used_names.append(n)

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
                wi = self._deterministic_orthogonal(gi)
                wi_norm = torch.linalg.norm(wi).clamp_min(self.eps)
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
        """
        必须在 scaler.unscale_(optimizer) 之后、clip_grad/optimizer.step 之前调用。
        覆盖 shared grads 为 RGB 的 v；DDP 下会先同步 v 再覆盖。
        """
        if self._v is None or self._splits is None:
            return

        # build / verify shared list (same order)
        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return

        if self.lock_shared_params and self._shared_params is not None:
            # 强制使用锁定列表，避免外部传参顺序不一致
            shared = self._shared_params
            cur_id = [int(p.data_ptr()) for p in shared_in]
            if len(cur_id) != len(self._shared_id) or any(a != b for a, b in zip(cur_id, self._shared_id)):
                raise RuntimeError(
                    "RGBBalancer: shared_params identity/order mismatch at apply_after_unscale(). "
                    "Do not change shared param list/order between update() and apply_after_unscale()."
                )
        else:
            shared = shared_in

        splits = [int(p.numel()) for p in shared]
        if sum(splits) != int(self._v.numel()):
            raise RuntimeError(
                f"RGBBalancer: split mismatch, sum(splits)={sum(splits)} vs v.numel()={self._v.numel()}"
            )

        v = self._v.to(device=shared[0].device, dtype=torch.float32)

        # ---- DDP sync: overwrite bypasses DDP reduction, so we must sync v ourselves ----
        if self.ddp_sync_v and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            world = torch.distributed.get_world_size()
            if world > 1:
                v.div_(float(world))

        # ---- write back to grads (respect AMP dtype) ----
        offset = 0
        for p, n in zip(shared, splits):
            piece = v[offset : offset + n].view_as(p)
            offset += n

            if p.grad is None:
                # 如果 shared 参数梯度为空，直接创建
                p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device)

            # 注意：unscale 后 p.grad 的 dtype 可能是 fp16/bf16/fp32，必须按其 dtype 写回
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

    def _deterministic_orthogonal(self, g: torch.Tensor) -> torch.Tensor:
        """
        构造一个确定性正交向量：选最不对齐的基向量 e_k，再做投影去除。
        使用通用投影公式，避免 g 非严格单位向量时的数值问题。
        """
        abs_g = g.abs()
        k = int(torch.argmin(abs_g).item())
        e = torch.zeros_like(g)
        e[k] = 1.0
        gg = torch.dot(g, g).clamp_min(self.eps)
        u = e - (torch.dot(e, g) / gg) * g
        u = u / torch.linalg.norm(u).clamp_min(self.eps)
        return u

    def _update_dt(self, names: list[str], task_losses: dict[str, torch.Tensor], shared: list[torch.Tensor]) -> None:
        """
        warmup 时用于更新 dt，避免 warmup 结束后 dt 从零开始剧烈抖动。
        这里使用按梯度范数加权的均值，降低噪声任务把 dt 拉偏的概率。
        """
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
        w_task = (g_norm / g_norm.sum().clamp_min(self.eps)).detach()
        Gbar = G / g_norm[:, None]

        y = (w_task[:, None] * Gbar).sum(dim=0)
        y = y / torch.linalg.norm(y).clamp_min(self.eps)

        if self._dt is None or self._dt.numel() != y.numel() or self._dt.device != y.device:
            self._dt = y.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * y).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)

