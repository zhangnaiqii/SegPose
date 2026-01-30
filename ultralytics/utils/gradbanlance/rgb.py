from __future__ import annotations

import math
from typing import Iterable

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class RGBBalancer(GradientBalancer):
    """
    RGB (Rotation-Based Gradient Balancing) for shared-parameter gradient surgery.

    设计要点（对齐论文核心套路）：
      1) 对每个任务的 shared-grad 做归一化，避免幅度主导方向；
      2) 用 EMA 维护一个“共识方向” d_t；
      3) 在 span(g_i, d_t) 的二维子空间内，为每个任务学一个旋转角 alpha_i：
            r_i = cos(alpha_i)*g_i + sin(alpha_i)*w_i
         其中 w_i 是 g_i 在该平面内的正交方向；
      4) 用 r_i 的平均作为 shared 参数的更新方向 v，并在 optimizer step 前覆盖 shared grads。

    参数映射（兼容你现有 CLI）：
      - alpha: 论文目标里的 proximity 权重 λ（越大越保守，越像“别乱转”）
      - lr:    内部优化 alpha_i 的学习率（不是模型 lr）
      - interval/warmup: 复用你现有机制
    """

    def __init__(
        self,
        task_names: list[str] | None = None,
        interval: int = 1,
        warmup: int = 0,
        alpha: float = 0.5,   # λ
        lr: float = 0.1,      # alpha_i 优化步长
        mu: float = 0.9,      # EMA for d_t
        steps: int = 10,      # 每次更新 alpha_i 的内循环步数
        eps: float = 1e-8,
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
        self.lam = float(alpha)
        self.alpha_lr = float(lr)
        self.mu = float(mu)
        self.steps = int(steps)
        self.eps = float(eps)

        self._dt: torch.Tensor | None = None          # EMA consensus direction (flat)
        self._v: torch.Tensor | None = None           # latest shared update direction (flat)
        self._splits: list[int] | None = None         # per-param numel splits for shared params
        self._device: torch.device | None = None

    def update(
        self, task_losses: dict[str, torch.Tensor], shared_params: Iterable[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self.step += 1
        if not task_losses:
            return {}

        # names: keep intersection to avoid silent mismatch
        if self.task_names is None:
            names = list(task_losses.keys())
        else:
            names = [n for n in self.task_names if n in task_losses]
        if not names:
            # fall back to all-ones
            return {k: torch.ones((), device=v.device, dtype=v.dtype) for k, v in task_losses.items()}

        shared = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared:
            return {k: torch.ones((), device=v.device, dtype=v.dtype) for k, v in task_losses.items()}

        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype
        self._device = device

        # interval caching
        if self.step <= self.warmup:
            self._v = None
            self._splits = None
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses}
            # 仍然更新 d_t（让 warmup 结束后不至于从零开始抖）
            self._update_dt(names, task_losses, shared)
            return self._last_weights

        if (self.step - 1) % self.interval != 0 and self._last_weights is not None:
            return self._last_weights

        # --- compute normalized task gradients on shared params ---
        g_list = []
        for n in names:
            grads = torch.autograd.grad(
                task_losses[n],
                shared,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            flat = self._flatten_grads(shared, grads, device=device).float()
            g_list.append(flat)

        G = torch.stack(g_list, dim=0)  # [T, D]
        g_norm = torch.linalg.norm(G, dim=1).clamp_min(self.eps)
        Gbar = G / g_norm[:, None]      # normalized gradients

        # --- update EMA consensus direction d_t ---
        y = Gbar.mean(dim=0)
        y = y / torch.linalg.norm(y).clamp_min(self.eps)
        if self._dt is None or self._dt.numel() != y.numel() or self._dt.device != y.device:
            self._dt = y.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * y).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)

        # --- build orthogonal directions w_i in span(g_i, d_t) ---
        dt = self._dt
        W = []
        for i in range(Gbar.shape[0]):
            gi = Gbar[i]
            proj = torch.dot(dt, gi)
            wi = dt - proj * gi
            wi_norm = torch.linalg.norm(wi)
            if wi_norm <= self.eps:
                wi = self._deterministic_orthogonal(gi)
                wi_norm = torch.linalg.norm(wi).clamp_min(self.eps)
            wi = wi / wi_norm
            W.append(wi)
        W = torch.stack(W, dim=0)  # [T, D]

        # --- optimize rotation angles alpha_i ---
        alphas = torch.zeros((Gbar.shape[0],), device=device, dtype=torch.float32, requires_grad=True)

        for _ in range(max(self.steps, 1)):
            R = torch.cos(alphas)[:, None] * Gbar + torch.sin(alphas)[:, None] * W
            R = R / torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(self.eps)

            # conflict term: mean_{i<j} (1 - cos)/2
            cosmat = R @ R.t()
            t = R.shape[0]
            if t > 1:
                iu = torch.triu_indices(t, t, offset=1, device=device)
                cos_ij = cosmat[iu[0], iu[1]]
                conflict = ((1.0 - cos_ij) * 0.5).mean()
            else:
                conflict = torch.zeros((), device=device, dtype=torch.float32)

            # proximity term: mean_i ||r_i - g_i||^2 / 4
            prox = ((R - Gbar).pow(2).sum(dim=1) * 0.25).mean()

            obj = conflict + self.lam * prox

            (grad_alpha,) = torch.autograd.grad(obj, (alphas,), retain_graph=False, create_graph=False)
            with torch.no_grad():
                alphas -= self.alpha_lr * grad_alpha
                alphas.clamp_(0.0, 0.5 * math.pi)
            alphas.requires_grad_(True)

        # --- final shared update direction v ---
        with torch.no_grad():
            R = torch.cos(alphas)[:, None] * Gbar + torch.sin(alphas)[:, None] * W
            R = R / torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(self.eps)
            v = R.mean(dim=0)  # [D]
            self._v = v.detach()

            # cache splits for apply_after_unscale
            self._splits = [int(p.numel()) for p in shared]

        # weights: keep loss computation unchanged (RGB acts in optimizer step)
        self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses}
        return self._last_weights

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        """
        Call this AFTER scaler.unscale_(optimizer), BEFORE grad clipping / optimizer.step().
        It overwrites shared parameter grads with RGB direction v.
        """
        if self._v is None or self._splits is None:
            return

        shared = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared:
            return

        # recompute splits defensively if mismatch
        splits = [int(p.numel()) for p in shared]
        if sum(splits) != int(self._v.numel()):
            # cannot safely reshape, better crash loudly than corrupt training
            raise RuntimeError(f"RGBBalancer: split mismatch, sum(splits)={sum(splits)} vs v.numel()={self._v.numel()}")

        v = self._v.to(device=shared[0].device, dtype=torch.float32)
        offset = 0
        for p, n in zip(shared, splits):
            piece = v[offset : offset + n].view_as(p)
            offset += n
            if p.grad is None:
                p.grad = piece.clone()
            else:
                p.grad.detach_()
                p.grad.copy_(piece)

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
        # pick a basis vector e_k that is least aligned with g, then orthogonalize
        abs_g = g.abs()
        k = int(torch.argmin(abs_g).item())
        e = torch.zeros_like(g)
        e[k] = 1.0
        u = e - torch.dot(e, g) * g
        u = u / torch.linalg.norm(u).clamp_min(self.eps)
        return u

    def _update_dt(self, names, task_losses, shared):
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
            flat = self._flatten_grads(shared, grads, device=device).float()
            g_list.append(flat)
        G = torch.stack(g_list, dim=0)
        g_norm = torch.linalg.norm(G, dim=1).clamp_min(self.eps)
        Gbar = G / g_norm[:, None]
        y = Gbar.mean(dim=0)
        y = y / torch.linalg.norm(y).clamp_min(self.eps)
        if self._dt is None or self._dt.numel() != y.numel() or self._dt.device != y.device:
            self._dt = y.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * y).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)
