# ultralytics/utils/gradbalance/rgb.py (或你存放 DAGR2Balancer 的文件)
from __future__ import annotations

import math
import torch
from typing import Iterable, Literal
from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class DAGR2Balancer(GradientBalancer):
    """
    DAGR v2.0 (Stable): Density-Aware Analytic Gradient Harmonization

    修复版：
    1. [Fix] 添加了 alpha, lr, **kwargs 以修复 train.py 传参导致的初始化报错。
    2. Residual Blending: 避免完全破坏原始梯度的动量特性。
    3. Adaptive Beta: 降低冲突解决的激进程度。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 500,
            # ---- 兼容性修复 (train.py 默认会传这些) ----
            alpha: float = 0.5,  # 接收但不使用，防止报错
            lr: float = 0.1,     # 接收但不使用，防止报错
            # ---- DA-AGH v2 args ----
            gamma: float = 0.2,
            beta_base: float = 0.3,
            quantile: float = 0.8,
            blend_ratio: float = 1.0,
            rescale_grads: bool = True,
            **kwargs  # [关键] 吸收所有其他未定义的参数
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        # 存储兼容性参数 (虽然 DAGR2 不直接用它们，但存下来是个好习惯)
        self.alpha = float(alpha)
        self.lr = float(lr)

        # Core Hyperparameters
        self.gamma = float(gamma)
        self.beta_base = float(beta_base)
        self.quantile = float(quantile)
        self.blend_ratio = float(blend_ratio)
        self.rescale_grads = bool(rescale_grads)
        self.eps = 1e-6

        # Caches
        self._v: torch.Tensor | None = None
        self._splits: list[int] | None = None

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        计算阶段：在 total_loss.backward() 之前运行。
        """
        self.step += 1

        # 1. Check intervals and warmup to save time (Speedup Trick)
        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            return {k: torch.ones((), device=next(iter(task_losses.values())).device) for k in task_losses}

        # 2. Filter valid params
        shared = [p for p in shared_params if p.requires_grad]
        if not shared or not task_losses:
            return {}

        names = list(task_losses.keys())
        device = shared[0].device

        # 3. Compute Per-Task Gradients
        g_list = []

        for n in names:
            grads = torch.autograd.grad(
                task_losses[n],
                shared,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            flat = self._flatten_grads(shared, grads)
            g_list.append(flat)

        if not g_list:
            return {k: torch.ones_like(v) for k, v in task_losses.items()}

        G = torch.stack(g_list, dim=0)  # [T, D]

        # 4. Density Calculation
        g_norms = torch.linalg.norm(G, dim=1).clamp_min(self.eps)

        l1 = G.abs().sum(dim=1)
        D = G.shape[1]
        density = l1 / (g_norms * math.sqrt(D) + self.eps)

        rho_logits = -self.gamma * density
        rho = torch.softmax(rho_logits, dim=0)

        # 5. Consensus Direction d*
        G_unit = G / g_norms[:, None]
        d_star = (rho[:, None] * G_unit).sum(dim=0)
        d_star_norm = torch.linalg.norm(d_star)

        if d_star_norm <= self.eps:
            d_star = G_unit.mean(dim=0)
        else:
            d_star = d_star / d_star_norm

        # 6. Harmonization
        cos_sim = torch.mv(G_unit, d_star).clamp(-1, 1)

        # Adaptive Beta
        conflict_score = (1.0 - cos_sim).clamp(0, 2) / 2.0
        beta = self.beta_base * conflict_score
        beta = beta.view(-1, 1)

        # v_i' = (1-beta)g_i + beta * ||g_i|| * d*
        G_harm = (1.0 - beta) * G + beta * g_norms.view(-1, 1) * d_star.view(1, -1)

        # 7. Aggregation
        v_final = G_harm.mean(dim=0)

        # 8. Magnitude Restoration
        current_mag = torch.linalg.norm(v_final).clamp_min(self.eps)
        target_mag = torch.quantile(g_norms, self.quantile).clamp_min(self.eps)

        if self.rescale_grads:
            v_final = v_final * (target_mag / current_mag)

        self._v = v_final.detach()
        self._splits = [p.numel() for p in shared]

        return {k: torch.ones((), device=device) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        """
        Gradient Surgery 实施阶段
        """
        if self._v is None:
            return

        shared = [p for p in shared_params if p.requires_grad]
        if not shared:
            return

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._v, op=torch.distributed.ReduceOp.SUM)
            self._v /= torch.distributed.get_world_size()

        offset = 0
        v_flat = self._v

        for p, numel in zip(shared, self._splits):
            if p.grad is None:
                offset += numel
                continue

            dagr_grad = v_flat[offset: offset + numel].view_as(p)
            offset += numel

            if self.blend_ratio >= 1.0:
                p.grad.copy_(dagr_grad)
            else:
                p.grad.mul_(1.0 - self.blend_ratio).add_(dagr_grad, alpha=self.blend_ratio)

        self._v = None

    def _flatten_grads(self, params, grads):
        return torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
            for p, g in zip(params, grads)
        ])