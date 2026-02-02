# ultralytics/utils/gradbalance/rgb.py
from __future__ import annotations

import math
import torch
from typing import Iterable, Literal
from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class DAGR2Balancer(GradientBalancer):
    """
    DAGR v2.0 (Stable): Density-Aware Analytic Gradient Harmonization

    改进特性：
    1. Residual Blending: 避免完全破坏原始梯度的动量特性。
    2. Adaptive Beta: 降低冲突解决的激进程度，保护任务特异性特征。
    3. Lazy Evaluation: 允许跳过部分 step 的繁重计算以恢复训练速度。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,  # 建议设为 2 或 4 以提速
            warmup: int = 500,  # 增加 warmup 步数
            # ---- DA-AGH v2 args ----
            gamma: float = 0.2,  # 降低密度敏感度，避免对噪声过敏
            beta_base: float = 0.3,  # [关键] 大幅降低：从 0.8 降至 0.3，避免过度“拉架”
            quantile: float = 0.8,  # 模长恢复分位数
            blend_ratio: float = 1.0,  # 1.0 = 完全使用 DAGR 梯度 (v1行为); 0.5 = 50% 原梯度 + 50% DAGR
            rescale_grads: bool = True,  # 是否重新缩放模长
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

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

        # Engineering optimization: Pre-allocate buffer logic could go here

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
            self._v = None  # Clear cache means "do nothing this step"
            # Return ones to act as Identity
            return {k: torch.ones((), device=next(iter(task_losses.values())).device) for k in task_losses}

        # 2. Filter valid params
        shared = [p for p in shared_params if p.requires_grad]
        if not shared or not task_losses:
            return {}

        names = list(task_losses.keys())
        device = shared[0].device

        # 3. Compute Per-Task Gradients (The Expensive Part)
        # 优化：使用 no_grad 处理中间变量，只保留计算图边缘
        g_list = []

        for n in names:
            # retain_graph=True is unavoidable for PCGrad-like methods
            grads = torch.autograd.grad(
                task_losses[n],
                shared,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            # Flatten immediately to save reference overhead
            flat = self._flatten_grads(shared, grads)
            g_list.append(flat)

        if not g_list:
            return {k: torch.ones_like(v) for k, v in task_losses.items()}

        G = torch.stack(g_list, dim=0)  # [T, D]

        # 4. Density Calculation (Robust Version)
        # 原始 L1/L2 在高维下容易失效，增加 log 缩放或单纯依靠 L2 相对模长
        g_norms = torch.linalg.norm(G, dim=1).clamp_min(self.eps)

        # 计算每个任务梯度的“稀疏度”作为密度代理
        # 越稀疏 (L1 close to L2) -> 密度越高 -> 权重越低
        # 越稠密 (L1 >> L2) -> 信息熵高 -> 权重越高?
        # DAGR v1 逻辑: 密度高(冲突多) -> 降权。
        l1 = G.abs().sum(dim=1)
        D = G.shape[1]
        # Adding stability term
        density = l1 / (g_norms * math.sqrt(D) + self.eps)

        # Softmax based weighting allows for smoother competition
        # rho ~ exp(-gamma * density)
        rho_logits = -self.gamma * density
        rho = torch.softmax(rho_logits, dim=0)

        # 5. Consensus Direction d*
        # 加权平均方向
        G_unit = G / g_norms[:, None]
        d_star = (rho[:, None] * G_unit).sum(dim=0)
        d_star_norm = torch.linalg.norm(d_star)

        if d_star_norm <= self.eps:
            # Fallback: if consensus is zero, define d_star as mean
            d_star = G_unit.mean(dim=0)
        else:
            d_star = d_star / d_star_norm

        # 6. Harmonization (Analytic Rotation)
        # Cosine similarity
        cos_sim = torch.mv(G_unit, d_star).clamp(-1, 1)

        # Adaptive Beta: Only intervene when conflict is high (cos < 0)
        # If cos > 0 (aligned), let it be. If cos < 0 (conflict), pull back.
        # v1 logic: abs(cos). v2 logic: focus on negative.
        conflict_score = (1.0 - cos_sim).clamp(0, 2) / 2.0  # 0(aligned) -> 1(opposed)
        beta = self.beta_base * conflict_score  # [T]

        beta = beta.view(-1, 1)

        # v_i' = (1-beta)g_i + beta * ||g_i|| * d*
        G_harm = (1.0 - beta) * G + beta * g_norms.view(-1, 1) * d_star.view(1, -1)

        # 7. Aggregation
        v_final = G_harm.mean(dim=0)

        # 8. Magnitude Restoration
        # 使用简单的均值恢复通常比 quantile 更稳定，除非有极端的离群任务
        current_mag = torch.linalg.norm(v_final).clamp_min(self.eps)
        target_mag = torch.quantile(g_norms, self.quantile).clamp_min(self.eps)

        if self.rescale_grads:
            v_final = v_final * (target_mag / current_mag)

        self._v = v_final.detach()  # Cache logic
        self._splits = [p.numel() for p in shared]

        # Return ones because we handle grads in apply_after_unscale
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

        # DDP Sync Logic
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._v, op=torch.distributed.ReduceOp.SUM)
            self._v /= torch.distributed.get_world_size()

        # Distribution to params
        offset = 0
        v_flat = self._v

        for p, numel in zip(shared, self._splits):
            if p.grad is None:
                offset += numel
                continue

            # Slice the calculated DAGR gradient
            dagr_grad = v_flat[offset: offset + numel].view_as(p)
            offset += numel

            # --- Key Fix: Residual Blending ---
            # Don't just overwrite. Blend with the naturally accumulated gradient.
            # p.grad contains the 'natural' gradient from total_loss.backward()

            if self.blend_ratio >= 1.0:
                p.grad.copy_(dagr_grad)
            else:
                # p.grad = (1 - ratio) * orig + ratio * dagr
                p.grad.mul_(1.0 - self.blend_ratio).add_(dagr_grad, alpha=self.blend_ratio)

        # Release memory
        self._v = None

    def _flatten_grads(self, params, grads):
        # Optimized flattening
        return torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
            for p, g in zip(params, grads)
        ])