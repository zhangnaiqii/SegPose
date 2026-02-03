# ultralytics/utils/gradbanlance/rgb.py
from __future__ import annotations

import math
import torch
from typing import Iterable, Literal
from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class RGBBalancer(GradientBalancer):
    """
    RGB v2.0 (Robust Gradient Balance)
    修复改进版：
    1. 兼容性修复：接受并处理 alpha/lr 参数，防止初始化报错。
    2. MagBal (Magnitude Balancing): 默认开启，强制拉平各任务梯度模长，解决 Seg 被淹没的问题。
    3. Lazy Conflict Resolution: 仅惩罚反向冲突，完全容忍正交梯度。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # ---- 兼容 train.py 的传参 ----
            alpha: float = 0.5,  # 实际上对于 RGB 这是一个保留参数
            lr: float = 0.1,  # 同上
            # ---- RGB v2 核心参数 ----
            blend_ratio: float = 1.0,  # 1.0=全用新梯度
            mag_balance: bool = True,  # [关键] 默认开启量级平衡
            conflict_threshold: float = 0.0,  # 0.0 表示只处理 >90度的冲突
            rescale_grads: bool = True,  # 保持总体梯度尺度不变
            eps: float = 1e-8,
            **kwargs  # 吸收多余参数
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        # 存储参数 (虽然部分可能不直接用于计算，但为了兼容性保留)
        self.alpha = float(alpha)
        self.lr = float(lr)

        self.blend_ratio = float(blend_ratio)
        self.mag_balance = bool(mag_balance)
        self.conflict_threshold = float(conflict_threshold)
        self.rescale_grads = bool(rescale_grads)
        self.eps = float(eps)

        # Caches
        self._v: torch.Tensor | None = None
        self._splits: list[int] | None = None

        # Engineering optimization
        self._shared_params: list[torch.Tensor] | None = None

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        计算阶段：在 total_loss.backward() 之前运行。
        """
        self.step += 1

        # 1. Check intervals and warmup
        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            return {k: torch.ones((), device=next(iter(task_losses.values())).device) for k in task_losses}

        # 2. Filter valid params
        if self._shared_params is None:
            self._shared_params = [p for p in shared_params if p.requires_grad]
        shared = self._shared_params

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
            flat = self._flatten_grads(shared, grads, device)
            g_list.append(flat)

        if not g_list:
            self._v = None
            return {k: torch.ones((), device=device) for k in task_losses}

        G = torch.stack(g_list, dim=0)  # [T, D]

        # 计算原始总梯度模长
        G_raw_sum = G.sum(dim=0)
        original_mag = torch.linalg.norm(G_raw_sum).clamp_min(self.eps)

        # ============================================================
        # [Step 1] Magnitude Balancing (量级均衡)
        # ============================================================
        g_norms = torch.linalg.norm(G, dim=1).clamp_min(self.eps)  # [T]

        if self.mag_balance:
            # 目标模长：取所有任务模长的平均值
            target_norm = g_norms.mean()
            # 计算缩放系数
            mag_weights = target_norm / g_norms
            G_bal = G * mag_weights.view(-1, 1)
        else:
            G_bal = G

        # ============================================================
        # [Step 2] Consensus & Conflict Resolution
        # ============================================================
        # 计算共识方向
        G_unit = G_bal / torch.linalg.norm(G_bal, dim=1, keepdim=True).clamp_min(self.eps)
        d_consensus = G_unit.mean(dim=0)

        # 归一化共识方向
        d_norm = torch.linalg.norm(d_consensus).clamp_min(self.eps)
        d_consensus = d_consensus / d_norm

        # 检查每个任务是否与共识冲突
        G_final_list = []
        for i in range(G_bal.shape[0]):
            gi = G_bal[i]
            # 投影: gi 在共识方向上的分量
            proj_len = torch.dot(gi, d_consensus)

            # 冲突判定: 仅处理反向冲突 (< 0)
            if proj_len < self.conflict_threshold:
                # 移除冲突分量
                gi_new = gi - proj_len * d_consensus
            else:
                gi_new = gi
            G_final_list.append(gi_new)

        # 聚合修正后的梯度
        v_final = torch.stack(G_final_list, dim=0).sum(dim=0)

        # ============================================================
        # [Step 3] Scale Restoration
        # ============================================================
        if self.rescale_grads:
            current_mag = torch.linalg.norm(v_final).clamp_min(self.eps)
            scale_factor = original_mag / current_mag
            v_final = v_final * scale_factor

        self._v = v_final.detach()
        self._splits = [p.numel() for p in shared]

        return {k: torch.ones((), device=device) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        """
        应用阶段：将计算好的 v 写入参数的 .grad 属性
        """
        if self._v is None:
            return

        if self._shared_params:
            shared = self._shared_params
        else:
            shared = [p for p in shared_params if p.requires_grad]

        if not shared:
            return

        # DDP Sync
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._v, op=torch.distributed.ReduceOp.SUM)
            self._v /= torch.distributed.get_world_size()

        offset = 0
        v_flat = self._v

        for p, numel in zip(shared, self._splits):
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            # 取出该参数对应的 RGB 梯度片段
            rgb_grad = v_flat[offset: offset + numel].view_as(p).to(p.grad.dtype)
            offset += numel

            if self.blend_ratio >= 1.0:
                p.grad.copy_(rgb_grad)
            else:
                p.grad.mul_(1.0 - self.blend_ratio).add_(rgb_grad, alpha=self.blend_ratio)

        self._v = None

    def _flatten_grads(self, params, grads, device):
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros((p.numel(),), device=device, dtype=torch.float32))
            else:
                flat.append(g.reshape(-1).to(device, dtype=torch.float32))
        return torch.cat(flat, dim=0)