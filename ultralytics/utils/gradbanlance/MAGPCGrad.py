# ultralytics/utils/gradbanlance/MagPCGrad.py
from __future__ import annotations

import torch
import random
from typing import Iterable

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class MagPCGradBalancer(GradientBalancer):
    """
    Magnitude Balanced Sequential PCGrad

    1. Magnitude Balancing：强制各任务梯度模长相等（均值目标）。
    2. Sequential PCGrad：原论文推荐（随机顺序 + 顺序投影已修正梯度），彻底移除冲突方向，减少顺序bias。
    3. Rescale：恢复原始总梯度模长，保持优化动量和lr schedule有效。

    专为seg+pose等冲突强烈的多任务设计，常能实现正迁移（超单任务）。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # 兼容性参数
            alpha: float = 0.0,
            lr: float = 0.0,
            mag_balance: bool = True,
            rescale_grads: bool = True,
            eps: float = 1e-8,
            **kwargs
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
        self.mag_balance = bool(mag_balance)
        self.rescale_grads = bool(rescale_grads)
        self.eps = float(eps)

        # Caches
        self._v: torch.Tensor | None = None
        self._splits: list[int] | None = None
        self._shared_params: list[torch.Tensor] | None = None

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        self.step += 1

        # Warmup 或非interval步，直接返回1权重（不干预）
        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            return {k: torch.ones((), device=next(iter(task_losses.values())).device) for k in task_losses}

        # 缓存共享参数（带requires_grad）
        if self._shared_params is None:
            self._shared_params = [p for p in shared_params if p.requires_grad]
        shared = self._shared_params

        if not shared or not task_losses:
            return {}

        names = list(task_losses.keys())
        device = shared[0].device

        # 计算各任务梯度并展平
        grads_task = []
        for name in names:
            grads = torch.autograd.grad(
                task_losses[name],
                shared,
                retain_graph=True,
                allow_unused=True
            )
            flat = self._flatten_grads(shared, grads, device)
            grads_task.append(flat)

        G = torch.stack(grads_task)  # [num_tasks, D]
        num_tasks = G.shape[0]

        # 单任务直接求和
        if num_tasks < 2:
            self._v = G.sum(0).detach()
            self._splits = [p.numel() for p in shared]
            return {k: torch.ones((), device=device) for k in task_losses}

        # 原始总梯度模长（用于最后恢复）
        original_mag = torch.linalg.norm(G.sum(dim=0)).clamp_min(self.eps)

        # Step 1: Magnitude Balancing
        if self.mag_balance:
            g_norms = torch.linalg.norm(G, dim=1).clamp_min(self.eps)  # [T]
            target_norm = g_norms.mean()
            scales = target_norm / g_norms
            G_bal = G * scales.unsqueeze(1)
        else:
            G_bal = G

        # Step 2: Sequential PCGrad（原论文推荐实现）
        grads_list = [G_bal[i].clone() for i in range(num_tasks)]
        perm = list(range(num_tasks))
        random.shuffle(perm)  # 随机顺序减bias

        for curr_pos in range(num_tasks):
            i = perm[curr_pos]
            g_i = grads_list[i]

            for prev_pos in range(curr_pos):
                j = perm[prev_pos]
                g_j = grads_list[j]  # 已修正的先前梯度

                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    denom = torch.dot(g_j, g_j).clamp_min(self.eps)
                    g_i = g_i - (dot / denom) * g_j

            grads_list[i] = g_i

        v_final = torch.stack(grads_list).sum(0)

        # Step 3: Rescale 恢复原始总模长
        if self.rescale_grads:
            current_mag = torch.linalg.norm(v_final).clamp_min(self.eps)
            v_final = v_final * (original_mag / current_mag)

        self._v = v_final.detach()
        self._splits = [p.numel() for p in shared]

        # 返回全1权重（实际梯度已在apply中覆盖）
        return {k: torch.ones((), device=device) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        if self._v is None:
            return

        shared = self._shared_params or [p for p in shared_params if p.requires_grad]
        if not shared:
            return

        # DDP sync
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._v, op=torch.distributed.ReduceOp.SUM)
            self._v /= torch.distributed.get_world_size()

        offset = 0
        for p, numel in zip(shared, self._splits):
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            grad_piece = self._v[offset:offset + numel].view_as(p).to(p.grad.dtype)
            p.grad.copy_(grad_piece)
            offset += numel

        self._v = None
        self._splits = None

    def _flatten_grads(self, params, grads, device):
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
            else:
                flat.append(g.reshape(-1).to(device, dtype=torch.float32))
        return torch.cat(flat)
