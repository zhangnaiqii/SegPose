# ultralytics/utils/gradbanlance/pcgrad.py
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable
import math
import random

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class PCGradBalancer(GradientBalancer):
    """
    Project Conflicting Gradients (PCGrad).
    Paper: https://arxiv.org/abs/2001.06782
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # 兼容性参数：接收但不使用 alpha 和 lr，防止报错
            alpha: float = 0.0,
            lr: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
        # PCGrad 不需要 alpha 和 lr，这里只是为了占位防止报错

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        self.step += 1

        # 1. Warmup 或 非 Interval 步数，直接跳过
        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            return {k: torch.ones((), device=next(iter(task_losses.values())).device) for k in task_losses}

        # 2. 准备梯度
        shared = [p for p in shared_params if p.requires_grad]
        if not shared or not task_losses:
            return {}

        device = shared[0].device
        names = list(task_losses.keys())

        # 计算每个任务的梯度
        grads_task = []
        for name in names:
            g = torch.autograd.grad(
                task_losses[name],
                shared,
                retain_graph=True,
                allow_unused=True
            )
            # 展平
            g_flat = self._flatten_grads(shared, g, device)
            grads_task.append(g_flat)

        if not grads_task:
            self._v = None
            return {k: torch.ones((), device=device) for k in task_losses}

        # 堆叠梯度 G: [num_tasks, num_params]
        G = torch.stack(grads_task)
        num_tasks = G.shape[0]

        # 3. PCGrad 核心逻辑 (投影冲突梯度)
        # 随机打乱任务顺序以避免偏差
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)

        final_grads = []

        for i in task_indices:
            g_i = G[i]
            # 与其它所有任务的梯度进行对比和投影
            for j in list(range(num_tasks)):  # 也可以再次随机 shuffle
                if i == j:
                    continue

                g_j = G[j]

                # 计算点积
                g_i_g_j = torch.dot(g_i, g_j)

                # 如果冲突 (夹角 > 90度, 点积 < 0)
                if g_i_g_j < 0:
                    # 将 g_i 投影到 g_j 的法平面上
                    # g_i = g_i - (g_i . g_j) / ||g_j||^2 * g_j
                    denom = torch.dot(g_j, g_j)
                    if denom > 1e-8:  # 避免除以零
                        g_i = g_i - (g_i_g_j / denom) * g_j

            final_grads.append(g_i)

        # 4. 合并梯度
        # 将所有修正后的任务梯度相加
        self._v = torch.stack(final_grads).sum(dim=0)

        # 记录 split 信息供 apply 使用
        self._splits = [p.numel() for p in shared]

        return {k: torch.ones((), device=device) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        if self._v is None:
            return

        shared = [p for p in shared_params if p.requires_grad]
        if not shared:
            return

        # DDP Sync
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._v, op=torch.distributed.ReduceOp.SUM)
            self._v /= torch.distributed.get_world_size()

        offset = 0
        for p, numel in zip(shared, self._splits):
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            new_grad = self._v[offset:offset + numel].view_as(p).to(p.grad.dtype)
            p.grad.copy_(new_grad)
            offset += numel

        self._v = None

    def _flatten_grads(self, params, grads, device):
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros(p.numel(), device=device))
            else:
                flat.append(g.reshape(-1).to(device))
        return torch.cat(flat)