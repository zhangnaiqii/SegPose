# ultralytics/utils/gradbalance/AnchorMagPCGrad.py
from __future__ import annotations

import torch
import random
from typing import Iterable

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class Anchor2MagPCGradBalancer(GradientBalancer):
    """
    Anchor Magnitude-Balanced Sequential PCGrad

    核心：
    1. Active filtering：梯度范数 < min_norm 的任务不参与（防稀疏 pose batch 噪声）
    2. Magnitude balancing：活跃任务梯度模长均衡到均值
    3. Anchored Sequential PCGrad：anchor task（默认 seg）梯度不投影，其他任务顺序投影掉与 anchor 冲突的部分
    4. Rescale：恢复原始总模长（保持 lr/momentum 有效）

    专为“主任务强 + 辅任务稀疏”设计，极大概率实现正迁移 > 单任务。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            anchor_task: str = "seg",  # 主任务 key（seg_total），它不被投影
            mag_balance: bool = True,
            rescale_grads: bool = True,
            min_norm_threshold: float = 1e-5,  # 低于此范数的任务视为 inactive
            eps: float = 1e-8,
            **kwargs
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
        self.anchor_task = str(anchor_task)
        self.mag_balance = bool(mag_balance)
        self.rescale_grads = bool(rescale_grads)
        self.min_norm = float(min_norm_threshold)
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

        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            return {k: torch.ones((), device=next(iter(task_losses.values())).device) for k in task_losses}

        if self._shared_params is None:
            self._shared_params = [p for p in shared_params if p.requires_grad]
        shared = self._shared_params

        if not shared or not task_losses:
            return {k: torch.ones((), device=shared[0].device) if shared else torch.ones(()) for k in task_losses}

        names = list(task_losses.keys())
        device = shared[0].device

        # 计算各任务梯度并展平
        grads_task = {}
        for name in names:
            grads = torch.autograd.grad(
                task_losses[name],
                shared,
                retain_graph=True,
                allow_unused=True
            )
            flat = self._flatten_grads(shared, grads, device)
            grads_task[name] = flat

        G = torch.stack([grads_task[n] for n in names])  # [num_tasks, D]
        norms = torch.linalg.norm(G, dim=1).clamp_min(self.eps)

        # Active tasks：排除范数太小的（稀疏 batch）
        active_mask = norms >= self.min_norm
        if active_mask.sum() < 2:
            # 只有一个活跃任务，直接用原始总梯度
            self._v = G.sum(0).detach()
            self._splits = [p.numel() for p in shared]
            return {k: torch.ones((), device=device) for k in task_losses}

        active_names = [n for n, active in zip(names, active_mask) if active]
        G_active = G[active_mask]

        # 原始总梯度模长（用于 rescale）
        original_mag = torch.linalg.norm(G.sum(0)).clamp_min(self.eps)

        # Step 1: Magnitude Balancing（只对 active）
        if self.mag_balance:
            active_norms = norms[active_mask]
            target_norm = active_norms.mean()
            scales = target_norm / active_norms
            G_bal = G_active * scales.unsqueeze(1)
        else:
            G_bal = G_active

        # Step 2: Anchored Sequential PCGrad
        num_active = len(active_names)
        grads_list = [G_bal[i].clone() for i in range(num_active)]
        perm = list(range(num_active))
        random.shuffle(perm)  # 随机顺序减 bias（但 anchor 优先固定）

        # 确保 anchor 在序列最前（如果它是 active）
        anchor_idx = None
        if self.anchor_task in active_names:
            anchor_local_idx = active_names.index(self.anchor_task)
            perm = [anchor_local_idx] + [i for i in perm if i != anchor_local_idx]

        for curr_pos in range(num_active):
            i = perm[curr_pos]
            g_i = grads_list[i]

            # anchor task 不投影（curr_pos == 0 时若为 anchor 则跳过）
            if active_names[i] == self.anchor_task and curr_pos > 0:
                continue  # 实际上因放在最前，只会执行一次不投影

            for prev_pos in range(curr_pos):
                j = perm[prev_pos]
                g_j = grads_list[j]

                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    denom = torch.dot(g_j, g_j).clamp_min(self.eps)
                    g_i = g_i - (dot / denom) * g_j

            grads_list[i] = g_i

        v_final = torch.stack(grads_list).sum(0)

        # 非 active 任务梯度补 0（已排除）
        full_v = torch.zeros_like(G[0])
        active_offset = 0
        for idx, name in enumerate(names):
            if active_mask[idx]:
                # 映射回 full（但因我们只改 active，inactive 保持 0）
                pass
        # 直接用 active sum 作为 final（inactive 已自然为 0）
        v_final = v_final

        # Step 3: Rescale
        if self.rescale_grads:
            current_mag = torch.linalg.norm(v_final).clamp_min(self.eps)
            v_final = v_final * (original_mag / current_mag)

        # 补回所有任务原始 grad（inactive 保持原小 grad？这里我们用 balanced 结果）
        self._v = v_final.detach()
        self._splits = [p.numel() for p in shared]

        return {k: torch.ones((), device=device) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        if self._v is None:
            return

        shared = self._shared_params or [p for p in shared_params if p.requires_grad]
        if not shared:
            return

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
