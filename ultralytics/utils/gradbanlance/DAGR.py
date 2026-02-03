# ultralytics/utils/gradbanlance/DAGR.py
from __future__ import annotations

import math
from typing import Iterable, Literal

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class DAGRBalancer(GradientBalancer):
    """
    Balanced-DAGR: A Robust Gradient Harmonization Method
    修复改进版：
    1. Magnitude Balancing: 强制均衡各任务梯度模长，防止强势任务主导。
    2. Soft Conflict Penalty: 仅惩罚反向冲突 (Cos < 0)，允许正交梯度共存。
    3. Safe Consensus: 基于均衡后的梯度计算共识方向。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # ---- Optimization args ----
            alpha: float = 0.5,  # 正则项系数 (保持梯度不偏离太远)
            lr: float = 0.05,  # 角度优化的学习率 (稍作降低以求稳定)
            mu: float = 0.9,  # EMA 动量
            steps: int = 10,  # 内部优化步数
            eps: float = 1e-8,
            # ---- Engineering ----
            lock_shared_params: bool = True,
            ddp_sync_v: bool = True,
            # ---- Args kept for compatibility (not used in new logic) ----
            magnitude: str = "mean",
            alpha_min: float = 0.0,
            alpha_max: float = 0.5 * math.pi,
            gamma: float = 0.5,
            beta_base: float = 0.8,
            quantile: float = 0.75,
            use_abs_cos: bool = True,
            aggregate: str = "mean",
            use_ema_consensus: bool = False,
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        self.lam = float(alpha)
        self.alpha_lr = float(lr)
        self.mu = float(mu)
        self.steps = int(steps)
        self.eps = float(eps)

        self.lock_shared_params = bool(lock_shared_params)
        self.ddp_sync_v = bool(ddp_sync_v)

        # Caches
        self._dt: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._splits: list[int] | None = None

        # Engineering safety
        self._shared_params: list[torch.Tensor] | None = None
        self._shared_id: list[int] | None = None

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        核心更新逻辑
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
                if len(cur_id) != len(self._shared_id):
                    self._shared_params = list(shared_in)
            shared = self._shared_params
        else:
            shared = shared_in

        # ---- Interval & Warmup Check (Fast Skip) ----
        # 在 warmup 期间或非 interval 步数，清空 _v 并返回 1.0 权重 (使用原始梯度)
        # 注意：这里我们选择在 warmup 期间不做任何干预，让模型先快速收敛一波
        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            self._splits = None
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            return self._last_weights

        # ---- Compute Gradients ----
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
            # 过滤无效梯度
            if not math.isfinite(gn) or gn <= self.eps:
                flat = torch.zeros_like(flat)
            g_list.append(flat)

        if not g_list:
            self._v = None
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}

        # G: [T, D] - 原始梯度矩阵
        G = torch.stack(g_list, dim=0)
        t = int(G.shape[0])

        # 记录原始总梯度的模长，用于最后恢复尺度，保证 lr schedule 有效
        g_sum_raw = G.sum(dim=0)
        base_mag = torch.linalg.norm(g_sum_raw).clamp_min(self.eps)

        # 单任务或无梯度情况，直接返回
        if t < 2:
            self._v = g_sum_raw.detach()
            self._splits = [int(p.numel()) for p in shared]
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}

        # ============================================================
        # [Step 1] Magnitude Balancing (量级均衡)
        # ============================================================
        # 计算每个任务的 L2 范数
        g_norms = torch.linalg.norm(G, dim=1, keepdim=True)  # [T, 1]

        # 目标模长：取平均值 (Simple GradNorm)
        # 这确保了 Seg 和 Pose 对最终方向有相同的贡献权
        target_norm = g_norms.mean()

        # 计算均衡系数
        # balance_scales: [T, 1]
        balance_scales = target_norm / g_norms.clamp_min(self.eps)

        # 获得均衡后的梯度矩阵 G_bal
        G_bal = G * balance_scales

        # ============================================================
        # [Step 2] Compute Consensus Direction & Orthogonal Basis
        # ============================================================
        # 使用均衡后的梯度计算共识方向，此时小任务不再被忽略

        # 归一化方向向量
        G_unit = G_bal / torch.linalg.norm(G_bal, dim=1, keepdim=True).clamp_min(self.eps)

        # 简单的平均方向作为共识 (Consensus)
        d_current = G_unit.mean(dim=0)
        d_current = d_current / torch.linalg.norm(d_current).clamp_min(self.eps)

        # 更新 EMA 方向
        if self._dt is None or self._dt.numel() != d_current.numel():
            self._dt = d_current.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * d_current).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)

        dt = self._dt

        # 构建正交基 W (Gram-Schmidt 类似思路)
        # w_i 垂直于 dt，位于 (g_i, dt) 平面内
        W_list = []
        for i in range(t):
            gi = G_unit[i]
            proj = torch.dot(dt, gi)
            wi = dt - proj * gi
            wi_norm = torch.linalg.norm(wi)
            if wi_norm <= self.eps:
                wi = torch.zeros_like(gi)
            else:
                wi = wi / wi_norm
            W_list.append(wi)
        W = torch.stack(W_list, dim=0)  # [T, D]

        # ============================================================
        # [Step 3] Relaxed Optimization (松弛优化)
        # ============================================================
        # 优化旋转角 alphas，但在目标函数中仅惩罚冲突
        alphas = torch.zeros((t,), device=device, dtype=torch.float32, requires_grad=True)

        # 预计算 G_unit 和 W 的点积矩阵，加速循环
        # (这部分如果不做复杂的 autograd 也可以手写，但用 autograd 比较方便维护)

        for _ in range(max(self.steps, 1)):
            # 旋转后的方向 R (基于均衡后的 Unit Vectors)
            # R_i = cos(a)*G_i + sin(a)*W_i
            # 注意：这里我们对 G_unit 进行旋转，只关心方向
            R = torch.cos(alphas)[:, None] * G_unit + torch.sin(alphas)[:, None] * W

            # 归一化 (理论上旋转后模长应为1，但数值上可能有误差)
            R = R / torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(self.eps)

            # 计算余弦相似度矩阵
            cos_mat = R @ R.t()  # [T, T]

            # 提取上三角 (不含对角线)
            iu = torch.triu_indices(t, t, offset=1, device=device)
            cos_ij = cos_mat[iu[0], iu[1]]

            # [CRITICAL FIX]: Conflict Loss
            # 只惩罚负余弦 (角度 > 90度)。如果 cos_ij >= 0 (0~90度)，loss 为 0。
            # 这允许正交任务共存。
            conflict_loss = torch.relu(-cos_ij).sum()

            # Proximal Loss: 防止旋转角度过大，偏离原始意图
            prox_loss = alphas.pow(2).mean()

            # Total Loss
            obj = conflict_loss + self.lam * prox_loss

            if obj.item() <= 1e-6:
                break

            (grad_a,) = torch.autograd.grad(obj, (alphas,), retain_graph=False, create_graph=False)

            with torch.no_grad():
                alphas -= self.alpha_lr * grad_a
                # 限制旋转角度在 -90 到 +90 度之间，防止反向
                alphas.clamp_(-0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4)
            alphas.requires_grad_(True)

        # ============================================================
        # [Step 4] Final Synthesis (最终合成)
        # ============================================================
        with torch.no_grad():
            # 使用优化好的 alphas，旋转 **均衡后(Balanced)** 的梯度 G_bal
            # 这样 Seg 任务依然保有足够的模长
            sin_a = torch.sin(alphas)[:, None]
            cos_a = torch.cos(alphas)[:, None]

            # R_final: [T, D]
            # 我们用 W (单位向量) * G_bal的模长 来保持尺度一致性
            # 或者更简单：直接旋转 G_bal
            # G_bal 分解为：模长 * G_unit
            bal_norms = torch.linalg.norm(G_bal, dim=1, keepdim=True)
            R_final = (cos_a * G_unit + sin_a * W) * bal_norms

            # 求和得到最终方向
            v_final = R_final.sum(dim=0)

            # [Final Scale Restoration]
            # 恢复到原始梯度的量级，保证 Learning Rate Schedule 不会失效
            # 我们使用均衡后的方向，但是使用原始的模长
            v_norm = torch.linalg.norm(v_final).clamp_min(self.eps)
            v = v_final / v_norm * base_mag

            self._v = v.detach()
            self._splits = [int(p.numel()) for p in shared]

        # 返回全1权重，因为真正的梯度修改已经存储在 self._v 中，将通过 apply_after_unscale 应用
        self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
        return self._last_weights

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        """
        将计算好的梯度 v 覆盖到 shared_params.grad 中
        """
        if self._v is None or self._splits is None:
            return

        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return

        if self.lock_shared_params and self._shared_params is not None:
            shared = self._shared_params
        else:
            shared = shared_in

        # 简单的校验
        splits = [int(p.numel()) for p in shared]
        if sum(splits) != int(self._v.numel()):
            return

        # DDP Sync (如果需要)
        v = self._v.to(device=shared[0].device, dtype=torch.float32)
        if self.ddp_sync_v and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            world = torch.distributed.get_world_size()
            if world > 1:
                v.div_(float(world))

        # 写入梯度
        offset = 0
        for p, n in zip(shared, splits):
            piece = v[offset: offset + n].view_as(p)
            offset += n
            if p.grad is None:
                p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device)

            # 覆盖原梯度
            p.grad.detach_().copy_(piece.to(dtype=p.grad.dtype))

    def _flatten_grads(self, shared, grads, device):
        flat = []
        for p, g in zip(shared, grads):
            if g is None:
                flat.append(torch.zeros((p.numel(),), device=device, dtype=torch.float32))
            else:
                flat.append(g.reshape(-1).to(device=device, dtype=torch.float32))
        return torch.cat(flat, dim=0)