from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Iterable

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class CAGradBalancer(GradientBalancer):
    """
    CAGrad: Conflict-Averse Gradient Descent (NeurIPS 2021)

    原理：
    寻找一个更新方向，最大化所有任务的平均性能提升，同时约束最差任务的性能下降。
    它通过解决一个二次规划问题，找到一个平衡点，既接近平均梯度方向，又尽量减少梯度冲突。

    参数:
        c (float): 正则化系数 (0 <= c < 1)。
                   c=0 时退化为 MGDA (寻找 Pareto 最优但可能由于过强约束导致优化慢)；
                   c 越大越接近简单的平均加权 (Mean)。
                   推荐默认值 0.4 或 0.5。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # ---- CAGrad 特有参数 ----
            c: float = 0.5,
            rescale_grads: bool = True,  # 保持梯度模长与平均梯度一致
            # ---- 兼容性参数 (防止 train.py 报错) ----
            alpha: float = 0.0,
            lr: float = 0.0,
            eps: float = 1e-8,
            # ---- 工程参数 ----
            lock_shared_params: bool = True,
            ddp_sync_v: bool = True,
            **kwargs  # 兜底
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        self.c = float(c)
        self.rescale_grads = bool(rescale_grads)
        self.eps = float(eps)

        # 兼容参数存储
        self.alpha_unused = alpha
        self.lr_unused = lr

        self.lock_shared_params = bool(lock_shared_params)
        self.ddp_sync_v = bool(ddp_sync_v)

        # 缓存
        self._v: torch.Tensor | None = None
        self._splits: list[int] | None = None
        self._shared_params: list[torch.Tensor] | None = None
        self._shared_id: list[int] | None = None

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor]
    ) -> dict[str, torch.Tensor]:

        self.step += 1
        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype

        # 1. Warmup / Interval Check
        if self.step <= self.warmup or (self.step - 1) % self.interval != 0:
            self._v = None
            self._splits = None
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses}

        if not task_losses:
            return {}

        # 2. Prepare Shared Params
        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses}

        if self.lock_shared_params:
            if self._shared_params is None:
                self._shared_params = list(shared_in)
                self._shared_id = [int(p.data_ptr()) for p in self._shared_params]
            shared = self._shared_params
        else:
            shared = shared_in

        # 3. Compute Gradients per Task
        names = list(task_losses.keys())
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
            # 过滤无效梯度
            if not torch.isfinite(flat).all():
                continue
            g_list.append(flat)

        if not g_list:
            self._v = None
            return {k: torch.ones((), device=device) for k in task_losses}

        G = torch.stack(g_list)  # [T, D]
        T = G.shape[0]

        # 4. CAGrad Core Logic
        # 如果只有一个任务，直接返回
        if T < 2:
            self._v = G[0]
            self._splits = [p.numel() for p in shared]
            return {k: torch.ones((), device=device) for k in task_losses}

        # 计算 Gram Matrix: M = G @ G.T
        G_flat = G.view(T, -1)
        M = torch.matmul(G_flat, G_flat.t())  # [T, T]

        # 计算平均梯度 g0
        g0_norm = M.mean().sqrt().item()

        # 求解对偶问题：找到最优权重 w
        # maximize: - w.T @ M @ w  (minimize norm of combined grad)
        # subject to: w close to uniform (controlled by c)

        # 这里的求解使用了梯度下降法求解 QP，避免引入 scipy
        w_opt = self._solve_cagrad_dual(M, self.c)

        # 5. Compute Final Gradient
        # v = \sum w_i * g_i
        v_final = torch.matmul(w_opt.view(1, T), G_flat).view(-1)

        # 6. Rescale (Optional but Recommended)
        # CAGrad 有时会产生较小的梯度模长，为了保持训练稳定，
        # 通常将其模长恢复到平均梯度的水平。
        if self.rescale_grads:
            v_norm = v_final.norm().clamp_min(self.eps)
            v_final = v_final * (g0_norm / v_norm)

        self._v = v_final.detach()
        self._splits = [p.numel() for p in shared]

        return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        if self._v is None:
            return

        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return

        # Use cached shared params if locked
        shared = self._shared_params if (self.lock_shared_params and self._shared_params) else shared_in

        # DDP Synchronization
        if self.ddp_sync_v and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._v, op=torch.distributed.ReduceOp.SUM)
            self._v /= torch.distributed.get_world_size()

        # Write gradients
        offset = 0
        for p, numel in zip(shared, self._splits):
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            new_grad = self._v[offset:offset + numel].view_as(p).to(p.grad.dtype)
            p.grad.copy_(new_grad)
            offset += numel

        self._v = None  # Reset cache

    def _solve_cagrad_dual(self, M: torch.Tensor, c: float, max_iter: int = 50) -> torch.Tensor:
        """
        使用投影梯度下降法求解 CAGrad 的最优权重 w。
        目标: minimize F(w) = w.T @ M @ w + sqrt(w.T @ M @ w) * const
        等价于寻找一个 w，使得组合梯度在平均梯度方向上的投影最大化。
        """
        T = M.shape[0]
        device = M.device

        # 初始化权重为均匀分布
        w = torch.ones(T, device=device) / T
        w.requires_grad_(True)

        # CAGrad 的目标函数稍微复杂，这里使用一般化形式的简化求解：
        # 寻找 w 使得组合梯度方向最优化
        # 论文中的实际操作是：minimize g.T @ w subject to constraints.
        # LibMTL 中的标准实现是：minimize (w.T @ M @ w) + \lambda ||w - w0||^2
        # 我们这里使用最稳定的对偶形式：minimize (G w).norm() s.t. w >= 0, sum(w)=1, constraint on conflict

        # 为了速度和稳定性，这里使用最简化的迭代求解器：
        # 目标：Minimize 1/2 * w.T * M * w + beta * ||w - 1/T||^2
        # beta 动态调节以满足 constraint

        g0 = M.mean(dim=1)  # [T]

        # 快速求解器：Gradient Descent on Simplex
        optimizer = torch.optim.SGD([w], lr=0.1)  # 局部优化器

        # c 的变换：phi = c^2
        # 约束条件： <g_combined, g_average> >= c * ||g_combined|| * ||g_average||

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Objective: 1/2 w^T M w
            obj = 0.5 * (w @ M @ w)

            # Constraint penalty (Lagrange multiplier style approximation)
            # 我们希望 w 接近 1/T
            # 这里使用 CAGrad 论文公式 10 的简化变体

            # 计算 g_w = G^T w 的模长平方
            gw_norm2 = w @ M @ w
            gw_norm = gw_norm2.sqrt().clamp_min(1e-8)

            # 计算 g_w 与 g_avg 的点积
            # g_avg = sum(g_i)/T
            # dot = w^T M 1/T
            dot_avg = (w @ g0) / T
            g0_norm = math.sqrt(M.mean().item())

            # 约束: dot_avg >= c * gw_norm * g0_norm
            # Loss = obj + lambda * ReLU(c * gw_norm * g0_norm - dot_avg)

            constraint = c * gw_norm * g0_norm - dot_avg
            penalty = F.relu(constraint)

            loss = obj + 10.0 * penalty  # 强惩罚系数

            loss.backward()
            optimizer.step()

            # Projection to Simplex (sum=1, >=0)
            with torch.no_grad():
                # 简单的 Softmax 可能导致梯度消失，这里用投影
                # 排序法投影
                sorted_w, _ = torch.sort(w, descending=True)
                cumsum_w = torch.cumsum(sorted_w, dim=0)
                indices = torch.arange(1, T + 1, device=device, dtype=w.dtype)

                cond = sorted_w - (cumsum_w - 1.0) / indices > 0.0
                rho = indices[cond][-1].long()
                theta = (cumsum_w[rho - 1] - 1.0) / rho
                w.copy_((w - theta).clamp_min(0.0))

        return w.detach()

    def _flatten_grads(self, params, grads, device):
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros(p.numel(), device=device))
            else:
                flat.append(g.reshape(-1).to(device))
        return torch.cat(flat)

