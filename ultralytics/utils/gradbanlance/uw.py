# ultralytics/utils/gradbanlance/uw.py
from __future__ import annotations

import torch
import math
from typing import Iterable

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class UWBalancer(GradientBalancer):
    """
    Uncertainty Weighting (CVPR 2018)
    Paper: Multi-Task Learning Using Uncertainty to Weigh Losses

    原理：
    通过学习每个任务的同方差不确定性（Homoscedastic Uncertainty, sigma）来动态调整权重。
    Loss = sum( (1 / (2 * sigma^2)) * loss + log(sigma) )

    实现技巧：
    令 s = log(sigma^2)，则 Loss = sum( 0.5 * exp(-s) * loss + 0.5 * s )
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # ---- 兼容 train.py 的传参 ----
            lr: float = 0.025,  # 这里用作学习 sigma 的学习率
            alpha: float = 0.0,  # UW 不需要 alpha，占位防止报错
            **kwargs  # 吸收其他多余参数
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        # 这里的 lr 专门用于更新权重参数 s
        self.lr = float(lr)

        # 存储每个任务的 log(sigma^2) 参数
        # 使用字典存储，支持动态任务名
        self.log_vars: dict[str, torch.Tensor] = {}

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor]
    ) -> dict[str, torch.Tensor]:

        self.step += 1
        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype

        # 1. 初始化参数 (Lazy Initialization)
        # 第一次运行时，根据传入的 loss keys 初始化参数 s
        # 初始化为 0，意味着 sigma=1，初始权重为 0.5
        if not self.log_vars:
            for name in task_losses.keys():
                # 这是一个需要梯度的参数，但我们手动更新它
                self.log_vars[name] = torch.zeros(
                    (), device=device, dtype=dtype, requires_grad=False
                )

        # 2. Warmup 或 Interval 检查
        # 如果还在 warmup，或者没到更新间隔，返回默认权重 (全1，保持原样)
        if self.step <= self.warmup:
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses}

        # 3. 计算权重并更新参数
        # UW 的核心：Loss_total = 0.5 * exp(-s) * L_i + 0.5 * s
        # 对 s 求导： d(Loss)/ds = -0.5 * exp(-s) * L_i + 0.5

        weights = {}
        for name, loss in task_losses.items():
            if name not in self.log_vars:
                # 如果突然出现了新任务，动态添加
                self.log_vars[name] = torch.zeros((), device=device, dtype=dtype)

            s = self.log_vars[name]

            # 计算当前权重: w = 0.5 * exp(-s)
            # 注意：有些实现会去掉 0.5，但为了严格符合论文公式，这里保留 0.5
            # 或者为了保持梯度量级，许多复现代码使用 w = exp(-s)。
            # 这里我们采用标准推导：w = 1 / (2 * sigma^2) = 0.5 * exp(-s)
            with torch.no_grad():
                weight = 0.5 * torch.exp(-s)

                # ---- 手动优化步骤 ----
                # 计算 s 的梯度: grad = 0.5 - 0.5 * exp(-s) * loss
                # 注意：这里 loss 需要 detach，避免反向传播影响模型主干
                loss_val = loss.detach()
                grad_s = 0.5 - weight * loss_val

                # 执行 SGD 更新: s = s - lr * grad
                s_new = s - self.lr * grad_s
                self.log_vars[name].copy_(s_new)

                weights[name] = weight

        self._last_weights = weights
        return weights

    # UW 是 Loss Weighting 方法，不需要在 backward 后操作梯度
    # 所以这里留空即可
    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        pass



