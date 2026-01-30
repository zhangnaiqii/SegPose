from __future__ import annotations

from typing import Iterable

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class GradNormBalancer(GradientBalancer):
    """
    Faster GradNorm:
      - No create_graph (no 2nd-order graph build).
      - No autograd.grad(grad_loss, weights).
      - Use analytic gradient for weights:
            gnorm_i = w_i * ||dL_i/dW||
            d|gnorm_i - target_i|/dw_i = sign(gnorm_i - target_i) * ||dL_i/dW||
    This keeps the same target definition (mean detached) as the original code.
    """

    def __init__(
        self,
        task_names: list[str] | None = None,
        interval: int = 1,
        warmup: int = 0,
        alpha: float = 1.5,
        lr: float = 0.025,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.eps = float(eps)
        self.weights: torch.Tensor | None = None
        self.initial_losses: torch.Tensor | None = None

    def update(
        self, task_losses: dict[str, torch.Tensor], shared_params: Iterable[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self.step += 1
        if self.task_names is None:
            self.task_names = list(task_losses.keys())
        names = self.task_names
        if not names or any(name not in task_losses for name in names):
            return self._fallback_weights(task_losses)

        losses = torch.stack([task_losses[name] for name in names])
        n = len(names)

        # init weights
        if self.weights is None or self.weights.numel() != n or self.weights.device != losses.device:
            self.weights = torch.ones(n, device=losses.device, dtype=torch.float32)
            self._last_weights = {name: torch.ones((), device=losses.device, dtype=losses.dtype) for name in names}

        if self.step <= self.warmup:
            return self._last_weights or self._fallback_weights(task_losses)
        if (self.step - 1) % self.interval != 0 and self._last_weights is not None:
            return self._last_weights

        shared = [p for p in shared_params if p.requires_grad]
        if not shared:
            return self._last_weights or self._fallback_weights(task_losses)

        losses_fp = losses.float()
        if self.initial_losses is None:
            self.initial_losses = losses_fp.detach().clamp(min=self.eps)

        # Compute ||dL_i/dW|| with create_graph=False (no 2nd-order graph)
        base_gnorm = losses_fp.new_empty(n)
        for i in range(n):
            grads = torch.autograd.grad(
                losses_fp[i],
                shared,
                retain_graph=True,   # must keep graph for the real backward later
                create_graph=False,
                allow_unused=True,
            )
            # L2 norm over concatenated gradients (in fp32 for stability)
            g2 = losses_fp.new_zeros(())
            for g in grads:
                if g is None:
                    continue
                g2 = g2 + g.float().pow(2).sum()
            base_gnorm[i] = torch.sqrt(g2 + self.eps)

        # gnorm_i = w_i * ||dL_i/dW||
        w = self.weights
        grad_norms = w * base_gnorm

        # same target as original: mean detached
        loss_ratio = losses_fp.detach().clamp(min=self.eps) / self.initial_losses
        inv_rate = loss_ratio / loss_ratio.mean()
        target = grad_norms.mean().detach() * (inv_rate ** self.alpha)

        # Analytic gradient for weights (subgradient of abs)
        # d/dw_i sum |gnorm_i - target_i| = sign(gnorm_i - target_i) * base_gnorm_i
        grad_w = torch.sign(grad_norms - target) * base_gnorm

        with torch.no_grad():
            w = w - self.lr * grad_w
            w.clamp_(min=self.eps)
            w *= n / w.sum()
            self.weights = w

        self._last_weights = {name: self.weights[i].to(losses.dtype).detach() for i, name in enumerate(names)}
        return self._last_weights

    def _fallback_weights(self, task_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._last_weights is not None:
            return self._last_weights
        if not task_losses:
            return {}
        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype
        return {name: torch.ones((), device=device, dtype=dtype) for name in task_losses}
