# AnchorMagPCGradBalancer.py
# ------------------------------------------------------------
# Anchor + Magnitude-Balanced + Sequential PCGrad + Active filtering
#
# 适用：seg + pose 这类“主任务很强、aux 任务会拖累”的多任务训练
# 目标：尽量保证 seg 不掉点，同时让 pose 在不冲突子空间里带来正迁移
#
# 接口假设：
#   - update(task_losses: Dict[str, Tensor], shared_params: Iterable[Tensor]) -> Dict[str, Tensor]
#   - apply_after_unscale(shared_params: Iterable[Tensor]) -> None
#
# 你需要把 shared_params 传“共享 trunk 参数”（建议 backbone+neck），不要把各 head 参数塞进来做 surgery。
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


# 尝试兼容你 repo 里的注册/基类位置（你按自己工程实际路径微调即可）
try:
    # 常见写法：ultralytics/utils/grad_balance/__init__.py 导出
    from . import GradientBalancer, register_grad_balancer  # type: ignore
except Exception:  # pragma: no cover
    try:
        from .base import GradientBalancer, register_grad_balancer  # type: ignore
    except Exception:  # pragma: no cover
        # 最差兜底：不影响你复制粘贴，只是没法自动注册
        class GradientBalancer:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass

        def register_grad_balancer(cls):  # type: ignore
            return cls


@dataclass
class _FlatPack:
    vec: torch.Tensor
    splits: List[int]
    params: List[torch.Tensor]


def _flatten_grads(
    params: List[torch.Tensor],
    grads: Tuple[Optional[torch.Tensor], ...],
    device: torch.device,
) -> torch.Tensor:
    """Flatten grads into a single FP32 vector (missing grads -> zeros)."""
    flat: List[torch.Tensor] = []
    for p, g in zip(params, grads):
        if g is None:
            flat.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
        else:
            flat.append(g.reshape(-1).to(device=device, dtype=torch.float32))
    return torch.cat(flat, dim=0)


def _unflatten_to_params(
    flat_vec: torch.Tensor,
    params: List[torch.Tensor],
    splits: List[int],
) -> None:
    """Write flat_vec back into params[i].grad (dtype matches existing grad if any)."""
    offset = 0
    for p, n in zip(params, splits):
        piece = flat_vec[offset : offset + n].view_as(p)
        offset += n

        if p.grad is None:
            # 在 unscale 后写入，通常这里应该是 FP32
            p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device)

        piece = piece.to(dtype=p.grad.dtype, device=p.grad.device)
        p.grad.copy_(piece)


def _safe_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.linalg.norm(x).clamp_min(eps)


def _project_conflicting(g_i: torch.Tensor, g_j: torch.Tensor, eps: float) -> torch.Tensor:
    """PCGrad projection: if dot(g_i, g_j) < 0, remove component along g_j."""
    dot = torch.dot(g_i, g_j)
    if dot >= 0:
        return g_i
    denom = torch.dot(g_j, g_j).clamp_min(eps)
    return g_i - (dot / denom) * g_j


@register_grad_balancer
class AnchorMagPCGradBalancer(GradientBalancer):
    """
    Anchor + Magnitude-Balanced + Sequential PCGrad + Active filtering

    参数建议（你可以从这组开始）：
      - anchor_task="seg"（或 "seg_total"）
      - interval=1
      - warmup=200~1000（视 batch_size / lr 而定）
      - min_norm=1e-6（pose 很稀疏可试 1e-5 或 1e-4）
      - mag_balance=True
      - rescale_grads=True
      - aux_weight=1.0（如果 seg 仍掉点，可以试 0.5~0.8）
    """

    def __init__(
        self,
        task_names: Optional[List[str]] = None,
        interval: int = 1,
        warmup: int = 0,
        *,
        anchor_task: str = "seg",
        min_norm: float = 1e-6,
        mag_balance: bool = True,
        rescale_grads: bool = True,
        aux_weight: float = 1.0,
        eps: float = 1e-12,
        deterministic: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)
        self.anchor_task = str(anchor_task)
        self.min_norm = float(min_norm)
        self.mag_balance = bool(mag_balance)
        self.rescale_grads = bool(rescale_grads)
        self.aux_weight = float(aux_weight)
        self.eps = float(eps)
        self.deterministic = bool(deterministic)

        self.step = 0
        self._cached: Optional[_FlatPack] = None

    @torch.no_grad()
    def _select_shared_params(self, shared_params: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        # 只取 requires_grad 的共享参数（建议你传 backbone+neck）
        return [p for p in shared_params if p is not None and p.requires_grad]

    def update(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: Iterable[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.step += 1

        # warmup / interval：直接不干预
        if self.step <= self.warmup or ((self.step - 1) % self.interval != 0):
            self._cached = None
            # 返回 1 权重以保持接口一致（你们若忽略权重也没事）
            if not task_losses:
                return {}
            dev = next(iter(task_losses.values())).device
            return {k: torch.ones((), device=dev) for k in task_losses}

        params = self._select_shared_params(shared_params)
        if not params or not task_losses:
            self._cached = None
            return {}

        device = params[0].device
        splits = [p.numel() for p in params]

        # ---- 1) 计算每个 task 的 flat grad（用 autograd.grad，不污染 .grad）----
        names = list(task_losses.keys())
        flats: Dict[str, torch.Tensor] = {}

        # 注意：这里需要 retain_graph=True，因为要对同一 forward 图求多次 grad
        for name in names:
            loss = task_losses[name]
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            flat = _flatten_grads(params, grads, device=device)
            flats[name] = flat

        # ---- 2) active-task filtering（解决某些 batch pose 无有效标注导致的爆缩放/噪声）----
        active: List[str] = []
        norms: Dict[str, torch.Tensor] = {}
        for name, g in flats.items():
            n = _safe_norm(g, self.eps)
            norms[name] = n
            if n.item() >= self.min_norm:
                active.append(name)

        # 只有一个 active task：等价单任务更新（不做 surgery）
        if len(active) <= 1:
            v = flats[active[0]] if active else torch.zeros(sum(splits), device=device, dtype=torch.float32)
            self._cached = _FlatPack(vec=v.detach(), splits=splits, params=params)
            dev = next(iter(task_losses.values())).device
            return {k: torch.ones((), device=dev) for k in task_losses}

        # ---- 3) 组装：anchor 置前，其它任务后（锚定主任务，保护强基线）----
        anchor = self.anchor_task if self.anchor_task in active else None
        ordered = []
        if anchor is not None:
            ordered.append(anchor)
        ordered.extend([n for n in active if n != anchor])

        G = [flats[n].clone() for n in ordered]  # list of [D]
        T = len(G)

        # 原始“active tasks 总梯度”的模长，用于最后 rescale（保持有效 LR 尺度）
        original_sum = torch.stack([flats[n] for n in active], dim=0).sum(dim=0)
        original_mag = _safe_norm(original_sum, self.eps)

        # ---- 4) Magnitude balance：把每个任务的模长拉到同一目标 ----
        if self.mag_balance:
            # 优先用 anchor 的 norm 当 target：更能“保护主任务”
            if anchor is not None:
                target = norms[anchor].detach()
            else:
                target = torch.stack([norms[n] for n in active]).mean().detach()

            for i in range(T):
                n = _safe_norm(G[i], self.eps)
                G[i] = G[i] * (target / n)

        # ---- 5) Sequential PCGrad（anchor 不动；其它任务依次对已处理梯度投影去冲突）----
        # deterministic=True 时使用固定顺序：anchor -> aux1 -> aux2...
        # 如果你更想要“去顺序 bias”，可以 deterministic=False，然后对 aux 部分随机 permute
        if not self.deterministic and T > 2:
            # 保持 anchor 在 0，打乱 aux 的顺序
            aux = G[1:]
            perm = torch.randperm(len(aux), device=device)
            aux = [aux[i] for i in perm.tolist()]
            G = [G[0]] + aux

        # anchor 梯度不投影（i=0）
        processed: List[torch.Tensor] = [G[0]]
        for i in range(1, T):
            g_i = G[i]
            # 依次对已处理梯度投影（包括 anchor）
            for g_j in processed:
                g_i = _project_conflicting(g_i, g_j, eps=self.eps)
            processed.append(g_i)

        # ---- 6) 合成：anchor + aux（可选 aux_weight）----
        v = processed[0].clone()
        if T > 1:
            aux_sum = torch.stack(processed[1:], dim=0).sum(dim=0)
            v = v + (self.aux_weight * aux_sum)

        # ---- 7) Rescale：把合成后梯度模长拉回 original_mag ----
        if self.rescale_grads:
            cur_mag = _safe_norm(v, self.eps)
            v = v * (original_mag / cur_mag)

        self._cached = _FlatPack(vec=v.detach(), splits=splits, params=params)

        dev = next(iter(task_losses.values())).device
        return {k: torch.ones((), device=dev) for k in task_losses}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        # 在 AMP scaler.unscale_(optimizer) 之后调用，覆写共享 trunk 的 .grad
        if self._cached is None:
            return

        pack = self._cached
        _unflatten_to_params(pack.vec, pack.params, pack.splits)

        # 清空 cache（避免误用到下一 step）
        self._cached = None

