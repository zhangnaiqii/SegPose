# ultralytics/utils/gradbalance/DAGR.py
from __future__ import annotations

import math
from typing import Iterable, Literal

import torch

from . import GradientBalancer, register_grad_balancer


@register_grad_balancer
class DAGRBalancer(GradientBalancer):
    """
    Density-Aware Analytic Gradient Harmonization (DA-AGH)
    implemented as a drop-in replacement for the original RGBBalancer interface.

    这是“覆盖 shared grads 的手术刀”，不是 loss-weighting。
    正确调用顺序（单卡 / DDP 都适用）：
      1) forward 得到 task_losses（每个任务一个标量 loss，保留计算图）
      2) bal.update(task_losses, shared_params=shared_params)      # 必须在 backward 之前
      3) total_loss.backward()                                     # 正常 backward
      4) scaler.unscale_(optimizer)
      5) bal.apply_after_unscale(shared_params=shared_params)      # 覆盖 shared grads（DDP 下内部 all_reduce v）
      6) clip_grad / scaler.step / scaler.update

    DA-AGH 核心：
      - 计算每个任务在 shared-parameter 上的梯度 g_i；
      - 用梯度密度度量 GDM 得到密度感知权重 rho_i；
      - 用 rho_i 计算加权共识方向 d*；
      - 动态计算冲突插值系数 beta，对梯度进行“旋转协调”；
      - 最终方向 v 经过幅值恢复后写回。
    """

    def __init__(
            self,
            task_names: list[str] | None = None,
            interval: int = 1,
            warmup: int = 0,
            # ---- legacy RGB args (kept for backward compatibility; unused by DA-AGH) ----
            alpha: float = 0.5,
            lr: float = 0.1,
            mu: float = 0.9,
            steps: int = 10,
            eps: float = 1e-8,
            magnitude: Literal["rms", "mean", "max"] = "rms",
            alpha_min: float = 0.0,
            alpha_max: float = 0.5 * math.pi,
            # ---- engineering ----
            lock_shared_params: bool = True,
            ddp_sync_v: bool = True,
            # ---- DA-AGH args ----
            gamma: float = 0.5,  # 密度敏感度（保留参数接口）
            beta_base: float = 0.8,  # 插值刚性（越大越向共识方向靠拢）
            quantile: float = 0.75,  # 幅值恢复分位数（0~1）
            use_abs_cos: bool = True,  # 是否使用 |cos|
            aggregate: Literal["mean", "density_weighted"] = "mean",  # 最终聚合方式
            use_ema_consensus: bool = False,  # 可选：对 d* 做 EMA
    ) -> None:
        super().__init__(task_names=task_names, interval=interval, warmup=warmup)

        # legacy kept
        self.lam = float(alpha)
        self.alpha_lr = float(lr)
        self.mu = float(mu)
        self.steps = int(steps)

        self.eps = float(eps)
        self.magnitude = str(magnitude)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if not (self.alpha_min <= self.alpha_max):
            raise ValueError(f"RGBBalancer: alpha_min must <= alpha_max, got {self.alpha_min} > {self.alpha_max}")

        self.lock_shared_params = bool(lock_shared_params)
        self.ddp_sync_v = bool(ddp_sync_v)

        # DA-AGH
        self.gamma = float(gamma)
        self.beta_base = float(beta_base)
        self.quantile = float(quantile)
        if not (0.0 <= self.quantile <= 1.0):
            raise ValueError(f"RGBBalancer(DA-AGH): quantile must be in [0,1], got {self.quantile}")
        if self.beta_base < 0.0:
            raise ValueError(f"RGBBalancer(DA-AGH): beta_base must be >= 0, got {self.beta_base}")

        self.use_abs_cos = bool(use_abs_cos)
        self.aggregate = str(aggregate)
        if self.aggregate not in {"mean", "density_weighted"}:
            raise ValueError(f"RGBBalancer(DA-AGH): unknown aggregate='{self.aggregate}'")
        self.use_ema_consensus = bool(use_ema_consensus)

        # caches
        self._dt: torch.Tensor | None = None  # optional EMA consensus
        self._v: torch.Tensor | None = None  # latest shared update direction
        self._splits: list[int] | None = None  # per-param numel splits

        # engineering safety
        self._shared_params: list[torch.Tensor] | None = None
        self._shared_id: list[int] | None = None

    def compute_density_weight(self, G: torch.Tensor) -> torch.Tensor:
        """
        计算密度感知权重 (Gradient Density Metric, GDM)。
        逻辑：
        1. Density = L1 / (L2 * sqrt(D))。
           - 稀疏任务 (Sparse): Density -> 0
           - 稠密任务 (Dense): Density -> 1
        2. 我们希望保护稀疏任务，权重应与密度负相关。
        """
        T, D = G.shape
        if D == 0:
            return torch.ones(T, device=G.device) / T

        # 1. 计算 GDM 密度
        g_l1 = torch.linalg.norm(G, ord=1, dim=1)
        g_l2 = torch.linalg.norm(G, ord=2, dim=1).clamp_min(1e-8)
        # density 取值范围约为 [0, 1]
        density = g_l1 / (g_l2 * math.sqrt(D))

        # 2. 转换为权重：稀疏优先 (Sparse-First)
        # 使用 (1 - density) 使得低密度任务获得高权重
        raw_weights = (1.0 - density).clamp_min(1e-6)

        # 3. 归一化
        rho_weights = raw_weights / raw_weights.sum()
        return rho_weights

    def update(
            self,
            task_losses: dict[str, torch.Tensor],
            shared_params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        DAGR 核心更新逻辑
        """
        self.step += 1
        if not task_losses:
            return {}

        # ---- select task names ----
        if self.task_names is None:
            names = list(task_losses.keys())
        else:
            names = [n for n in self.task_names if n in task_losses]

        if not names:
            any_loss = next(iter(task_losses.values()))
            return {k: torch.ones((), device=any_loss.device, dtype=any_loss.dtype) for k in task_losses.keys()}

        # ---- build / lock shared param list ----
        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            any_loss = next(iter(task_losses.values()))
            return {k: torch.ones((), device=any_loss.device, dtype=any_loss.dtype) for k in task_losses.keys()}

        if self.lock_shared_params:
            if self._shared_params is None:
                self._shared_params = list(shared_in)
                self._shared_id = [int(p.data_ptr()) for p in self._shared_params]
                # [FIXED] 必须在这里赋值 shared，否则第一次运行会报 UnboundLocalError
                shared = self._shared_params
            else:
                if len(shared_in) != len(self._shared_params):
                    self._shared_params = list(shared_in)
                shared = self._shared_params
        else:
            shared = shared_in

        device = next(iter(task_losses.values())).device
        dtype = next(iter(task_losses.values())).dtype

        # ---- warmup / interval check ----
        if self.step <= self.warmup:
            self._v = None
            self._splits = None
            self._last_weights = {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}
            if self.use_ema_consensus:
                self._update_dt(names, task_losses, shared)
            return self._last_weights

        if (self.step - 1) % self.interval != 0 and self._last_weights is not None:
            return self._last_weights

        # ---- compute per-task gradients ----
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
            if not torch.isfinite(flat).all():
                continue
            g_list.append(flat)

        if not g_list:
            return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}

        # Shape: (T, D)
        G = torch.stack(g_list)

        # 1. 计算密度权重
        rho_weights = self.compute_density_weight(G)  # (T,)

        # 2. 计算共识方向 d_star
        G_normalized = torch.nn.functional.normalize(G, p=2, dim=1, eps=1e-8)
        d_star = (rho_weights.view(-1, 1) * G_normalized).sum(dim=0)
        d_star_norm = torch.linalg.norm(d_star)

        if d_star_norm <= self.eps:
            d_star = G_normalized.mean(dim=0)
            d_star = torch.nn.functional.normalize(d_star, p=2, dim=0, eps=1e-8)
        else:
            d_star = d_star / d_star_norm

        # EMA Consensus Logic
        if self.use_ema_consensus:
            if self._dt is None or self._dt.numel() != d_star.numel():
                self._dt = d_star.detach()
            else:
                self._dt = (self.mu * self._dt + (1.0 - self.mu) * d_star).detach()
                self._dt = torch.nn.functional.normalize(self._dt, p=2, dim=0, eps=1e-8)
            d_ref = self._dt
        else:
            d_ref = d_star

        # 3. 动态计算 Beta 并旋转
        # cos_sim in [-1, 1]
        cos_sim = torch.mv(G_normalized, d_ref)

        # 修正: 线性映射，冲突越大(cos=-1) beta越大(1.0)，正交(cos=0) beta中等(0.5)，一致(cos=1) beta最小(0.0)
        beta = self.beta_base * 0.5 * (1.0 - cos_sim).view(-1, 1)

        # 旋转公式: g_harm = (1-beta)*g + beta*|g|*d_ref
        G_mags = torch.linalg.norm(G, dim=1, keepdim=True)
        G_harm = (1.0 - beta) * G + beta * G_mags * d_ref.view(1, -1)

        # 4. 幅值恢复与聚合
        target_mag = self._compute_target_magnitude(G_mags.squeeze(1))

        if self.aggregate == "density_weighted":
            v_final = (rho_weights.view(-1, 1) * G_harm).sum(dim=0)
        else:
            v_final = G_harm.mean(dim=0)

        v_norm = torch.linalg.norm(v_final)
        if v_norm > self.eps:
            v_final = v_final / v_norm * target_mag

        # Cache for apply_after_unscale
        self._v = v_final
        self._splits = []
        for p in shared:
            self._splits.append(p.numel())

        return {k: torch.ones((), device=device, dtype=dtype) for k in task_losses.keys()}

    def apply_after_unscale(self, shared_params: Iterable[torch.Tensor]) -> None:
        """
        必须在 scaler.unscale_(optimizer) 之后、clip_grad/optimizer.step 之前调用。
        覆盖 shared grads 为缓存的 v；DDP 下会先同步 v 再覆盖。
        """
        if self._v is None or self._splits is None:
            return

        shared_in = [p for p in shared_params if getattr(p, "requires_grad", False)]
        if not shared_in:
            return

        if self.lock_shared_params and self._shared_params is not None:
            shared = self._shared_params
            cur_id = [int(p.data_ptr()) for p in shared_in]
            # 允许顺序一致但对象重建的情况（某些DDP实现），但严格检查数量
            if len(cur_id) != len(self._shared_id):
                # 这种情况下通常是 shared params 发生了变化，安全起见直接返回不覆盖，
                # 或者抛出警告。为了不中断训练，这里选择不做操作。
                return
        else:
            shared = shared_in

        splits = [int(p.numel()) for p in shared]
        if sum(splits) != int(self._v.numel()):
            return

        v = self._v.to(device=shared[0].device, dtype=torch.float32)

        # ---- DDP sync ----
        if self.ddp_sync_v and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            world = torch.distributed.get_world_size()
            if world > 1:
                v.div_(float(world))

        # ---- write back to grads ----
        offset = 0
        for p, n in zip(shared, splits):
            piece = v[offset: offset + n].view_as(p)
            offset += n

            if p.grad is None:
                p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device)

            piece_cast = piece.to(dtype=p.grad.dtype, device=p.grad.device)
            p.grad.detach_()
            p.grad.copy_(piece_cast)

    # ----------------------- helpers -----------------------

    def _flatten_grads(
            self,
            shared: list[torch.Tensor],
            grads: tuple[torch.Tensor | None, ...],
            device: torch.device,
    ) -> torch.Tensor:
        flat = []
        for p, g in zip(shared, grads):
            if g is None:
                flat.append(torch.zeros((p.numel(),), device=device, dtype=torch.float32))
            else:
                flat.append(g.reshape(-1).to(device=device, dtype=torch.float32))
        return torch.cat(flat, dim=0)

    def _compute_target_magnitude(self, mags: torch.Tensor) -> torch.Tensor:
        """
        mags: [T] float32.
        """
        if math.isnan(self.quantile):
            return self._legacy_target_magnitude(mags)

        q = float(self.quantile)
        if q <= 0.0:
            return mags.min().clamp_min(self.eps)
        if q >= 1.0:
            return mags.max().clamp_min(self.eps)

        target = torch.quantile(mags, q).clamp_min(self.eps)
        if not torch.isfinite(target):
            return self._legacy_target_magnitude(mags)
        return target

    def _legacy_target_magnitude(self, mags: torch.Tensor) -> torch.Tensor:
        if self.magnitude == "rms":
            return torch.sqrt((mags * mags).mean()).clamp_min(self.eps)
        if self.magnitude == "mean":
            return mags.mean().clamp_min(self.eps)
        if self.magnitude == "max":
            return mags.max().clamp_min(self.eps)
        raise ValueError(f"RGBBalancer(DA-AGH): unknown magnitude='{self.magnitude}'")

    def _update_dt_from_direction(self, d: torch.Tensor) -> None:
        if self._dt is None or self._dt.numel() != d.numel() or self._dt.device != d.device:
            self._dt = d.detach()
        else:
            self._dt = (self.mu * self._dt + (1.0 - self.mu) * d).detach()
            self._dt = self._dt / torch.linalg.norm(self._dt).clamp_min(self.eps)

    def _update_dt(self, names: list[str], task_losses: dict[str, torch.Tensor], shared: list[torch.Tensor]) -> None:
        """
        Warmup helper for EMA consensus.
        """
        g_list = []
        device = next(iter(task_losses.values())).device
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
            if not math.isfinite(gn) or gn <= self.eps:
                continue
            g_list.append(flat)

        if not g_list:
            return

        G = torch.stack(g_list, dim=0)
        g_norm = torch.linalg.norm(G, dim=1).clamp_min(self.eps)
        G_unit = G / g_norm[:, None]

        # Use unified density weight calculation
        rho = self.compute_density_weight(G)
        d_star = (rho[:, None] * G_unit).sum(dim=0)
        d_star = d_star / torch.linalg.norm(d_star).clamp_min(self.eps)

        self._update_dt_from_direction(d_star)