from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Iterable

import torch

_REGISTRY: dict[str, type] = {}


def register_grad_balancer(cls: type) -> type:
    _REGISTRY[cls.__name__] = cls
    return cls


def discover_grad_balancers() -> None:
    pkg_path = Path(__file__).resolve().parent
    for module in pkgutil.iter_modules([str(pkg_path)]):
        name = module.name
        if name.startswith("_") or name == "__init__":
            continue
        importlib.import_module(f"{__name__}.{name}")


class GradientBalancer:
    def __init__(self, task_names: list[str] | None = None, interval: int = 1, warmup: int = 0) -> None:
        self.task_names = list(task_names) if task_names else None
        self.interval = max(int(interval), 1)
        self.warmup = max(int(warmup), 0)
        self.step = 0
        self._last_weights: dict[str, torch.Tensor] | None = None

    def __call__(
        self, task_losses: dict[str, torch.Tensor], shared_params: Iterable[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self.update(task_losses, shared_params)

    def update(
        self, task_losses: dict[str, torch.Tensor], shared_params: Iterable[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def build_grad_balancer(name: str | None, **kwargs) -> GradientBalancer | None:
    if not name or str(name).lower() in {"none", "null", "false", "0"}:
        return None
    discover_grad_balancers()
    cls = _REGISTRY.get(str(name))
    if cls is None:
        raise KeyError(f"Unknown grad balancer: {name}")
    return cls(**kwargs)


def get_shared_params(model, strategy=None) -> list[torch.Tensor]:
    """
    Select shared parameters for gradient balancing.

    strategy can be:
      - None: default to "body"
      - str: one of {"all","body","backbone_neck","trunk","backbone","neck","head"}
      - list/tuple: mixed selectors, e.g.
          [11, 14, 17, 20]          -> pick specific module indices from model.model
          ["neck"]                  -> pick neck modules
          ["backbone", 12, 13]      -> union of backbone modules and explicit indices
          ["body", "head"]          -> union (not recommended but allowed)

    Notes:
      - integer indices refer to the index in list(model.model). Negative indices are supported.
      - "neck" is approximated as modules[backbone_len:-1] (exclude head).
      - returns a de-duplicated list of parameters (by id), only those with requires_grad=True.
      - invalid selectors raise ValueError (no silent fallback to "all").
    """
    # fallback: if model has no staged modules, return all trainable params
    if not hasattr(model, "model"):
        return [p for p in model.parameters() if p.requires_grad]

    modules = list(model.model)
    n = len(modules)
    if n == 0:
        return [p for p in model.parameters() if p.requires_grad]

    # normalize strategy into a list of selectors
    if strategy is None:
        selectors = ["body"]
    elif isinstance(strategy, (list, tuple)):
        if len(strategy) == 0:
            raise ValueError("get_shared_params: strategy list is empty.")
        selectors = list(strategy)
    else:
        selectors = [strategy]

    # helpers
    def _norm_idx(i: int) -> int:
        ii = i + n if i < 0 else i
        if ii < 0 or ii >= n:
            raise ValueError(f"get_shared_params: module index out of range: {i} (n={n}).")
        return ii

    def _dedup_params(mods: list) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        seen = set()
        for m in mods:
            for p in m.parameters(recurse=True):
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    # keyword -> index set
    backbone_len = len(getattr(model, "yaml", {}).get("backbone", [])) if hasattr(model, "yaml") else 0

    def _indices_for_keyword(k: str) -> set[int]:
        kk = k.lower().strip()
        if kk == "all":
            return set(range(n))
        if kk in {"body", "backbone_neck", "trunk"}:
            # exclude last head module
            return set(range(max(0, n - 1)))
        if kk == "backbone":
            if backbone_len and 0 < backbone_len <= n:
                return set(range(backbone_len))
            # if backbone_len unknown, fallback to body (exclude head) rather than silently all
            return set(range(max(0, n - 1)))
        if kk == "neck":
            # approx neck: backbone_len .. -1 (exclude head)
            if backbone_len and backbone_len < max(0, n - 1):
                return set(range(backbone_len, max(0, n - 1)))
            # if unknown, fallback to body (still exclude head)
            return set(range(max(0, n - 1)))
        if kk == "head":
            return {n - 1}
        raise ValueError(
            f"get_shared_params: unknown keyword selector '{k}'. "
            "Supported: all/body/backbone_neck/trunk/backbone/neck/head"
        )

    # build selected module indices
    selected: set[int] = set()
    for sel in selectors:
        if isinstance(sel, int):
            selected.add(_norm_idx(sel))
        elif isinstance(sel, str):
            selected |= _indices_for_keyword(sel)
        else:
            raise ValueError(
                f"get_shared_params: invalid selector type {type(sel)}. "
                "Use int (module index) or str (keyword like 'neck')."
            )

    if not selected:
        raise ValueError(f"get_shared_params: empty selection for strategy={strategy}.")

    mods = [modules[i] for i in range(n) if i in selected]
    return _dedup_params(mods)

