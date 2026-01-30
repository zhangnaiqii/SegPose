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


def get_shared_params(model, strategy: str | None = None) -> list[torch.Tensor]:
    strategy = (strategy or "body").lower()
    if strategy == "all":
        return [p for p in model.parameters() if p.requires_grad]

    modules = None
    if hasattr(model, "model"):
        modules = list(model.model)
        if strategy in {"body", "backbone_neck", "trunk"}:
            modules = modules[:-1]
        elif strategy == "backbone":
            backbone_len = len(getattr(model, "yaml", {}).get("backbone", []))
            modules = modules[:backbone_len] if backbone_len else modules[:-1]
        elif strategy == "head":
            modules = modules[-1:]

    if modules is None:
        return [p for p in model.parameters() if p.requires_grad]

    params: list[torch.Tensor] = []
    seen = set()
    for module in modules:
        for param in module.parameters(recurse=True):
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params
