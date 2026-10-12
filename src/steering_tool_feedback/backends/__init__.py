"""Бэкенды моделей: load_backend(cfg.model) -> HFBackend | MockBackend."""

from __future__ import annotations

from .base import Backend, Decision, Prompt, ResolvedSteer, auto_layers
from .mock import MockBackend

__all__ = ["Backend", "Decision", "MockBackend", "Prompt", "ResolvedSteer", "auto_layers",
           "load_backend"]


def load_backend(model_cfg) -> Backend:
    if model_cfg.name == "mock":
        return MockBackend()
    from .hf import HFBackend

    return HFBackend.load(model_cfg)
