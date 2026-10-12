"""Интерфейс бэкенда. Пайплайн знает только его — модель за ним может быть
настоящей (HFBackend: CPU / MPS / CUDA) или синтетической (MockBackend)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..steering import SteeringVector
from ..windows import PositionWindow


@dataclass
class Prompt:
    key: str                                   # хеш входа модели — ключ кэшей
    windows: dict[str, PositionWindow]
    n_tokens: int
    messages: list[dict[str, Any]]
    tools: list[dict] | None
    prepared: Any = None                       # PreparedPrompt у HF
    meta: dict[str, Any] = field(default_factory=dict)  # только для MockBackend


@dataclass
class ResolvedSteer:
    vector: SteeringVector
    window: PositionWindow
    alpha: float
    normalize: bool = True


@dataclass
class Decision:
    action: str
    scores: dict[str, float] | None = None     # scored: log-prob вызова каждого действия
    text: str | None = None                    # generate: сырой текст ответа
    n_prompt_tokens: int = 0
    n_new_tokens: int = 0
    n_model_calls: int = 1
    latency_ms: float = 0.0
    hook_applied: int | None = None
    positions_source: str | None = None


class Backend(Protocol):
    model_key: str
    n_layers: int
    hidden_size: int

    def info(self) -> dict[str, Any]: ...

    def prompt(self, messages: list[dict[str, Any]], essential_field: str | None,
               tools: list[dict] | None, meta: dict[str, Any] | None = None,
               k_post: int = 4) -> Prompt: ...

    def decide(self, prompt: Prompt, mode: str, steer: ResolvedSteer | None,
               temperature: float, trial: int) -> Decision: ...

    def decide_restate(self, prompt: Prompt, mode: str, temperature: float,
                       trial: int) -> Decision: ...

    def record(self, prompt: Prompt, layers: list[int],
               windows: list[str]) -> dict[tuple[int, str], np.ndarray]: ...


def auto_layers(n_layers: int, k: int) -> list[int]:
    """~k слоёв равномерно по глубине, без первого и последнего."""
    if n_layers <= 2:
        return list(range(n_layers))
    k = max(1, min(k, n_layers - 2))
    xs = np.linspace(1, n_layers - 2, k)
    return sorted({int(round(x)) for x in xs})
