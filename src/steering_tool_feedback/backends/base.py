"""Интерфейс бэкенда. Пайплайн знает только его — модель за ним может быть
настоящей (HFBackend: CPU / MPS / CUDA) или синтетической (MockBackend)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..steering import DriftAxis, SteeringVector
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
class ResolvedTact:
    """Адаптация TACT-Gate: оси дрейфа на нескольких слоях, окно, полоса k·sigma, сила lam."""

    axes: list[DriftAxis]
    window: PositionWindow
    k: float
    lam: float


@dataclass
class ResolvedAdapter:
    """LoRA-адаптер: включается только на время решения, база не меняется."""

    adapter_id: str
    path: str


Intervention = ResolvedSteer | ResolvedTact


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
    hook_changed: int | None = None            # TACT: сколько позиций реально сдвинуто
    arguments: dict[str, Any] | None = None    # with_args=True: аргументы вызова от модели


class Backend(Protocol):
    model_key: str
    n_layers: int
    hidden_size: int

    def info(self) -> dict[str, Any]: ...

    def prompt(self, messages: list[dict[str, Any]], essential_field: str | None,
               tools: list[dict] | None, meta: dict[str, Any] | None = None,
               k_post: int = 4) -> Prompt: ...

    def decide(self, prompt: Prompt, mode: str, steer: Intervention | None,
               temperature: float, trial: int, adapter: ResolvedAdapter | None = None,
               with_args: bool = False, arg_tokens: int = 40) -> Decision: ...

    def decide_two_step(self, prompt: Prompt, ask: str, go: str, ask_tokens: int, mode: str,
                        temperature: float, trial: int, adapter: ResolvedAdapter | None = None,
                        with_args: bool = False, arg_tokens: int = 40) -> Decision: ...

    def record(self, prompt: Prompt, layers: list[int],
               windows: list[str]) -> dict[tuple[int, str], np.ndarray]: ...

    def sft_example(self, prompt: Prompt, action: str,
                    arguments: dict[str, Any]) -> tuple[list[int], list[int]]: ...

    def train_adapter(self, examples: list[tuple[list[int], list[int]]], lora_cfg: Any,
                      lr: float, epochs: int, out_dir: str) -> dict[str, Any]: ...


def auto_layers(n_layers: int, k: int) -> list[int]:
    """~k слоёв равномерно по глубине, без первого и последнего."""
    if n_layers <= 2:
        return list(range(n_layers))
    k = max(1, min(k, n_layers - 2))
    xs = np.linspace(1, n_layers - 2, k)
    return sorted({int(round(x)) for x in xs})
