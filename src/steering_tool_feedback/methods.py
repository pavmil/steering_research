"""Методы, которые сравниваются в одном scaffold (docs/EXPERIMENT_PLAN.md).

Метод = как строится контекст + (опционально) какое вмешательство включено.
Всё, что отличает метод, входит в его fingerprint — по нему кэшируются решения.

Добавить новый метод = добавить запись в METHODS и, если нужно, ветку в
build_messages. Пайплайн, метрики и отчёт подхватят его сами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import simulator as S
from .store import sha

RESTATE_ASK = "In one sentence, restate exactly what the last tool result says."
RESTATE_GO = "Now choose your next action."


@dataclass(frozen=True)
class MethodSpec:
    name: str
    variant: str                    # baseline | strong_prompt | restate
    steer: bool = False             # использует выбранный steering-вектор
    sham: str | None = None         # вид sham-контроля для steering
    implemented: bool = True
    note: str = ""


METHODS: dict[str, MethodSpec] = {
    "baseline": MethodSpec("baseline", "baseline"),
    "strong_prompt": MethodSpec("strong_prompt", "strong_prompt",
                                note="Явная инструкция учитывать ответ инструмента в system."),
    "restate": MethodSpec("restate", "restate",
                          note="Модель сначала пересказывает ответ, затем выбирает действие. "
                               "Два вызова модели — дорого, учитывать в цене."),
    "steering": MethodSpec("steering", "baseline", steer=True),
    "lora": MethodSpec("lora", "baseline", implemented=False,
                       note="LoRA/SFT на тех же данных — ещё не реализован (peft)."),
}

for _s in ("random_direction", "permuted_coords", "layer_shift", "window_shift"):
    METHODS[f"sham:{_s}"] = MethodSpec(f"sham:{_s}", "baseline", steer=True, sham=_s)


def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {sorted(METHODS)}")
    m = METHODS[name]
    if not m.implemented:
        raise NotImplementedError(f"method {name!r} is not implemented yet: {m.note}")
    return m


def build_messages(task: S.PairedTask, branch: int, variant: str,
                   strong_prompt: str = "") -> list[dict[str, Any]]:
    """Контекст для метода. Только то, что агент видел бы и так — без парной ветки и gold."""
    msgs = task.prompt(branch)
    if variant == "baseline" or variant == "restate":
        return msgs
    if variant == "strong_prompt":
        sys = dict(msgs[0])
        sys["content"] = f"{sys['content']}\n\n{strong_prompt}"
        return [sys, *msgs[1:]]
    raise ValueError(f"unknown variant {variant!r}")


@dataclass
class SteerSpec:
    """Конкретное вмешательство: вектор + окно + сила."""

    layer: int
    window: str
    alpha: float
    vector_id: str                 # хеш вектора — часть fingerprint
    normalize: bool = True
    window_mode: str = "exact"     # exact | before_tool (sham: то же окно, но ДО ответа инструмента)
    tag: str = ""

    def fingerprint(self) -> str:
        return sha([self.layer, self.window, round(self.alpha, 6), self.vector_id,
                    self.normalize, self.window_mode], 12)


@dataclass
class MethodRun:
    """Метод + параметры конкретного прогона (для fingerprint и кэша)."""

    spec: MethodSpec
    strong_prompt: str = ""
    steer: SteerSpec | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        parts: list[Any] = [self.spec.variant]
        if self.spec.variant == "strong_prompt":
            parts.append(sha(self.strong_prompt))
        if self.spec.variant == "restate":
            parts.append(sha([RESTATE_ASK, RESTATE_GO]))
        if self.steer is not None:
            parts.append(self.steer.fingerprint())
        return sha(parts, 12)
