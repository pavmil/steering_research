"""Методы, которые сравниваются в одном scaffold (docs/EXPERIMENT_PLAN.md).

Метод = как строится контекст + (опционально) какое вмешательство включено.
Всё, что отличает метод, входит в его fingerprint — по нему кэшируются решения.

| метод            | EXPERIMENT_PLAN | что делает                                          |
|------------------|-----------------|-----------------------------------------------------|
| baseline         | 0               | без вмешательства                                   |
| strong_prompt    | 1               | явная инструкция учитывать ответ (в system)         |
| restate          | 2               | модель пересказывает ответ, затем выбирает действие |
| explicit_state   | 3               | OODA: модель пишет явное состояние задачи, затем действует |
| lora             | 4               | LoRA/SFT на тех же данных, что и вектор              |
| steering         | 5               | CAA-вектор в окне после ответа, всегда включён       |
| steering_gated   | 6               | тот же вектор, но только если обучаемый гейт сработал |
| tact             | адаптация       | TACT-Gate: оттягивание вдоль оси «сбой» в точке действия |
| sham:*           | контроли H3     | случайное/переставленное направление, соседний слой, окно до ответа |

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

# Явное ведение состояния (OODA-подобный контроллер). Состояние пишет сама модель —
# обвязка не знает, какое поле ответа существенно (иначе это была бы подсказка-оракул).
# Пересчитывается на каждом шаге из полной истории.
STATE_ASK = (
    "Before acting, write down your task state in exactly this format:\n"
    "GOAL: <what the user wants>\n"
    "OBSERVED: <what the latest tool result says>\n"
    "PLAN STILL VALID: <yes or no, and why>\n"
    "NEXT STEP: <what to do next>"
)
STATE_GO = "Now call the tool for your next step."

# variant -> (просьба, команда действовать, лимит токенов на текст)
TWO_STEP: dict[str, tuple[str, str, int]] = {
    "restate": (RESTATE_ASK, RESTATE_GO, 48),
    "explicit_state": (STATE_ASK, STATE_GO, 96),
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    variant: str                    # baseline | strong_prompt | restate | explicit_state
    steer: bool = False             # CAA-вектор в окне (SteerSpec)
    sham: str | None = None         # вид sham-контроля для steering
    gate: bool = False              # steering только при срабатывании гейта (GateSpec)
    tact: bool = False              # адаптация TACT (TactSpec)
    adapter: bool = False           # LoRA-адаптер (AdapterSpec)
    fitted: bool = False            # подбирается в стадии fit на val
    implemented: bool = True
    note: str = ""


METHODS: dict[str, MethodSpec] = {
    "baseline": MethodSpec("baseline", "baseline"),
    "strong_prompt": MethodSpec("strong_prompt", "strong_prompt",
                                note="Явная инструкция учитывать ответ инструмента в system."),
    "restate": MethodSpec("restate", "restate",
                          note="Модель сначала пересказывает ответ, затем выбирает действие. "
                               "Два вызова модели — дорого, учитывать в цене."),
    "explicit_state": MethodSpec("explicit_state", "explicit_state",
                                 note="OODA-подобное явное состояние, которое пишет модель "
                                      "перед действием. Главный конкурент по дешевизне."),
    "steering": MethodSpec("steering", "baseline", steer=True),
    "steering_gated": MethodSpec("steering_gated", "baseline", gate=True, fitted=True,
                                 note="Вектор steering включается, только если гейт на "
                                      "активациях предсказывает ошибку базы."),
    "tact": MethodSpec("tact", "baseline", tact=True, fitted=True,
                       note="Адаптация TACT-Gate (2605.05980): одна ось (overacting vs "
                            "calibrated), без оси overthinking — решение без рассуждения."),
    "lora": MethodSpec("lora", "baseline", adapter=True, fitted=True,
                       note="LoRA/SFT на тех же данных, что и вектор (меняющие пары train)."),
}

for _s in ("random_direction", "permuted_coords", "layer_shift", "window_shift"):
    METHODS[f"sham:{_s}"] = MethodSpec(f"sham:{_s}", "baseline", steer=True, sham=_s)

FITTED = tuple(n for n, m in METHODS.items() if m.fitted)


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
    if variant in ("baseline", "restate", "explicit_state"):
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
class GateSpec:
    """steering_gated: вектор + гейт (логистическая регрессия на активациях окна) + порог."""

    steer: SteerSpec
    gate_id: str
    threshold: float

    def fingerprint(self) -> str:
        return sha([self.steer.fingerprint(), self.gate_id, round(self.threshold, 6)], 12)


@dataclass
class TactSpec:
    """tact: оси дрейфа (по слоям), окно, полоса k·sigma, сила lam."""

    axes_id: str
    layers: tuple[int, ...]
    window: str
    k: float
    lam: float

    def fingerprint(self) -> str:
        return sha([self.axes_id, list(self.layers), self.window, round(self.k, 6),
                    round(self.lam, 6)], 12)


@dataclass
class AdapterSpec:
    """lora: обученный адаптер (id = хеш данных + гиперпараметров)."""

    adapter_id: str
    path: str

    def fingerprint(self) -> str:
        return sha(["lora", self.adapter_id], 12)


@dataclass
class MethodRun:
    """Метод + параметры конкретного прогона (для fingerprint и кэша)."""

    spec: MethodSpec
    strong_prompt: str = ""
    steer: SteerSpec | None = None
    gate: GateSpec | None = None
    tact: TactSpec | None = None
    adapter: AdapterSpec | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        parts: list[Any] = [self.spec.variant]
        if self.spec.variant == "strong_prompt":
            parts.append(sha(self.strong_prompt))
        if self.spec.variant in TWO_STEP:
            parts.append(sha(list(TWO_STEP[self.spec.variant])))
        if self.steer is not None:
            parts.append(self.steer.fingerprint())
        for extra in (self.gate, self.tact, self.adapter):
            if extra is not None:
                parts.append(extra.fingerprint())
        return sha(parts, 12)
