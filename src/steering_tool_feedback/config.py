"""Конфиг эксперимента: один объект, который полностью описывает прогон.

Новый эксперимент = новый ноутбук с другим набором переопределений:

    cfg = ExperimentConfig.make(
        name="h2-baseline-qwen3-1.7b",
        hypothesis="H2",
        overrides={"model.name": "Qwen/Qwen3-1.7B", "data.n_tasks": 160},
    )

Хеш конфига (`cfg.hash()`) — ключ кэшей и часть run_id; при любом изменении
параметров результаты не перепутаются со старыми. Поля `name`, `notes` и
`notebook` и `hypothesis` в хеш не входят: они описывают прогон, а не меняют вычисления.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WINDOW_KINDS = ("tool_content", "field_read", "post_message", "pre_action")
SHAMS = ("random_direction", "permuted_coords", "layer_shift", "window_shift")

DEFAULT_STRONG_PROMPT = (
    "Before choosing your next action, re-read the most recent tool result. "
    "Base the next action strictly on what that result says. If it contradicts "
    "your current plan (for example, something is unavailable or access is denied), "
    "change the plan accordingly instead of continuing as before."
)


@dataclass
class ModelCfg:
    name: str = "mock"                 # HF id, локальная папка или "mock"
    device: str = "auto"               # auto | cpu | mps | cuda
    dtype: str = "auto"                # auto | float32 | float16 | bfloat16
    enable_thinking: bool | None = False   # в chat template, если он это понимает
    max_new_tokens: int = 96           # только для decision.mode="generate"
    action_prefix: str | None = None   # если формат вызова не определился из шаблона
    action_suffix: str | None = None


@dataclass
class DataCfg:
    n_tasks: int = 120
    seed: int = 0
    split: tuple[float, float, float] = (0.5, 0.2, 0.3)
    use_tools_schema: bool = True
    k_post: int = 4


@dataclass
class DecisionCfg:
    # scored   — выбор действия по log-prob вызова каждого инструмента сразу после
    #            промпта (без рассуждения). Детерминированно и быстро: основной режим.
    # generate — модель генерирует ответ (с рассуждением, если оно включено),
    #            действие извлекается парсером. Медленно; для финальной проверки.
    mode: str = "scored"
    n_trials: int = 1                  # для generate с семплированием (pass^k)
    temperature: float = 0.0           # 0 = жадно


@dataclass
class ProbeCfg:
    layers: list[int] | str = "auto"   # "auto" = ~6 слоёв равномерно по глубине
    n_auto_layers: int = 6
    windows: list[str] = field(default_factory=lambda: ["field_read", "post_message", "pre_action"])


@dataclass
class SteerCfg:
    layers: list[int] | str = "auto"   # "auto" = лучшие по probe (top_k_layers)
    top_k_layers: int = 3
    windows: list[str] = field(default_factory=lambda: ["field_read", "post_message", "pre_action"])
    alphas: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0])
    normalize: bool = True
    min_class_examples: int = 4        # минимум верных И неверных решений на train для CAA
    max_specificity_drop: float = 0.05 # допустимое падение специфичности на val (METRICS.md)


@dataclass
class EvalCfg:
    methods: list[str] = field(default_factory=lambda: ["baseline", "strong_prompt", "steering"])
    shams: list[str] = field(default_factory=lambda: list(SHAMS))
    strong_prompt: str = DEFAULT_STRONG_PROMPT
    n_boot: int = 10_000
    h2_max_sensitivity: float = 0.9    # выше — ошибок после ответа мало, разрыва нет (H2 против)
    reopen_test_reason: str | None = None  # повторное открытие test — только с причиной


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    hypothesis: str = "explore"        # H1 | H2 | H3 | H4 | explore
    notebook: str = ""
    notes: str = ""
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)
    probe: ProbeCfg = field(default_factory=ProbeCfg)
    steer: SteerCfg = field(default_factory=SteerCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    _DESCRIPTIVE = ("name", "hypothesis", "notes", "notebook")

    # ------------------------------------------------------------------
    @classmethod
    def make(cls, name: str | None = None, hypothesis: str | None = None,
             overrides: dict[str, Any] | None = None, base: str | Path | None = None,
             **top: Any) -> "ExperimentConfig":
        """Базовый конфиг (по умолчанию или из YAML) + переопределения.

        overrides — точечные ключи: {"model.name": "...", "steer.alphas": [1, 2]}
        или вложенные словари: {"model": {"name": "..."}}.
        name / hypothesis перекрывают значения из YAML, только если заданы.
        """
        cfg = load_yaml(base) if base else cls()
        if name is not None:
            cfg.name = name
        if hypothesis is not None:
            cfg.hypothesis = hypothesis
        for k, v in top.items():
            _set(cfg, k, v)
        for k, v in _flatten(overrides or {}).items():
            _set(cfg, k, v)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def compute_dict(self) -> dict[str, Any]:
        """То, что влияет на вычисления (без описательных полей)."""
        d = self.to_dict()
        for k in self._DESCRIPTIVE:
            d.pop(k, None)
        d["eval"].pop("reopen_test_reason", None)   # причина — описание, не параметр
        return d

    def hash(self) -> str:
        blob = json.dumps(self.compute_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def diff_from_default(self) -> dict[str, tuple[Any, Any]]:
        """Что поменяли относительно конфига по умолчанию — для журнала."""
        base = _flatten(ExperimentConfig().compute_dict())
        cur = _flatten(self.compute_dict())
        return {k: (base.get(k), v) for k, v in cur.items() if base.get(k) != v}

    def validate(self) -> None:
        if self.decision.mode not in ("scored", "generate"):
            raise ValueError("decision.mode must be 'scored' or 'generate'")
        if abs(sum(self.data.split) - 1.0) > 1e-9:
            raise ValueError("data.split must sum to 1")
        for w in list(self.probe.windows) + list(self.steer.windows):
            if w not in WINDOW_KINDS:
                raise ValueError(f"unknown window {w!r}; choose from {WINDOW_KINDS}")
        for s in self.eval.shams:
            if s not in SHAMS:
                raise ValueError(f"unknown sham {s!r}; choose from {SHAMS}")
        if self.decision.mode == "scored" and self.decision.n_trials != 1:
            raise ValueError("scored mode is deterministic: decision.n_trials must be 1")
        if self.decision.n_trials > 1 and self.decision.temperature <= 0:
            raise ValueError("n_trials > 1 needs temperature > 0, otherwise trials are identical")

    def save_yaml(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False))


def load_yaml(path: str | Path) -> ExperimentConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = ExperimentConfig()
    for k, v in _flatten(raw).items():
        _set(cfg, k, v)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------


_SECTIONS = ("model", "data", "decision", "probe", "steer", "eval")


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and (k in _SECTIONS or prefix):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _set(cfg: ExperimentConfig, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise KeyError(f"unknown config section {p!r} in {dotted!r}")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not dataclasses.is_dataclass(obj) or leaf not in {f.name for f in dataclasses.fields(obj)}:
        raise KeyError(f"unknown config key {dotted!r}")
    if isinstance(getattr(obj, leaf), tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(obj, leaf, copy.deepcopy(value))
