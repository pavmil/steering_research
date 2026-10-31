"""Метрики проекта. Определения зафиксированы в docs/METRICS.md.

Зависит только от numpy, чтобы метрики можно было считать и тестировать
без модели и без torch.

Главное правило, встроенное в API: sensitivity и specificity возвращаются
только вместе (см. PairedResult.target_effect). Метод, поднявший переключения
на меняющих парах за счёт инвариантных, ничего не улучшил.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

PairType = Literal["changing", "invariant", "stress", "global", "control"]

# Целевые и охранные пары (как в simulator.TARGET_TYPES / GUARD_TYPES). На глобальном срезе (H4′)
# global — принятие правила, control — сохранность; в таблицах это те же sensitivity / specificity.
TARGET_TYPES = ("changing", "global")
GUARD_TYPES = ("invariant", "control")
MAIN_TYPES = TARGET_TYPES + GUARD_TYPES


# --------------------------------------------------------------------------
# Слой 1. Успех
# --------------------------------------------------------------------------


def pass_hat_k(successes_per_task: Sequence[Sequence[bool]], k: int) -> float:
    """pass^k из tau-bench (2406.12045): вероятность, что ВСЕ k прогонов успешны.

    Несмещённая оценка. Для задачи с n прогонами и c успехами
    P(все k успешны) = C(c, k) / C(n, k), затем усреднение по задачам.

    Нужна именно эта метрика, а не средняя: для агента важна стабильность,
    средняя прячет метод, который иногда ломает задачу.

    Задачи с n < k пропускаются (оценка неопределена).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    per_task = []
    for trials in successes_per_task:
        n = len(trials)
        if n < k:
            continue
        c = int(sum(bool(t) for t in trials))
        per_task.append(math.comb(c, k) / math.comb(n, k) if c >= k else 0.0)

    if not per_task:
        raise ValueError(f"no task has at least k={k} trials")
    return float(np.mean(per_task))


def episode_success(successes: Sequence[bool]) -> float:
    """Доля успешно завершённых эпизодов."""
    if not successes:
        raise ValueError("empty successes")
    return float(np.mean([bool(s) for s in successes]))


# --------------------------------------------------------------------------
# Слой 2. Целевой эффект
# --------------------------------------------------------------------------


def normalized_gain(y_steered: float, y_base: float) -> float:
    """EFFECTIVENESS из SteeringControl (2509.13450): (y_s - y) / (1 - y).

    Нормировка на остаток до потолка. Нужна, когда у методов разная база
    и абсолютные приросты несравнимы.

    При y_base == 1.0 роста быть не может: возвращаем 0.0, если не упали,
    иначе отрицательную величину относительно самого падения.
    """
    if not (0.0 <= y_base <= 1.0 and 0.0 <= y_steered <= 1.0):
        raise ValueError("scores must lie in [0, 1]")
    headroom = 1.0 - y_base
    if headroom <= 1e-12:
        return 0.0 if y_steered >= y_base else float(y_steered - y_base)
    return float((y_steered - y_base) / headroom)


@dataclass(frozen=True)
class PairOutcome:
    """Результат одной пары.

    task_id    — исходная задача; обе ветки и все повторы делят один id,
                 по нему группируем при bootstrap.
    pair_type  — changing / invariant / stress (условный срез), global / control (глобальный, H4′).
    action_ok_0, action_ok_1 — попало ли следующее действие в множество
                 допустимых A* в каждой ветке (не совпадение со строкой-эталоном!).
    action_changed — различается ли выбранное действие между ветками.
    success_0, success_1 — успех всего эпизода в каждой ветке.
    """

    task_id: str
    pair_type: PairType
    action_ok_0: bool
    action_ok_1: bool
    action_changed: bool
    success_0: bool
    success_1: bool

    @property
    def behaved_correctly(self) -> bool:
        """Для changing действие должно смениться, для invariant и global — нет;
        для control и stress — верно в обеих ветках."""
        if self.pair_type == "changing":
            return self.action_changed and self.action_ok_0 and self.action_ok_1
        if self.pair_type in ("invariant", "global"):
            return (not self.action_changed) and self.action_ok_0 and self.action_ok_1
        return self.action_ok_0 and self.action_ok_1


@dataclass
class TargetEffect:
    """Чувствительность и специфичность. Намеренно в одном объекте."""

    sensitivity: float
    specificity: float
    n_changing: int
    n_invariant: int

    def __str__(self) -> str:
        return (
            f"sensitivity={self.sensitivity:.3f} (n={self.n_changing}) | "
            f"specificity={self.specificity:.3f} (n={self.n_invariant})"
        )


def target_effect(outcomes: Sequence[PairOutcome]) -> TargetEffect:
    """Доля корректных переключений на changing и корректных сохранений на invariant
    (на глобальном срезе — принятие правила на global и сохранность на control).

    Отчитываются только вместе — порознь они не интерпретируются.
    """
    changing = [o for o in outcomes if o.pair_type in TARGET_TYPES]
    invariant = [o for o in outcomes if o.pair_type in GUARD_TYPES]
    if not changing or not invariant:
        raise ValueError(
            "need both target (changing/global) and guard (invariant/control) pairs; "
            "reporting sensitivity without specificity is meaningless"
        )
    return TargetEffect(
        sensitivity=float(np.mean([o.behaved_correctly for o in changing])),
        specificity=float(np.mean([o.behaved_correctly for o in invariant])),
        n_changing=len(changing),
        n_invariant=len(invariant),
    )


def switch_rate(outcomes: Sequence[PairOutcome]) -> float:
    """Частота смены действия, по всем парам.

    Диагностика, НЕ метрика качества. Рост частоты смены действия сам по себе
    не улучшение — это самая вероятная ошибка интерпретации в проекте.
    """
    if not outcomes:
        raise ValueError("empty outcomes")
    return float(np.mean([o.action_changed for o in outcomes]))


# --------------------------------------------------------------------------
# Слой 3. Побочные эффекты
# --------------------------------------------------------------------------


@dataclass
class EpisodeDelta:
    """Испорченные и исправленные эпизоды относительно базы."""

    fixed: int
    broken: int
    unchanged: int

    @property
    def net(self) -> int:
        return self.fixed - self.broken


def episode_delta(base: Sequence[bool], method: Sequence[bool]) -> EpisodeDelta:
    """Поэпизодное сравнение: что метод исправил, а что сломал.

    Средний прирост может быть положительным при большом числе сломанных
    эпизодов — поэтому считаем обе величины отдельно.
    """
    if len(base) != len(method):
        raise ValueError("base and method must align episode-by-episode")
    fixed = broken = unchanged = 0
    for b, m in zip(base, method):
        if m and not b:
            fixed += 1
        elif b and not m:
            broken += 1
        else:
            unchanged += 1
    return EpisodeDelta(fixed=fixed, broken=broken, unchanged=unchanged)


def entanglement(
    base_by_task: dict[str, float], steered_by_task: dict[str, float]
) -> float:
    """ENTANGLEMENT из SteeringControl: сдвиг на поведениях, которые не трогали.

    Считается по НЕцелевым (OOD) наборам: средний абсолютный сдвиг счёта.
    Берём абсолютное значение, потому что любой сдвиг на нецелевом поведении —
    побочный эффект, даже если он в «хорошую» сторону.
    """
    shared = sorted(set(base_by_task) & set(steered_by_task))
    if not shared:
        raise ValueError("no shared off-target datasets")
    return float(
        np.mean([abs(steered_by_task[t] - base_by_task[t]) for t in shared])
    )


# --------------------------------------------------------------------------
# Парная оценка эффекта с bootstrap
# --------------------------------------------------------------------------


@dataclass
class PairedEstimate:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_tasks: int
    excludes_zero: bool = field(init=False)

    def __post_init__(self) -> None:
        self.excludes_zero = (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def __str__(self) -> str:
        verdict = "excludes 0" if self.excludes_zero else "includes 0"
        return (
            f"diff={self.mean_diff:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"({verdict}, n={self.n_tasks})"
        )


def paired_bootstrap(
    base_by_task: dict[str, Sequence[bool]],
    method_by_task: dict[str, Sequence[bool]],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedEstimate:
    """Средняя по задачам разность успеха (метод - база) с bootstrap-интервалом.

    Ресемплинг по ЗАДАЧАМ, а не по эпизодам: обе ветки пары и все повторы
    одной задачи зависимы, поэтому ресемплить их по отдельности нельзя —
    интервал получится узким и соврёт.
    """
    tasks = sorted(set(base_by_task) & set(method_by_task))
    if not tasks:
        raise ValueError("no shared tasks")

    per_task = np.array(
        [
            np.mean([bool(x) for x in method_by_task[t]])
            - np.mean([bool(x) for x in base_by_task[t]])
            for t in tasks
        ],
        dtype=float,
    )

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_task), size=(n_boot, len(per_task)))
    boot = per_task[idx].mean(axis=1)

    return PairedEstimate(
        mean_diff=float(per_task.mean()),
        ci_low=float(np.quantile(boot, alpha / 2)),
        ci_high=float(np.quantile(boot, 1 - alpha / 2)),
        n_tasks=len(tasks),
    )


@dataclass(frozen=True)
class InteractionEstimate:
    """H4′: (метод − ориентир) в срезе A минус то же в срезе B, с интервалом уровня level.
    delta_a, delta_b — парные разности в каждом срезе с интервалами того же уровня."""

    delta_a: PairedEstimate
    delta_b: PairedEstimate
    mean: float
    ci_low: float
    ci_high: float
    level: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def interaction_bootstrap(
    ref_a: dict[str, Sequence[bool]],
    method_a: dict[str, Sequence[bool]],
    ref_b: dict[str, Sequence[bool]],
    method_b: dict[str, Sequence[bool]],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> InteractionEstimate:
    """Разность парных эффектов двух срезов (docs/METRICS.md «Глобальный срез»).

    В каждом срезе — средняя по задачам разность (метод − ориентир), как в paired_bootstrap.
    Срезы — разные наборы задач, поэтому ресемплинг независимый: интервал разности строится
    из разностей независимых bootstrap-средних.
    """
    def per_task(ref: dict, m: dict) -> np.ndarray:
        tasks = sorted(set(ref) & set(m))
        if not tasks:
            raise ValueError("no shared tasks")
        return np.array([np.mean([bool(x) for x in m[t]]) - np.mean([bool(x) for x in ref[t]])
                         for t in tasks], dtype=float)

    da, db = per_task(ref_a, method_a), per_task(ref_b, method_b)
    rng = np.random.default_rng(seed)
    ba = da[rng.integers(0, len(da), size=(n_boot, len(da)))].mean(axis=1)
    bb = db[rng.integers(0, len(db), size=(n_boot, len(db)))].mean(axis=1)
    lo, hi = alpha / 2, 1 - alpha / 2

    def est(d: np.ndarray, b: np.ndarray) -> PairedEstimate:
        return PairedEstimate(mean_diff=float(d.mean()), ci_low=float(np.quantile(b, lo)),
                              ci_high=float(np.quantile(b, hi)), n_tasks=len(d))

    boot = ba - bb
    return InteractionEstimate(est(da, ba), est(db, bb), float(da.mean() - db.mean()),
                               float(np.quantile(boot, lo)), float(np.quantile(boot, hi)), 1 - alpha)


# --------------------------------------------------------------------------
# Слой 4. Цена
# --------------------------------------------------------------------------


@dataclass
class CostReport:
    """Без этого слоя нельзя ответить на главный вопрос проекта —
    выгоднее ли steering, а не просто работает ли."""

    tokens_total: int
    tool_calls_total: int
    extra_tool_calls_vs_base: int
    latency_p50_ms: float
    latency_p95_ms: float
    peak_vram_mb: float | None = None
    train_cost_note: str = ""

    def __str__(self) -> str:
        vram = f"{self.peak_vram_mb:.0f}MB" if self.peak_vram_mb else "n/a"
        return (
            f"tokens={self.tokens_total} calls={self.tool_calls_total} "
            f"(+{self.extra_tool_calls_vs_base}) "
            f"p50={self.latency_p50_ms:.0f}ms p95={self.latency_p95_ms:.0f}ms "
            f"vram={vram}"
        )


def cost_report(
    tokens: Sequence[int],
    tool_calls: Sequence[int],
    latencies_ms: Sequence[float],
    base_tool_calls: Sequence[int] | None = None,
    peak_vram_mb: float | None = None,
    train_cost_note: str = "",
) -> CostReport:
    if not latencies_ms:
        raise ValueError("empty latencies")
    lat = np.asarray(latencies_ms, dtype=float)
    total_calls = int(sum(tool_calls))
    extra = total_calls - int(sum(base_tool_calls)) if base_tool_calls else 0
    return CostReport(
        tokens_total=int(sum(tokens)),
        tool_calls_total=total_calls,
        extra_tool_calls_vs_base=extra,
        latency_p50_ms=float(np.percentile(lat, 50)),
        latency_p95_ms=float(np.percentile(lat, 95)),
        peak_vram_mb=peak_vram_mb,
        train_cost_note=train_cost_note,
    )


# --------------------------------------------------------------------------
# Сводка по методу
# --------------------------------------------------------------------------


@dataclass
class MethodReport:
    """Полная карточка метода. Печатается целиком, чтобы нельзя было
    отчитаться одним выгодным числом."""

    name: str
    success: float
    pass_k: dict[int, float]
    effect: TargetEffect
    switch_rate: float
    delta: EpisodeDelta
    paired: PairedEstimate | None
    cost: CostReport | None
    entanglement: float | None = None

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  success        {self.success:.3f}",
            "  pass^k         "
            + ", ".join(f"k={k}:{v:.3f}" for k, v in sorted(self.pass_k.items())),
            f"  target         {self.effect}",
            f"  switch_rate    {self.switch_rate:.3f}  (diagnostic only)",
            f"  episodes       fixed={self.delta.fixed} broken={self.delta.broken} "
            f"net={self.delta.net:+d}",
        ]
        if self.paired is not None:
            lines.append(f"  paired_diff    {self.paired}")
        if self.entanglement is not None:
            lines.append(f"  entanglement   {self.entanglement:.4f}")
        if self.cost is not None:
            lines.append(f"  cost           {self.cost}")
        return "\n".join(lines)
