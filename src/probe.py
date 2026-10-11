"""Проверка H1: различает ли модель ответы инструмента.

Probe — ИНСТРУМЕНТ ОТБОРА слоя и окна, а не результат исследования
(docs/HYPOTHESES.md). Положительная accuracy не доказывает H3: классификатор
может читать формулировку ответа или позицию токена, а не то, чем агент
реально пользуется при выборе действия.

Поэтому здесь считаются три вещи сразу:
  * accuracy/AUROC против честной базовой частоты;
  * контроль на перестановке метк (должен упасть до случайного);
  * проверка, не читает ли probe поверхностную форму (shortcut-тест
    на инвариантных парах, где форма меняется, а смысл нет).

Только sklearn/numpy — без torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    layer: int
    window_kind: str
    accuracy: float
    auroc: float
    majority_baseline: float
    permuted_accuracy: float
    n: int
    shortcut_auroc: float | None = None

    # Запас, который должен быть у сигнала над обеими нулевыми моделями.
    MARGIN = 0.05

    @property
    def beats_baseline(self) -> bool:
        """Сигнал должен обойти ДВЕ нулевые модели:

          * базовую частоту (самый частый класс);
          * эмпирический null — probe на перестановленных метках.

        Сравнение с перестановкой, а не с фиксированным порогом, потому что
        при CV на небольшом n случайная точность сама гуляет около 0.55-0.60.
        Фиксированный порог здесь даёт ложные отрицания.
        """
        return (
            self.accuracy > self.majority_baseline + self.MARGIN
            and self.accuracy > self.permuted_accuracy + self.MARGIN
        )

    @property
    def suspected_shortcut(self) -> bool:
        """Probe уверенно отличает ИНВАРИАНТНЫЕ ветки -> читает форму, не смысл."""
        return self.shortcut_auroc is not None and self.shortcut_auroc > 0.65

    def __str__(self) -> str:
        s = (
            f"L{self.layer:<3} {self.window_kind:<13} "
            f"acc={self.accuracy:.3f} auroc={self.auroc:.3f} "
            f"base={self.majority_baseline:.3f} perm={self.permuted_accuracy:.3f} "
            f"n={self.n}"
        )
        if self.shortcut_auroc is not None:
            s += f" shortcut={self.shortcut_auroc:.3f}"
        flags = []
        if not self.beats_baseline:
            flags.append("NO SIGNAL")
        if self.suspected_shortcut:
            flags.append("SHORTCUT?")
        return s + ("  [" + ", ".join(flags) + "]" if flags else "")


def _fit_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int
) -> tuple[float, float]:
    """Кросс-валидация с группировкой по задаче (иначе утечка между ветками)."""
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    accs, aurocs = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        accs.append(float(np.mean((p > 0.5).astype(int) == y[te])))
        aurocs.append(float(roc_auc_score(y[te], p)))

    if not accs:
        raise ValueError("no usable CV fold (labels not separable across groups)")
    return float(np.mean(accs)), float(np.mean(aurocs))


def run_probe(
    activations: Sequence[np.ndarray],
    labels: Sequence[int],
    task_ids: Sequence[str],
    layer: int,
    window_kind: str,
    seed: int = 0,
    n_splits: int = 5,
    invariant_activations: Sequence[np.ndarray] | None = None,
    invariant_labels: Sequence[int] | None = None,
    invariant_task_ids: Sequence[str] | None = None,
) -> ProbeResult:
    """Обучить probe и посчитать все контроли.

    activations — пулинг по окну, одна строка на ветку.
    labels      — существенное состояние среды (0/1), НЕ действие агента.
    task_ids    — для группировки: обе ветки одной задачи в один фолд.

    invariant_* — опционально: ветки инвариантных пар с метками «какая форма».
                  Если probe их хорошо различает, он читает форму, а не смысл.
    """
    X = np.stack([np.asarray(a, dtype=float) for a in activations])
    y = np.asarray(labels, dtype=int)
    g = np.asarray(task_ids)

    if X.shape[0] != y.shape[0] or X.shape[0] != g.shape[0]:
        raise ValueError("activations, labels and task_ids must align")
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes present")

    acc, auroc = _fit_cv(X, y, g, seed, n_splits)

    # Базовая частота: доля самого частого класса.
    majority = float(max(np.mean(y == c) for c in np.unique(y)))

    # Контроль: перестановка метк ВНУТРИ групп сохраняет структуру групп,
    # но разрушает связь с таргетом.
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    rng.shuffle(y_perm)
    try:
        perm_acc, _ = _fit_cv(X, y_perm, g, seed, n_splits)
    except ValueError:
        perm_acc = majority

    shortcut = None
    if invariant_activations is not None and invariant_labels is not None:
        Xi = np.stack([np.asarray(a, dtype=float) for a in invariant_activations])
        yi = np.asarray(invariant_labels, dtype=int)
        gi = np.asarray(
            invariant_task_ids if invariant_task_ids is not None
            else np.arange(len(yi)).astype(str)
        )
        if len(np.unique(yi)) >= 2:
            try:
                _, shortcut = _fit_cv(Xi, yi, gi, seed, n_splits)
            except ValueError:
                shortcut = None

    return ProbeResult(
        layer=layer,
        window_kind=window_kind,
        accuracy=acc,
        auroc=auroc,
        majority_baseline=majority,
        permuted_accuracy=perm_acc,
        n=int(X.shape[0]),
        shortcut_auroc=shortcut,
    )


def sweep_layers(
    activations_by_layer: dict[int, Sequence[np.ndarray]],
    labels: Sequence[int],
    task_ids: Sequence[str],
    window_kind: str,
    seed: int = 0,
) -> list[ProbeResult]:
    """Прогон probe по слоям. Возвращает отсортированный по AUROC список.

    Выбор слоя по этому результату делается на TRAIN/VALIDATION и затем
    замораживается (AGENTS.md).
    """
    out = []
    for layer in sorted(activations_by_layer):
        try:
            out.append(
                run_probe(activations_by_layer[layer], labels, task_ids,
                          layer, window_kind, seed)
            )
        except ValueError as e:
            print(f"  L{layer}: skipped ({e})")
    return sorted(out, key=lambda r: r.auroc, reverse=True)


def gap_report(
    probe_ok: bool, acts_differ: bool, action_differs: bool
) -> str:
    """Интерпретация связки H1 + H2 — целевой разрыв проекта.

    Это та развилка, из-за которой probe нельзя считать результатом.
    """
    if probe_ok and not action_differs:
        return (
            "TARGET GAP: модель различает ответы, но не меняет действие. "
            "Это случай, на который нацелен steering (H3)."
        )
    if probe_ok and action_differs:
        return (
            "NO GAP: модель и различает, и действует правильно. "
            "Ошибки редки -> пересмотреть постановку задач (HYPOTHESES: stop H2)."
        )
    if not probe_ok:
        return (
            "NO SIGNAL: ответ не декодируется из активаций. Steering скорее не "
            "поможет. НО это не значит 'только дообучение' - проверить явное "
            "состояние, разбор ответа, сильный промпт (HYPOTHESES: stop H1)."
        )
    return "inconclusive"
