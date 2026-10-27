"""Диагностика LoRA как «маркера» (JOURNAL, 2026-10-27): где и в какой форме дообучение меняет
активации — чтобы понять, может ли вмешательство без градиентного обучения привести модель туда же.

1. Подстановка (activation patching) LoRA → база. Прогоняем модель с LoRA и запоминаем скрытые
   состояния; затем в базовой модели на слое L в позициях P заменяем состояние на LoRA-шное и
   смотрим, стало ли решение как у LoRA. Эффект нормируется: 0 — как у базы, 1 — как у LoRA:
       effect = (d_patched − d_base) / (d_lora − d_base),  d = log p(действие LoRA) − log p(действие базы).
   Позиции «all» на слое L — сколько эффекта LoRA уже записано в поток к слою L; «last» — хватит ли
   одной точки решения; «tool_content» / «field_read» — меняет ли LoRA то, как прочитан ответ.

2. Форма разницы Δh = h_LoRA − h_база (по слоям, в точке решения и в окнах). Какую долю её энергии
   на НОВЫХ задачах (кросс-валидация по задачам) воспроизводит:
       const   — один постоянный вектор (класс CAA);
       state   — вектор по истинному состоянию «подходит / не подходит» (оракул, только верхняя граница);
       probe   — вектор по состоянию, которое probe читает из активаций базы (без оракула);
       linear  — линейное отображение состояния базы (Δh ≈ W·h + b, W низкого ранга).
   R² считается относительно нуля: 1 − Σ‖Δh − предсказание‖² / Σ‖Δh‖².

Решение читается по первому токену имени действия (у всех действий он разный) — один прогон на
вариант; согласие с полным скорингом проверяется и пишется в сводку.

Числовая часть (разбор Δh, эффект) — только numpy/sklearn и тестируется без модели.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Sequence

import numpy as np

from . import steering as ST
from .windows import PositionWindow

POSITION_SETS = ("all", "tool_content", "field_read", "decision", "last")


# --------------------------------------------------------------------------
# Числа
# --------------------------------------------------------------------------


def patch_effect(base: np.ndarray, lora: np.ndarray, patched: np.ndarray, a_base: int,
                 a_lora: int) -> float:
    """Нормированный эффект подстановки по логарифмам вероятностей действий."""
    d = lambda x: float(x[a_lora] - x[a_base])  # noqa: E731
    den = d(lora) - d(base)
    if abs(den) < 1e-6:
        return float("nan")
    return (d(patched) - d(base)) / den


def _r2(delta: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((delta - pred) ** 2) / max(np.sum(delta ** 2), 1e-12))


def delta_decomposition(delta: np.ndarray, base: np.ndarray, states: Sequence[int],
                        groups: Sequence[str], rank: int = 8, n_pcs: int = 32,
                        ridge: float = 10.0, n_splits: int = 5, seed: int = 0) -> dict[str, float]:
    """Какую долю Δh (R² относительно нуля, на отложенных задачах) воспроизводит каждая форма.

    delta, base — [N, d]; states — 0/1 (подходит / не подходит); groups — задача (CV по задачам).
    linear: PCA базы (n_pcs) → ridge → проекция на rank главных направлений Δh (низкий ранг W).
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    delta = np.asarray(delta, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    s = np.asarray(states, dtype=int)
    g = np.asarray(groups)
    preds = {k: np.zeros_like(delta) for k in ("const", "state", "probe", "linear")}
    k = max(2, min(n_splits, len(np.unique(g))))
    for tr, te in GroupKFold(n_splits=k).split(delta, s, g):
        m = delta[tr].mean(0)
        preds["const"][te] = m
        by = {c: (delta[tr][s[tr] == c].mean(0) if np.any(s[tr] == c) else m) for c in (0, 1)}
        preds["state"][te] = np.stack([by[c] for c in s[te]])
        sc = StandardScaler().fit(base[tr])
        if len(np.unique(s[tr])) > 1:
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(sc.transform(base[tr]), s[tr])
            ps = clf.predict(sc.transform(base[te]))
        else:
            ps = np.full(len(te), s[tr][0])
        preds["probe"][te] = np.stack([by[c] for c in ps])
        n_pc = max(1, min(n_pcs, len(tr) - 1, base.shape[1]))
        pca = PCA(n_components=n_pc, random_state=seed).fit(sc.transform(base[tr]))
        ztr, zte = pca.transform(sc.transform(base[tr])), pca.transform(sc.transform(base[te]))
        reg = Ridge(alpha=ridge).fit(ztr, delta[tr])
        p_tr, p_te = reg.predict(ztr), reg.predict(zte)
        # низкий ранг: оставить rank главных направлений предсказанной добавки
        _u, _sv, vt = np.linalg.svd(p_tr - p_tr.mean(0), full_matrices=False)
        basis = vt[: max(1, min(rank, vt.shape[0]))]
        mu = p_tr.mean(0)
        preds["linear"][te] = mu + (p_te - mu) @ basis.T @ basis
    out = {f"r2_{k}": _r2(delta, v) for k, v in preds.items()}
    energy = np.sum(delta ** 2, axis=1)
    out["norm_mean"] = float(np.sqrt(energy).mean())
    _u, sv, _vt = np.linalg.svd(delta - delta.mean(0), full_matrices=False)
    cum = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-12)
    out["rank90_centered"] = int(np.searchsorted(cum, 0.9) + 1)
    out["n"] = int(len(delta))
    return out


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------


def first_tokens(be: Any) -> dict[str, int]:
    """Первый токен имени каждого действия; диагностика читает решение по нему."""
    ft = {a: c[0] for a, c in be._candidates().items()}
    if len(set(ft.values())) != len(ft):
        raise ValueError("actions share a first token: first-token readout is ambiguous")
    return ft


def position_windows(p: Any, n_total: int) -> dict[str, PositionWindow]:
    """Наборы позиций в последовательности промпт + префикс вызова."""
    w = p.windows
    return {
        "all": PositionWindow(0, n_total, "tool_content"),
        "tool_content": w["tool_content"],
        "field_read": w["field_read"],
        "decision": PositionWindow(w["pre_action"].start, n_total, "pre_action"),
        "last": PositionWindow(n_total - 1, n_total, "action_point"),
    }


@contextmanager
def _attached(model: Any, hook: Any) -> Iterator[Any]:
    hook.attach(model)
    try:
        yield hook
    finally:
        hook.detach()


def decision_logprobs(be: Any, ids: list[int], first: dict[str, int], hook: Any = None,
                      prefix: tuple[int, Any] = (0, None)) -> np.ndarray:
    """log p(первый токен каждого действия) в точке решения; hook — вмешательство на время прогона."""
    import torch

    with torch.no_grad(), (_attached(be.model, hook) if hook is not None else nullcontext()):
        out = be._forward_from(ids, prefix, use_cache=True, logits_to_keep=1)
    if hook is not None and hook.n_applied != len(hook.window):
        raise RuntimeError(f"hook changed {hook.n_applied} positions, expected {len(hook.window)}")
    lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    return np.array([float(lp[t]) for t in first.values()], dtype=np.float64)


def hidden_all(be: Any, ids: list[int], layers: Sequence[int], first: dict[str, int],
               adapter: Any = None) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Полный прогон (с адаптером или без): скрытые состояния всех позиций на слоях + решение."""
    import torch

    with be._adapter_ctx(adapter), torch.no_grad(), ST.ActivationRecorder(be.model, layers) as rec:
        out = be.model(input_ids=be._tensor(ids), logits_to_keep=1)
        hs = {L: rec.raw(L).astype(np.float32) for L in layers}
    lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    return hs, np.array([float(lp[t]) for t in first.values()], dtype=np.float64)


def patch_sweep(be: Any, p: Any, ids: list[int], lora_hidden: dict[int, np.ndarray],
                base_lp: np.ndarray, lora_lp: np.ndarray, first: dict[str, int],
                layers: Sequence[int], sets: Sequence[str] = POSITION_SETS) -> list[dict[str, Any]]:
    """Подстановка LoRA → база по (слой × набор позиций) для одного промпта."""
    a_b, a_l = int(np.argmax(base_lp)), int(np.argmax(lora_lp))
    wins = position_windows(p, len(ids))
    rows = []
    for L in layers:
        for s in sets:
            w = wins[s]
            hook = ST.PatchHook(L, w, lora_hidden[L][w.start:w.end])
            prefix = be._system_prefix(ids, p.messages, p.tools, hook)
            lp = decision_logprobs(be, ids, first, hook, prefix)
            rows.append({"layer": L, "set": s, "n_positions": len(w),
                         "effect": patch_effect(base_lp, lora_lp, lp, a_b, a_l),
                         "to_lora": int(np.argmax(lp)) == a_l, "stays_base": int(np.argmax(lp)) == a_b})
    return rows


def add_vector_decision(be: Any, p: Any, ids: list[int], first: dict[str, int], layer: int,
                        vec: np.ndarray) -> np.ndarray:
    """Решение при добавке вектора в точке решения (последний токен) на слое — без нормировки."""
    w = PositionWindow(len(ids) - 1, len(ids), "action_point")
    sv = ST.SteeringVector(layer, np.asarray(vec, dtype=np.float32), "action_point", {})
    hook = ST.SteeringHook(sv, ST.SteerConfig(alpha=1.0, window=w, normalize=False))
    prefix = be._system_prefix(ids, p.messages, p.tools, hook)
    return decision_logprobs(be, ids, first, hook, prefix)
