"""Вмешательства в активации в точке решения (JOURNAL, 2026-10-30): приблизиться к LoRA без
градиентного обучения весов модели.

Диагностика 07: эффект LoRA передаётся через одну позицию — последний токен перед именем действия
(«точка вызова», call_point) — на слоях 22+; изменение, которое делает LoRA, в основном постоянный
сдвиг плюс линейная функция состояния базы. Поэтому все методы здесь — аффинная поправка в одной
позиции одного слоя:  h ← h + α·(h·U·P + a0),  U ∈ R^{d×r}, P ∈ R^{r×d} (ранг r).

linmap_lora — Δh = h_LoRA − h_база, подогнанный в закрытой форме (PCA → ridge → ранг r): LoRA как учитель.
linmap      — без учителя: цель для неверного решения базы — центроид верных состояний с тем же
              правильным действием; для верного — ноль. Та же подгонка в закрытой форме.
reft        — LoReFT (Wu et al., 2404.03592): h + Rᵀ(W h + b − R h), R с ортонормированными строками,
              обучается градиентом по CE имени верного действия; при выводе — та же аффинная форма.
CAST        — Lee et al. (2409.05907): вектор условия (1-я главная компонента центрированного контраста
              «не подходит / подходит»), проверка sim(h, tanh(proj_c h)) против порога θ (слой, θ и
              направление — по F1 на train), вектор поведения добавляется, только если условие сработало.

Здесь — подгонка (numpy / torch); применение — хуки steering.AffineHook и механизм гейта пайплайна.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class AffineMap:
    """δ(h) = h·U·P + a0 (строки — позиции). Ранг ≤ r = U.shape[1]."""

    U: np.ndarray      # [d, r]
    P: np.ndarray      # [r, d]
    a0: np.ndarray     # [d]
    info: dict

    def __call__(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) @ self.U @ self.P + self.a0

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    def n_params(self) -> int:
        return int(self.U.size + self.P.size + self.a0.size)


def fit_affine(H: np.ndarray, D: np.ndarray, rank: int, n_pcs: int = 32, ridge: float = 10.0,
               seed: int = 0) -> AffineMap:
    """Закрытая форма: D ≈ δ(H). Стандартизация → PCA(n_pcs) → ridge → проекция на rank главных
    направлений предсказаний. Всё аффинно, поэтому сворачивается в δ(h) = h·U·P + a0."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    H = np.asarray(H, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    m, s = H.mean(0), H.std(0) + 1e-6
    Z = (H - m) / s
    k = max(1, min(n_pcs, len(H) - 1, H.shape[1]))
    pca = PCA(n_components=k, random_state=seed).fit(Z)
    reg = Ridge(alpha=ridge).fit(pca.transform(Z), D)
    Y = reg.predict(pca.transform(Z))
    mu = Y.mean(0)
    _u, _sv, vt = np.linalg.svd(Y - mu, full_matrices=False)
    P = vt[: max(1, min(rank, vt.shape[0]))]                     # [r, d]
    # ŷ(h) = ((h − m)/s − pca.mean_)·Cᵀ·coefᵀ + intercept;  проекция: mu + (ŷ − mu)·Pᵀ·P
    L = (np.diag(1.0 / s) @ pca.components_.T @ reg.coef_.T)    # [d, d_out]
    U = L @ P.T                                                  # [d, r]
    y0 = ((-m / s) - pca.mean_) @ pca.components_.T @ reg.coef_.T + reg.intercept_
    a0 = mu + (y0 - mu) @ P.T @ P
    r2 = float(1 - np.sum((D - (H @ U @ P + a0)) ** 2) / max(np.sum(D ** 2), 1e-12))
    return AffineMap(U.astype(np.float32), P.astype(np.float32), a0.astype(np.float32),
                     {"rank": int(P.shape[0]), "n_pcs": k, "ridge": ridge, "train_r2": r2, "n": len(H)})


def contrast_targets(H: np.ndarray, correct: Sequence[bool], gold: Sequence[str]) -> np.ndarray:
    """Цель без учителя: для неверного решения — центроид верных состояний с тем же правильным
    действием минус текущее состояние; для верного — ноль (не трогать то, что уже работает).
    Действие без единого верного примера — центроид всех верных."""
    H = np.asarray(H, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    g = np.asarray(gold)
    if not ok.any():
        raise ValueError("no correct base decisions to aim at")
    allc = H[ok].mean(0)
    cent = {a: (H[ok & (g == a)].mean(0) if np.any(ok & (g == a)) else allc) for a in set(g)}
    D = np.zeros_like(H)
    for i in np.where(~ok)[0]:
        D[i] = cent[g[i]] - H[i]
    return D


# --------------------------------------------------------------------------
# CAST
# --------------------------------------------------------------------------


def pc1_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """1-я главная компонента центрированного контраста (CAST): центр μ = (среднее pos + среднее neg)/2;
    знак — от neg к pos. Единичная норма."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    mu = (pos.mean(0) + neg.mean(0)) / 2
    X = np.concatenate([pos - mu, neg - mu])
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    if (pos.mean(0) - neg.mean(0)) @ v < 0:
        v = -v
    return v / np.linalg.norm(v)


def cast_similarity(h: np.ndarray, c: np.ndarray) -> np.ndarray:
    """sim(h, tanh(proj_c h)) — как в CAST (2409.05907): косинус состояния и сжатой проекции.

    Знаковонезависима: tanh нечётна, поэтому h·tanh(proj_c h) ≥ 0 — мера выравнивания h с c
    (величины проекции), а не стороны. Классы разделяются, если их проекции различаются по модулю —
    у настоящих скрытых состояний так и есть (большая средняя компонента).
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    c = np.asarray(c, dtype=np.float64)
    proj = (h @ c)[:, None] * c[None, :] / (c @ c)
    t = np.tanh(proj)
    num = np.sum(h * t, axis=1)
    den = np.linalg.norm(h, axis=1) * np.linalg.norm(t, axis=1) + 1e-12
    return num / den


def cast_condition(sim: np.ndarray, labels: Sequence[int], n_grid: int = 50) -> dict:
    """Порог θ и направление (> или <) по максимуму F1 на train — как в CAST."""
    y = np.asarray(labels, dtype=int)
    best = {"f1": -1.0}
    for theta in np.quantile(sim, np.linspace(0.02, 0.98, n_grid)):
        for direction in (">", "<"):
            pred = sim > theta if direction == ">" else sim < theta
            tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best["f1"]:
                best = {"theta": float(theta), "direction": direction, "f1": float(f1)}
    return best


def cast_fires(sim: float, cond: dict) -> bool:
    return bool(sim > cond["theta"]) if cond["direction"] == ">" else bool(sim < cond["theta"])


# --------------------------------------------------------------------------
# LoReFT
# --------------------------------------------------------------------------


def train_loreft(be: Any, examples: Sequence[tuple[list[int], list[int]]], layer: int, rank: int,
                 lr: float, epochs: int, seed: int = 0) -> tuple[AffineMap, dict]:
    """LoReFT в точке вызова одного слоя: h + Rᵀ(W h + b − R h), R — ортонормированные строки
    (параметризация orthogonal). Обучение — CE имени верного действия; веса модели заморожены.

    examples — (ids промпта + префикса вызова, токены имени действия). Возвращает аффинную форму
    для вывода: δ(h) = h·(Wᵀ − Rᵀ)·R + b·R, и сведения о цене обучения.
    """
    import time

    import torch
    from torch.nn.utils import parametrizations

    from . import steering as ST

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    d = be.hidden_size

    class LoReFT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.R = torch.nn.Linear(d, rank, bias=False)
            parametrizations.orthogonal(self.R)
            self.W = torch.nn.Linear(d, rank)

        def forward(self, h):                          # h: [1, d] float32 на CPU
            Rw = self.R.weight
            return (self.W(h) - h @ Rw.T) @ Rw         # Rᵀ(W h + b − R h)

    mod = LoReFT().float()                             # маленький модуль живёт на CPU
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=0.0)
    state = {"pos": -1}

    def hook(_m, _a, out):
        h, rest = ST._split_output(out)
        pos = state["pos"]
        if pos < 0 or h.shape[1] <= pos:
            return out
        x = h[:, pos, :]
        delta = mod(x.float().cpu()).to(device=h.device, dtype=h.dtype)
        h = h.clone()
        h[:, pos, :] = x + delta
        return ST._rejoin_output(h, rest)

    frozen = [p for p in be.model.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    handle = ST.get_layers(be.model)[layer].register_forward_hook(hook)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    try:
        with torch.enable_grad():
            for _ in range(epochs):
                ep = []
                for i in rng.permutation(len(examples)):
                    ids, tgt = examples[i]
                    state["pos"] = len(ids) - 1
                    out = be.model(input_ids=be._tensor(list(ids) + list(tgt[:-1])),
                                   logits_to_keep=len(tgt))
                    logits = out.logits[0].float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, torch.tensor(tgt, dtype=torch.long, device=logits.device))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    ep.append(float(loss.detach()))
                losses.append(float(np.mean(ep)))
    finally:
        handle.remove()
        for p in frozen:
            p.requires_grad_(True)
    with torch.no_grad():
        R = mod.R.weight.detach().double().numpy()           # [r, d]
        W = mod.W.weight.detach().double().numpy()           # [r, d]
        b = mod.W.bias.detach().double().numpy()             # [r]
    am = AffineMap((W.T - R.T).astype(np.float32), R.astype(np.float32), (b @ R).astype(np.float32),
                   {"rank": rank, "layer": layer, "lr": lr, "epochs": epochs})
    info = {"train_seconds": round(time.perf_counter() - t0, 1),
            "epoch_losses": [round(x, 4) for x in losses],
            "n_trainable": int(sum(p.numel() for p in mod.parameters())), "n_examples": len(examples)}
    return am, info
