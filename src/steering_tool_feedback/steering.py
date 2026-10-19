"""Steering: съём активаций, построение вектора, вмешательство, sham-контроли.

Границы метода (AGENTS.md, нарушать нельзя):
  * веса базовой модели НЕ меняются;
  * вмешательство только ПОСЛЕ ответа инструмента, в заданном окне
    (окна строит windows.prepare_prompt — см. там, почему не арифметикой);
  * слой, alpha, окно и правило включения выбираются ТОЛЬКО на validation;
  * нельзя подставлять patching-ом сам ответ инструмента или готовое действие.

Позиции. Хук не считает позиции сам, а берёт их из того, что модель передаёт
слою: cache_position, иначе position_ids. Это корректно и на prefill, и на
декодировании с KV-кэшем, и при нескольких прогонах в одном контексте.
Счётчик — только запасной путь; какой источник использован, видно в
SteeringHook.positions_source.

Проверено (tests/test_hooks.py) на случайно инициализированных Llama и Qwen3.5
(гибрид линейного и полного внимания, как у Qwen3.x) и на реальном токенизаторе
Qwen3.8. На обученных весах ещё не запускалось.

ОГРАНИЧЕНИЕ: batch = 1.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from .windows import PositionWindow, Window

__all__ = [
    "ActivationRecorder", "DriftAxis", "HookGroup", "PositionWindow", "ProjectionHook",
    "SteerConfig", "SteeringHook", "SteeringVector", "drift_axis", "drift_clamp",
    "contrastive_vector", "dose_response", "get_layers", "lowrank_subspace", "n_layers",
    "permuted_direction", "random_direction_like", "shifted_layer", "shifted_window", "steering",
]


# --------------------------------------------------------------------------
# Доступ к слоям
# --------------------------------------------------------------------------

_LAYER_PATHS = (
    "model.layers",                    # Llama, Qwen2/3, Qwen3.5 text, Mistral, Gemma
    "model.language_model.layers",     # мультимодальные обёртки (Qwen3.5 и др.)
    "language_model.model.layers",
    "transformer.h",                   # GPT-2
    "gpt_neox.layers",
    "model.decoder.layers",            # OPT
)


def get_layers(model: Any) -> Sequence[Any]:
    """Список декодерных блоков. Поддерживает распространённые раскладки HF."""
    for path in _LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            return obj  # type: ignore[return-value]
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__}; add its layout to _LAYER_PATHS"
    )


def n_layers(model: Any) -> int:
    return len(get_layers(model))


def _split_output(out: Any):
    """Блок может вернуть тензор (transformers 5) или кортеж (hidden, ...)."""
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def _rejoin_output(hidden: Any, rest: Any):
    return hidden if rest is None else (hidden, *rest)


def _positions_from_call(args: tuple, kwargs: dict, seq: int):
    """Абсолютные позиции токенов текущего вызова слоя или None.

    position_ids бывает [bs, seq], а у M-RoPE моделей [3|4, bs, seq] —
    для чисто текстового входа первая плоскость равна текстовым позициям.
    """
    import torch

    cp = kwargs.get("cache_position")
    if isinstance(cp, torch.Tensor) and cp.dim() == 1 and cp.shape[0] == seq:
        return cp.long(), "cache_position"

    pid = kwargs.get("position_ids")
    if isinstance(pid, torch.Tensor):
        while pid.dim() > 1:
            pid = pid[0]
        if pid.shape[0] == seq:
            return pid.long(), "position_ids"
    return None, "counter"


# --------------------------------------------------------------------------
# Съём активаций
# --------------------------------------------------------------------------


class ActivationRecorder:
    """Пишет residual stream (выход блока) на выбранных слоях.

        with ActivationRecorder(model, layers=[10, 14]) as rec:
            model(**prepared.model_inputs())
        acts = rec.pooled(prepared.windows["post_message"])   # {layer: [hidden]}
    """

    def __init__(self, model: Any, layers: Sequence[int]):
        self.model = model
        self.layers = list(layers)
        self._store: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _hook(self, layer_idx: int):
        def fn(_module, _inp, out):
            hidden, _ = _split_output(out)
            if hidden.shape[0] != 1:
                raise RuntimeError("ActivationRecorder supports batch=1 only")
            # Копия обязательна: буфер переиспользуется.
            self._store[layer_idx] = hidden.detach().to("cpu", copy=True)
            return out
        return fn

    def __enter__(self) -> "ActivationRecorder":
        blocks = get_layers(self.model)
        for i in self.layers:
            self._handles.append(blocks[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def raw(self, layer: int) -> np.ndarray:
        """[seq, hidden] для слоя."""
        return self._store[layer][0].float().numpy()

    def pooled(self, window: PositionWindow) -> dict[int, np.ndarray]:
        """Среднее по позициям окна. Окно за пределами последовательности — ошибка."""
        out: dict[int, np.ndarray] = {}
        for layer, h in self._store.items():
            seq = h.shape[1]
            if window.end > seq:
                raise ValueError(
                    f"window [{window.start},{window.end}) exceeds recorded length {seq}"
                )
            out[layer] = h[0, window.start:window.end, :].float().mean(dim=0).numpy()
        return out


# --------------------------------------------------------------------------
# Вектор
# --------------------------------------------------------------------------


@dataclass
class SteeringVector:
    """Направление вмешательства на одном слое.

    provenance — на каких данных и в каком окне получен вектор. Без этого
    прогон невоспроизводим (AGENTS.md: манифест обязателен).
    """

    layer: int
    vector: np.ndarray
    window_kind: Window
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError("vector must be 1-D [hidden]")

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    def unit(self) -> np.ndarray:
        n = self.norm
        if n < 1e-12:
            raise ValueError("degenerate steering vector (zero norm)")
        return self.vector / n

    def save(self, path: str) -> None:
        import json

        np.savez(
            path,
            vector=self.vector,
            layer=self.layer,
            window_kind=self.window_kind,
            provenance=json.dumps(self.provenance, default=str, ensure_ascii=False),
        )

    @staticmethod
    def load(path: str) -> "SteeringVector":
        import json

        z = np.load(path, allow_pickle=False)
        return SteeringVector(
            layer=int(z["layer"]),
            vector=z["vector"],
            window_kind=str(z["window_kind"]),  # type: ignore[arg-type]
            provenance=json.loads(str(z["provenance"])),
        )


def contrastive_vector(
    acts_correct: Sequence[np.ndarray],
    acts_incorrect: Sequence[np.ndarray],
    layer: int,
    window_kind: Window,
    provenance: dict[str, Any] | None = None,
) -> SteeringVector:
    """Contrastive activation addition: mean(correct) - mean(incorrect). Только на train."""
    if not acts_correct or not acts_incorrect:
        raise ValueError("need both correct and incorrect activations")
    a = np.stack(acts_correct).mean(axis=0)
    b = np.stack(acts_incorrect).mean(axis=0)
    prov = dict(provenance or {})
    prov.update({"n_correct": len(acts_correct), "n_incorrect": len(acts_incorrect),
                 "method": "contrastive_activation_addition"})
    return SteeringVector(layer=layer, vector=a - b, window_kind=window_kind, provenance=prov)


def lowrank_subspace(
    acts_correct: Sequence[np.ndarray],
    acts_incorrect: Sequence[np.ndarray],
    rank: int = 4,
) -> np.ndarray:
    """Низкоранговое подпространство различия, [rank, hidden].
    Только если простой вектор не сработал (EXPERIMENT_PLAN, шаг 2)."""
    a, b = np.stack(acts_correct), np.stack(acts_incorrect)
    diffs = np.concatenate([a.mean(axis=0) - b, a - b.mean(axis=0)], axis=0)
    diffs = diffs - diffs.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(diffs, full_matrices=False)
    return vt[:rank]


# --------------------------------------------------------------------------
# Sham-контроли
# --------------------------------------------------------------------------


def random_direction_like(v: SteeringVector, seed: int = 0) -> SteeringVector:
    """Случайное направление ТОЙ ЖЕ нормы. Обязательный контроль для H3."""
    rng = np.random.default_rng(seed)
    r = rng.normal(size=v.vector.shape)
    r = r / np.linalg.norm(r) * v.norm
    return SteeringVector(v.layer, r, v.window_kind,
                          {**v.provenance, "sham": "random_direction", "seed": seed})


def permuted_direction(v: SteeringVector, seed: int = 0) -> SteeringVector:
    """Перестановка координат: норма и спектр сохранены, смысл разрушен."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(v.vector.shape[0])
    return SteeringVector(v.layer, v.vector[p], v.window_kind,
                          {**v.provenance, "sham": "permuted_coords", "seed": seed})


def shifted_layer(v: SteeringVector, delta: int, max_layer: int) -> SteeringVector:
    """Тот же вектор на соседнем слое — проверка специфичности слоя."""
    new = max(0, min(max_layer - 1, v.layer + delta))
    if new == v.layer:
        raise ValueError("shifted layer equals original; widen delta")
    return SteeringVector(new, v.vector, v.window_kind,
                          {**v.provenance, "sham": f"layer_shift{delta:+d}"})


def shifted_window(w: PositionWindow, shift: int) -> PositionWindow:
    """Сдвинутое окно той же длины — проверка, что важен именно момент после ответа."""
    start = max(0, w.start + shift)
    return PositionWindow(start, start + len(w), w.kind, w.apply_to_generated)


# --------------------------------------------------------------------------
# Вмешательство
# --------------------------------------------------------------------------


@dataclass
class SteerConfig:
    """alpha, слой, окно и правило включения — выбираются только на validation."""

    alpha: float
    window: PositionWindow
    normalize: bool = True   # масштаб по норме hidden в позиции: переносимо между слоями
    enabled: bool = True


class _WindowHook:
    """Общая часть хуков: позиции из модели, маска окна, учёт срабатываний.

    Учёт:
      n_applied        — сколько позиций окна прошло через хук (аудит: 0 = окно мимо);
      positions_source — откуда взяты позиции (cache_position / position_ids / counter);
      n_sequences      — сколько последовательностей прошло через хук.
    """

    layer: int
    window: PositionWindow

    def __init__(self) -> None:
        self._handle: Any = None
        self._counter = 0
        self._prompt_len: int | None = None
        self.n_applied = 0
        self.n_sequences = 0
        self.positions_source: str | None = None

    def reset(self) -> None:
        self._counter = 0
        self._prompt_len = None
        self.n_applied = 0
        self.n_sequences = 0
        self.positions_source = None

    def _active(self) -> bool:
        return True

    def _apply(self, hidden: Any, mask: Any) -> Any:
        raise NotImplementedError

    def _fn(self, _module, args, kwargs, out):
        import torch

        hidden, rest = _split_output(out)
        bs, seq = hidden.shape[0], hidden.shape[1]
        if bs != 1:
            raise RuntimeError(f"{type(self).__name__} supports batch=1 only (windows are per-sequence)")

        pos, src = _positions_from_call(args, kwargs, seq)
        if pos is None:
            if seq > 1:            # prefill = новая последовательность
                self._counter = 0
            pos = torch.arange(self._counter, self._counter + seq, device=hidden.device)
            self._counter += seq
        self.positions_source = src

        first = int(pos[0])
        if first == 0 or self._prompt_len is None:
            self._prompt_len = int(pos[-1]) + 1
            self.n_sequences += 1

        if not self._active():
            return out

        w = self.window
        pos = pos.to(hidden.device)
        mask = (pos >= w.start) & (pos < w.end)
        if w.apply_to_generated:
            mask = mask | (pos >= self._prompt_len)
        n = int(mask.sum())
        if n == 0:
            return out
        hidden = self._apply(hidden.clone(), mask)
        self.n_applied += n
        return _rejoin_output(hidden, rest)

    def attach(self, model: Any) -> None:
        if self._handle is not None:
            raise RuntimeError("hook already attached")
        self.reset()
        self._handle = get_layers(model)[self.layer].register_forward_hook(self._fn, with_kwargs=True)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class SteeringHook(_WindowHook):
    """Добавляет alpha * v к residual stream в позициях окна."""

    def __init__(self, vector: SteeringVector, cfg: SteerConfig):
        super().__init__()
        self.vector = vector
        self.cfg = cfg
        self.layer = vector.layer
        self.window = cfg.window

    def _active(self) -> bool:
        return self.cfg.enabled and self.cfg.alpha != 0.0

    def _apply(self, hidden: Any, mask: Any) -> Any:
        import torch

        v = torch.as_tensor(
            self.vector.unit() if self.cfg.normalize else self.vector.vector,
            dtype=hidden.dtype, device=hidden.device,
        )
        sel = hidden[:, mask, :]
        if self.cfg.normalize:
            delta = self.cfg.alpha * v * sel.norm(dim=-1, keepdim=True)
        else:
            delta = self.cfg.alpha * v
        hidden[:, mask, :] = sel + delta
        return hidden


@dataclass
class DriftAxis:
    """Ось дрейфа на одном слое (адаптация TACT, 2605.05980).

    unit  — направление «откалиброванное -> сбой» (норма 1);
    mu    — центроид откалиброванных состояний;
    sigma — std проекций откалиброванных состояний на ось.
    """

    layer: int
    unit: np.ndarray
    mu: np.ndarray
    sigma: float


class ProjectionHook(_WindowHook):
    """TACT-Gate: p = (h - mu)·v; вне полосы |p| <= k·sigma сдвиг h <- h - lam·excess·v.

    excess = p - tau при p > tau, p + tau при p < -tau, иначе 0 (lam = 1 — TACT-Cap).
    Внутри полосы состояние не меняется. n_changed — сколько позиций реально сдвинуто.
    """

    def __init__(self, axis: DriftAxis, window: PositionWindow, k: float, lam: float):
        super().__init__()
        if axis.sigma <= 0:
            raise ValueError("sigma must be positive")
        self.axis, self.window, self.k, self.lam = axis, window, float(k), float(lam)
        self.layer = axis.layer
        self.n_changed = 0
        self.projections: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.n_changed = 0
        self.projections = []

    def _active(self) -> bool:
        return self.lam != 0.0

    def _apply(self, hidden: Any, mask: Any) -> Any:
        import torch

        v = torch.as_tensor(self.axis.unit, dtype=torch.float32, device=hidden.device)
        mu = torch.as_tensor(self.axis.mu, dtype=torch.float32, device=hidden.device)
        sel = hidden[:, mask, :].float()
        p = (sel - mu) @ v                                   # [1, n]
        tau = self.k * self.axis.sigma
        excess = torch.where(p > tau, p - tau, torch.where(p < -tau, p + tau, torch.zeros_like(p)))
        self.projections += [float(x) for x in p.detach().flatten()]
        self.n_changed += int((excess != 0).sum())
        hidden[:, mask, :] = (sel - self.lam * excess.unsqueeze(-1) * v).to(hidden.dtype)
        return hidden


class HookGroup:
    """Несколько хуков одного вмешательства (TACT — по хуку на слой)."""

    def __init__(self, hooks: list[_WindowHook]):
        self.hooks = hooks

    @property
    def n_applied(self) -> int:
        """Позиций окна на слой; -1, если слои отработали по-разному (рассинхрон)."""
        vals = {h.n_applied for h in self.hooks}
        return vals.pop() if len(vals) == 1 else -1

    @property
    def n_changed(self) -> int:
        return sum(getattr(h, "n_changed", h.n_applied) for h in self.hooks)

    @property
    def positions_source(self) -> str | None:
        srcs = {h.positions_source for h in self.hooks}
        return "counter" if "counter" in srcs else next(iter(srcs), None)


@contextmanager
def drift_clamp(model: Any, axes: Sequence[DriftAxis], window: PositionWindow, k: float,
                lam: float) -> Iterator[HookGroup]:
    """TACT-Gate на нескольких слоях сразу. Проверять group.n_applied == len(window)."""
    hooks = [ProjectionHook(a, window, k, lam) for a in axes]
    for h in hooks:
        h.attach(model)
    try:
        yield HookGroup(hooks)
    finally:
        for h in hooks:
            h.detach()


def drift_axis(acts_failure: Sequence[np.ndarray], acts_ok: Sequence[np.ndarray],
               groups_failure: Sequence[str], groups_ok: Sequence[str], layer: int,
               acts_calibrated: Sequence[np.ndarray] | None = None) -> tuple[DriftAxis, dict]:
    """Ось «верно -> сбой» по TACT: средняя по задачам разность средних (per-problem balanced).

    Берутся задачи, где есть оба класса; если таких нет — разность глобальных средних
    (в info["mode"] это видно). mu и sigma — по откалиброванным состояниям
    (acts_calibrated, по умолчанию acts_ok).
    """
    if not acts_failure or not acts_ok:
        raise ValueError("need both failure and ok activations")
    by_f: dict[str, list[np.ndarray]] = {}
    by_o: dict[str, list[np.ndarray]] = {}
    for a, g in zip(acts_failure, groups_failure):
        by_f.setdefault(g, []).append(np.asarray(a, dtype=np.float64))
    for a, g in zip(acts_ok, groups_ok):
        by_o.setdefault(g, []).append(np.asarray(a, dtype=np.float64))
    both = sorted(set(by_f) & set(by_o))
    if both:
        d = np.mean([np.mean(by_f[g], axis=0) - np.mean(by_o[g], axis=0) for g in both], axis=0)
        mode = "per_problem"
    else:
        d = np.mean(np.stack(acts_failure), axis=0) - np.mean(np.stack(acts_ok), axis=0)
        mode = "global_means"
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("degenerate drift axis (zero norm)")
    unit = d / n
    cal = np.stack([np.asarray(a, dtype=np.float64) for a in (acts_calibrated or acts_ok)])
    mu = cal.mean(axis=0)
    sigma = float(np.std((cal - mu) @ unit))
    if sigma < 1e-8:
        sigma = 1e-8
    info = {"mode": mode, "n_problems": len(both), "n_failure": len(acts_failure),
            "n_ok": len(acts_ok), "n_calibrated": len(cal)}
    return DriftAxis(layer, unit.astype(np.float32), mu.astype(np.float32), sigma), info


@contextmanager
def steering(model: Any, vector: SteeringVector, cfg: SteerConfig) -> Iterator[SteeringHook]:
    """Контекст с включённым вмешательством.

        with steering(model, vec, cfg) as hook:
            out = model.generate(**prepared.model_inputs())
        assert hook.n_applied > 0, "hook never fired - window is wrong"

    Проверка n_applied > 0 обязательна: молча не сработавший хук даёт
    «нулевой эффект», который легко принять за опровержение H3.
    """
    hook = SteeringHook(vector, cfg)
    hook.attach(model)
    try:
        yield hook
    finally:
        hook.detach()


def dose_response(alphas: Sequence[float], run_fn: Any) -> list[tuple[float, float]]:
    """Прогон по сетке alpha. run_fn(alpha) -> метрика.

    Монотонность — слабое свидетельство в пользу механизма; перелом или
    немонотонность требуют объяснения до выводов (EXPERIMENT_PLAN, шаг 5).
    """
    return [(a, float(run_fn(a))) for a in alphas]
