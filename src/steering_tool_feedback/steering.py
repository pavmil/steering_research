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
    "ActivationRecorder", "PositionWindow", "SteerConfig", "SteeringHook", "SteeringVector",
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


class SteeringHook:
    """Добавляет alpha * v к residual stream в позициях окна.

    Учёт:
      n_applied        — сколько позиций реально изменено (аудит: 0 = окно мимо);
      positions_source — откуда взяты позиции (cache_position / position_ids / counter);
      n_sequences      — сколько последовательностей прошло через хук.
    """

    def __init__(self, vector: SteeringVector, cfg: SteerConfig):
        self.vector = vector
        self.cfg = cfg
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

    def _fn(self, _module, args, kwargs, out):
        import torch

        hidden, rest = _split_output(out)
        bs, seq = hidden.shape[0], hidden.shape[1]
        if bs != 1:
            raise RuntimeError("SteeringHook supports batch=1 only (windows are per-sequence)")

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

        if not self.cfg.enabled or self.cfg.alpha == 0.0:
            return out

        w = self.cfg.window
        pos = pos.to(hidden.device)
        mask = (pos >= w.start) & (pos < w.end)
        if w.apply_to_generated:
            mask = mask | (pos >= self._prompt_len)
        n = int(mask.sum())
        if n == 0:
            return out

        v = torch.as_tensor(
            self.vector.unit() if self.cfg.normalize else self.vector.vector,
            dtype=hidden.dtype, device=hidden.device,
        )
        sel = hidden[:, mask, :]
        if self.cfg.normalize:
            delta = self.cfg.alpha * v * sel.norm(dim=-1, keepdim=True)
        else:
            delta = self.cfg.alpha * v
        hidden = hidden.clone()
        hidden[:, mask, :] = sel + delta
        self.n_applied += n
        return _rejoin_output(hidden, rest)

    def attach(self, model: Any) -> None:
        if self._handle is not None:
            raise RuntimeError("hook already attached")
        self.reset()
        layer = get_layers(model)[self.vector.layer]
        self._handle = layer.register_forward_hook(self._fn, with_kwargs=True)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
