"""Steering: съём активаций, построение вектора, вмешательство, sham-контроли.

Границы метода (AGENTS.md, нарушать нельзя):
  * веса базовой модели НЕ меняются;
  * вмешательство только ПОСЛЕ ответа инструмента, в заданном окне;
  * слой, alpha, окно и правило включения выбираются ТОЛЬКО на validation;
  * нельзя подставлять patching-ом сам ответ инструмента или готовое действие.

torch импортируется лениво, чтобы metrics/simulator/probe работали без него.

ВНИМАНИЕ: хуки написаны под HF-совместимые декодерные модели
(model.model.layers[i]), но на реальной модели в этом окружении не проверялись —
torch не установился. Перед первым экспериментом прогнать tests/test_hooks.py
на крошечной модели (hf-internal-testing/tiny-random-LlamaForCausalLM).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence

import numpy as np

Window = Literal["field_read", "post_message", "pre_action"]


# --------------------------------------------------------------------------
# Доступ к слоям
# --------------------------------------------------------------------------


def get_layers(model: Any) -> Sequence[Any]:
    """Список декодерных блоков. Поддерживает распространённые раскладки HF."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj  # type: ignore[return-value]
        except AttributeError:
            continue
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__}; "
        "add its layout to get_layers()"
    )


def n_layers(model: Any) -> int:
    return len(get_layers(model))


def _split_output(out: Any):
    """Блок может вернуть тензор или кортеж (hidden, ...). Возвращаем (hidden, rest)."""
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def _rejoin_output(hidden: Any, rest: Any):
    return hidden if rest is None else (hidden, *rest)


# --------------------------------------------------------------------------
# Окно позиций
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionWindow:
    """Абсолютные позиции токенов (в полной последовательности), куда вмешиваемся.

    kind — какое из трёх окон локализации (docs/EXPERIMENT_PLAN.md, шаг 2).
    apply_to_generated — продолжать ли вмешиваться на этапе декодирования.
        Начинать с False: так эффект нельзя списать на удлинение рассуждения.
    """

    start: int
    end: int  # полуинтервал [start, end)
    kind: Window = "post_message"
    apply_to_generated: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"bad window [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start


def locate_windows(
    tokenizer: Any,
    messages: list[dict[str, str]],
    essential_field: str | None = None,
) -> dict[Window, PositionWindow]:
    """Три окна локализации относительно сообщения инструмента.

    field_read   — токены существенного поля внутри ответа (если найдено);
    post_message — хвост сообщения инструмента;
    pre_action   — последние позиции перед первым токеном следующего действия.

    Реализация опирается на то, что сообщение инструмента рендерится в шаблон
    непрерывным куском. Для конкретной модели ПРОВЕРИТЬ вручную: напечатать
    токены по индексам и глазами убедиться, что окна попали куда надо.
    """
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_msgs = messages[:-1]
    prefix = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)

    n_full = len(tokenizer(full, add_special_tokens=False)["input_ids"])
    n_prefix = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

    tool_start, tool_end = n_prefix, n_full
    windows: dict[Window, PositionWindow] = {
        "post_message": PositionWindow(max(tool_start, tool_end - 8), tool_end, "post_message"),
        "pre_action": PositionWindow(max(tool_start, n_full - 4), n_full, "pre_action"),
    }

    if essential_field:
        tool_text = full[len(prefix):]
        idx = tool_text.find(essential_field)
        if idx >= 0:
            before = tool_text[:idx]
            off = len(tokenizer(before, add_special_tokens=False)["input_ids"])
            n_field = len(tokenizer(essential_field, add_special_tokens=False)["input_ids"])
            s = tool_start + off
            windows["field_read"] = PositionWindow(s, min(s + max(n_field, 1), n_full), "field_read")

    return windows


# --------------------------------------------------------------------------
# Съём активаций
# --------------------------------------------------------------------------


class ActivationRecorder:
    """Пишет residual stream на выбранных слоях, усреднённый по окну.

    Использование:
        with ActivationRecorder(model, layers=[10, 14]) as rec:
            model(**batch)
        acts = rec.pooled(window)   # {layer: np.ndarray[hidden]}
    """

    def __init__(self, model: Any, layers: Sequence[int]):
        self.model = model
        self.layers = list(layers)
        self._store: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _hook(self, layer_idx: int):
        def fn(_module, _inp, out):
            hidden, _ = _split_output(out)
            # Копия обязательна: буфер переиспользуется на следующем шаге.
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

    def pooled(self, window: PositionWindow, batch_index: int = 0) -> dict[int, np.ndarray]:
        """Среднее по позициям окна. Форма hidden: [batch, seq, hidden]."""
        out: dict[int, np.ndarray] = {}
        for layer, h in self._store.items():
            seq = h.shape[1]
            end = min(window.end, seq)
            start = min(window.start, max(end - 1, 0))
            if end <= start:
                raise ValueError(
                    f"window [{window.start},{window.end}) outside sequence len {seq}"
                )
            out[layer] = h[batch_index, start:end, :].float().mean(dim=0).numpy()
        return out


# --------------------------------------------------------------------------
# Вектор
# --------------------------------------------------------------------------


@dataclass
class SteeringVector:
    """Направление вмешательства на одном слое.

    provenance фиксирует, на каких данных и в каком окне вектор получен —
    без этого прогон невоспроизводим (AGENTS.md: манифест обязателен).
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
        np.savez(
            path,
            vector=self.vector,
            layer=self.layer,
            window_kind=self.window_kind,
            provenance=np.array([repr(self.provenance)], dtype=object),
        )

    @staticmethod
    def load(path: str) -> "SteeringVector":
        z = np.load(path, allow_pickle=True)
        return SteeringVector(
            layer=int(z["layer"]),
            vector=z["vector"],
            window_kind=str(z["window_kind"]),  # type: ignore[arg-type]
            provenance=eval(str(z["provenance"][0])),  # noqa: S307 - свой же файл
        )


def contrastive_vector(
    acts_correct: Sequence[np.ndarray],
    acts_incorrect: Sequence[np.ndarray],
    layer: int,
    window_kind: Window,
    provenance: dict[str, Any] | None = None,
) -> SteeringVector:
    """Contrastive activation addition: mean(correct) - mean(incorrect).

    Собирается ТОЛЬКО на train (AGENTS.md).
    """
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
    """Низкоранговое подпространство различия (на случай, если одного вектора мало).

    Возвращает [rank, hidden] — главные компоненты разностей по классам.
    Применять, только если простой вектор не сработал (EXPERIMENT_PLAN, шаг 2).
    """
    a, b = np.stack(acts_correct), np.stack(acts_incorrect)
    diffs = a.mean(axis=0) - b
    diffs = np.concatenate([diffs, a - b.mean(axis=0)], axis=0)
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
    """Тот же вектор, но на соседнем слое — проверка специфичности слоя."""
    new = max(0, min(max_layer - 1, v.layer + delta))
    if new == v.layer:
        raise ValueError("shifted layer equals original; widen delta")
    return SteeringVector(new, v.vector, v.window_kind,
                          {**v.provenance, "sham": f"layer_shift{delta:+d}"})


def shifted_window(v: SteeringVector, w: PositionWindow, shift: int) -> PositionWindow:
    """Сдвинутое окно — проверка, что важен именно момент после ответа."""
    start = max(0, w.start + shift)
    return PositionWindow(start, start + len(w), w.kind, w.apply_to_generated)


SHAM_FACTORIES = {
    "random_direction": lambda v, **kw: random_direction_like(v, kw.get("seed", 0)),
    "permuted_coords": lambda v, **kw: permuted_direction(v, kw.get("seed", 0)),
}


# --------------------------------------------------------------------------
# Вмешательство
# --------------------------------------------------------------------------


@dataclass
class SteerConfig:
    """alpha, слой, окно и правило включения — выбираются только на validation."""

    alpha: float
    window: PositionWindow
    normalize: bool = True   # масштабировать по норме hidden: переносимо между слоями
    enabled: bool = True


class SteeringHook:
    """Добавляет alpha * v к residual stream в позициях окна.

    Корректно работает и на prefill (seq > 1), и на декодировании (seq == 1,
    KV-кэш): во втором случае вмешивается только если window.apply_to_generated.
    """

    def __init__(self, vector: SteeringVector, cfg: SteerConfig):
        self.vector = vector
        self.cfg = cfg
        self._handle: Any = None
        self._pos = 0          # сколько позиций уже обработано
        self.n_applied = 0     # сколько раз реально вмешались (для аудита)

    def reset(self) -> None:
        self._pos = 0
        self.n_applied = 0

    def _fn(self, _module, _inp, out):
        import torch

        hidden, rest = _split_output(out)
        if not self.cfg.enabled or self.cfg.alpha == 0.0:
            self._pos += hidden.shape[1]
            return out

        seq = hidden.shape[1]
        abs_start, abs_end = self._pos, self._pos + seq
        self._pos = abs_end

        w = self.cfg.window
        if seq == 1 and not w.apply_to_generated:
            return out

        lo, hi = max(w.start, abs_start), min(w.end, abs_end)
        if seq == 1 and w.apply_to_generated and abs_start >= w.end:
            lo, hi = abs_start, abs_end  # продолжаем на сгенерированных токенах
        if hi <= lo:
            return out

        v = torch.as_tensor(
            self.vector.unit() if self.cfg.normalize else self.vector.vector,
            dtype=hidden.dtype, device=hidden.device,
        )
        sl = slice(lo - abs_start, hi - abs_start)
        chunk = hidden[:, sl, :]
        if self.cfg.normalize:
            scale = chunk.norm(dim=-1, keepdim=True)
            delta = self.cfg.alpha * v * scale
        else:
            delta = self.cfg.alpha * v

        hidden = hidden.clone()
        hidden[:, sl, :] = chunk + delta
        self.n_applied += hi - lo
        return _rejoin_output(hidden, rest)

    def attach(self, model: Any) -> None:
        if self._handle is not None:
            raise RuntimeError("hook already attached")
        self.reset()
        self._handle = get_layers(model)[self.vector.layer].register_forward_hook(self._fn)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def steering(model: Any, vector: SteeringVector, cfg: SteerConfig) -> Iterator[SteeringHook]:
    """Контекст с включённым вмешательством.

        with steering(model, vec, cfg) as hook:
            out = model.generate(**batch)
        assert hook.n_applied > 0, "hook never fired - window is wrong"

    Проверка n_applied > 0 обязательна: молча не сработавший хук даёт
    "нулевой эффект", который легко принять за опровержение H3.
    """
    hook = SteeringHook(vector, cfg)
    hook.attach(model)
    try:
        yield hook
    finally:
        hook.detach()


def dose_response(
    alphas: Sequence[float],
    run_fn: Any,
) -> list[tuple[float, float]]:
    """Прогон по сетке alpha. run_fn(alpha) -> метрика.

    Монотонность — слабое свидетельство в пользу механизма; перелом или
    немонотонность требуют объяснения до любых выводов (EXPERIMENT_PLAN, шаг 5).
    """
    return [(a, float(run_fn(a))) for a in alphas]
