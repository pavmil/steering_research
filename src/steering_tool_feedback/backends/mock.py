"""Синтетический бэкенд: тот же интерфейс, что у модели, но мгновенно.

Нужен, чтобы проверять пайплайн, кэш, гейты, журнал и ноутбуки без GPU.
Его числа — НЕ результат исследования.

Как устроено:
  * у каждого слоя есть «направление состояния» (что сказал инструмент) и
    «направление использования» (агент реально опирается на ответ);
  * активации = шум + состояние + использование, с профилем силы по слоям
    и окнам — probe находит состояние, CAA по исходам находит использование;
  * steering по вектору, сонаправленному с «использованием», повышает долю
    верных переключений; любое возмущение (в т.ч. sham) с ростом alpha
    ломает инвариантные пары. Поэтому подбор alpha — настоящий компромисс.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import simulator as S
from ..store import sha
from ..windows import PositionWindow
from .base import Decision, Prompt, ResolvedSteer

_WIN = {"tool_content": (100, 120), "field_read": (105, 108),
        "post_message": (116, 124), "pre_action": (124, 128)}
_STATE_W = {"tool_content": 0.8, "field_read": 1.0, "post_message": 0.6, "pre_action": 0.4}
_USE_W = {"tool_content": 0.5, "field_read": 0.6, "post_message": 1.0, "pre_action": 0.8}


def _u(*parts: Any) -> float:
    return int(sha(list(parts), 12), 16) / 16 ** 12


class MockBackend:
    def __init__(self, n_layers: int = 12, hidden: int = 48, use_rate: float = 0.35,
                 gain: float = 0.9, cost: float = 0.015, sep: float = 1.6, seed: int = 0):
        self.model_key = f"mock-v1-L{n_layers}-h{hidden}-s{seed}"
        self.n_layers, self.hidden_size = n_layers, hidden
        self.use_rate, self.gain, self.cost, self.sep = use_rate, gain, cost, sep
        rng = np.random.default_rng(seed)
        self._state = self._unit(rng.normal(size=(n_layers, hidden)))
        self._use = self._unit(rng.normal(size=(n_layers, hidden)))
        self.calls = {"decide": 0, "record": 0}

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=-1, keepdims=True)

    def _profile(self, layer: int) -> float:
        mid = self.n_layers * 0.55
        return math.exp(-((layer - mid) / (self.n_layers * 0.22)) ** 2)

    def info(self) -> dict[str, Any]:
        return {"backend": "mock", "model": self.model_key, "device": "cpu", "dtype": "float32",
                "n_layers": self.n_layers, "hidden_size": self.hidden_size,
                "chat_template_hash": None, "action_format": None}

    # ------------------------------------------------------------------
    def prompt(self, messages, essential_field, tools, meta=None, k_post=4) -> Prompt:
        meta = dict(meta or {})
        key = sha([messages, essential_field, bool(tools)])
        wins = {k: PositionWindow(a, b, k) for k, (a, b) in _WIN.items()}  # type: ignore[arg-type]
        return Prompt(key=key, windows=wins, n_tokens=128, messages=messages, tools=tools,
                      meta=meta)

    def _would_use(self, p: Prompt, boost: float = 0.0, trial: int = 0) -> bool:
        return _u(p.meta.get("task_id"), p.meta.get("branch"), "use", trial) < self.use_rate + boost

    def record(self, prompt, layers, windows):
        self.calls["record"] += 1
        state = 1.0 if prompt.meta.get("state", 0) else -1.0
        used = 1.0 if self._would_use(prompt) else -1.0
        out = {}
        for L in layers:
            for w in windows:
                rng = np.random.default_rng(int(sha([prompt.key, L, w], 8), 16))
                a = rng.normal(size=self.hidden_size)
                a += self.sep * self._profile(L) * _STATE_W[w] * state * self._state[L]
                a += self.sep * self._profile(L) * _USE_W[w] * used * self._use[L] * 0.8
                out[(L, w)] = a.astype(np.float32)
        return out

    def _steer_effect(self, steer: ResolvedSteer | None) -> tuple[float, float]:
        if steer is None:
            return 0.0, 0.0
        L = steer.vector.layer
        cos = float(steer.vector.unit() @ self._use[L])
        # Вмешательство до ответа инструмента не может помочь его учесть.
        w = 0.0 if steer.window.end <= _WIN["tool_content"][0] else _USE_W.get(steer.window.kind, 0.3)
        a = steer.alpha
        good = self.gain * max(cos, 0.0) * self._profile(L) * w * (1 - math.exp(-a))
        bad = self.cost * a * (1.0 + abs(min(cos, 0.0)))
        return good, min(bad, 0.9)

    def _decide(self, p: Prompt, boost: float, harm: float, trial: int) -> str:
        allowed = p.meta["allowed"]
        other = p.meta["other_allowed"]
        kind = p.meta["pair_type"]
        if kind == "invariant":
            if _u(p.meta["task_id"], p.meta["branch"], "harm", trial) < harm:
                return sorted(set(S.ACTIONS) - set(allowed))[0]
            return allowed[0]
        if self._would_use(p, boost, trial):
            return allowed[0]
        return (other or allowed)[0]

    def _decision(self, p: Prompt, action: str, calls: int = 1,
                  steer: ResolvedSteer | None = None) -> Decision:
        scores = {a: (-0.1 if a == action else -3.0 - _u(p.key, a)) for a in S.ACTIONS}
        return Decision(action=action, scores=scores, n_prompt_tokens=p.n_tokens,
                        n_new_tokens=4, n_model_calls=calls,
                        latency_ms=p.n_tokens * 0.05 * calls,
                        hook_applied=len(steer.window) if steer else None,
                        positions_source="mock" if steer else None)

    def decide(self, prompt, mode, steer, temperature, trial):
        self.calls["decide"] += 1
        boost, harm = self._steer_effect(steer)
        if prompt.meta.get("variant") == "strong_prompt":
            boost += 0.12
        return self._decision(prompt, self._decide(prompt, boost, harm, trial), steer=steer)

    def decide_restate(self, prompt, mode, temperature, trial):
        self.calls["decide"] += 1
        return self._decision(prompt, self._decide(prompt, 0.18, 0.0, trial), calls=2)
