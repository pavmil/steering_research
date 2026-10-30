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
    ломает инвариантные пары. Поэтому подбор alpha — настоящий компромисс;
  * TACT, LoRA и двухшаговые методы дают свои синтетические эффекты;
  * на шагах продолжения эпизода агент почти всегда прав; аргументы
    вызова берутся из последнего ответа инструмента.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .. import simulator as S
from ..store import sha
from ..windows import PositionWindow
from .base import Decision, Prompt, ResolvedAffine, ResolvedAffineStack, ResolvedSteer, ResolvedTact

_WIN = {"tool_content": (100, 120), "field_read": (105, 108),
        "post_message": (116, 124), "pre_action": (124, 128), "action_point": (127, 128)}
_STATE_W = {"tool_content": 0.8, "field_read": 1.0, "post_message": 0.6, "pre_action": 0.4,
            "action_point": 0.4}
_USE_W = {"tool_content": 0.5, "field_read": 0.6, "post_message": 1.0, "pre_action": 0.8,
          "action_point": 0.9}


def _u(*parts: Any) -> float:
    return int(sha(list(parts), 12), 16) / 16 ** 12


class MockBackend:
    def __init__(self, n_layers: int = 12, hidden: int = 48, use_rate: float = 0.35,
                 gain: float = 0.9, cost: float = 0.015, sep: float = 1.6, seed: int = 0):
        self.model_key = f"mock-v2-L{n_layers}-h{hidden}-s{seed}"
        self.n_layers, self.hidden_size = n_layers, hidden
        self.use_rate, self.gain, self.cost, self.sep = use_rate, gain, cost, sep
        rng = np.random.default_rng(seed)
        self._state = self._unit(rng.normal(size=(n_layers, hidden)))
        self._use = self._unit(rng.normal(size=(n_layers, hidden)))
        self.calls = {"decide": 0, "record": 0, "train": 0}

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

    def _steer_effect(self, steer) -> tuple[float, float]:
        if steer is None:
            return 0.0, 0.0
        if isinstance(steer, ResolvedAffine):
            r = steer.U.shape[1] if hasattr(steer.U, "shape") else 1
            good = self.gain * 0.7 * (1 - math.exp(-steer.alpha)) * min(1.0, r / 8)
            return good, min(self.cost * 0.5 * steer.alpha, 0.9)
        if isinstance(steer, ResolvedAffineStack):
            # Сумма по слоям: та же форма, что у одного слоя, с суммарной силой и лучшим рангом.
            a = sum(p.alpha for p in steer.parts)
            r = max(p.U.shape[1] for p in steer.parts)
            good = self.gain * 0.75 * (1 - math.exp(-a)) * min(1.0, r / 8)
            return good, min(self.cost * 0.45 * a, 0.9)
        if isinstance(steer, ResolvedTact):
            cos = float(np.mean([abs(float(ax.unit @ self._use[ax.layer])) * self._profile(ax.layer)
                                 for ax in steer.axes]))
            good = self.gain * 0.8 * cos * (1 - math.exp(-steer.lam))
            return good, min(self.cost * 0.4 * steer.lam, 0.9)
        assert isinstance(steer, ResolvedSteer)
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
        other = p.meta.get("other_allowed", [])
        kind = p.meta["pair_type"]
        if kind == "continuation":
            if _u(p.key, "cont", trial) < 0.08 + harm:
                return sorted(set(S.ACTIONS) - set(allowed))[0]
            return allowed[0]
        if kind == "invariant":
            if _u(p.meta["task_id"], p.meta["branch"], "harm", trial) < harm:
                return sorted(set(S.ACTIONS) - set(allowed))[0]
            return allowed[0]
        if self._would_use(p, boost, trial):
            return allowed[0]
        return (other or allowed)[0]

    @staticmethod
    def _call_arg(msgs: list[dict], key: str) -> Any:
        for m in reversed(msgs):
            for tc in m.get("tool_calls") or []:
                v = tc["function"].get("arguments", {}).get(key)
                if v:
                    return v
        return None

    def _arguments(self, p: Prompt, action: str) -> dict[str, Any]:
        """Аргументы «как у разумного агента»: из последнего ответа инструмента."""
        last = next((m for m in reversed(p.messages) if m.get("role") == "tool"), None)
        try:
            payload = json.loads(last["content"]) if last else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        item = payload.get("item") or payload.get("query") or self._call_arg(p.messages, "item")
        if action == "place_order":
            alts = [a["item"] for a in payload.get("alternatives", []) if a.get("in_stock")]
            return {"item": alts[0] if alts else item}
        if action in ("search_alternative", "check_stock", "verify_source"):
            return {"item": item}
        if action == "request_access":
            return {"doc_id": payload.get("doc_id") or self._call_arg(p.messages, "doc_id")}
        if action == "ask_user":
            return {"question": "Is an alternative fine?"}
        if action == "stop_no_option":
            return {"reason": "no option"}
        return {}

    def _decision(self, p: Prompt, action: str, calls: int = 1, steer=None,
                  with_args: bool = False) -> Decision:
        scores = {a: (-0.1 if a == action else -3.0 - _u(p.key, a)) for a in S.ACTIONS}
        d = Decision(action=action, scores=scores, n_prompt_tokens=p.n_tokens,
                     n_new_tokens=4, n_model_calls=calls,
                     latency_ms=p.n_tokens * 0.05 * calls,
                     hook_applied=(len(steer.window) if getattr(steer, "window", None) else 1) if steer else None,
                     positions_source="mock" if steer else None)
        if isinstance(steer, ResolvedTact):
            d.hook_changed = len(steer.axes)
        if with_args:
            d.arguments = self._arguments(p, action)
            d.n_new_tokens += 12
            d.n_model_calls += 1
        return d

    def decide(self, prompt, mode, steer, temperature, trial, adapter=None, with_args=False,
               arg_tokens=40):
        self.calls["decide"] += 1
        boost, harm = self._steer_effect(steer)
        if prompt.meta.get("variant") == "strong_prompt":
            boost += 0.12
        if prompt.meta.get("variant") == "two_step":
            boost += 0.24
        if adapter is not None:
            boost, harm = boost + 0.3, harm + 0.04
        return self._decision(prompt, self._decide(prompt, boost, harm, trial), steer=steer,
                              with_args=with_args)

    def decide_two_step(self, prompt, ask, go, ask_tokens, mode, temperature, trial,
                        adapter=None, with_args=False, arg_tokens=40):
        self.calls["decide"] += 1
        boost = 0.24 if "STATE" in ask.upper() else 0.18
        return self._decision(prompt, self._decide(prompt, boost, 0.0, trial), calls=2,
                              with_args=with_args)

    def two_step_prompt(self, prompt, ask, go, ask_tokens):
        """Промпт второго шага: тот же ключ задачи, вариант — двухшаговый (учитель-промпт)."""
        meta = {**prompt.meta, "variant": "two_step"}
        return Prompt(key=sha([prompt.key, ask, go]), windows=dict(prompt.windows),
                      n_tokens=prompt.n_tokens + 64, messages=prompt.messages, tools=prompt.tools,
                      meta=meta)

    def decide_restate(self, prompt, mode, temperature, trial):
        return self.decide_two_step(prompt, "restate", "go", 48, mode, temperature, trial)

    def record_call_point(self, prompt, layers, adapter=None):
        rec = self.record(prompt, layers, ["action_point"])
        shift = 0.5 if adapter is not None else 0.3 if prompt.meta.get("variant") == "two_step" else 0.0
        return {L: rec[(L, "action_point")] + shift * self._use[L].astype(np.float32) for L in layers}

    # --- «обучение» адаптера --------------------------------------------
    def sft_example(self, prompt, action, arguments):
        return [len(prompt.key)], [len(action)]

    def reft_example(self, prompt, action):
        return [len(prompt.key)], [len(action)]

    def train_reft(self, examples, layer, rank, lr, epochs, seed=0):
        from ..interventions import AffineMap

        self.calls["train"] += 1
        rng = np.random.default_rng(seed + layer)
        d = self.hidden_size
        am = AffineMap((0.01 * rng.normal(size=(d, rank))).astype(np.float32),
                       np.linalg.qr(rng.normal(size=(d, rank)))[0].T.astype(np.float32),
                       np.zeros(d, np.float32), {"rank": rank, "layer": layer, "lr": lr, "epochs": epochs})
        return am, {"train_seconds": 0.0, "epoch_losses": [], "n_trainable": 2 * rank * d + rank,
                    "n_examples": len(examples)}

    def train_adapter(self, examples, lora_cfg, lr, epochs, out_dir):
        self.calls["train"] += 1
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "adapter.json").write_text(json.dumps({"lr": lr, "epochs": epochs}))
        return {"train_seconds": 0.0, "n_examples": len(examples), "epochs": epochs, "lr": lr,
                "n_trainable": 0, "train_tokens": 0, "epoch_losses": []}
