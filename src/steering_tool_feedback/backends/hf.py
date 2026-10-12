"""Бэкенд на transformers: CPU, Apple MPS, CUDA.

Mac (MPS):
  * dtype по умолчанию float16; PYTORCH_ENABLE_MPS_FALLBACK=1 ставится до
    импорта torch — редкие операции без MPS-ядра уйдут на CPU, а не упадут;
  * гибридные модели с линейным вниманием (Qwen3.5/3.6) на MPS идут через
    медленную референсную реализацию — для пилота лучше обычные 1–3B модели;
  * логиты всегда только по нужным позициям (logits_to_keep): у словарей
    на 150–250k токенов полные логиты на ~1k токенов весят ~1 ГБ.

Решение о действии:
  scored   — prefill промпта (с вмешательством) + префикс вызова, затем
             log-prob имени каждого действия из общего KV-кэша. Детерминированно;
             вызов идёт сразу, без рассуждения.
  generate — обычная генерация (рассуждение — если включено в шаблоне),
             действие — parse_action.
"""

from __future__ import annotations

import copy
import os
import time
from contextlib import nullcontext
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

from .. import steering as ST  # noqa: E402
from ..actions import ActionFormat, detect_action_format, parse_action  # noqa: E402
from ..methods import RESTATE_ASK, RESTATE_GO  # noqa: E402
from ..simulator import ACTIONS  # noqa: E402
from ..store import sha  # noqa: E402
from ..windows import prepare_prompt  # noqa: E402
from .base import Decision, Prompt, ResolvedSteer  # noqa: E402


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(pref: str, device: str):
    import torch

    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if pref != "auto":
        return table[pref]
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


class HFBackend:
    def __init__(self, model: Any, tokenizer: Any, name: str, device: str,
                 enable_thinking: bool | None = False, max_new_tokens: int = 96,
                 action_prefix: str | None = None, action_suffix: str | None = None):
        self.model, self.tok, self.name, self.device = model, tokenizer, name, device
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        self.n_layers = ST.n_layers(model)
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.hidden_size = int(cfg.hidden_size)
        self.dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        tmpl = getattr(tokenizer, "chat_template", None)
        self.template_hash = sha(str(tmpl)) if tmpl else None
        self.model_key = f"{name}|{self.dtype}|{device}|tmpl-{self.template_hash}"
        self._fmt: ActionFormat | None = (
            ActionFormat(action_prefix, action_suffix or "", "config") if action_prefix else None
        )
        self._cand_ids: dict[str, list[int]] | None = None
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, cfg) -> "HFBackend":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = pick_device(cfg.device)
        dtype = pick_dtype(cfg.dtype, device)
        tok = AutoTokenizer.from_pretrained(cfg.name)
        model = AutoModelForCausalLM.from_pretrained(cfg.name, dtype=dtype)
        model.to(device).eval()
        torch.set_grad_enabled(False)
        return cls(model, tok, cfg.name, device, cfg.enable_thinking, cfg.max_new_tokens,
                   cfg.action_prefix, cfg.action_suffix)

    def info(self) -> dict[str, Any]:
        fmt = self._fmt
        return {"backend": "hf", "model": self.name, "device": self.device, "dtype": self.dtype,
                "n_layers": self.n_layers, "hidden_size": self.hidden_size,
                "chat_template_hash": self.template_hash,
                "template_kwargs": self.template_kwargs,
                "action_format": None if fmt is None else
                {"prefix": fmt.prefix, "suffix": fmt.suffix, "source": fmt.source}}

    def free_memory(self) -> None:
        import torch

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def prompt(self, messages, essential_field, tools, meta=None, k_post=4) -> Prompt:
        p = prepare_prompt(self.tok, messages, essential_field=essential_field, tools=tools,
                           k_post=k_post, template_kwargs=self.template_kwargs)
        if self._fmt is None:
            self._fmt = detect_action_format(self.tok, messages, tools, self.template_kwargs)
        return Prompt(key=sha(p.input_ids), windows=dict(p.windows), n_tokens=len(p.input_ids),
                      messages=messages, tools=tools, prepared=p, meta=dict(meta or {}))

    @property
    def action_format(self) -> ActionFormat:
        if self._fmt is None:
            raise RuntimeError("action format unknown: build at least one prompt first")
        return self._fmt

    def _ids(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def _candidates(self) -> dict[str, list[int]]:
        if self._cand_ids is None:
            self._cand_ids = {a: self._ids(a + self.action_format.suffix) for a in ACTIONS}
        return self._cand_ids

    def _tensor(self, ids: list[int]):
        import torch

        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _steer_ctx(self, steer: ResolvedSteer | None):
        if steer is None:
            return nullcontext(None)
        return ST.steering(self.model, steer.vector,
                           ST.SteerConfig(alpha=steer.alpha, window=steer.window,
                                          normalize=steer.normalize))

    @staticmethod
    def _check_hook(hook, steer: ResolvedSteer | None, n_seq: int = 1) -> None:
        if hook is None:
            return
        expected = len(steer.window) * n_seq
        if hook.n_applied != expected:
            raise RuntimeError(
                f"steering hook changed {hook.n_applied} positions, expected {expected}: "
                "window and model input are out of sync"
            )

    # ------------------------------------------------------------------
    def score_actions(self, prompt_ids: list[int], steer: ResolvedSteer | None,
                      shared_prefix: bool = True) -> tuple[dict[str, float], Any]:
        """log p(prefix-continuation of each action | prompt). Возвращает (scores, hook)."""
        import torch

        pre = self._ids(self.action_format.prefix)
        ids = prompt_ids + pre
        cands = self._candidates()
        scores: dict[str, float] = {}
        with torch.no_grad(), self._steer_ctx(steer) as hook:
            if shared_prefix:
                out = self.model(input_ids=self._tensor(ids), use_cache=True, logits_to_keep=1)
                last = torch.log_softmax(out.logits[0, -1].float(), -1)
                cache = out.past_key_values
                for a, c in cands.items():
                    lp = float(last[c[0]])
                    if len(c) > 1:
                        o2 = self.model(input_ids=self._tensor(c[:-1]),
                                        past_key_values=copy.deepcopy(cache), use_cache=True)
                        lps = torch.log_softmax(o2.logits[0].float(), -1)
                        lp += float(sum(lps[i, c[i + 1]] for i in range(len(c) - 1)))
                    scores[a] = lp
                self._check_hook(hook, steer, 1)
            else:
                for a, c in cands.items():
                    out = self.model(input_ids=self._tensor(ids + c), logits_to_keep=len(c) + 1)
                    lps = torch.log_softmax(out.logits[0, : len(c)].float(), -1)
                    scores[a] = float(sum(lps[i, c[i]] for i in range(len(c))))
                self._check_hook(hook, steer, len(cands))
        return scores, hook

    def _generate(self, prompt_ids: list[int], steer: ResolvedSteer | None, temperature: float,
                  seed: int) -> tuple[str, int, Any]:
        import torch

        torch.manual_seed(seed)
        x = self._tensor(prompt_ids)
        kw: dict[str, Any] = {"max_new_tokens": self.max_new_tokens,
                              "pad_token_id": self.tok.pad_token_id}
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        with torch.no_grad(), self._steer_ctx(steer) as hook:
            out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), **kw)
            self._check_hook(hook, steer, 1)
        new = out[0, len(prompt_ids):].tolist()
        return self.tok.decode(new, skip_special_tokens=False), len(new), hook

    def decide(self, prompt, mode, steer, temperature, trial) -> Decision:
        t0 = time.perf_counter()
        ids = prompt.prepared.input_ids
        if mode == "scored":
            scores, hook = self.score_actions(ids, steer)
            action = max(scores, key=scores.get)
            d = Decision(action=action, scores=scores, n_prompt_tokens=len(ids),
                         n_new_tokens=0, n_model_calls=1 + len(scores))
        else:
            text, n_new, hook = self._generate(ids, steer, temperature,
                                               seed=int(sha([prompt.key, trial], 8), 16))
            d = Decision(action=parse_action(text), text=text, n_prompt_tokens=len(ids),
                         n_new_tokens=n_new, n_model_calls=1)
        d.latency_ms = (time.perf_counter() - t0) * 1000
        if hook is not None:
            d.hook_applied, d.positions_source = hook.n_applied, hook.positions_source
        return d

    def decide_restate(self, prompt, mode, temperature, trial) -> Decision:
        """Явный пересказ ответа инструмента моделью, затем выбор действия."""
        t0 = time.perf_counter()
        ask = prompt.messages + [{"role": "user", "content": RESTATE_ASK}]
        p1 = prepare_prompt(self.tok, ask, tools=prompt.tools, locate=False,
                            template_kwargs=self.template_kwargs)
        keep, self.max_new_tokens = self.max_new_tokens, 48
        try:
            restated, n1, _ = self._generate(p1.input_ids, None, 0.0, seed=0)
        finally:
            self.max_new_tokens = keep
        clean = restated.split("<|im_end|>")[0].replace("<think>", "").replace("</think>", "").strip()
        msgs = ask + [{"role": "assistant", "content": clean},
                      {"role": "user", "content": RESTATE_GO}]
        p2 = prepare_prompt(self.tok, msgs, tools=prompt.tools, locate=False,
                            template_kwargs=self.template_kwargs)
        sub = Prompt(key=sha(p2.input_ids), windows={}, n_tokens=len(p2.input_ids),
                     messages=msgs, tools=prompt.tools, prepared=p2)
        d = self.decide(sub, mode, None, temperature, trial)
        d.n_prompt_tokens += len(p1.input_ids)
        d.n_new_tokens += n1
        d.n_model_calls += 1
        d.text = f"[restated] {clean}\n" + (d.text or "")
        d.latency_ms = (time.perf_counter() - t0) * 1000
        return d

    # ------------------------------------------------------------------
    def record(self, prompt, layers, windows) -> dict[tuple[int, str], np.ndarray]:
        import torch

        with torch.no_grad(), ST.ActivationRecorder(self.model, layers) as rec:
            self.model(**prompt.prepared.model_inputs(self.device), logits_to_keep=1)
            out = {}
            for w in windows:
                if w not in prompt.windows:
                    continue
                pooled = rec.pooled(prompt.windows[w])
                for L in layers:
                    out[(L, w)] = pooled[L].astype(np.float32)
        return out
