"""Бэкенд на transformers: CPU, Apple MPS, CUDA.

Mac (MPS):
  * dtype по умолчанию float16; PYTORCH_ENABLE_MPS_FALLBACK=1 ставится до
    импорта torch — редкие операции без MPS-ядра уйдут на CPU, а не упадут;
  * HF_DEACTIVATE_ASYNC_LOAD=1 — веса грузятся в одном потоке (см. ниже, зачем);
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
with_args=True — модель дописывает и аргументы вызова (нужно в многошаговых
эпизодах: среде надо знать, какой товар заказывать).

Вмешательства: ResolvedSteer (CAA-вектор в окне) и ResolvedTact (адаптация
TACT-Gate на нескольких слоях). LoRA-адаптер (ResolvedAdapter) включается только
внутри решения и выключается в finally — база остаётся базой.
"""

from __future__ import annotations

import copy
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Параллельная загрузка весов (пул потоков transformers 5) взаимно блокируется внутри
# safetensors при подключении LoRA-адаптера: один поток ждёт GIL в OnceCell, другой держит
# GIL и ждёт тот же OnceCell (зависание на 30+ мин, 04 на Qwen3-1.7B, 2026-10-19).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import numpy as np  # noqa: E402

from .. import steering as ST  # noqa: E402
from ..actions import (ActionFormat, detect_action_format, name_tokens, parse_action,  # noqa: E402
                       parse_arguments)
from ..simulator import ACTIONS, tool_call_message  # noqa: E402
from ..store import sha  # noqa: E402
from ..windows import prepare_prompt  # noqa: E402
from .base import (Decision, Prompt, ResolvedAdapter, ResolvedAffine, ResolvedAffineStack,  # noqa: E402
                   ResolvedSteer, ResolvedTact)


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
            ActionFormat(action_prefix, (action_suffix or "")[:1], "config", action_suffix or "")
            if action_prefix else None
        )
        self._cand_ids: dict[str, list[int]] | None = None
        self._pre_ids: list[int] = []
        self._sys_ids: dict[str, list[int]] = {}
        self._sys_cache: dict[tuple, Any] = {}
        self._adapters: set[str] = set()
        self._adapter_on: str | None = None
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
                {"prefix": fmt.prefix, "suffix": fmt.suffix, "source": fmt.source,
                 "scoring": "canonical-name-tokens"}}

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
        """Токены имени каждого действия — из канонической токенизации вызова (actions.name_tokens)."""
        if self._cand_ids is None:
            self._pre_ids, self._cand_ids = name_tokens(self.tok, self.action_format, ACTIONS)
        return self._cand_ids

    def _prefix_ids(self) -> list[int]:
        self._candidates()
        return self._pre_ids

    def _tensor(self, ids: list[int]):
        import torch

        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _steer_ctx(self, steer: ResolvedSteer | ResolvedTact | ResolvedAffine | None):
        if steer is None:
            return nullcontext(None)
        if isinstance(steer, ResolvedTact):
            return ST.drift_clamp(self.model, steer.axes, steer.window, steer.k, steer.lam)
        if isinstance(steer, ResolvedAffine):
            return ST.affine(self.model, steer.layer, steer.window, steer.U, steer.P, steer.a0, steer.alpha)
        if isinstance(steer, ResolvedAffineStack):
            return ST.affine_stack(self.model, steer.parts)
        return ST.steering(self.model, steer.vector,
                           ST.SteerConfig(alpha=steer.alpha, window=steer.window,
                                          normalize=steer.normalize))

    @staticmethod
    def _check_hook(hook, steer, n_seq: int = 1) -> None:
        if hook is None:
            return
        expected = len(steer.window) * n_seq
        if hook.n_applied != expected:
            raise RuntimeError(
                f"steering hook changed {hook.n_applied} positions, expected {expected}: "
                "window and model input are out of sync"
            )

    # --- LoRA ----------------------------------------------------------
    @contextmanager
    def _adapter_ctx(self, adapter: ResolvedAdapter | None) -> Iterator[None]:
        if adapter is None:
            if self._adapter_on is not None:
                raise RuntimeError(f"adapter {self._adapter_on} is still enabled: base decision "
                                   "would be contaminated")
            yield
            return
        if adapter.adapter_id not in self._adapters:
            self.model.load_adapter(adapter.path, adapter_name=adapter.adapter_id)
            self._adapters.add(adapter.adapter_id)
        self.model.set_adapter(adapter.adapter_id)
        self.model.enable_adapters()
        self._adapter_on = adapter.adapter_id
        try:
            yield
        finally:
            self.model.disable_adapters()
            self._adapter_on = None

    # --- кэш системного хода ----------------------------------------------
    def _system_prefix(self, ids: list[int], messages: list[dict], tools: list[dict] | None,
                       steer) -> tuple[int, Any]:
        """KV-кэш системного хода (system + схемы инструментов): он одинаков у всех задач
        и занимает ~85% промпта. Окна вмешательства всегда после него — иначе кэш не берётся.
        Ключ включает активный LoRA-адаптер: адаптер меняет k/v и в префиксе."""
        import torch

        if not messages or messages[0].get("role") != "system":
            return 0, None
        k = sha([messages[0], tools, self.template_kwargs])
        if k not in self._sys_ids:
            text = self.tok.apply_chat_template([messages[0]], tools=tools, tokenize=False,
                                                add_generation_prompt=False, **self.template_kwargs)
            self._sys_ids[k] = self._ids(text) if isinstance(text, str) else []
        sys_ids = self._sys_ids[k]
        n = len(sys_ids)
        if n == 0 or n >= len(ids) or ids[:n] != sys_ids:
            return 0, None
        if steer is not None and steer.window.start < n:
            return 0, None
        ck = (k, self._adapter_on)
        if ck not in self._sys_cache:
            with torch.no_grad():
                out = self.model(input_ids=self._tensor(sys_ids), use_cache=True, logits_to_keep=1)
            if len(self._sys_cache) >= 8:
                self._sys_cache.pop(next(iter(self._sys_cache)))
            self._sys_cache[ck] = out.past_key_values
        return n, self._sys_cache[ck]

    def _forward_from(self, ids: list[int], prefix: tuple[int, Any], **kw):
        n, cache = prefix
        if cache is None:
            return self.model(input_ids=self._tensor(ids), **kw)
        return self.model(input_ids=self._tensor(ids[n:]), past_key_values=copy.deepcopy(cache), **kw)

    # ------------------------------------------------------------------
    def score_actions(self, prompt_ids: list[int], steer: ResolvedSteer | ResolvedTact | None,
                      shared_prefix: bool = True,
                      prefix: tuple[int, Any] = (0, None)) -> tuple[dict[str, float], Any]:
        """log p(токены имени действия | промпт + префикс вызова). Возвращает (scores, hook).

        shared_prefix=True: один prefill (вмешательство — только в нём: окна лежат в промпте),
        затем продолжения всех кандидатов одним батчем из общего KV-кэша.
        shared_prefix=False: отдельный полный прогон на каждого кандидата — эталон для тестов.
        """
        import torch

        cands = self._candidates()
        ids = prompt_ids + self._prefix_ids()
        scores: dict[str, float] = {}
        if not shared_prefix:
            with torch.no_grad(), self._steer_ctx(steer) as hook:
                for a, c in cands.items():
                    out = self.model(input_ids=self._tensor(ids + c), logits_to_keep=len(c) + 1)
                    lps = torch.log_softmax(out.logits[0, : len(c)].float(), -1)
                    scores[a] = float(sum(lps[i, c[i]] for i in range(len(c))))
                self._check_hook(hook, steer, len(cands))
            return scores, hook
        with torch.no_grad():
            with self._steer_ctx(steer) as hook:
                out = self._forward_from(ids, prefix, use_cache=True, logits_to_keep=1)
                self._check_hook(hook, steer, 1)
            last = torch.log_softmax(out.logits[0, -1].float(), -1)
            names = list(cands)
            rest = [cands[a][:-1] for a in names]
            width = max(len(r) for r in rest)
            lps = None
            cache = out.past_key_values
            batchable = all(hasattr(x, "batch_repeat_interleave") for x in getattr(cache, "layers", []))
            if width and not batchable:
                # Гибриды с линейным вниманием (Qwen3.5/3.6): кэш не размножается по батчу —
                # кандидаты по одному, каждый со своей копией кэша.
                for a, c in cands.items():
                    lp = float(last[c[0]])
                    if len(c) > 1:
                        o2 = self.model(input_ids=self._tensor(c[:-1]),
                                        past_key_values=copy.deepcopy(cache), use_cache=True)
                        step = torch.log_softmax(o2.logits[0].float(), -1)
                        lp += float(sum(step[i, c[i + 1]] for i in range(len(c) - 1)))
                    scores[a] = lp
                return scores, hook
            if width:
                n_ctx, b = len(ids), len(names)
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                x = torch.tensor([r + [pad] * (width - len(r)) for r in rest], dtype=torch.long,
                                 device=self.device)
                mask = torch.zeros(b, n_ctx + width, dtype=torch.long, device=self.device)
                mask[:, :n_ctx] = 1
                for j, r in enumerate(rest):
                    mask[j, n_ctx:n_ctx + len(r)] = 1
                cache.batch_repeat_interleave(b)
                o2 = self.model(input_ids=x, attention_mask=mask, past_key_values=cache, use_cache=True)
                lps = torch.log_softmax(o2.logits.float(), -1)
            for j, a in enumerate(names):
                c = cands[a]
                lp = float(last[c[0]])
                for i in range(len(c) - 1):
                    lp += float(lps[j, i, c[i + 1]])
                scores[a] = lp
        return scores, hook

    def _generate(self, prompt_ids: list[int], steer, temperature: float, seed: int,
                  max_new_tokens: int | None = None, stop_strings: list[str] | None = None,
                  prefix: tuple[int, Any] = (0, None)) -> tuple[str, int, Any]:
        import torch

        torch.manual_seed(seed)
        x = self._tensor(prompt_ids)
        kw: dict[str, Any] = {"max_new_tokens": max_new_tokens or self.max_new_tokens,
                              "pad_token_id": self.tok.pad_token_id}
        if stop_strings:
            kw.update(stop_strings=stop_strings, tokenizer=self.tok)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if prefix[1] is not None:
            kw["past_key_values"] = copy.deepcopy(prefix[1])
        with torch.no_grad(), self._steer_ctx(steer) as hook:
            out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), **kw)
            self._check_hook(hook, steer, 1)
        new = out[0, len(prompt_ids):].tolist()
        return self.tok.decode(new, skip_special_tokens=False), len(new), hook

    def _arguments(self, prompt_ids: list[int], action: str, steer, max_new_tokens: int,
                   prefix: tuple[int, Any] = (0, None)) -> tuple[dict[str, Any], int]:
        """Модель дописывает аргументы после выбранного имени действия (жадно)."""
        head = self._prefix_ids() + self._candidates()[action]
        text, n_new, _ = self._generate(prompt_ids + head, steer, 0.0, seed=0,
                                        max_new_tokens=max_new_tokens, stop_strings=["</tool_call>"],
                                        prefix=prefix)
        return parse_arguments(self.tok.decode(head, skip_special_tokens=False) + text), n_new

    def _bind(self, steer, ids: list[int], mode: str):
        """Аффинное вмешательство без окна — в точку вызова: последний токен промпта + префикса
        вызова (scored) или последний токен промпта (generate: префикс пишет сама модель)."""
        if isinstance(steer, (ResolvedAffine, ResolvedAffineStack)) and steer.window is None:
            import dataclasses

            from ..windows import PositionWindow

            i = len(ids) + (len(self._prefix_ids()) if mode == "scored" else 0) - 1
            w = PositionWindow(i, i + 1, "action_point")
            if isinstance(steer, ResolvedAffineStack):
                return ResolvedAffineStack(tuple(dataclasses.replace(p, window=w) for p in steer.parts))
            return dataclasses.replace(steer, window=w)
        return steer

    def record_call_point(self, prompt, layers, adapter=None) -> dict[int, np.ndarray]:
        """Скрытое состояние в точке вызова (последний токен промпта + префикса) на слоях."""
        import torch

        ids = list(prompt.prepared.input_ids) + self._prefix_ids()
        with self._adapter_ctx(adapter), torch.no_grad(), ST.ActivationRecorder(self.model, layers) as rec:
            self.model(input_ids=self._tensor(ids), logits_to_keep=1)
            return {L: rec._store[L][0, -1].float().numpy().astype(np.float32) for L in layers}

    def decide(self, prompt, mode, steer, temperature, trial, adapter=None, with_args=False,
               arg_tokens=40) -> Decision:
        t0 = time.perf_counter()
        ids = prompt.prepared.input_ids
        steer = self._bind(steer, ids, mode)
        with self._adapter_ctx(adapter):
            prefix = self._system_prefix(ids, prompt.messages, prompt.tools, steer)
            if mode == "scored":
                scores, hook = self.score_actions(ids, steer, prefix=prefix)
                action = max(scores, key=scores.get)
                d = Decision(action=action, scores=scores, n_prompt_tokens=len(ids),
                             n_new_tokens=0, n_model_calls=1 + len(scores))
                if with_args:
                    d.arguments, n_arg = self._arguments(ids, action, steer, arg_tokens, prefix)
                    d.n_new_tokens += n_arg
                    d.n_model_calls += 1
            else:
                text, n_new, hook = self._generate(ids, steer, temperature,
                                                   seed=int(sha([prompt.key, trial], 8), 16),
                                                   prefix=prefix)
                d = Decision(action=parse_action(text), text=text, n_prompt_tokens=len(ids),
                             n_new_tokens=n_new, n_model_calls=1)
                if with_args:
                    d.arguments = parse_arguments(text)
        d.latency_ms = (time.perf_counter() - t0) * 1000
        if hook is not None:
            d.hook_applied, d.positions_source = hook.n_applied, hook.positions_source
            d.hook_changed = getattr(hook, "n_changed", None)
        return d

    def decide_two_step(self, prompt, ask, go, ask_tokens, mode, temperature, trial,
                        adapter=None, with_args=False, arg_tokens=40) -> Decision:
        """Модель сначала пишет текст по просьбе ask (пересказ ответа, явное состояние),
        затем по просьбе go выбирает действие. Два вызова модели — учитывать в цене."""
        t0 = time.perf_counter()
        sub, (clean, n1, n_ask_prompt) = self._two_step(prompt, ask, go, ask_tokens, adapter)
        d = self.decide(sub, mode, None, temperature, trial, adapter, with_args, arg_tokens)
        d.n_prompt_tokens += n_ask_prompt
        d.n_new_tokens += n1
        d.n_model_calls += 1
        d.text = f"[note] {clean}\n" + (d.text or "")
        d.latency_ms = (time.perf_counter() - t0) * 1000
        return d

    def _two_step(self, prompt, ask, go, ask_tokens, adapter=None) -> tuple[Prompt, tuple[str, int, int]]:
        """Первый шаг двухшагового метода: модель жадно пишет текст по просьбе ask; возвращает
        промпт второго шага (… + текст + go) и (текст, его длина, длина промпта первого шага)."""
        ask_msgs = prompt.messages + [{"role": "user", "content": ask}]
        p1 = prepare_prompt(self.tok, ask_msgs, tools=prompt.tools, locate=False,
                            template_kwargs=self.template_kwargs)
        with self._adapter_ctx(adapter):
            prefix = self._system_prefix(p1.input_ids, ask_msgs, prompt.tools, None)
            note, n1, _ = self._generate(p1.input_ids, None, 0.0, seed=0, max_new_tokens=ask_tokens,
                                         prefix=prefix)
        for stop in ("<|im_end|>", "<|eot_id|>", self.tok.eos_token or "\0"):
            note = note.split(stop)[0]
        clean = note.replace("<think>", "").replace("</think>", "").strip()
        msgs = ask_msgs + [{"role": "assistant", "content": clean},
                           {"role": "user", "content": go}]
        p2 = prepare_prompt(self.tok, msgs, tools=prompt.tools, locate=False,
                            template_kwargs=self.template_kwargs)
        sub = Prompt(key=sha(p2.input_ids), windows={}, n_tokens=len(p2.input_ids),
                     messages=msgs, tools=prompt.tools, prepared=p2, meta=dict(prompt.meta))
        return sub, (clean, n1, len(p1.input_ids))

    def two_step_prompt(self, prompt, ask, go, ask_tokens) -> Prompt:
        """Промпт, на котором двухшаговый метод выбирает действие (учитель-промпт для linmap_prompt)."""
        return self._two_step(prompt, ask, go, ask_tokens)[0]

    def decide_restate(self, prompt, mode, temperature, trial) -> Decision:
        from ..methods import RESTATE_ASK, RESTATE_GO

        return self.decide_two_step(prompt, RESTATE_ASK, RESTATE_GO, 48, mode, temperature, trial)

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

    # --- обучение адаптера -----------------------------------------------
    def sft_example(self, prompt: Prompt, action: str,
                    arguments: dict[str, Any]) -> tuple[list[int], list[int]]:
        """(вход, цель): вход — ровно тот промпт, что видит оценка; цель — вызов в формате шаблона."""
        msgs = prompt.messages
        gen = self.tok.apply_chat_template(msgs, tools=prompt.tools, tokenize=False,
                                           add_generation_prompt=True, **self.template_kwargs)
        full = self.tok.apply_chat_template(msgs + [tool_call_message(action, arguments)],
                                            tools=prompt.tools, tokenize=False,
                                            add_generation_prompt=False, **self.template_kwargs)
        if not full.startswith(gen):
            raise ValueError("rendered tool call does not extend the generation prompt")
        if gen != prompt.prepared.text:
            raise ValueError("SFT prompt differs from the evaluated prompt")
        return list(prompt.prepared.input_ids), self._ids(full[len(gen):])

    def reft_example(self, prompt: Prompt, action: str) -> tuple[list[int], list[int]]:
        """(промпт + префикс вызова, токены имени действия) — пример для LoReFT в точке вызова."""
        return list(prompt.prepared.input_ids) + self._prefix_ids(), list(self._candidates()[action])

    def train_reft(self, examples, layer, rank, lr, epochs, seed=0):
        from ..interventions import train_loreft

        return train_loreft(self, examples, layer, rank, lr, epochs, seed)

    def train_adapter(self, examples, lora_cfg, lr, epochs, out_dir) -> dict[str, Any]:
        from ..finetune import train_lora

        return train_lora(self.model, self.device, examples, lora_cfg, lr, epochs, out_dir)
