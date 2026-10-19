"""Окна позиций для вмешательства и их проверка глазами.

Почему этот модуль отдельно и почему так.

Первая версия считала окна арифметикой: длина токенов шаблона без последнего
сообщения vs с ним. На реальном шаблоне Qwen3.x это ломается молча:
ответ инструмента заворачивается в <tool_response> внутри user-хода, а в конец
добавляется generation prompt `<|im_start|>assistant\\n<think>\\n`.
В итоге «post_message» целиком попадал в служебные токены и не содержал
ни одного токена самого ответа, а «pre_action» с ним перекрывался.

Здесь окна считаются иначе:
  1. рендерим шаблон ОДИН раз в строку;
  2. токенизируем ЭТУ ЖЕ строку с offset_mapping;
  3. ищем символьные ориентиры (текст ответа, существенное поле, начало
     generation prompt) и переводим их в индексы токенов через offsets.

input_ids возвращаются вместе с окнами в PreparedPrompt — в модель надо подавать
именно их, тогда окна и вход не могут разъехаться.

ОГРАНИЧЕНИЕ: batch = 1. Окна — индексы в одной последовательности без паддинга.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Window = Literal["tool_content", "field_read", "post_message", "pre_action", "action_point"]

WINDOW_ORDER: tuple[Window, ...] = ("tool_content", "field_read", "post_message", "pre_action",
                                    "action_point")


class WindowError(ValueError):
    """Окно нельзя построить надёжно. Не ловить молча: это ошибка данных или шаблона."""


@dataclass(frozen=True)
class PositionWindow:
    """Полуинтервал [start, end) абсолютных позиций токенов.

    apply_to_generated — продолжать ли вмешательство на сгенерированных токенах.
    По умолчанию False: так эффект нельзя списать на удлинение рассуждения.
    """

    start: int
    end: int
    kind: Window = "post_message"
    apply_to_generated: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"bad window [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start

    def contains(self, other: "PositionWindow") -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass
class PreparedPrompt:
    """Отрендеренный промпт + токены + окна. Единый источник правды для прогона."""

    text: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    windows: dict[Window, PositionWindow]
    char_spans: dict[str, tuple[int, int]]
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.input_ids)

    def window(self, kind: Window, apply_to_generated: bool = False) -> PositionWindow:
        w = self.windows[kind]
        if apply_to_generated == w.apply_to_generated:
            return w
        return PositionWindow(w.start, w.end, w.kind, apply_to_generated)

    def model_inputs(self, device: Any = None) -> dict[str, Any]:
        """Тензоры для model(**inputs) / model.generate(**inputs). batch = 1."""
        import torch

        ids = torch.tensor([self.input_ids], dtype=torch.long)
        mask = torch.ones_like(ids)
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return {"input_ids": ids, "attention_mask": mask}


# --------------------------------------------------------------------------
# Построение
# --------------------------------------------------------------------------


def _tokens_overlapping(offsets: Sequence[tuple[int, int]], a: int, b: int) -> tuple[int, int]:
    """Индексы токенов, пересекающих символьный интервал [a, b). (lo, hi) или (-1, -1)."""
    hits = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
    if not hits:
        return -1, -1
    return hits[0], hits[-1] + 1


def _first_token_at_or_after(offsets: Sequence[tuple[int, int]], char_pos: int) -> int:
    for i, (s, e) in enumerate(offsets):
        if e > s and s >= char_pos:
            return i
    return len(offsets)


def _render(tokenizer: Any, messages: list[dict[str, Any]], gen: bool,
            tools: list[dict] | None, template_kwargs: dict[str, Any]) -> str:
    out = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=gen, **template_kwargs
    )
    if not isinstance(out, str):
        raise WindowError("apply_chat_template(tokenize=False) did not return a string")
    return out


def prepare_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    essential_field: str | None = None,
    tools: list[dict] | None = None,
    k_post: int = 4,
    k_pre_fallback: int = 4,
    require_field: bool = True,
    template_kwargs: dict[str, Any] | None = None,
    locate: bool = True,
) -> PreparedPrompt:
    """Отрендерить промпт и построить окна.

    messages        — последний элемент обязан быть сообщением инструмента (role="tool").
    essential_field — подстрока ответа, несущая существенное различие веток,
                      напр. '"in_stock": false'. Ищется ТОЛЬКО внутри ответа.
    k_post          — сколько последних токенов ответа включить в post_message
                      (плюс закрывающая обёртка шаблона до generation prompt).

    Окна:
      tool_content — все токены текста ответа инструмента;
      field_read   — токены существенного поля;
      post_message — хвост ответа + закрывающая обёртка (</tool_response><|im_end|>…);
      pre_action   — generation prompt: позиции прямо перед первым токеном действия;
      action_point — последний токен промпта: точка, где модель выбирает действие
                     (аналог позиции </think> в TACT; у Qwen3 без рассуждения это
                     токен сразу после пустого блока <think></think>).

    locate=False — только рендер и токены, без окон (для методов, где после
    ответа инструмента идут ещё сообщения и вмешательства нет).
    """
    if not locate:
        kw = dict(template_kwargs or {})
        text = _render(tokenizer, messages, True, tools, kw)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return PreparedPrompt(text=text, input_ids=list(enc["input_ids"]),
                              offsets=[tuple(o) for o in enc["offset_mapping"]],
                              windows={}, char_spans={})
    if not messages or messages[-1].get("role") != "tool":
        raise WindowError("last message must have role='tool'")
    if not getattr(tokenizer, "is_fast", False):
        raise WindowError("need a fast tokenizer (offset_mapping); got a slow one")

    kw = dict(template_kwargs or {})
    full = _render(tokenizer, messages, True, tools, kw)
    nogen = _render(tokenizer, messages, False, tools, kw)

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
    notes: list[str] = []

    # --- ответ инструмента ----------------------------------------------
    obs = str(messages[-1]["content"])
    obs_start = full.rfind(obs)
    if obs_start < 0:
        raise WindowError(
            "tool content not found verbatim in the rendered prompt: "
            "the template rewrites it, windows cannot be located reliably"
        )
    obs_end = obs_start + len(obs)
    if full.count(obs) > 1:
        notes.append("tool content occurs more than once; using the LAST occurrence")

    t_lo, t_hi = _tokens_overlapping(offsets, obs_start, obs_end)
    if t_lo < 0:
        raise WindowError("no tokens overlap the tool content")

    # --- generation prompt ---------------------------------------------
    if full.startswith(nogen) and len(nogen) < len(full):
        gen_char = len(nogen)
        g_lo = _first_token_at_or_after(offsets, gen_char)
    else:
        g_lo = max(t_hi, len(ids) - k_pre_fallback)
        notes.append(
            "template has no separable generation prompt; pre_action = last "
            f"{len(ids) - g_lo} tokens"
        )
        gen_char = offsets[g_lo][0] if g_lo < len(offsets) else len(full)

    if g_lo < t_hi:
        raise WindowError("generation prompt starts inside the tool content")

    windows: dict[Window, PositionWindow] = {
        "tool_content": PositionWindow(t_lo, t_hi, "tool_content"),
    }

    # --- существенное поле ----------------------------------------------
    if essential_field:
        rel = full.find(essential_field, obs_start, obs_end)
        if rel < 0:
            msg = f"essential field {essential_field!r} not found inside the tool content"
            if require_field:
                raise WindowError(msg)
            notes.append(msg)
        else:
            f_lo, f_hi = _tokens_overlapping(offsets, rel, rel + len(essential_field))
            windows["field_read"] = PositionWindow(f_lo, f_hi, "field_read")

    # --- хвост ответа + закрывающая обёртка -----------------------------
    p_lo = max(t_lo, t_hi - k_post)
    p_hi = max(g_lo, t_hi)
    windows["post_message"] = PositionWindow(p_lo, p_hi, "post_message")

    # --- перед действием ------------------------------------------------
    if g_lo >= len(ids):
        raise WindowError("empty generation prompt and no tokens after the tool content")
    windows["pre_action"] = PositionWindow(g_lo, len(ids), "pre_action")
    windows["action_point"] = PositionWindow(len(ids) - 1, len(ids), "action_point")

    prepared = PreparedPrompt(
        text=full,
        input_ids=ids,
        offsets=offsets,
        windows=windows,
        char_spans={
            "tool_content": (obs_start, obs_end),
            "generation_prompt": (gen_char, len(full)),
        },
        warnings=notes,
    )
    validate_windows(prepared)

    # Сверка с тем, что дал бы apply_chat_template(tokenize=True).
    try:
        direct = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True, **kw
        )
        if hasattr(direct, "keys"):
            direct = direct["input_ids"]
        if direct and isinstance(direct[0], list):
            direct = direct[0]
        if list(direct) != ids:
            prepared.warnings.append(
                "apply_chat_template(tokenize=True) gives different ids; "
                "ALWAYS feed PreparedPrompt.input_ids to the model"
            )
    except Exception as e:  # noqa: BLE001 — сверка необязательна
        prepared.warnings.append(f"could not cross-check tokenization: {e}")

    for n in prepared.warnings:
        warnings.warn(n, stacklevel=2)
    return prepared


def validate_windows(p: PreparedPrompt) -> None:
    """Инварианты, нарушение которых означает, что вмешательство попадёт не туда."""
    w = p.windows
    n = len(p.input_ids)
    for kind, win in w.items():
        if win.end > n:
            raise WindowError(f"{kind} [{win.start},{win.end}) exceeds sequence length {n}")
    tc, pm, pa = w["tool_content"], w["post_message"], w["pre_action"]
    if "field_read" in w and not tc.contains(w["field_read"]):
        raise WindowError("field_read is not inside tool_content")
    if not (pm.start >= tc.start and pm.start < tc.end):
        raise WindowError("post_message must start inside the tool content")
    if pm.end > pa.start:
        raise WindowError("post_message overlaps pre_action")
    if pa.end != n:
        raise WindowError("pre_action must end at the last prompt token")
    ap = w.get("action_point")
    if ap is not None and (len(ap) != 1 or ap.end != n):
        raise WindowError("action_point must be exactly the last prompt token")


# --------------------------------------------------------------------------
# Печать: проверка глазами
# --------------------------------------------------------------------------


def _piece(tokenizer: Any, tid: int) -> str:
    s = tokenizer.decode([tid], skip_special_tokens=False)
    return repr(s)[1:-1] or "∅"


def describe_windows(
    tokenizer: Any,
    p: PreparedPrompt,
    context: int = 3,
    kinds: Sequence[Window] | None = None,
) -> str:
    """Человекочитаемая распечатка окон: индексы, токены, контекст по краям.

        window post_message  [57, 64)  7 tok
          … 'qty' '":' ' ' ⟦ '0' '}' '\\n' '</tool_response>' '<|im_end|>' '\\n' ⟧ '<|im_start|>' …
          text: '0}\\n</tool_response><|im_end|>\\n'

    Обязательно посмотреть на реальной модели до первого прогона.
    """
    lines = [f"sequence: {len(p.input_ids)} tokens"]
    for kind in kinds or WINDOW_ORDER:
        if kind not in p.windows:
            lines.append(f"window {kind:<13} — not built")
            continue
        w = p.windows[kind]
        left = range(max(0, w.start - context), w.start)
        right = range(w.end, min(len(p.input_ids), w.end + context))
        body = " ".join(f"'{_piece(tokenizer, p.input_ids[i])}'" for i in range(w.start, w.end))
        lctx = " ".join(f"'{_piece(tokenizer, p.input_ids[i])}'" for i in left)
        rctx = " ".join(f"'{_piece(tokenizer, p.input_ids[i])}'" for i in right)
        s_char, e_char = p.offsets[w.start][0], p.offsets[w.end - 1][1]
        lines.append(f"window {kind:<13} [{w.start}, {w.end})  {len(w)} tok")
        lines.append(f"  … {lctx} ⟦ {body} ⟧ {rctx} …")
        lines.append(f"  text: {p.text[s_char:e_char]!r}")
    if p.warnings:
        lines.append("warnings:")
        lines += [f"  - {x}" for x in p.warnings]
    return "\n".join(lines)


def print_windows(tokenizer: Any, p: PreparedPrompt, **kw: Any) -> None:
    print(describe_windows(tokenizer, p, **kw))
