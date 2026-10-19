"""Формат вызова инструмента у конкретной модели и разбор действия из текста.

Разные семейства пишут вызов по-разному:
    Qwen3.5/3.x   <tool_call>\\n<function=NAME>
    Qwen3, Qwen2.5 <tool_call>\\n{"name": "NAME", ...}
    Llama 3.x     {"name": "NAME", "parameters": ...}

Формат не хардкодится, а выводится из самого chat template: рендерим
ассистентский вызов с именем-маркером и смотрим, что стоит между generation
prompt и маркером (prefix) и после него (rest). Если шаблон устроен
нестандартно — задать model.action_prefix / action_suffix в конфиге.

Скоринг берёт токены имени из КАНОНИЧЕСКОЙ токенизации prefix + имя + rest.
Раньше к имени дописывался отдельный токен закрывающей кавычки: у Qwen3 модель
всегда пишет слитный `",`, поэтому одиночный `"` стоил ~−37 nats и по-разному
для разных действий — штраф артефакта мог перевернуть решение (найдено на
Qwen3-1.7B, 2026-10-19).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .simulator import ACTIONS

_MARK = "ZZQXACTION"


@dataclass(frozen=True)
class ActionFormat:
    prefix: str        # текст от конца промпта до имени действия
    suffix: str        # символ сразу после имени (справочно)
    source: str        # detected | config
    rest: str = ""     # весь текст вызова после имени — для канонической токенизации

    def continuation(self, action: str) -> str:
        return f"{self.prefix}{action}{self.suffix}"


class ActionFormatError(ValueError):
    pass


def detect_action_format(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    template_kwargs: dict[str, Any] | None = None,
) -> ActionFormat:
    kw = dict(template_kwargs or {})
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": _MARK, "arguments": {}}}]}
    gen = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                        add_generation_prompt=True, **kw)
    full = tokenizer.apply_chat_template(messages + [call], tools=tools, tokenize=False,
                                         add_generation_prompt=False, **kw)
    i = full.rfind(_MARK)
    if i < 0:
        raise ActionFormatError("chat template does not render tool_calls")
    if not full.startswith(gen):
        raise ActionFormatError(
            "generation prompt is not a prefix of a rendered tool call; "
            "set model.action_prefix / model.action_suffix in the config"
        )
    prefix = full[len(gen):i]
    rest = full[i + len(_MARK):]
    suffix = rest[:1]
    if not prefix.strip() and not suffix:
        raise ActionFormatError("could not isolate the tool-call prefix")
    return ActionFormat(prefix=prefix, suffix=suffix, source="detected", rest=rest)


def name_tokens(tokenizer: Any, fmt: ActionFormat, actions: tuple[str, ...]
                ) -> tuple[list[int], dict[str, list[int]]]:
    """(общий префикс, токены имени каждого действия) из канонической токенизации вызова.

    Токены имени — от первого токена, задевающего начало имени, до последнего,
    задевающего его конец: ровно то, что модель пишет сама. Префикс обязан
    токенизироваться одинаково для всех действий, и ни одна последовательность
    имени не может быть префиксом другой — иначе сравнение log-prob нечестно.
    """
    pre: list[int] | None = None
    cands: dict[str, list[int]] = {}
    for a in actions:
        text = fmt.prefix + a + fmt.rest
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        s, e = len(fmt.prefix), len(fmt.prefix) + len(a)
        i0 = next(i for i, (x, y) in enumerate(offs) if y > s)
        i1 = max(i for i, (x, y) in enumerate(offs) if x < e and y > x)
        if pre is None:
            pre = ids[:i0]
        elif ids[:i0] != pre:
            raise ActionFormatError("tool-call prefix tokenizes differently for different actions")
        cands[a] = ids[i0:i1 + 1]
    for a, c in cands.items():
        for b, d in cands.items():
            if a != b and d[:len(c)] == c:
                raise ActionFormatError(f"action {a!r} tokens are a prefix of {b!r}")
    return pre or [], cands


_THINK = re.compile(r"<think>.*?</think>", re.S)
_PATTERNS = (
    re.compile(r"<function=([A-Za-z_][\w-]*)"),
    re.compile(r'"name"\s*:\s*"([A-Za-z_][\w-]*)"'),
    re.compile(r"<tool_call>\s*([A-Za-z_][\w-]*)"),
)


def parse_action(text: str) -> str:
    """Действие из сгенерированного текста.

    Сначала выкидывается рассуждение <think>…</think> (и незакрытый <think>
    тоже: если модель не дошла до вызова — это 'unparsed', а не действие,
    упомянутое в рассуждении). Затем ищется структурированный вызов, и только
    потом — упоминание имени словами.
    """
    body = _THINK.sub(" ", text)
    if "<think>" in body:
        body = body.split("<think>", 1)[0]
    for pat in _PATTERNS:
        for m in pat.finditer(body):
            if m.group(1) in ACTIONS:
                return m.group(1)
    low = body.lower()
    for action in sorted(ACTIONS, key=len, reverse=True):
        if action in low or action.replace("_", " ") in low:
            return action
    return "unparsed"


_PARAM_XML = re.compile(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_arguments(text: str) -> dict[str, Any]:
    """Аргументы вызова из текста вызова (после <think>, если он есть).

    JSON-формы: "arguments": {...} (Qwen3/2.5), "parameters": {...} (Llama);
    XML-форма Qwen3.5: <parameter=item>\nblue mug\n</parameter>.
    Не нашлось — {} (среда ответит ошибкой: это ошибка модели, не обвязки).
    """
    body = _THINK.sub(" ", text)
    if "<think>" in body:
        body = body.split("<think>", 1)[0]
    dec = json.JSONDecoder()
    for key in ("arguments", "parameters"):
        for m in re.finditer(r'"%s"\s*:\s*' % key, body):
            try:
                obj, _ = dec.raw_decode(body[m.end():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):          # аргументы строкой с JSON внутри
                try:
                    inner = json.loads(obj)
                    if isinstance(inner, dict):
                        return inner
                except json.JSONDecodeError:
                    pass
    params = _PARAM_XML.findall(body)
    if params:
        return {k: v for k, v in params}
    return {}
