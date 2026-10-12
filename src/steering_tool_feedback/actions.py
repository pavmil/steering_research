"""Формат вызова инструмента у конкретной модели и разбор действия из текста.

Разные семейства пишут вызов по-разному:
    Qwen3.5/3.x   <tool_call>\\n<function=NAME>
    Qwen3, Qwen2.5 <tool_call>\\n{"name": "NAME", ...}
    Llama 3.x     {"name": "NAME", "parameters": ...}

Формат не хардкодится, а выводится из самого chat template: рендерим
ассистентский вызов с именем-маркером и смотрим, что стоит между generation
prompt и маркером (prefix) и сразу после него (suffix). Если шаблон устроен
нестандартно — задать model.action_prefix / action_suffix в конфиге.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .simulator import ACTIONS

_MARK = "ZZQXACTION"


@dataclass(frozen=True)
class ActionFormat:
    prefix: str        # текст от конца промпта до имени действия
    suffix: str        # символ(ы) сразу после имени — закрывает имя при скоринге
    source: str        # detected | config

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
    suffix = full[i + len(_MARK): i + len(_MARK) + 1]
    if not prefix.strip() and not suffix:
        raise ActionFormatError("could not isolate the tool-call prefix")
    return ActionFormat(prefix=prefix, suffix=suffix, source="detected")


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
