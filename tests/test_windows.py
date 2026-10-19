"""Окна позиций: построение по offsets, инварианты, печать токенов.

Ключевой регрессионный тест — test_old_arithmetic_missed_tool_content:
первая версия (арифметика длин префиксов) клала post_message в generation
prompt, не задевая ни одного токена ответа инструмента.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from steering_tool_feedback import simulator as S  # noqa: E402
from steering_tool_feedback.windows import (  # noqa: E402
    WindowError,
    describe_windows,
    prepare_prompt,
)


def _text(p, kind):
    w = p.windows[kind]
    return p.text[p.offsets[w.start][0]: p.offsets[w.end - 1][1]]


def _changing_task():
    return next(t for t in S.build_pilot_set(60, seed=0)
                if t.pair_type == "changing" and t.domain == "inventory")


def _check_semantics(tok, p, task, branch):
    obs = task.branch(branch).observation()
    tc = _text(p, "tool_content")
    assert obs in tc or tc in obs, "tool_content must cover the tool response text"

    field = task.essential_field(branch)
    assert field in _text(p, "field_read"), "field_read must cover key+value"

    post = _text(p, "post_message")
    assert "</tool_response>" in post, "post_message must include the closing wrapper"
    assert any(ch in post for ch in obs[-3:]), "post_message must include the end of the response"

    pre = _text(p, "pre_action")
    assert "assistant" in pre, "pre_action must be the generation prompt"
    assert "tool_response" not in pre

    w = p.windows
    assert w["post_message"].end <= w["pre_action"].start
    assert w["pre_action"].end == len(p.input_ids)


# --------------------------------------------------------------------------
# Синтетический ChatML-токенизатор (запускается везде, где есть transformers)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("branch", [0, 1])
def test_windows_on_chatml(chatml_tokenizer, branch):
    t = _changing_task()
    p = prepare_prompt(chatml_tokenizer, t.prompt(branch),
                       essential_field=t.essential_field(branch), tools=None)
    _check_semantics(chatml_tokenizer, p, t, branch)
    assert _text(p, "pre_action") == "<|im_start|>assistant\n"


def test_old_arithmetic_missed_tool_content(chatml_tokenizer):
    """Документирует баг первой версии и проверяет, что новая его не повторяет."""
    tok = chatml_tokenizer
    t = _changing_task()
    msgs = t.prompt(1)
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    n_full, n_pre = len(ids), len(tok(prefix, add_special_tokens=False)["input_ids"])
    old_post = tok.decode(ids[max(n_pre, n_full - 8): n_full])
    old_pre = (max(n_pre, n_full - 4), n_full)
    old_post_span = (max(n_pre, n_full - 8), n_full)

    obs_value = t.branch(1).result.payload["in_stock"]
    assert str(obs_value).lower() not in old_post.lower(), "old window happened to hit the value"
    assert old_post_span[1] > old_pre[0], "old post_message overlapped pre_action"

    p = prepare_prompt(tok, msgs, essential_field=t.essential_field(1))
    tc = p.windows["tool_content"]
    pm = p.windows["post_message"]
    assert pm.start < tc.end, "new post_message reaches into the tool content"
    assert pm.end <= p.windows["pre_action"].start, "new windows do not overlap"


def test_every_pilot_prompt_builds(chatml_tokenizer):
    for t in S.build_pilot_set(120, seed=0):
        for b in (0, 1):
            prepare_prompt(chatml_tokenizer, t.prompt(b), essential_field=t.essential_field(b))


def test_describe_prints_tokens(chatml_tokenizer):
    t = _changing_task()
    p = prepare_prompt(chatml_tokenizer, t.prompt(0), essential_field=t.essential_field(0))
    out = describe_windows(chatml_tokenizer, p)
    for kind in ("tool_content", "field_read", "post_message", "pre_action"):
        assert f"window {kind}" in out
    assert "⟦" in out and "⟧" in out and "text:" in out


def test_errors_are_loud(chatml_tokenizer):
    t = _changing_task()
    with pytest.raises(WindowError):
        prepare_prompt(chatml_tokenizer, t.prompt(0), essential_field="no_such_field")
    with pytest.raises(WindowError):
        prepare_prompt(chatml_tokenizer, t.prompt(0)[:-1])
    p = prepare_prompt(chatml_tokenizer, t.prompt(0), essential_field="no_such_field",
                       require_field=False)
    assert "field_read" not in p.windows and p.warnings


def test_model_inputs_match_windows(chatml_tokenizer):
    pytest.importorskip("torch")
    t = _changing_task()
    p = prepare_prompt(chatml_tokenizer, t.prompt(0), essential_field=t.essential_field(0))
    x = p.model_inputs()
    assert x["input_ids"].shape == (1, len(p.input_ids))
    assert x["input_ids"][0].tolist() == p.input_ids


# --------------------------------------------------------------------------
# Реальный токенизатор (STF_TOKENIZER)
# --------------------------------------------------------------------------


@pytest.mark.real_tokenizer
@pytest.mark.parametrize("branch", [0, 1])
def test_windows_on_real_template(real_tokenizer, branch):
    t = _changing_task()
    p = prepare_prompt(real_tokenizer, t.prompt(branch),
                       essential_field=t.essential_field(branch), tools=S.tools_schema())
    _check_semantics(real_tokenizer, p, t, branch)
    print("\n" + describe_windows(real_tokenizer, p))


@pytest.mark.real_tokenizer
def test_every_pilot_prompt_builds_on_real_template(real_tokenizer):
    tools = S.tools_schema()
    for t in S.build_pilot_set(120, seed=0):
        for b in (0, 1):
            prepare_prompt(real_tokenizer, t.prompt(b), essential_field=t.essential_field(b),
                           tools=tools)


# --------------------------------------------------------------------------
# Скоринг: токены имени действия = каноническая токенизация вызова
# --------------------------------------------------------------------------


def _check_name_tokens(tok, kw):
    from steering_tool_feedback.actions import detect_action_format, name_tokens

    t = _changing_task()
    msgs = t.prompt(1)
    fmt = detect_action_format(tok, msgs, S.tools_schema(), kw)
    pre, cands = name_tokens(tok, fmt, S.ACTIONS)
    gen = tok.apply_chat_template(msgs, tools=S.tools_schema(), tokenize=False,
                                  add_generation_prompt=True, **kw)
    for a, c in cands.items():
        full = tok.apply_chat_template(msgs + [S.tool_call_message(a, {})], tools=S.tools_schema(),
                                       tokenize=False, add_generation_prompt=False, **kw)
        canon = tok(full[len(gen):], add_special_tokens=False)["input_ids"]
        assert canon[:len(pre) + len(c)] == pre + c, f"{a}: scored tokens are not what the model writes"
        assert a in tok.decode(c), a


def test_scored_tokens_are_canonical_chatml(chatml_tokenizer):
    _check_name_tokens(chatml_tokenizer, {})


def test_scored_tokens_are_canonical_real(real_tokenizer):
    """Регрессия (Qwen3-1.7B): отдельный токен `"` после имени стоил ~−37 nats."""
    _check_name_tokens(real_tokenizer, {"enable_thinking": False})
