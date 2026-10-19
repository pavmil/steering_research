"""Многошаговая среда: правила совпадают с A*, verifier ловит ошибки чтения ответа.

Без модели и torch — агент здесь скриптовый.
"""

from __future__ import annotations

import pytest

from steering_tool_feedback import episodes as E
from steering_tool_feedback import simulator as S


@pytest.fixture(scope="module")
def tasks():
    return [t for t in S.build_pilot_set(160, seed=0) if t.pair_type != "stress"]


def _agent(policy):
    """Агент из политики (expected, step) -> action; аргументы — очевидные, из теневой среды."""
    def make(task, branch):
        shadow = E.Env(task, branch)

        def agent(msgs, essential, expected, i):
            a = policy(expected, i, msgs)
            args = shadow.canonical_arguments(a)
            if not shadow.done:
                shadow.step(a, args)
            return a, args, {}
        return agent
    return make


def _oracle(expected, i, msgs):
    return "place_order" if "place_order" in expected else sorted(expected)[0]


def test_oracle_solves_every_episode(tasks):
    for t in tasks:
        for b in (0, 1):
            ep = E.rollout(t, b, t.prompt(b), _agent(_oracle)(t, b), max_steps=4)
            assert ep.success, ep.to_dict()
            assert ep.steps[0]["expected"] == sorted(t.branch(b).allowed_actions)


def test_first_step_equals_paired_decision(tasks):
    """Первое решение эпизода — тот же промпт, что и в одношаговом протоколе."""
    t = tasks[0]
    seen = []

    def agent(msgs, essential, expected, i):
        seen.append((msgs, essential))
        return "stop_no_option", {}, {}

    E.rollout(t, 1, t.prompt(1), agent)
    assert seen[0][0] == t.prompt(1) and seen[0][1] == t.essential_field(1)


def test_ignoring_the_result_fails_with_violation(tasks):
    """Агент, который продолжает план по инерции, проваливает «плохую» ветку."""
    shop = next(t for t in tasks if t.pair_type == "changing" and t.domain == "inventory")
    always_order = _agent(lambda exp, i, m: "place_order")
    good = E.rollout(shop, 0, shop.prompt(0), always_order(shop, 0))
    bad = E.rollout(shop, 1, shop.prompt(1), always_order(shop, 1))
    assert good.success
    assert not bad.success and not bad.critical_ok
    assert any("unavailable" in v for v in bad.violations)

    doc = next(t for t in tasks if t.domain == "documents")
    always_go = _agent(lambda exp, i, m: "proceed")
    assert E.rollout(doc, 0, doc.prompt(0), always_go(doc, 0)).success
    ep = E.rollout(doc, 1, doc.prompt(1), always_go(doc, 1))
    assert not ep.success and any("without access" in v for v in ep.violations)


def test_continuation_must_read_the_second_result(tasks):
    """Верная реакция на критический ответ не спасает, если агент не прочёл список альтернатив."""
    t = next(t for t in tasks if t.pair_type == "changing" and t.domain == "inventory")
    env = E.Env(t, 1)
    item = t.tool_call["arguments"]["item"]
    res = env.step("search_alternative", {"item": item})
    alts = res.result.payload["alternatives"]
    assert alts and all(a["in_stock"] for a in alts)
    assert env.expected() == {"place_order", "ask_user"}
    env.step("place_order", {"item": item})        # снова исходный товар
    assert env.violations and not env.done

    env2 = E.Env(t, 1)
    env2.step("search_alternative", {"item": item})
    env2.step("place_order", {"item": alts[0]["item"].upper()})   # регистр не важен
    assert env2.done and env2.goal and not env2.violations


def test_documents_access_flow(tasks):
    t = next(t for t in tasks if t.domain == "documents")
    env = E.Env(t, 1)
    assert env.expected() == {"request_access"}
    r = env.step("request_access", {"doc_id": t.tool_call["arguments"]["doc_id"]})
    assert r.result.payload["access"] == "granted" and r.result.payload["content"]
    assert env.expected() == {"proceed"}
    assert env.step("proceed", {}) is None and env.goal


def test_wrong_arguments_are_not_fixed(tasks):
    t = next(t for t in tasks if t.pair_type == "changing" and t.domain == "inventory")
    env = E.Env(t, 1)
    r = env.step("search_alternative", {"item": "something else"})
    assert r.result.payload["alternatives"] == [] and not env.searched
    assert env.step("place_order", {}).result.payload["status"] == "error"


def test_budget_and_unparsed(tasks):
    t = next(t for t in tasks if t.pair_type == "changing" and t.domain == "inventory")
    loop = _agent(lambda exp, i, m: "check_stock")
    ep = E.rollout(t, 1, t.prompt(1), loop(t, 1), max_steps=3)
    assert len(ep.steps) == 3 and "budget" in ep.reason and not ep.success

    ep = E.rollout(t, 0, t.prompt(0), lambda m, e, x, i: ("unparsed", {}, {}))
    assert not ep.success and "unparsed" in ep.reason and len(ep.steps) == 1


def test_branches_do_not_share_state(tasks):
    t = next(t for t in tasks if t.pair_type == "changing" and t.domain == "inventory")
    e0 = E.Env(t, 0)
    e0.step("place_order", {"item": t.tool_call["arguments"]["item"]})
    assert t.world_0.state["stock"][t.tool_call["arguments"]["item"]] > 0
    assert E.Env(t, 0).target_available()


def test_stress_is_single_step_only():
    t = next(t for t in S.build_pilot_set(160, seed=0) if t.pair_type == "stress")
    with pytest.raises(ValueError):
        E.Env(t, 0)


def test_late_position_has_real_history(tasks):
    late = [t for t in tasks if t.critical_step_position == "late"]
    early = [t for t in tasks if t.critical_step_position == "early"]
    assert late and early
    for t in late:
        msgs = t.prompt(0)
        n_calls = sum(1 for m in msgs if m.get("tool_calls"))
        assert n_calls >= 2, "late: до критического вызова должен быть хотя бы один полный шаг"
        assert msgs[-1]["content"] == t.world_0.observation()
        assert t.prompt(0)[:-1] == t.prompt(1)[:-1], "история и вызов одинаковы в обеих ветках"
    for t in early:
        assert sum(1 for m in t.prompt(0) if m.get("tool_calls")) == 1
