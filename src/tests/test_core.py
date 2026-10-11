"""Тесты метрик, симулятора и probe. Запуск: python3 src/tests/test_core.py

Торч не нужен. Тесты хуков — отдельно, в test_hooks.py (нужен torch).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import metrics as M
import probe as P
import simulator as S

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def expect_raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name} (wrong exception: {type(e).__name__}: {e})")
        FAILED.append(name)
    else:
        print(f"  FAIL {name} (no exception)")
        FAILED.append(name)


# --------------------------------------------------------------------------
print("\n[metrics] pass^k")

# Все прогоны успешны -> 1.0 при любом k.
check("all success k=3", math.isclose(M.pass_hat_k([[True] * 4], 3), 1.0))
# Ни одного успеха -> 0.0
check("no success", M.pass_hat_k([[False] * 4], 1) == 0.0)
# c=2, n=4, k=2 -> C(2,2)/C(4,2) = 1/6
check("unbiased estimator", math.isclose(
    M.pass_hat_k([[True, True, False, False]], 2), 1 / 6))
# pass^1 == средняя доля успеха
check("pass^1 == mean", math.isclose(
    M.pass_hat_k([[True, False, True, True]], 1), 0.75))
# pass^k не растёт с k
seq = [[True, True, False, True, True]]
check("monotone in k", M.pass_hat_k(seq, 1) >= M.pass_hat_k(seq, 2) >= M.pass_hat_k(seq, 3))
expect_raises("k too large everywhere", lambda: M.pass_hat_k([[True, False]], 5))
expect_raises("k < 1", lambda: M.pass_hat_k([[True]], 0))

print("\n[metrics] normalized gain")
check("half the headroom", math.isclose(M.normalized_gain(0.75, 0.5), 0.5))
check("no change", math.isclose(M.normalized_gain(0.5, 0.5), 0.0))
check("regression is negative", M.normalized_gain(0.3, 0.6) < 0)
check("ceiling base", math.isclose(M.normalized_gain(1.0, 1.0), 0.0))
expect_raises("out of range", lambda: M.normalized_gain(1.5, 0.5))

print("\n[metrics] target effect requires both pair types")


def _out(pair_type, changed, ok0=True, ok1=True, tid="t1"):
    return M.PairOutcome(tid, pair_type, ok0, ok1, changed, ok0, ok1)


expect_raises(
    "sensitivity alone is refused",
    lambda: M.target_effect([_out("changing", True)]),
)

mixed = [
    _out("changing", True, tid="a"),
    _out("changing", False, tid="b"),
    _out("invariant", False, tid="c"),
    _out("invariant", True, tid="d"),
]
te = M.target_effect(mixed)
check("sensitivity", math.isclose(te.sensitivity, 0.5), f"got {te.sensitivity}")
check("specificity", math.isclose(te.specificity, 0.5), f"got {te.specificity}")

# Главное смысловое правило: смена действия на инвариантной паре — ошибка.
check("invariant switch counts as wrong", not _out("invariant", True).behaved_correctly)
check("changing switch counts as right", _out("changing", True).behaved_correctly)
check("changing hold counts as wrong", not _out("changing", False).behaved_correctly)
# Действие вне A* не спасает даже при правильном переключении.
check("out-of-A* fails", not _out("changing", True, ok0=False).behaved_correctly)

print("\n[metrics] episode delta")
d = M.episode_delta([True, False, True, False], [True, True, False, False])
check("fixed=1", d.fixed == 1, f"got {d.fixed}")
check("broken=1", d.broken == 1, f"got {d.broken}")
check("net=0", d.net == 0)
expect_raises("misaligned", lambda: M.episode_delta([True], [True, False]))

print("\n[metrics] paired bootstrap")
base = {f"t{i}": [False, False] for i in range(40)}
better = {f"t{i}": [True, True] for i in range(40)}
est = M.paired_bootstrap(base, better, n_boot=2000, seed=0)
check("detects large effect", est.excludes_zero and est.mean_diff > 0.9, str(est))

same = {f"t{i}": [True, False] for i in range(40)}
est0 = M.paired_bootstrap(same, dict(same), n_boot=2000, seed=0)
check("no effect -> includes zero", not est0.excludes_zero, str(est0))
check("zero diff", math.isclose(est0.mean_diff, 0.0, abs_tol=1e-9))
expect_raises("no shared tasks",
              lambda: M.paired_bootstrap({"a": [True]}, {"b": [True]}))

print("\n[metrics] entanglement")
check("abs shift", math.isclose(
    M.entanglement({"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.4}), 0.1))
check("zero when unchanged", math.isclose(
    M.entanglement({"x": 0.5}, {"x": 0.5}), 0.0))

print("\n[metrics] cost")
c = M.cost_report([10, 20], [1, 2], [100.0, 200.0, 300.0], base_tool_calls=[1, 1])
check("extra calls", c.extra_tool_calls_vs_base == 1, f"got {c.extra_tool_calls_vs_base}")
check("tokens", c.tokens_total == 30)

# --------------------------------------------------------------------------
print("\n[simulator] pair validation")

tasks = S.build_pilot_set(120, seed=0)
check("built 120", len(tasks) == 120, f"got {len(tasks)}")

rep = S.audit(tasks)
check("audit clean", rep.ok, str(rep.problems[:3]))
check("has changing", rep.by_pair_type.get("changing", 0) > 0)
check("has invariant", rep.by_pair_type.get("invariant", 0) > 0)

# changing: множества допустимых действий не пересекаются
for t in tasks:
    if t.pair_type == "changing":
        assert not (t.world_0.allowed_actions & t.world_1.allowed_actions)
check("changing A* disjoint", True)

# invariant: A* совпадают, но ФОРМА ответа различается
for t in tasks:
    if t.pair_type == "invariant":
        assert t.world_0.allowed_actions == t.world_1.allowed_actions
        assert t.world_0.observation() != t.world_1.observation()
check("invariant same A*, different surface", True)

# Плохие пары должны отвергаться
bad_world = S.World("w", {}, S.ToolResult("x", {"a": 1}), frozenset({"place_order"}))
expect_raises(
    "changing with identical A* refused",
    lambda: S.PairedTask(
        "bad1", "changing", "i", [], {"tool": "x"},
        bad_world,
        S.World("w2", {}, S.ToolResult("x", {"a": 2}), frozenset({"place_order"})),
    ),
)
expect_raises(
    "invariant with identical surface refused",
    lambda: S.PairedTask(
        "bad2", "invariant", "i", [], {"tool": "x"},
        S.World("w1", {}, S.ToolResult("x", {"a": 1}), frozenset({"place_order"})),
        S.World("w2", {}, S.ToolResult("x", {"a": 1}), frozenset({"place_order"})),
    ),
)
expect_raises(
    "unknown action refused",
    lambda: S.World("w", {}, S.ToolResult("x", {}), frozenset({"fly_to_mars"})),
)
expect_raises(
    "empty A* refused",
    lambda: S.World("w", {}, S.ToolResult("x", {}), frozenset()),
)
expect_raises(
    "non-stress pair with untruthful result refused",
    lambda: S.PairedTask(
        "bad3", "changing", "i", [], {"tool": "x"},
        S.World("w1", {}, S.ToolResult("x", {"a": 1}, truthful=False),
                frozenset({"place_order"})),
        S.World("w2", {}, S.ToolResult("x", {"a": 2}), frozenset({"search_alternative"})),
    ),
)

print("\n[simulator] splits do not leak")
splits = S.split_tasks(tasks, seed=0)
ids = {k: {t.task_id for t in v} for k, v in splits.items()}
check("train/test disjoint", not (ids["train"] & ids["test"]))
check("train/val disjoint", not (ids["train"] & ids["val"]))
check("val/test disjoint", not (ids["val"] & ids["test"]))
check("all tasks assigned",
      sum(len(v) for v in splits.values()) == len(tasks),
      f"{sum(len(v) for v in splits.values())} vs {len(tasks)}")

print("\n[simulator] prompt hides the paired branch")
t = next(x for x in tasks if x.pair_type == "changing")
p0 = t.prompt(0)
joined = " ".join(m["content"] for m in p0)
check("branch 1 observation absent from branch 0 prompt",
      t.world_1.observation() not in joined)
check("branch 0 observation present", t.world_0.observation() in joined)
# Условная инструкция в system-промпте — ЧАСТЬ ЗАДАЧИ (condition_explicit),
# поэтому она законно описывает политику. Утечкой было бы сериализованное
# множество допустимых действий, gold-метка или verifier.
check("A* set not serialised into prompt",
      "allowed_actions" not in joined and "world_id" not in joined)
check("no verifier or gold field in prompt",
      not any(k in joined for k in ("gold", "verifier", "pair_type")))

print("\n[simulator] action scoring")
check("in A* passes", S.score_action("place_order",
      S.World("w", {}, S.ToolResult("x", {}), frozenset({"place_order", "ask_user"}))))
check("multiple A* both pass", S.score_action("ask_user",
      S.World("w", {}, S.ToolResult("x", {}), frozenset({"place_order", "ask_user"}))))
check("outside A* fails", not S.score_action("place_order",
      S.World("w", {}, S.ToolResult("x", {}), frozenset({"search_alternative"}))))
check("parse finds action", S.parse_action("I will search_alternative now")
      == "search_alternative")
check("parse handles spaces", S.parse_action("Let me search alternative")
      == "search_alternative")
check("parse returns unparsed", S.parse_action("hmm, not sure") == "unparsed")

print("\n[simulator] render styles differ, payload does not")
tr = S.ToolResult("check_stock", {"item": "mug", "in_stock": False})
check("json != verbose", tr.render("json") != tr.render("verbose_json"))
check("json != prose", tr.render("json") != tr.render("prose"))
check("all mention the essential field",
      all("in_stock" in tr.render(s) or "in_stock" in tr.render(s).lower()
          for s in ("json", "verbose_json", "prose")))

# --------------------------------------------------------------------------
print("\n[probe] signal, no-signal, shortcut")

rng = np.random.default_rng(0)
H = 32
direction = rng.normal(size=H)
direction /= np.linalg.norm(direction)

# Сигнал есть: классы разнесены.
acts, labels, ids = [], [], []
for i in range(60):
    for branch in (0, 1):
        acts.append(rng.normal(size=H) + 3.0 * branch * direction)
        labels.append(branch)
        ids.append(f"t{i}")
r = P.run_probe(acts, labels, ids, layer=5, window_kind="post_message", seed=0)
check("detects real signal", r.beats_baseline, str(r))
check("auroc high", r.auroc > 0.8, f"auroc={r.auroc:.3f}")
check("permutation control drops", r.permuted_accuracy < 0.7,
      f"perm={r.permuted_accuracy:.3f}")
check("signal clears empirical null",
      r.accuracy > r.permuted_accuracy + P.ProbeResult.MARGIN)

# Сигнала нет: метки случайны относительно активаций.
acts2, labels2, ids2 = [], [], []
for i in range(60):
    for branch in (0, 1):
        acts2.append(rng.normal(size=H))
        labels2.append(branch)
        ids2.append(f"t{i}")
r2 = P.run_probe(acts2, labels2, ids2, layer=5, window_kind="post_message", seed=0)
check("no false signal", not r2.beats_baseline, str(r2))

# Shortcut: probe различает инвариантные ветки -> читает форму.
inv_acts, inv_labels, inv_ids = [], [], []
for i in range(60):
    for branch in (0, 1):
        inv_acts.append(rng.normal(size=H) + 3.0 * branch * direction)
        inv_labels.append(branch)
        inv_ids.append(f"i{i}")
r3 = P.run_probe(acts, labels, ids, layer=5, window_kind="post_message", seed=0,
                 invariant_activations=inv_acts, invariant_labels=inv_labels,
                 invariant_task_ids=inv_ids)
check("flags shortcut", r3.suspected_shortcut, f"shortcut={r3.shortcut_auroc}")

expect_raises("single class refused",
              lambda: P.run_probe(acts[:10], [0] * 10, ids[:10], 1, "w"))
expect_raises("misaligned inputs refused",
              lambda: P.run_probe(acts[:10], labels[:5], ids[:10], 1, "w"))

print("\n[probe] gap interpretation")
check("target gap", "TARGET GAP" in P.gap_report(True, True, False))
check("no gap", "NO GAP" in P.gap_report(True, True, True))
check("no signal", "NO SIGNAL" in P.gap_report(False, False, False))
check("no-signal advice is not fine-tuning-only",
      "явное состояние" in P.gap_report(False, False, False))

# --------------------------------------------------------------------------
print("\n[steering] numpy-only parts")
try:
    import steering as ST

    v = ST.SteeringVector(5, np.array([1.0, 0.0, 0.0]), "post_message", {})
    check("norm", math.isclose(v.norm, 1.0))
    check("unit", np.allclose(v.unit(), [1.0, 0.0, 0.0]))

    cv = ST.contrastive_vector(
        [np.array([2.0, 0.0]), np.array([2.0, 0.0])],
        [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
        layer=3, window_kind="post_message",
    )
    check("contrastive = mean diff", np.allclose(cv.vector, [2.0, 0.0]))
    check("provenance recorded", cv.provenance["n_correct"] == 2)

    sham = ST.random_direction_like(cv, seed=0)
    check("sham preserves norm", math.isclose(sham.norm, cv.norm, rel_tol=1e-9))
    check("sham differs in direction", not np.allclose(sham.unit(), cv.unit()))

    perm = ST.permuted_direction(cv, seed=1)
    check("permuted preserves norm", math.isclose(perm.norm, cv.norm, rel_tol=1e-9))

    sl = ST.shifted_layer(cv, +2, max_layer=12)
    check("layer shifted", sl.layer == 5)

    w = ST.PositionWindow(10, 18, "post_message")
    check("window len", len(w) == 8)
    sw = ST.shifted_window(cv, w, -6)
    check("shifted window keeps length", len(sw) == 8 and sw.start == 4)
    expect_raises("bad window refused", lambda: ST.PositionWindow(5, 5))
    expect_raises("degenerate vector refused",
                  lambda: ST.SteeringVector(0, np.zeros(4), "post_message", {}).unit())
    expect_raises("2-D vector refused",
                  lambda: ST.SteeringVector(0, np.zeros((2, 2)), "post_message", {}))

    sub = ST.lowrank_subspace([np.array([2.0, 0.0]), np.array([2.2, 0.1])],
                              [np.array([0.0, 0.0]), np.array([0.1, 0.0])], rank=2)
    check("subspace shape", sub.shape == (2, 2), f"got {sub.shape}")

    expect_raises("empty contrastive refused",
                  lambda: ST.contrastive_vector([], [np.array([1.0])], 0, "post_message"))

    dr = ST.dose_response([0.0, 0.5, 1.0], lambda a: a * 2)
    check("dose response", dr == [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0)])
except ImportError as e:
    print(f"  skip steering tests: {e}")

# --------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILED:
    print(f"FAILED {len(FAILED)}: {FAILED}")
    raise SystemExit(1)
print("ALL PASSED")
