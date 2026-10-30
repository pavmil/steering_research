"""Метрики, симулятор, probe, numpy-часть steering. torch не нужен."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steering_tool_feedback import metrics as M
from steering_tool_feedback import probe as P
from steering_tool_feedback import simulator as S

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_pass_hat_k():
    assert math.isclose(M.pass_hat_k([[True] * 4], 3), 1.0)
    assert M.pass_hat_k([[False] * 4], 1) == 0.0
    assert math.isclose(M.pass_hat_k([[True, True, False, False]], 2), 1 / 6)  # C(2,2)/C(4,2)
    assert math.isclose(M.pass_hat_k([[True, False, True, True]], 1), 0.75)
    seq = [[True, True, False, True, True]]
    assert M.pass_hat_k(seq, 1) >= M.pass_hat_k(seq, 2) >= M.pass_hat_k(seq, 3)
    with pytest.raises(ValueError):
        M.pass_hat_k([[True, False]], 5)
    with pytest.raises(ValueError):
        M.pass_hat_k([[True]], 0)


def test_normalized_gain():
    assert math.isclose(M.normalized_gain(0.75, 0.5), 0.5)
    assert math.isclose(M.normalized_gain(0.5, 0.5), 0.0)
    assert M.normalized_gain(0.3, 0.6) < 0
    assert math.isclose(M.normalized_gain(1.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        M.normalized_gain(1.5, 0.5)


def _out(pair_type, changed, ok0=True, ok1=True, tid="t1"):
    return M.PairOutcome(tid, pair_type, ok0, ok1, changed, ok0, ok1)


def test_target_effect_requires_both_types():
    with pytest.raises(ValueError):
        M.target_effect([_out("changing", True)])
    te = M.target_effect([
        _out("changing", True, tid="a"), _out("changing", False, tid="b"),
        _out("invariant", False, tid="c"), _out("invariant", True, tid="d"),
    ])
    assert math.isclose(te.sensitivity, 0.5) and math.isclose(te.specificity, 0.5)


def test_behaved_correctly_semantics():
    assert not _out("invariant", True).behaved_correctly      # смена на инвариантной — ошибка
    assert _out("changing", True).behaved_correctly
    assert not _out("changing", False).behaved_correctly
    assert not _out("changing", True, ok0=False).behaved_correctly  # вне A* не спасает


def test_episode_delta():
    d = M.episode_delta([True, False, True, False], [True, True, False, False])
    assert (d.fixed, d.broken, d.net) == (1, 1, 0)
    with pytest.raises(ValueError):
        M.episode_delta([True], [True, False])


def test_paired_bootstrap():
    base = {f"t{i}": [False, False] for i in range(40)}
    better = {f"t{i}": [True, True] for i in range(40)}
    est = M.paired_bootstrap(base, better, n_boot=2000, seed=0)
    assert est.excludes_zero and est.mean_diff > 0.9
    same = {f"t{i}": [True, False] for i in range(40)}
    est0 = M.paired_bootstrap(same, dict(same), n_boot=2000, seed=0)
    assert not est0.excludes_zero and math.isclose(est0.mean_diff, 0.0, abs_tol=1e-9)
    with pytest.raises(ValueError):
        M.paired_bootstrap({"a": [True]}, {"b": [True]})


def test_entanglement_and_cost():
    assert math.isclose(M.entanglement({"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.4}), 0.1)
    assert math.isclose(M.entanglement({"x": 0.5}, {"x": 0.5}), 0.0)
    c = M.cost_report([10, 20], [1, 2], [100.0, 200.0, 300.0], base_tool_calls=[1, 1])
    assert c.extra_tool_calls_vs_base == 1 and c.tokens_total == 30


# --------------------------------------------------------------------------
# simulator
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tasks():
    return S.build_pilot_set(120, seed=0)


def test_pilot_set_audit(tasks):
    assert len(tasks) == 120
    rep = S.audit(tasks)
    assert rep.ok, rep.problems[:3]
    assert rep.by_pair_type.get("changing") and rep.by_pair_type.get("invariant")


def test_pair_invariants(tasks):
    for t in tasks:
        if t.pair_type == "changing":
            assert not (t.world_0.allowed_actions & t.world_1.allowed_actions)
        if t.pair_type == "invariant":
            assert t.world_0.allowed_actions == t.world_1.allowed_actions
            assert t.world_0.observation() != t.world_1.observation()


def test_bad_pairs_rejected():
    w = lambda a, acts, **kw: S.World("w", {}, S.ToolResult("x", {"a": a}, **kw), frozenset(acts))  # noqa: E731
    with pytest.raises(ValueError):
        S.PairedTask("b1", "changing", "i", [], {"tool": "x"}, w(1, {"place_order"}), w(2, {"place_order"}))
    with pytest.raises(ValueError):
        S.PairedTask("b2", "invariant", "i", [], {"tool": "x"}, w(1, {"place_order"}), w(1, {"place_order"}))
    with pytest.raises(ValueError):
        w(1, {"fly_to_mars"})
    with pytest.raises(ValueError):
        w(1, set())
    with pytest.raises(ValueError):
        S.PairedTask("b3", "changing", "i", [], {"tool": "x"},
                     w(1, {"place_order"}, truthful=False), w(2, {"search_alternative"}))
    with pytest.raises(ValueError):
        S.World("w", {}, S.ToolResult("x", {"a": 1}), frozenset({"proceed"}), essential_key="zzz")


def test_splits_do_not_leak(tasks):
    sp = S.split_tasks(tasks, seed=0)
    ids = {k: {t.task_id for t in v} for k, v in sp.items()}
    assert not (ids["train"] & ids["test"] or ids["train"] & ids["val"] or ids["val"] & ids["test"])
    assert sum(len(v) for v in sp.values()) == len(tasks)


def test_prompt_structure_and_no_leaks(tasks):
    t = next(x for x in tasks if x.pair_type == "changing")
    msgs = t.prompt(0)
    assert msgs[-1]["role"] == "tool"
    call = msgs[-2]
    assert call["role"] == "assistant" and call["tool_calls"][0]["function"]["name"] == t.tool_call["tool"]
    assert isinstance(call["tool_calls"][0]["function"]["arguments"], dict)
    joined = " ".join(str(m.get("content", "")) for m in msgs)
    assert t.world_1.observation() not in joined
    assert t.world_0.observation() in joined
    assert not any(k in joined for k in ("allowed_actions", "world_id", "gold", "verifier", "pair_type"))


def test_essential_field_in_every_style(tasks):
    for t in tasks:
        for b in (0, 1):
            f = t.essential_field(b)
            assert f and f in t.branch(b).observation()
            assert t.branch(b).essential_key in f


def test_tools_schema():
    names = {s["function"]["name"] for s in S.tools_schema()}
    assert set(S.ACTIONS) <= names
    assert "check_stock" in names and "fetch_doc" in names


def test_action_scoring_and_parsing():
    world = S.World("w", {}, S.ToolResult("x", {}), frozenset({"place_order", "ask_user"}))
    assert S.score_action("place_order", world) and S.score_action("ask_user", world)
    assert not S.score_action("search_alternative", world)
    assert S.parse_action("I will search_alternative now") == "search_alternative"
    assert S.parse_action("Let me search alternative") == "search_alternative"
    assert S.parse_action("<tool_call>\n<function=place_order>") == "place_order"
    assert S.parse_action("hmm, not sure") == "unparsed"


def test_render_styles():
    tr = S.ToolResult("check_stock", {"item": "mug", "in_stock": False})
    styles = ("json", "verbose_json", "prose")
    assert len({tr.render(s) for s in styles}) == 3
    assert all("in_stock" in tr.render(s) for s in styles)


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


def _blobs(sep, n=60, h=32, seed=0, prefix="t"):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=h)
    d /= np.linalg.norm(d)
    acts, labels, ids = [], [], []
    for i in range(n):
        for b in (0, 1):
            acts.append(rng.normal(size=h) + sep * b * d)
            labels.append(b)
            ids.append(f"{prefix}{i}")
    return acts, labels, ids


def test_probe_detects_signal():
    acts, labels, ids = _blobs(3.0)
    r = P.run_probe(acts, labels, ids, layer=5, window_kind="post_message", seed=0)
    assert r.beats_baseline, str(r)
    assert r.auroc > 0.8 and r.permuted_accuracy < 0.7
    assert r.accuracy > r.permuted_accuracy + P.ProbeResult.MARGIN


def test_probe_no_false_signal():
    acts, labels, ids = _blobs(0.0, seed=1)
    r = P.run_probe(acts, labels, ids, layer=5, window_kind="post_message", seed=0)
    assert not r.beats_baseline, str(r)


def _form_blobs(form_dir, sep_state=3.0, sep_form=3.0, n=40, h=32, seed=5):
    """Меняющие пары: состояние по оси d. Инвариантные: состояние 1, форма по оси form_dir."""
    rng = np.random.default_rng(seed)
    d = np.zeros(h)
    d[0] = 1.0
    acts, labels, ids = [], [], []
    for i in range(n):
        for b in (0, 1):
            acts.append(rng.normal(size=h) + sep_state * b * d)
            labels.append(b)
            ids.append(f"t{i}")
    ia, il, ii = [], [], []
    for i in range(n):
        for f in (0, 1):
            ia.append(rng.normal(size=h) + sep_state * d + sep_form * (f - 0.5) * form_dir)
            il.append(f)
            ii.append(f"i{i}")
    return acts, labels, ids, ia, il, ii


def test_probe_flags_shortcut():
    """Форма сдвигает оценку probe состояния -> shortcut."""
    h = 32
    along = np.zeros(h)
    along[0] = 1.0
    acts, labels, ids, ia, il, ii = _form_blobs(along, sep_form=4.0)
    r = P.run_probe(acts, labels, ids, 5, "post_message", seed=0, invariant_activations=ia,
                    invariant_labels=il, invariant_task_ids=ii, invariant_states=[1] * len(il))
    assert r.suspected_shortcut


def test_probe_form_on_orthogonal_axis_is_not_a_shortcut():
    """Регрессия: форма различима, но probe состояния её не читает -> не shortcut.
    Старый тест обучал отдельный классификатор формы и срабатывал бы всегда."""
    h = 32
    ortho = np.zeros(h)
    ortho[1] = 1.0
    acts, labels, ids, ia, il, ii = _form_blobs(ortho, sep_form=4.0)
    r = P.run_probe(acts, labels, ids, 5, "post_message", seed=0, invariant_activations=ia,
                    invariant_labels=il, invariant_task_ids=ii, invariant_states=[1] * len(il))
    assert not r.suspected_shortcut and r.beats_baseline, str(r)
    assert r.invariant_transfer > 0.7


def test_probe_input_validation():
    acts, labels, ids = _blobs(3.0)
    with pytest.raises(ValueError):
        P.run_probe(acts[:10], [0] * 10, ids[:10], 1, "w")
    with pytest.raises(ValueError):
        P.run_probe(acts[:10], labels[:5], ids[:10], 1, "w")


def test_gap_report():
    assert "TARGET GAP" in P.gap_report(True, True, False)
    assert "NO GAP" in P.gap_report(True, True, True)
    no_sig = P.gap_report(False, False, False)
    assert "NO SIGNAL" in no_sig and "явное состояние" in no_sig


# --------------------------------------------------------------------------
# steering: numpy-часть (без torch)
# --------------------------------------------------------------------------


def test_steering_vectors_numpy():
    from steering_tool_feedback import steering as ST
    from steering_tool_feedback.windows import PositionWindow

    v = ST.SteeringVector(5, np.array([1.0, 0.0, 0.0]), "post_message", {})
    assert math.isclose(v.norm, 1.0) and np.allclose(v.unit(), [1, 0, 0])

    cv = ST.contrastive_vector([np.array([2.0, 0.0])] * 2, [np.array([0.0, 0.0])] * 2,
                               layer=3, window_kind="post_message")
    assert np.allclose(cv.vector, [2.0, 0.0]) and cv.provenance["n_correct"] == 2

    sham = ST.random_direction_like(cv, seed=0)
    assert math.isclose(sham.norm, cv.norm, rel_tol=1e-9) and not np.allclose(sham.unit(), cv.unit())
    assert math.isclose(ST.permuted_direction(cv, seed=1).norm, cv.norm, rel_tol=1e-9)
    assert ST.shifted_layer(cv, +2, max_layer=12).layer == 5

    w = PositionWindow(10, 18, "post_message")
    sw = ST.shifted_window(w, -6)
    assert len(sw) == 8 and sw.start == 4

    with pytest.raises(ValueError):
        PositionWindow(5, 5)
    with pytest.raises(ValueError):
        ST.SteeringVector(0, np.zeros(4), "post_message", {}).unit()
    with pytest.raises(ValueError):
        ST.SteeringVector(0, np.zeros((2, 2)), "post_message", {})
    with pytest.raises(ValueError):
        ST.contrastive_vector([], [np.array([1.0])], 0, "post_message")

    sub = ST.lowrank_subspace([np.array([2.0, 0.0]), np.array([2.2, 0.1])],
                              [np.array([0.0, 0.0]), np.array([0.1, 0.0])], rank=2)
    assert sub.shape == (2, 2)
    assert ST.dose_response([0.0, 0.5, 1.0], lambda a: a * 2) == [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0)]


def test_cli_mock_run(tmp_path, monkeypatch):
    from steering_tool_feedback import run_experiment as R

    monkeypatch.setenv("STF_JOURNAL", str(tmp_path / "J.md"))
    rc = R.main(["--mock", "--name", "cli", "--until", "sweep", "--results-dir",
                 str(tmp_path / "res"), "--quiet", "--set", "eval.n_boot=200"])
    assert rc == 0
    assert list((tmp_path / "res" / "runs").glob("*/manifest.json"))
    assert (tmp_path / "J.md").exists()


def test_module_entrypoint(tmp_path):
    """`python -m steering_tool_feedback ...` работает без консольных команд в PATH."""
    import subprocess
    import sys

    run = lambda *a: subprocess.run([sys.executable, "-m", "steering_tool_feedback", *a],  # noqa: E731
                                    capture_output=True, text=True, timeout=120)
    r = run("run", "--mock", "--until", "baseline", "--results-dir", str(tmp_path), "--no-journal",
            "--quiet")
    assert r.returncode == 0, r.stderr[-500:]
    assert list(tmp_path.glob("runs/*/manifest.json"))
    r = run("doctor")
    assert r.returncode == 0 and "core deps OK" in r.stdout, r.stdout[-500:]
    assert run("nope").returncode == 2


# --------------------------------------------------------------------------
# диагностика LoRA: форма Δh (numpy-часть)
# --------------------------------------------------------------------------


def _dd_data(kind, n=120, d=40, seed=0):
    rng = np.random.default_rng(seed)
    states = np.array([i % 2 for i in range(n)])
    groups = np.array([f"t{i // 2}" for i in range(n)])
    u = np.zeros(d)
    u[0] = 1.0
    base = rng.normal(size=(n, d)) + 6.0 * states[:, None] * u      # база «знает» состояние
    v = rng.normal(size=d)
    if kind == "const":
        delta = np.tile(v, (n, 1))
    elif kind == "state":
        delta = np.where(states[:, None] == 1, v, -v)
    else:  # linear
        W = np.outer(rng.normal(size=d), rng.normal(size=d)) / d
        delta = base @ W.T
    return delta + 0.05 * rng.normal(size=(n, d)), base, states, groups


def test_delta_decomposition_recognises_each_form():
    from steering_tool_feedback.diagnose import delta_decomposition as dd

    r = dd(*_dd_data("const"))
    assert r["r2_const"] > 0.95
    r = dd(*_dd_data("state"))
    assert r["r2_const"] < 0.2 and r["r2_state"] > 0.95 and r["r2_probe"] > 0.8
    assert r["rank90_centered"] == 1
    r = dd(*_dd_data("linear"), n_pcs=40)
    assert r["r2_linear"] > 0.8 and r["r2_linear"] > r["r2_const"] + 0.3


def test_patch_effect():
    from steering_tool_feedback.diagnose import patch_effect

    base, lora = np.array([0.0, -5.0]), np.array([-5.0, 0.0])
    assert patch_effect(base, lora, base, 0, 1) == 0.0
    assert patch_effect(base, lora, lora, 0, 1) == 1.0
    assert abs(patch_effect(base, lora, np.array([-2.5, -2.5]), 0, 1) - 0.5) < 1e-9
    assert np.isnan(patch_effect(base, base, base, 0, 0))


# --------------------------------------------------------------------------
# вмешательства в точке вызова: подгонка в закрытой форме, CAST
# --------------------------------------------------------------------------


def test_fit_affine_recovers_low_rank_map():
    from steering_tool_feedback import interventions as IV

    rng = np.random.default_rng(0)
    n, d, r = 200, 24, 3
    H = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d) + rng.normal(size=d)
    A = rng.normal(size=(d, r)) @ rng.normal(size=(r, d)) / d
    c = rng.normal(size=d)
    D = H @ A + c
    am = IV.fit_affine(H, D, rank=r, n_pcs=d, ridge=1e-3)
    assert am.rank == r and am.U.shape == (d, r) and am.P.shape == (r, d)
    assert am.info["train_r2"] > 0.99
    Hn = rng.normal(size=(50, d)) * 2
    assert np.allclose(am(Hn), Hn @ A + c, atol=0.05 * np.abs(Hn @ A + c).max())


def test_contrast_targets():
    from steering_tool_feedback import interventions as IV

    H = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    correct = [True, True, False, False]
    gold = ["a", "b", "a", "b"]
    D = IV.contrast_targets(H, correct, gold)
    assert np.allclose(D[:2], 0)
    assert np.allclose(D[2], [0.0, -2.0]) and np.allclose(D[3], [-2.0, -4.0])
    with pytest.raises(ValueError):
        IV.contrast_targets(H, [False] * 4, gold)


def test_cast_pieces():
    from steering_tool_feedback import interventions as IV

    rng = np.random.default_rng(0)
    d = 16
    c_true = np.zeros(d)
    c_true[0] = 1.0
    # как у настоящих скрытых состояний: классы различаются величиной проекции, а не знаком
    # (sim из CAST знаковонезависима — меряет выравнивание с c)
    pos = rng.normal(size=(40, d)) + 8 * c_true
    neg = rng.normal(size=(40, d)) + 1 * c_true
    c = IV.pc1_direction(pos, neg)
    assert c @ c_true > 0.9 and abs(np.linalg.norm(c) - 1) < 1e-9
    X = np.concatenate([pos, neg])
    y = np.array([1] * 40 + [0] * 40)
    sim = IV.cast_similarity(X, c)
    cond = IV.cast_condition(sim, y)
    assert cond["f1"] > 0.95
    fired = np.array([IV.cast_fires(s, cond) for s in sim])
    assert (fired == (y == 1)).mean() > 0.95
