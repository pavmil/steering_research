"""Пайплайн, конфиг, кэш, протокол (train/val/test, test-lock), журнал, гейты, ноутбуки.

Все прогоны идут во временный корень: настоящие results/ и docs/JOURNAL.md не трогаются.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from steering_tool_feedback import gates as G
from steering_tool_feedback import simulator as S
from steering_tool_feedback.actions import parse_action
from steering_tool_feedback.backends import MockBackend
from steering_tool_feedback.config import ExperimentConfig, load_yaml
from steering_tool_feedback.pipeline import Experiment, StageOrderError, TestLockError

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """Временный корень проекта со своими docs/ и results/."""
    (tmp_path / "docs").mkdir()
    for f in ("HYPOTHESES.md", "METRICS.md"):
        shutil.copy(REPO / "docs" / f, tmp_path / "docs" / f)
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    monkeypatch.delenv("STF_RESULTS_DIR", raising=False)
    monkeypatch.delenv("STF_JOURNAL", raising=False)
    return tmp_path


def _exp(root, backend=None, **ov) -> Experiment:
    cfg = ExperimentConfig.make("t", "H3", notebook="test", overrides={"eval.n_boot": 500, **ov})
    return Experiment(cfg, backend=backend or MockBackend(), root=root, verbose=False)


# --------------------------------------------------------------------------
# конфиг
# --------------------------------------------------------------------------


def test_config_overrides_hash_and_diff(tmp_path):
    a = ExperimentConfig.make("a", "H2", overrides={"model.name": "x", "steer.alphas": [1, 2]})
    b = ExperimentConfig.make("b", "H3", notes="другое описание",
                              overrides={"model": {"name": "x"}, "steer.alphas": [1, 2]})
    assert a.model.name == "x" and a.steer.alphas == [1, 2]
    assert a.hash() == b.hash(), "имя/гипотеза/заметки не должны менять хеш вычислений"
    assert a.hash() != ExperimentConfig.make("a", "H2", overrides={"model.name": "y"}).hash()
    assert set(a.diff_from_default()) == {"model.name", "steer.alphas"}
    a.save_yaml(tmp_path / "c.yaml")
    c = load_yaml(tmp_path / "c.yaml")
    assert c.hash() == a.hash() and c.name == "a"
    d = ExperimentConfig.make(base=tmp_path / "c.yaml")
    assert d.name == "a" and d.hypothesis == "H2", "YAML-значения не должны затираться"


@pytest.mark.parametrize("bad", [
    {"nope.x": 1}, {"model.nope": 1}, {"decision.mode": "fast"}, {"steer.windows": ["x"]},
    {"eval.shams": ["x"]}, {"decision.n_trials": 3},
])
def test_config_rejects_bad(bad):
    with pytest.raises((KeyError, ValueError)):
        ExperimentConfig.make("x", overrides=bad)


def test_unknown_or_unimplemented_method_fails_early(root):
    with pytest.raises(KeyError):
        _exp(root, **{"eval.methods": ["baseline", "magic"]})
    with pytest.raises(NotImplementedError):
        _exp(root, **{"eval.methods": ["baseline", "lora"]})


# --------------------------------------------------------------------------
# данные
# --------------------------------------------------------------------------


def test_tasks_unique_and_no_leak_across_splits():
    ts = S.build_pilot_set(200, seed=3)
    assert len({S.signature(t) for t in ts}) == len(ts)
    sp = S.split_tasks(ts, (0.5, 0.2, 0.3), seed=3)
    sig = {k: {S.signature(t) for t in v} for k, v in sp.items()}
    assert not (sig["train"] & sig["test"] or sig["train"] & sig["val"] or sig["val"] & sig["test"])


# --------------------------------------------------------------------------
# полный прогон на mock
# --------------------------------------------------------------------------


def test_full_run_files_index_journal(root):
    exp = _exp(root)
    exp.run("test")
    assert exp.done == ["data", "baseline", "probe", "vectors", "sweep", "freeze", "test"]
    for f in ("config.yaml", "manifest.json", "data.json", "probe.csv", "vectors.json",
              "sweep.csv", "selection.json", "test.json", "log.txt"):
        assert (exp.dir / f).exists(), f
    md = exp.report(conclusion="вывод X", next_step="шаг Y")
    assert (exp.dir / "report.md").exists() and (exp.dir / "verdicts.json").exists()
    j = (root / "docs" / "JOURNAL.md").read_text()
    assert exp.run_id in j and "вывод X" in j and "шаг Y" in j and "mock-бэкенд" in j
    idx = exp.store.read_index()
    assert exp.run_id in set(idx["run_id"]) and "test" in idx.iloc[-1]["stages"]
    assert "| method |" in md
    res = json.loads((exp.dir / "test.json").read_text())
    assert {"baseline", "strong_prompt", "steering", "sham:random_direction",
            "sham:window_shift"} <= set(res["methods"])


def test_records_belong_to_their_own_tasks(root):
    """Регрессия: кэш по одинаковому промпту возвращал чужие task_id."""
    exp = _exp(root)
    exp.data()
    for split in ("train", "val"):
        recs = exp.evaluate("baseline", split, write=False)
        ids = [(r["task_id"], r["branch"]) for r in recs]
        want = [(t.task_id, b) for t in exp.state["splits"][split] for b in (0, 1)]
        assert sorted(ids) == sorted(want)


def test_cache_makes_rerun_free(root):
    _exp(root).run("sweep")
    be = MockBackend()
    exp2 = _exp(root, backend=be)
    exp2.run("sweep")
    assert be.calls == {"decide": 0, "record": 0}
    assert exp2._dcache.hits > 0


def test_stage_order_enforced(root):
    exp = _exp(root)
    with pytest.raises(StageOrderError):
        exp.sweep()
    exp.data()
    with pytest.raises(StageOrderError):
        exp.vectors()


def test_vectors_from_train_and_sweep_on_val_only(root):
    exp = _exp(root)
    exp.run("sweep")
    vinfo = json.loads((exp.dir / "vectors.json").read_text())
    assert vinfo["vectors"], vinfo
    v = next(iter(exp.vectors_by_id.values()))
    assert v.provenance["split"] == "train"
    sweep_recs = (exp.dir / "decisions" / "val__sweep.jsonl").read_text().splitlines()
    assert sweep_recs and all(json.loads(r)["split"] == "val" for r in sweep_recs)
    assert not list((exp.dir / "decisions").glob("test__*")), "test открыт до freeze"


def test_test_lock(root):
    _exp(root).run("test")
    exp2 = _exp(root)
    exp2.run("freeze")
    with pytest.raises(TestLockError):
        exp2.test()
    # параметры отчёта не меняют метод -> не дают обойти lock
    exp3 = _exp(root, **{"eval.n_boot": 777, "eval.shams": ["random_direction"]})
    exp3.run("freeze")
    assert exp3.state["frozen"]["freeze_id"] == exp2.state["frozen"]["freeze_id"]
    with pytest.raises(TestLockError):
        exp3.test()
    # повторно — только с причиной, и это видно в журнале
    exp4 = _exp(root, **{"eval.reopen_test_reason": "добавили сравнение"})
    exp4.run("freeze")
    exp4.test()
    assert any("ОТКРЫТ ПОВТОРНО" in f for f in exp4.flags)
    exp4.report()
    j = (root / "docs" / "JOURNAL.md").read_text()
    assert "ОТКРЫТ ПОВТОРНО" in j and "добавили сравнение" in j


def test_no_material_for_vectors_is_reported_not_crashed(root):
    be = MockBackend(use_rate=0.0)   # база никогда не права на меняющих парах
    exp = _exp(root, backend=be)
    exp.run("test")
    assert exp.state["vectors"]["vectors"] == []
    assert "мало материала" in exp.state["vectors"]["reason"]
    assert exp.state["selection"]["row"] is None
    assert "steering" not in exp.state["test"]["methods"]
    assert exp.verdicts()["H3"].status in ("against", "not_run", "inconclusive")


def test_estimate(root):
    df = _exp(root).estimate()
    assert list(df["stage"]) == ["baseline", "probe", "sweep", "test"]
    assert (df["calls"] > 0).all()


# --------------------------------------------------------------------------
# гейты
# --------------------------------------------------------------------------


def _probe_row(**kw):
    r = {"layer": 3, "window": "post_message", "accuracy": 0.8, "auroc": 0.85,
         "permuted_accuracy": 0.5, "beats_baseline": True, "suspected_shortcut": False}
    r.update(kw)
    return r


def test_gate_h1():
    assert G.h1(None).status == "not_run"
    assert G.h1([_probe_row()]).status == "support"
    assert G.h1([_probe_row(suspected_shortcut=True)]).status == "inconclusive"
    assert G.h1([_probe_row(beats_baseline=False)]).status == "against"


def test_gate_h2():
    ok = G.Verdict("support", "")
    assert G.h2(ok, {"sensitivity": 0.95, "specificity": 1}, 0.9).status == "against"
    assert G.h2(ok, {"sensitivity": 0.4, "specificity": 1}, 0.9).status == "support"
    assert G.h2(G.Verdict("against", ""), {"sensitivity": 0.4, "specificity": 1}, 0.9).status == "inconclusive"


def _test_res(diff, lo, hi, spec_b=1.0, spec_s=1.0, sham=None):
    p = {"mean_diff": diff, "ci_low": lo, "ci_high": hi, "excludes_zero": lo > 0 or hi < 0}
    m = {"baseline": {"specificity": spec_b, "paired": None},
         "steering": {"specificity": spec_s, "paired": p}}
    if sham:
        m["sham:random_direction"] = {"specificity": 1.0, "paired": sham}
    return {"methods": m}


def test_gate_h3():
    assert G.h3(_test_res(0.2, 0.1, 0.3), None, 0.05).status == "support"
    assert G.h3(_test_res(0.2, 0.1, 0.3, spec_s=0.8), None, 0.05).status == "against"
    assert G.h3(_test_res(-0.2, -0.3, -0.1), None, 0.05).status == "against"
    assert G.h3(_test_res(0.05, -0.05, 0.15), None, 0.05).status == "inconclusive"
    sham = {"mean_diff": 0.2, "ci_low": 0.1, "ci_high": 0.3, "excludes_zero": True}
    assert G.h3(_test_res(0.2, 0.1, 0.3, sham=sham), None, 0.05).status == "inconclusive"
    assert G.h3(None, {"row": None, "reason": "x"}, 0.05).preliminary


# --------------------------------------------------------------------------
# действия
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,want", [
    ("<tool_call>\n<function=place_order>\n</function>", "place_order"),
    ('<tool_call>\n{"name": "search_alternative", "arguments": {}}', "search_alternative"),
    ('{"name": "request_access", "parameters": {}}', "request_access"),
    ("<think>maybe place_order?</think>\n<tool_call>\n<function=search_alternative>", "search_alternative"),
    ("<think>I should place_order but I ran out of tokens", "unparsed"),
    ("I will ask user first", "ask_user"),
    ("hmm", "unparsed"),
])
def test_parse_action(text, want):
    assert parse_action(text) == want


# --------------------------------------------------------------------------
# HF-бэкенд на маленькой модели: скоринг и сквозной прогон
# --------------------------------------------------------------------------


@pytest.mark.model
def test_hf_scoring_shared_cache_equals_full_forward(chatml_tokenizer, tiny_model):
    torch = pytest.importorskip("torch")
    import numpy as np

    from conftest import MODEL_FACTORIES
    from steering_tool_feedback import steering as ST
    from steering_tool_feedback.backends.base import ResolvedSteer
    from steering_tool_feedback.backends.hf import HFBackend

    name, _ = tiny_model
    m = MODEL_FACTORIES[name](vocab=len(chatml_tokenizer))
    be = HFBackend(m, chatml_tokenizer, f"tiny-{name}", "cpu")
    t = next(x for x in S.build_pilot_set(20) if x.pair_type == "changing")
    p = be.prompt(t.prompt(1), t.essential_field(1), S.tools_schema())
    vec = ST.SteeringVector(1, np.random.default_rng(0).normal(size=be.hidden_size).astype("float32"),
                            "post_message", {})
    for steer in (None, ResolvedSteer(vec, p.windows["post_message"], 2.0)):
        a, ha = be.score_actions(p.prepared.input_ids, steer, shared_prefix=True)
        b, hb = be.score_actions(p.prepared.input_ids, steer, shared_prefix=False)
        assert max(abs(a[k] - b[k]) for k in a) < 1e-4
        if steer:
            assert ha.n_applied == len(steer.window)
            assert hb.n_applied == len(steer.window) * len(a)
    assert torch.is_grad_enabled() in (True, False)


@pytest.mark.model
def test_hf_pipeline_end_to_end(chatml_tokenizer, root):
    pytest.importorskip("torch")
    from conftest import make_tiny_llama
    from steering_tool_feedback.backends.hf import HFBackend

    be = HFBackend(make_tiny_llama(vocab=len(chatml_tokenizer)), chatml_tokenizer, "tiny", "cpu",
                   max_new_tokens=8)
    exp = _exp(root, backend=be, **{
        "model.name": "tiny", "data.n_tasks": 20, "steer.alphas": [1.0],
        "steer.min_class_examples": 1, "steer.top_k_layers": 1,
        "steer.windows": ["post_message"], "probe.n_auto_layers": 2,
        "eval.methods": ["baseline", "restate", "steering"]})
    exp.run("test")
    assert exp.done[-1] == "test"
    recs = [json.loads(x) for f in (exp.dir / "decisions").glob("test__*.jsonl")
            for x in f.read_text().splitlines()]
    steered = [r for r in recs if r["method"].startswith(("steering", "sham")) and r["hook_applied"]]
    assert all(r["positions_source"] in ("cache_position", "position_ids") for r in steered)
    restate = [r for r in recs if r["method"] == "restate"]
    assert restate and all(r["n_model_calls"] >= 2 for r in restate)


# --------------------------------------------------------------------------
# ноутбуки
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["01_smoke_mock.ipynb", "00_TEMPLATE.ipynb",
                                  "03_h1_h2_baseline.ipynb", "04_h3_steering.ipynb"])
def test_notebooks_execute_on_mock(name, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("STF_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("STF_JOURNAL", str(tmp_path / "JOURNAL.md"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(REPO / "notebooks" / name, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace('"Qwen/Qwen3-1.7B"', '"mock"')
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO / "notebooks")}}).execute()
    if name != "01_smoke_mock.ipynb":
        assert "прогон" in (tmp_path / "JOURNAL.md").read_text()
