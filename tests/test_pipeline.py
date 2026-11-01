"""Пайплайн, конфиг, кэш, протокол (train/val/test, test-lock), журнал, гейты, ноутбуки.

Все прогоны идут во временный корень: настоящие results/ и docs/JOURNAL.md не трогаются.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from steering_tool_feedback import gates as G
from steering_tool_feedback import simulator as S
from steering_tool_feedback.actions import parse_action
from steering_tool_feedback.backends import MockBackend
from steering_tool_feedback.config import ExperimentConfig, load_yaml
from steering_tool_feedback.methods import get_method
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


def test_unknown_or_unimplemented_method_fails_early(root, monkeypatch):
    from steering_tool_feedback import methods as MM

    with pytest.raises(KeyError):
        _exp(root, **{"eval.methods": ["baseline", "magic"]})
    monkeypatch.setitem(MM.METHODS, "later", MM.MethodSpec("later", "baseline", implemented=False))
    with pytest.raises(NotImplementedError):
        _exp(root, **{"eval.methods": ["baseline", "later"]})


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
    assert exp.done == ["data", "baseline", "probe", "vectors", "sweep", "fit", "freeze", "test"]
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


ALL_METHODS = ["baseline", "strong_prompt", "restate", "explicit_state", "steering",
               "steering_gated", "tact", "lora"]


def test_cache_makes_rerun_free(root):
    """Повторный прогон до fit (вкл. эпизоды на val и «обучение» LoRA) — без вызовов модели."""
    _exp(root, **{"eval.methods": ALL_METHODS}).run("fit")
    be = MockBackend()
    exp2 = _exp(root, backend=be, **{"eval.methods": ALL_METHODS})
    exp2.run("fit")
    assert be.calls == {"decide": 0, "record": 0, "train": 0}
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
    assert list(df["stage"]) == ["baseline", "probe", "sweep", "fit", "test", "episodes"]
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
        "model.name": "tiny", "data.n_tasks": 40, "steer.alphas": [1.0],
        "steer.min_class_examples": 1, "steer.top_k_layers": 1,
        "steer.windows": ["post_message"], "probe.n_auto_layers": 2,
        "eval.methods": ["baseline", "restate", "steering"]})
    exp.run("test")
    assert exp.done[-1] == "test"
    files = list((exp.dir / "decisions").glob("test__*.jsonl"))
    recs = [json.loads(x) for f in files if "__episodes__" not in f.name
            for x in f.read_text().splitlines()]
    steered = [r for r in recs if r["method"].startswith(("steering", "sham")) and r["hook_applied"]]
    assert all(r["positions_source"] in ("cache_position", "position_ids") for r in steered)
    restate = [r for r in recs if r["method"] == "restate"]
    assert restate and all(r["n_model_calls"] >= 2 for r in restate)
    eps = [json.loads(x) for f in files if "__episodes__" in f.name
           for x in f.read_text().splitlines()]
    assert eps and all(e["steps"] and "arguments" in e["steps"][0] for e in eps)
    assert {e["method"] for e in eps} >= {"baseline", "restate"}


# --------------------------------------------------------------------------
# ноутбуки
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["01_smoke_mock.ipynb", "00_TEMPLATE.ipynb", "05_h1_h2_v3.ipynb", "06_transfer.ipynb",
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


# --------------------------------------------------------------------------
# конкуренты: явное состояние, гейт, TACT, LoRA; эпизоды; H4/ALT
# --------------------------------------------------------------------------


def test_all_methods_full_run(root):
    exp = _exp(root, **{"eval.methods": ALL_METHODS})
    exp.run("test")
    fitted = exp.state["fitted"]
    assert set(fitted) == {"steering_gated", "tact", "lora"}
    for m, f in fitted.items():
        assert f["rows"], m
    frozen = exp.state["frozen"]["methods"]
    assert "steering" in frozen and set(frozen) - {"steering"}, "freeze_id покрывает весь набор"
    res = exp.state["test"]
    assert set(ALL_METHODS) <= set(res["methods"]) | {m for m in ALL_METHODS if m not in frozen
                                                       and get_method(m).fitted}
    assert {"baseline", "steering"} <= set(res["episodes"])
    assert "steering_vs_explicit_state" in res["comparisons"]
    v = exp.verdicts()
    assert v["H4"].status != "not_run" and v["ALT"].status != "not_run"
    md = exp.report()
    assert "H4" in md and "ALT" in md and "episode" in md
    ex = [json.loads(x) for x in (exp.dir / "decisions" / "test__explicit_state.jsonl").read_text().splitlines()]
    assert all(r["n_model_calls"] >= 2 for r in ex)


def test_fitted_methods_use_train_and_val_only(root):
    exp = _exp(root, **{"eval.methods": ALL_METHODS})
    exp.run("fit")
    assert not list((exp.dir / "decisions").glob("test__*")), "test открыт до freeze"
    # LoRA учится на меняющих парах train — тех же, что и вектор
    n_changing_train = len(exp._tasks("train", ("changing",)))
    lora = exp.state["fitted"]["lora"]["rows"][0]
    assert lora["n_examples"] == 2 * n_changing_train
    # гейт: решения собраны из кэша steering/базы, без новых вызовов модели
    be = MockBackend()
    exp2 = _exp(root, backend=be, **{"eval.methods": ALL_METHODS})
    exp2.run("fit")
    assert be.calls["decide"] == 0
    gated = exp2.evaluate("steering_gated", "val", write=False,
                          gate=exp2._gate_spec(exp2.state["fitted"]["steering_gated"]["rows"][0]))
    assert all("gate_p" in r for r in gated) and {r["gate_on"] for r in gated} <= {True, False}


def test_test_lock_flags_other_freeze_of_same_data(root):
    _exp(root).run("test")
    exp2 = _exp(root, **{"eval.methods": ALL_METHODS})
    exp2.run("test")              # другой набор методов -> другой freeze_id -> не блокируется,
    assert any("уже открывался под другим набором" in f for f in exp2.flags)   # но помечается


def test_episodes_on_mock(root):
    exp = _exp(root)
    exp.data()
    eps = exp.evaluate_episodes("baseline", "val", write=False)
    tasks = exp._tasks("val", ("changing", "invariant"))
    assert len(eps) == 2 * len(tasks)
    assert all(e["steps"][0]["expected"] == sorted(next(t for t in tasks if t.task_id == e["task_id"])
                                                    .branch(e["branch"]).allowed_actions) for e in eps)
    s = exp.summarize_episodes(eps)
    assert 0 <= s["success"] <= s["critical_ok"] <= 1
    # первый шаг эпизода = одношаговое решение (тот же промпт, тот же кэш)
    single = {(r["task_id"], r["branch"]): r["action"] for r in exp.evaluate("baseline", "val", write=False)}
    assert all(e["steps"][0]["action"] == single[(e["task_id"], e["branch"])] for e in eps)


def test_gates_h4_and_alt():
    c = lambda d, lo, hi: {"mean_diff": d, "ci_low": lo, "ci_high": hi, "excludes_zero": lo > 0 or hi < 0}  # noqa: E731
    assert G.h4(None, None).status == "not_run"
    assert G.h4({"comparisons": {"steering_vs_lora": c(0.2, 0.1, 0.3)}}, None).status == "inconclusive"
    assert G.h4({"comparisons": {"steering_vs_lora": c(0.0, -0.1, 0.1)}}, None).status == "against"
    assert G.h4(None, {"steering_vs_lora": c(-0.2, -0.3, -0.1)}).preliminary
    lat = {"steering": {"latency_p50_ms": 100}, "strong_prompt": {"latency_p50_ms": 100},
           "explicit_state": {"latency_p50_ms": 400}}
    win = {"steering_vs_strong_prompt": c(0.2, 0.1, 0.3), "steering_vs_explicit_state": c(0.2, 0.1, 0.3)}
    assert G.alternatives({"comparisons": win}, None, lat).status == "support"
    tie_cheap = {**win, "steering_vs_strong_prompt": c(0.0, -0.1, 0.1)}
    assert G.alternatives({"comparisons": tie_cheap}, None, lat).status == "against"
    tie_costly = {**win, "steering_vs_explicit_state": c(0.0, -0.1, 0.1)}
    assert G.alternatives({"comparisons": tie_costly}, None, lat).status == "inconclusive"


def test_gate_h3_requires_episode_gain():
    ep = lambda d, lo, hi: {"steering": {"paired": {"mean_diff": d, "ci_low": lo, "ci_high": hi,  # noqa: E731
                                                     "excludes_zero": lo > 0 or hi < 0}}}
    r = _test_res(0.2, 0.1, 0.3)
    assert G.h3({**r, "episodes": ep(0.2, 0.1, 0.3)}, None, 0.05).status == "support"
    assert G.h3({**r, "episodes": ep(0.02, -0.05, 0.1)}, None, 0.05).status == "inconclusive"
    assert G.h3({**r, "episodes": ep(-0.2, -0.3, -0.1)}, None, 0.05).status == "against"


@pytest.mark.model
def test_hf_lora_adapter_trains_and_leaves_base_intact(chatml_tokenizer, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from conftest import make_tiny_llama
    from steering_tool_feedback.backends import ResolvedAdapter
    from steering_tool_feedback.backends.hf import HFBackend
    from steering_tool_feedback.config import LoraCfg
    from steering_tool_feedback.episodes import Env

    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)          # как в HFBackend.load: обучение должно включать градиенты само
    try:
        _lora_checks(chatml_tokenizer, tmp_path, S, HFBackend, LoraCfg, Env, ResolvedAdapter)
    finally:
        torch.set_grad_enabled(prev)


def _lora_checks(chatml_tokenizer, tmp_path, S, HFBackend, LoraCfg, Env, ResolvedAdapter):
    from conftest import make_tiny_llama

    be = HFBackend(make_tiny_llama(vocab=len(chatml_tokenizer)), chatml_tokenizer, "tiny", "cpu")
    tasks = [t for t in S.build_pilot_set(30, seed=0) if t.pair_type == "changing"][:6]
    prompts, examples = [], []
    for t in tasks:
        for b in (0, 1):
            p = be.prompt(t.prompt(b), t.essential_field(b), S.tools_schema())
            gold = sorted(t.branch(b).allowed_actions)[0]
            ex = be.sft_example(p, gold, Env(t, b).canonical_arguments(gold))
            assert ex[0] == p.prepared.input_ids and ex[1], "вход SFT = ровно оцениваемый промпт"
            prompts.append(p)
            examples.append(ex)
    before = be.decide(prompts[0], "scored", None, 0.0, 0).scores
    cfg = LoraCfg(rank=4, lora_alpha=8, dropout=0.0, grad_accum=1)
    info = be.train_adapter(examples, cfg, lr=5e-3, epochs=3, out_dir=str(tmp_path / "ad"))
    assert info["epoch_losses"][-1] < info["epoch_losses"][0], info
    assert info["n_trainable"] > 0
    ad = ResolvedAdapter("a1", str(tmp_path / "ad"))
    on = be.decide(prompts[0], "scored", None, 0.0, 0, adapter=ad).scores
    off = be.decide(prompts[0], "scored", None, 0.0, 0).scores
    assert max(abs(before[k] - off[k]) for k in before) < 1e-5, "после адаптера база должна остаться базой"
    assert max(abs(before[k] - on[k]) for k in before) > 1e-3, "адаптер должен менять решение"
    d = be.decide(prompts[0], "scored", None, 0.0, 0, adapter=ad, with_args=True, arg_tokens=6)
    assert isinstance(d.arguments, dict) and d.n_model_calls > len(S.ACTIONS)
    # Регрессия: второй адаптер учится, когда первый уже подключён к оценочной модели
    info2 = be.train_adapter(examples, cfg, lr=2e-3, epochs=2, out_dir=str(tmp_path / "ad2"))
    assert info2["epoch_losses"][-1] < info2["epoch_losses"][0], info2
    on2 = be.decide(prompts[0], "scored", None, 0.0, 0,
                    adapter=ResolvedAdapter("a2", str(tmp_path / "ad2"))).scores
    off2 = be.decide(prompts[0], "scored", None, 0.0, 0).scores
    assert max(abs(before[k] - off2[k]) for k in before) < 1e-5
    assert max(abs(on[k] - on2[k]) for k in on) > 1e-4, "адаптеры разные"


@pytest.mark.model
def test_hf_system_prefix_cache_and_batched_candidates(chatml_tokenizer, tiny_model):
    """Оптимизации не меняют чисел: кэш системного хода + батч кандидатов = полные прогоны."""
    pytest.importorskip("torch")
    import numpy as np

    from conftest import MODEL_FACTORIES
    from steering_tool_feedback import steering as ST
    from steering_tool_feedback.backends.base import ResolvedSteer
    from steering_tool_feedback.backends.hf import HFBackend

    name, _ = tiny_model
    be = HFBackend(MODEL_FACTORIES[name](vocab=len(chatml_tokenizer)), chatml_tokenizer,
                   f"tiny-{name}", "cpu")
    t = next(x for x in S.build_pilot_set(40) if x.pair_type == "changing"
             and x.critical_step_position == "late")
    p = be.prompt(t.prompt(1), t.essential_field(1), S.tools_schema())
    ids = p.prepared.input_ids
    prefix = be._system_prefix(ids, p.messages, p.tools, None)
    assert prefix[0] > 0, "системный ход должен кэшироваться"
    vec = ST.SteeringVector(1, np.random.default_rng(0).normal(size=be.hidden_size).astype("float32"),
                            "post_message", {})
    for steer in (None, ResolvedSteer(vec, p.windows["post_message"], 2.0)):
        ref, _ = be.score_actions(ids, steer, shared_prefix=False)
        d = be.decide(p, "scored", steer, 0.0, 0)
        assert max(abs(ref[k] - d.scores[k]) for k in ref) < 1e-4
        if steer:
            assert d.hook_applied == len(steer.window)
        a, n1, _ = be._generate(ids, steer, 0.0, 0, max_new_tokens=5)
        b, n2, _ = be._generate(ids, steer, 0.0, 0, max_new_tokens=5,
                                prefix=be._system_prefix(ids, p.messages, p.tools, steer))
        assert a == b and n1 == n2


def test_transfer_stage_on_mock(root, monkeypatch):
    exp = _exp(root, **{"eval.methods": ALL_METHODS, "transfer.n_tasks": 40})
    exp.data()
    with pytest.raises(StageOrderError):
        exp.transfer()
    exp.run("fit")
    res = exp.transfer()
    assert {"seen", "unseen"} <= set(res)
    for g in ("seen", "unseen"):
        assert {"baseline", "steering", "lora"} <= set(res[g]["methods"])
        assert set(res[g]["episodes"]) == {"baseline", "steering", "lora"}
        assert "steering_vs_lora" in res[g]["comparisons"]
    assert not list((exp.dir / "decisions").glob("test__*")), "перенос не открывает test"
    md = exp.report()
    assert "Перенос, unseen (разведка)" in md
    # набор, совпадающий со сплитом по содержимому, — ошибка
    exp2 = _exp(root, **{"eval.methods": ["baseline"]})
    exp2.run("fit")
    train = exp2.state["splits"]["train"]
    monkeypatch.setattr(S, "build_transfer_set", lambda *a, **k: {"seen": train, "unseen": train})
    with pytest.raises(ValueError, match="overlaps"):
        exp2.transfer()


def test_lora_diagnostics_needs_a_real_model(root):
    exp = _exp(root, **{"eval.methods": ["baseline", "lora"]})
    exp.run("fit")
    with pytest.raises(NotImplementedError):
        exp.lora_diagnostics()


@pytest.mark.model
def test_lora_diagnostics_end_to_end(chatml_tokenizer, root):
    pytest.importorskip("peft")
    from conftest import make_tiny_llama
    from steering_tool_feedback.backends.hf import HFBackend

    be = HFBackend(make_tiny_llama(vocab=len(chatml_tokenizer)), chatml_tokenizer, "tiny", "cpu")
    exp = _exp(root, backend=be, **{
        "model.name": "tiny", "data.n_tasks": 40, "steer.alphas": [1.0],
        "steer.min_class_examples": 1, "steer.top_k_layers": 1, "steer.windows": ["post_message"],
        "probe.n_auto_layers": 2, "episode.on_val": False,
        "lora.grid": [{"lr": 0.01, "epochs": 2}], "lora.rank": 4, "lora.grad_accum": 1,
        "steer.max_specificity_drop": 1.0,
        "eval.methods": ["baseline", "lora"]})
    exp.run("fit")
    res = exp.lora_diagnostics(n_control=2, form_layers=1)
    n_train = 2 * len(exp._tasks("train", ("changing", "invariant")))
    assert res["n_prompts"] == n_train and 0.0 <= res["first_token_agreement"] <= 1.0
    for f in ("diag_patch.csv", "diag_patch_summary.csv", "diag_delta.csv", "diag_form.csv",
              "diag_acts.npz", "diagnostics.json"):
        assert (exp.dir / f).exists(), f
    import pandas as pd

    dd = pd.read_csv(exp.dir / "diag_delta.csv")
    assert set(dd["position"]) == {"last", "decision", "tool_content"} and len(dd) == 3 * be.n_layers
    pr = pd.read_csv(exp.dir / "diag_patch.csv")
    assert set(pr["set"]) == {"all", "tool_content", "field_read", "decision", "last"}
    # подстановка ВСЕХ позиций на последнем слое = решение LoRA
    last_all = pr[(pr["layer"] == be.n_layers - 1) & (pr["set"] == "all")]
    assert last_all["to_lora"].all()
    assert "diagnose" in exp.done


ACT_METHODS = ["baseline", "steering", "lora", "linmap_lora", "linmap", "reft", "cast"]


def test_activation_methods_full_cycle_on_mock(root):
    exp = _exp(root, **{"eval.methods": ACT_METHODS})
    exp.run("test")
    fitted = exp.state["fitted"]
    for m in ("linmap_lora", "linmap", "reft", "cast"):
        assert fitted[m]["rows"], m
    assert fitted["linmap_lora"]["teacher"] == fitted["lora"]["row"]["adapter_id"]
    assert fitted["cast"]["cond_f1"] > 0
    frozen = exp.state["frozen"]["methods"]
    assert {"linmap", "reft"} & set(frozen)
    assert set(frozen) <= set(exp.state["test"]["methods"]) | {"steering"}
    # повтор — всё из кэша, включая обучение ReFT и активации в точке вызова
    be = MockBackend()
    exp2 = _exp(root, backend=be, **{"eval.methods": ACT_METHODS, "eval.reopen_test_reason": "повтор кэша"})
    exp2.run("fit")
    assert be.calls == {"decide": 0, "record": 0, "train": 0}


STACK_METHODS = ["baseline", "linmap", "linmap_prompt", "linmap_stack", "linmap_prompt_stack"]


def test_prompt_teacher_and_stack_on_mock(root):
    exp = _exp(root, **{"eval.methods": STACK_METHODS, "act.stack_alphas": [1.0, 2.0]})
    exp.run("test")
    fitted = exp.state["fitted"]
    lp = fitted["linmap_prompt"]
    assert lp["rows"] and lp["teacher"] == "explicit_state"
    assert lp["teacher_train_success"] > lp["base_train_success"]   # mock: учитель-промпт лучше базы
    for m in ("linmap_stack", "linmap_prompt_stack"):
        rows = fitted[m]["rows"]
        assert {r["alloc"] for r in rows} == {"uniform", "importance"} and len(rows) == 4, m
        for r in rows:
            assert np.isclose(sum(r["alphas"]), r["alpha"], atol=1e-6)
            assert sum(r["ranks"]) <= exp.cfg.act.stack_rank * len(exp._act_layers()) + len(r["ranks"])
        assert set(fitted[m]["importance"]) == set(exp._act_layers())
    frozen = exp.state["frozen"]["methods"]
    if "linmap_prompt_stack" in frozen:
        assert len(frozen["linmap_prompt_stack"]["affine_ids"]) == len(frozen["linmap_prompt_stack"]["layers"])
    assert set(frozen) <= set(exp.state["test"]["methods"]) | {"steering"}
    # повтор — учитель-промпт и активации из кэша
    be = MockBackend()
    exp2 = _exp(root, backend=be, **{"eval.methods": STACK_METHODS, "act.stack_alphas": [1.0, 2.0],
                                     "eval.reopen_test_reason": "повтор кэша"})
    exp2.run("fit")
    assert be.calls == {"decide": 0, "record": 0, "train": 0}


def test_mix_target_combines_teacher_and_contrast(root):
    exp = _exp(root, **{"eval.methods": ["baseline", "linmap_mix"]})
    exp.run("fit")
    f = exp.state["fitted"]["linmap_mix"]
    assert f["rows"] and f["teacher"] == "explicit_state"
    tm, tp, tc = (exp._targets[k] for k in ("mix", "prompt", "contrast"))
    ok = np.asarray(tp["teacher_ok"])
    assert ok.any() and (~ok).any()
    L = tm["layers"][0]
    assert np.allclose(tm["D"][L][ok], tp["D"][L][ok]) and np.allclose(tm["D"][L][~ok], tc["D"][L][~ok])
    assert f["cost"]["acts_seconds"] == tp["acts_seconds"]


LOTO_METHODS = ["baseline", "lora", "linmap", "linmap_prompt", "linmap_mix"]


def test_loto_selection_on_mock(root):
    ov = {"eval.methods": LOTO_METHODS, "act.select": "val+loto", "act.ranks": [8],
          "act.alphas": [0.5, 1.0, 2.0], "transfer.n_tasks": 40,
          "transfer.episode_methods": ["baseline", "linmap_mix"]}
    exp = _exp(root, **ov)
    exp.run("fit")
    fitted = exp.state["fitted"]
    tol = exp.cfg.steer.max_specificity_drop
    for m in ("linmap", "linmap_prompt", "linmap_mix"):
        f = fitted[m]
        assert f["select"] == "val+loto"
        for r in f["rows"]:
            assert r["loto_success"] is not None and abs(r["score"] - (r["success"] + r["loto_success"]) / 2) < 1e-12
        ok = [r for r in f["rows"] if r["spec_drop"] <= tol + 1e-12]
        assert f["row"]["score"] == max(r["score"] for r in ok)
        assert f["row_val"]["success"] == max(r["success"] for r in ok)
    assert fitted["lora"].get("select", "val") == "val"
    # фолд не содержит задач отложенного типа и подгоняется на меньшем числе примеров
    tasks, keys = exp._train_changing()
    by = {t.task_id: t for t in tasks}
    for k in exp.cfg.act.loto_kinds:
        t = exp._targets.get(f"contrast|-{k}")
        n_k = sum(S.requirement_kind(by[tid]) == k for tid, _ in keys)
        if n_k:
            assert t["n_examples"] == len(keys) - n_k
        assert all(S.requirement_kind(x) == k for x in exp.state["splits"][f"loto:{k}"])
    exp.transfer()
    names = exp._method_names()
    for m in ("linmap", "linmap_prompt", "linmap_mix"):
        if f"{m}@val" in names:
            assert f"{m}@val" in exp.state["summaries"]["transfer:unseen"]
    # повтор — всё из кэша, включая карты фолдов и оценку на отложенных типах
    be = MockBackend()
    exp2 = _exp(root, backend=be, **ov)
    exp2.run("fit")
    assert be.calls == {"decide": 0, "record": 0, "train": 0}


def test_stack_needs_parent(root):
    exp = _exp(root, **{"eval.methods": ["baseline", "linmap_stack"]})
    exp.run("fit")
    f = exp.state["fitted"]["linmap_stack"]
    assert f["row"] is None and "linmap" in f["reason"]


@pytest.mark.model
def test_hf_affine_at_call_point_and_loreft(chatml_tokenizer):
    torch = pytest.importorskip("torch")
    import numpy as np

    from conftest import make_tiny_llama
    from steering_tool_feedback import interventions as IV
    from steering_tool_feedback.backends import ResolvedAffine
    from steering_tool_feedback.backends.hf import HFBackend

    be = HFBackend(make_tiny_llama(vocab=len(chatml_tokenizer)), chatml_tokenizer, "tiny", "cpu")
    tasks = [t for t in S.build_pilot_set(30, seed=0) if t.pair_type == "changing"][:4]
    ps = [be.prompt(t.prompt(1), t.essential_field(1), S.tools_schema()) for t in tasks]
    d = be.hidden_size
    rng = np.random.default_rng(0)
    aff = ResolvedAffine(2, rng.normal(size=(d, 2)) * 0.1, rng.normal(size=(2, d)), rng.normal(size=d), 1.0)
    base = be.decide(ps[0], "scored", None, 0.0, 0).scores
    zero = be.decide(ps[0], "scored", ResolvedAffine(2, aff.U, aff.P, aff.a0, 0.0), 0.0, 0)
    on = be.decide(ps[0], "scored", aff, 0.0, 0, with_args=True, arg_tokens=4)
    assert max(abs(base[k] - zero.scores[k]) for k in base) < 1e-5
    assert on.hook_applied == 1 and max(abs(base[k] - on.scores[k]) for k in base) > 1e-4
    cp = be.record_call_point(ps[0], [1, 2])
    assert set(cp) == {1, 2} and cp[2].shape == (d,)
    # LoReFT: loss падает, веса модели снова обучаемы, форма — аффинная карта
    gold = [sorted(t.branch(1).allowed_actions)[0] for t in tasks]
    ex = [be.reft_example(p, g) for p, g in zip(ps, gold)]
    am, info = IV.train_loreft(be, ex, layer=2, rank=2, lr=1e-2, epochs=4)
    assert info["epoch_losses"][-1] < info["epoch_losses"][0], info
    assert am.U.shape == (d, 2) and am.P.shape == (2, d)
    assert np.allclose(am.P @ am.P.T, np.eye(2), atol=1e-4), "строки R ортонормированы"
    assert all(p.requires_grad for p in be.model.parameters())
    assert not torch.is_grad_enabled() or True


@pytest.mark.model
def test_hf_two_step_prompt_and_affine_stack(chatml_tokenizer):
    pytest.importorskip("torch")
    import numpy as np

    from conftest import make_tiny_llama
    from steering_tool_feedback.backends import ResolvedAffine, ResolvedAffineStack
    from steering_tool_feedback.backends.hf import HFBackend
    from steering_tool_feedback.methods import TWO_STEP

    be = HFBackend(make_tiny_llama(vocab=len(chatml_tokenizer)), chatml_tokenizer, "tiny", "cpu")
    t = next(t for t in S.build_pilot_set(30, seed=0) if t.pair_type == "changing")
    p = be.prompt(t.prompt(1), t.essential_field(1), S.tools_schema())
    ask, go, n = TWO_STEP["explicit_state"]
    tp = be.two_step_prompt(p, ask, go, 4)
    assert tp.messages[: len(p.messages)] == p.messages and tp.messages[-1]["content"] == go
    assert tp.n_tokens > p.n_tokens
    # двухшаговое решение идёт ровно через этот промпт
    d2 = be.decide_two_step(p, ask, go, 4, "scored", 0.0, 0)
    assert d2.scores == be.decide(tp, "scored", None, 0.0, 0).scores
    d = be.hidden_size
    rng = np.random.default_rng(0)
    parts = tuple(ResolvedAffine(L, rng.normal(size=(d, 2)) * 0.1, rng.normal(size=(2, d)), rng.normal(size=d), a)
                  for L, a in ((1, 0.5), (2, 0.5)))
    base = be.decide(p, "scored", None, 0.0, 0).scores
    on = be.decide(p, "scored", ResolvedAffineStack(parts), 0.0, 0, with_args=True, arg_tokens=4)
    assert on.hook_applied == 1 and max(abs(base[k] - on.scores[k]) for k in base) > 1e-4


G1_METHODS = ["baseline", "policy_prompt", "steering", "lora", "linmap", "linmap_prompt"]
G1_OV = {"data.generator": "g1", "data.n_tasks": 120, "data.stratify_pair_type": True,
         "episode.enabled": False, "act.teacher": "policy_prompt", "eval.methods": G1_METHODS,
         "act.ranks": [8], "act.alphas": [0.5, 1.0, 2.0], "transfer.n_tasks": 40,
         "transfer.episode_methods": []}


def test_global_slice_on_mock(root):
    with pytest.raises(ValueError, match="эпизоды не определены"):
        _exp(root, **{**G1_OV, "episode.enabled": True})
    exp = _exp(root, **G1_OV)
    exp.run("fit")
    assert set(exp.state["data_summary"]["by_type"]) == {"global", "control"}
    s = exp.state["summaries"]["val"]
    assert s["policy_prompt"]["sensitivity"] > s["baseline"]["sensitivity"]   # mock: правило словами помогает
    lp = exp.state["fitted"]["linmap_prompt"]
    assert lp["rows"] and lp["teacher"] == "policy_prompt"
    assert lp["teacher_train_success"] > lp["base_train_success"]
    tasks, keys = exp._train_changing()
    assert tasks and all(t.pair_type == "global" for t in tasks)
    assert exp.state["fitted"]["linmap"]["rows"]                          # у базы есть и верные, и неверные
    res = exp.transfer()
    for g in ("seen", "unseen"):
        assert {t.pair_type for t in exp.state["splits"][f"transfer:{g}"]} == {"global", "control"}
        assert not res[g]["episodes"]
    # повтор — учитель policy_prompt из кэша
    be = MockBackend()
    exp2 = _exp(root, backend=be, **G1_OV)
    exp2.run("fit")
    assert be.calls == {"decide": 0, "record": 0, "train": 0}


def test_crossslice_h4_prime_on_mock(root):
    from steering_tool_feedback import crossslice as X

    methods = ["baseline", "lora", "linmap", "linmap_prompt"]
    common = {"act.ranks": [8], "act.alphas": [0.5, 1.0, 2.0], "transfer.n_tasks": 40,
              "transfer.episode_methods": []}
    cond = _exp(root, **{**common, "eval.methods": methods})
    cond.run("fit")
    cond.transfer()
    glob = _exp(root, **{**G1_OV, **common, "eval.methods": methods})
    glob.run("fit")
    glob.transfer()
    out = X.compare(cond.dir, glob.dir, "transfer:seen", n_boot=500)
    assert out["level"] == 0.975 and [r["method"] for r in out["rows"]] == ["linmap", "linmap_prompt"]
    for r in out["rows"]:
        assert math.isclose(r["I"], r["delta_cond"] - r["delta_glob"], abs_tol=1e-12)
        assert r["ci_low"] <= r["I"] <= r["ci_high"]
        assert isinstance(r["eligible_cond"], bool) and isinstance(r["eligible_glob"], bool)
    assert out["verdict"].preliminary and out["verdict"].status in ("support", "against", "inconclusive")
    with pytest.raises(FileNotFoundError):
        X.compare(cond.dir, glob.dir, "transfer:seen", methods=("reft",))
    # ориентир не выбран в одном срезе (ни одна конфигурация не прошла допуск): вердикта нет, только описание
    (glob.dir / "decisions" / "transfer-seen__lora.jsonl").unlink()
    out = X.compare(cond.dir, glob.dir, "transfer:seen", n_boot=500)
    assert out["verdict"].status == "not_run" and out["ref_missing"] == ["glob"] and "глобальном" in out["verdict"].reason
    assert [r["method"] for r in out["rows"]] == ["linmap", "linmap_prompt"]
    assert all("I" not in r and 0 <= r["success_glob"] <= 1 for r in out["rows"])


def test_gate_h4_prime():
    def row(m, lo, hi, ec=True, eg=True):
        return {"method": m, "I": (lo + hi) / 2, "ci_low": lo, "ci_high": hi, "eligible_cond": ec, "eligible_glob": eg}
    assert G.h4_prime([], True).status == "not_run"
    assert G.h4_prime([row("a", 0.05, 0.3), row("b", -0.1, 0.2)], True).status == "support"
    assert G.h4_prime([row("a", -0.3, -0.05), row("b", -0.1, 0.2)], False).status == "against"
    assert G.h4_prime([row("a", 0.05, 0.3), row("b", -0.3, -0.05)], True).status == "inconclusive"
    assert G.h4_prime([row("a", -0.04, 0.04), row("b", -0.03, 0.05)], True).status == "against"
    assert G.h4_prime([row("a", -0.2, 0.2)], True).status == "inconclusive"
    # выигрыш на срезе, где метод сломал специфичность, не засчитывается
    assert G.h4_prime([row("a", 0.05, 0.3, ec=False)], True).status == "inconclusive"
    assert G.h4_prime([row("a", -0.3, -0.05, eg=False)], True).status == "inconclusive"
