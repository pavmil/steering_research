"""Пайплайн проверки гипотез: один класс Experiment, стадии строго по протоколу.

    exp = Experiment(cfg)
    exp.data()        # задачи, аудит, разбиение по задачам
    exp.baseline()    # база на train+val — материал для H2 и для вектора
    exp.probe()       # H1: различает ли модель ответы (train, CV по задачам)
    exp.vectors()     # CAA: верные vs неверные решения базы (train, меняющие пары)
    exp.sweep()       # слой × окно × alpha на VALIDATION, выбор с ограничением
    exp.freeze()      # заморозка выбранного метода
    exp.test()        # test ОДИН раз: база, сравнения, steering, sham-контроли
    exp.report(conclusion="...", next_step="...")   # отчёт + запись в журнал

    exp.run(until="sweep")   # то же, по порядку, с пропуском сделанного

Что гарантирует пайплайн, чтобы в ноутбуке нельзя было случайно нарушить протокол:
  * вектор строится только на train, подбор — только на val;
  * test открывается один раз на замороженный метод (results/test_lock.jsonl);
    повторно — только с eval.reopen_test_reason, и это попадает в журнал;
  * каждое вмешательство проверяет, что хук изменил ровно len(окно) позиций;
  * чувствительность и специфичность считаются и показываются только вместе;
  * в отчёт попадают флаги: грязный git, изменённые HYPOTHESES/METRICS, mock.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import gates as G
from . import journal as J
from . import metrics as M
from . import probe as P
from . import simulator as S
from . import steering as ST
from .backends import Prompt, ResolvedSteer, auto_layers, load_backend
from .config import ExperimentConfig
from .methods import MethodRun, SteerSpec, build_messages, get_method
from .store import ActivationCache, DecisionCache, Store, env_info, git_info, sha, write_jsonl
from .windows import PositionWindow

STAGES = ("data", "baseline", "probe", "vectors", "sweep", "freeze", "test")


class TestLockError(RuntimeError):
    """Test для этого замороженного метода уже открывался."""


class StageOrderError(RuntimeError):
    pass


def _progress(it, desc: str, total: int | None = None, disable: bool = False):
    try:
        from tqdm.auto import tqdm

        return tqdm(it, desc=desc, total=total, leave=False, disable=disable)
    except Exception:  # noqa: BLE001
        return it


class Experiment:
    def __init__(self, cfg: ExperimentConfig, backend: Any = None, root: Path | None = None,
                 results_dir: Path | None = None, verbose: bool = True):
        cfg.validate()
        for m in cfg.eval.methods:
            get_method(m)  # неизвестный или нереализованный метод — ошибка сразу
        self.cfg = cfg
        self.store = Store(root, results_dir)
        self.verbose = verbose
        self._backend = backend
        self.run_id, self.dir = self.store.new_run_dir(cfg.name, cfg.hash())
        cfg.save_yaml(self.dir / "config.yaml")
        self.state: dict[str, Any] = {"summaries": defaultdict(dict)}
        self.done: list[str] = []
        self.flags: list[str] = []
        self.vectors_by_id: dict[str, ST.SteeringVector] = {}
        self._prompts: dict[tuple, Prompt] = {}
        self._dcache: DecisionCache | None = None
        self._acache: ActivationCache | None = None
        self.git = git_info(self.store.root)
        self.protocol = self.store.protocol_hashes()
        if self.git["dirty"]:
            self.flags.append("незакоммиченные изменения в src/docs/notebooks/configs — "
                              "прогон не воспроизводится по коммиту")
        if self.protocol["changed_since_first_run"]:
            self.flags.append("изменены зафиксированные документы с первого прогона: "
                              + ", ".join(self.protocol["changed_since_first_run"]))
        self._write_manifest()
        try:
            shown = self.dir.relative_to(self.store.root)
        except ValueError:
            shown = self.dir
        self.log(f"run {self.run_id}  ->  {shown}")

    # ------------------------------------------------------------------
    # служебное
    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.verbose:
            print(line)
        with (self.dir / "log.txt").open("a") as f:
            f.write(line + "\n")

    @property
    def backend(self):
        if self._backend is None:
            self.log(f"loading model {self.cfg.model.name} …")
            self._backend = load_backend(self.cfg.model)
        return self._backend

    def _caches(self) -> tuple[DecisionCache, ActivationCache]:
        if self._dcache is None:
            d = self.store.cache_dir(self.backend.model_key)
            self._dcache = DecisionCache(d / "decisions.jsonl")
            self._acache = ActivationCache(d / "acts")
            info = self.backend.info()
            if info["backend"] == "mock":
                self.flags.append("mock-бэкенд: числа синтетические, НЕ результат исследования")
            self._write_manifest(backend=info)
        return self._dcache, self._acache  # type: ignore[return-value]

    def _write_manifest(self, backend: dict | None = None) -> None:
        path = self.dir / "manifest.json"
        cur = json.loads(path.read_text()) if path.exists() else {}
        cur.update({"run_id": self.run_id, "name": self.cfg.name, "hypothesis": self.cfg.hypothesis,
                    "notebook": self.cfg.notebook, "config_hash": self.cfg.hash(),
                    "git": self.git, "env": env_info(), "protocol": self.protocol,
                    "stages": self.done, "flags": self.flags})
        if backend is not None:
            cur["backend"] = backend
        path.write_text(json.dumps(cur, indent=2, ensure_ascii=False, default=str))

    def _mark(self, stage: str) -> None:
        if stage not in self.done:
            self.done.append(stage)
        self._write_manifest()
        self._update_index()

    def _need(self, *stages: str) -> None:
        missing = [s for s in stages if s not in self.done]
        if missing:
            raise StageOrderError(f"сначала выполните: {', '.join(missing)}")

    def _save_json(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

    # ------------------------------------------------------------------
    # промпты и решения
    # ------------------------------------------------------------------
    def _prompt(self, task: S.PairedTask, branch: int, variant: str = "baseline") -> Prompt:
        k = (task.task_id, branch, variant)
        if k not in self._prompts:
            other = task.branch(1 - branch).allowed_actions if task.pair_type == "changing" else []
            meta = {"task_id": task.task_id, "branch": branch, "pair_type": task.pair_type,
                    "allowed": sorted(task.branch(branch).allowed_actions),
                    "other_allowed": sorted(other), "state": branch, "variant": variant}
            msgs = build_messages(task, branch, variant, self.cfg.eval.strong_prompt)
            tools = S.tools_schema() if self.cfg.data.use_tools_schema else None
            self._prompts[k] = self.backend.prompt(msgs, task.essential_field(branch), tools,
                                                   meta=meta, k_post=self.cfg.data.k_post)
        return self._prompts[k]

    def _resolve(self, spec: SteerSpec, p: Prompt) -> ResolvedSteer:
        vec = self.vectors_by_id[spec.vector_id]
        w = p.windows[spec.window]
        if spec.window_mode == "before_tool":
            tc = p.windows["tool_content"]
            end = max(1, tc.start - 1)
            w = PositionWindow(max(0, end - len(w)), end, w.kind)
        return ResolvedSteer(ST.SteeringVector(spec.layer, vec.vector, vec.window_kind,
                                               vec.provenance), w, spec.alpha, spec.normalize)

    def _decide(self, mr: MethodRun, task: S.PairedTask, branch: int, split: str,
                trial: int) -> dict[str, Any]:
        dcache, _ = self._caches()
        p = self._prompt(task, branch, mr.spec.variant)
        fmt = self.backend.info().get("action_format")
        dc = self.cfg.decision
        key = sha([self.backend.model_key, mr.fingerprint(), p.key, dc.mode,
                   dc.temperature, trial, fmt])
        # В кэше — только ответ модели. Всё, что относится к задаче (id, ветка,
        # оценка), пересчитывается: одинаковый промпт может быть у разных задач.
        out = dcache.get(key)
        cached = out is not None
        if out is None:
            if mr.spec.variant == "restate":
                d = self.backend.decide_restate(p, dc.mode, dc.temperature, trial)
            else:
                steer = self._resolve(mr.steer, p) if mr.steer else None
                d = self.backend.decide(p, dc.mode, steer, dc.temperature, trial)
            out = {"key": key, "action": d.action, "scores": d.scores,
                   "text": (d.text or "")[:2000], "n_prompt_tokens": d.n_prompt_tokens,
                   "n_new_tokens": d.n_new_tokens, "n_model_calls": d.n_model_calls,
                   "latency_ms": round(d.latency_ms, 2), "hook_applied": d.hook_applied,
                   "positions_source": d.positions_source}
            dcache.put(out)
        if out.get("positions_source") == "counter":
            self.flags.append("позиции хука взяты из счётчика, а не из модели — проверить окна")
        w = task.branch(branch)
        return {**out, "task_id": task.task_id, "branch": branch, "trial": trial,
                "pair_type": task.pair_type, "domain": task.domain,
                "explicit": task.condition_explicit, "position": task.critical_step_position,
                "ok": S.score_action(out["action"], w), "allowed": sorted(w.allowed_actions),
                "split": split, "method": mr.spec.name, "cached": cached}

    def _tasks(self, split: str, kinds=("changing", "invariant", "stress")) -> list[S.PairedTask]:
        self._need("data")
        return [t for t in self.state["splits"][split] if t.pair_type in kinds]

    def evaluate(self, method: str, split: str, steer: SteerSpec | None = None,
                 tag: str | None = None, write: bool = True) -> list[dict[str, Any]]:
        """Все пары сплита (обе ветки × n_trials) одним методом. Кэшируется."""
        mr = MethodRun(get_method(method), strong_prompt=self.cfg.eval.strong_prompt, steer=steer)
        if mr.spec.steer and steer is None:
            raise ValueError(f"method {method!r} needs a SteerSpec")
        tasks = self._tasks(split)
        jobs = [(t, b, k) for t in tasks for b in (0, 1) for k in range(self.cfg.decision.n_trials)]
        recs = [self._decide(mr, t, b, split, k)
                for t, b, k in _progress(jobs, f"{tag or method}/{split}", len(jobs),
                                         disable=not self.verbose)]
        if write:
            write_jsonl(self.dir / "decisions" / f"{split}__{(tag or method).replace(':', '-')}.jsonl",
                        recs)
        return recs

    # ------------------------------------------------------------------
    # сводка по методу
    # ------------------------------------------------------------------
    def summarize(self, recs: list[dict], base: list[dict] | None = None) -> dict[str, Any]:
        main = [r for r in recs if r["pair_type"] in ("changing", "invariant")]
        stress = [r for r in recs if r["pair_type"] == "stress"]
        by_pair: dict[tuple, dict[int, dict]] = defaultdict(dict)
        for r in main:
            by_pair[(r["task_id"], r["trial"])][r["branch"]] = r
        outcomes = []
        for (tid, _trial), br in sorted(by_pair.items()):
            if 0 in br and 1 in br:
                a, b = br[0], br[1]
                outcomes.append(M.PairOutcome(tid, a["pair_type"], a["ok"], b["ok"],
                                              a["action"] != b["action"], a["ok"], b["ok"]))
        eff = M.target_effect(outcomes)
        by_ep: dict[tuple, list[bool]] = defaultdict(list)
        by_task: dict[str, list[bool]] = defaultdict(list)
        for r in main:
            by_ep[(r["task_id"], r["branch"])].append(r["ok"])
            by_task[r["task_id"]].append(r["ok"])
        k_max = self.cfg.decision.n_trials
        lat = [r["latency_ms"] for r in recs]
        out: dict[str, Any] = {
            "n_pairs": len(outcomes),
            "success": M.episode_success([r["ok"] for r in main]),
            "pass_k": {k: M.pass_hat_k(list(by_ep.values()), k) for k in sorted({1, k_max})},
            "sensitivity": eff.sensitivity, "specificity": eff.specificity,
            "n_changing": eff.n_changing, "n_invariant": eff.n_invariant,
            "switch_rate": M.switch_rate(outcomes),
            "stress_success": M.episode_success([r["ok"] for r in stress]) if stress else None,
            "unparsed": sum(r["action"] == "unparsed" for r in recs),
            "tokens_mean": float(np.mean([r["n_prompt_tokens"] + r["n_new_tokens"] for r in recs])),
            "model_calls_mean": float(np.mean([r["n_model_calls"] for r in recs])),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "paired": None, "fixed": None, "broken": None,
        }
        if base is not None:
            b_main = [r for r in base if r["pair_type"] in ("changing", "invariant")]
            b_task: dict[str, list[bool]] = defaultdict(list)
            for r in b_main:
                b_task[r["task_id"]].append(r["ok"])
            pe = M.paired_bootstrap(b_task, by_task, n_boot=self.cfg.eval.n_boot,
                                    seed=self.cfg.data.seed)
            out["paired"] = {"mean_diff": pe.mean_diff, "ci_low": pe.ci_low,
                             "ci_high": pe.ci_high, "excludes_zero": pe.excludes_zero,
                             "n_tasks": pe.n_tasks}
            ka = {(r["task_id"], r["branch"], r["trial"]): r["ok"] for r in b_main}
            kb = {(r["task_id"], r["branch"], r["trial"]): r["ok"] for r in main}
            keys = sorted(set(ka) & set(kb))
            dlt = M.episode_delta([ka[k] for k in keys], [kb[k] for k in keys])
            out["fixed"], out["broken"] = dlt.fixed, dlt.broken
        return out

    # ------------------------------------------------------------------
    # стадии
    # ------------------------------------------------------------------
    def data(self) -> dict[str, Any]:
        c = self.cfg.data
        tasks = S.build_pilot_set(c.n_tasks, seed=c.seed)
        rep = S.audit(tasks)
        if not rep.ok:
            raise ValueError(f"аудит данных не пройден:\n{rep}")
        splits = S.split_tasks(tasks, tuple(c.split), seed=c.seed)
        self.state.update(tasks=tasks, splits=splits)
        summary = {"n_tasks": len(tasks), "by_type": rep.by_pair_type, "by_domain": rep.by_domain,
                   "splits": {k: len(v) for k, v in splits.items()},
                   "data_hash": sha([[t.task_id, t.world_0.observation(), t.world_1.observation()]
                                     for t in tasks])}
        self.state["data_summary"] = summary
        self._save_json("data.json", summary)
        self.log(f"data: {summary['splits']}  types={summary['by_type']}")
        self._mark("data")
        return summary

    def baseline(self) -> dict[str, Any]:
        self._need("data")
        for split in ("train", "val"):
            recs = self.evaluate("baseline", split)
            self.state.setdefault("records", {}).setdefault(split, {})["baseline"] = recs
            s = self.summarize(recs)
            self.state["summaries"][split]["baseline"] = s
            self.log(f"baseline/{split}: success={s['success']:.3f}  "
                     f"sens={s['sensitivity']:.3f}  spec={s['specificity']:.3f}  "
                     f"unparsed={s['unparsed']}")
        self._mark("baseline")
        return dict(self.state["summaries"]["val"])

    def _layers_for_probe(self) -> list[int]:
        c = self.cfg.probe
        layers = auto_layers(self.backend.n_layers, c.n_auto_layers) if c.layers == "auto" else list(c.layers)
        if self.cfg.steer.layers != "auto":
            layers = sorted(set(layers) | set(self.cfg.steer.layers))
        return layers

    def _acts(self, prompts: list[Prompt], layers: list[int], windows: list[str]) -> list[dict]:
        _, acache = self._caches()
        need = [ActivationCache.key(L, w) for L in layers for w in windows]
        out = []
        for p in _progress(prompts, "activations", len(prompts), disable=not self.verbose):
            got = acache.load(p.key)
            if not all(k in got for k in need):
                rec = self.backend.record(p, layers, windows)
                acache.save(p.key, {ActivationCache.key(L, w): v for (L, w), v in rec.items()})
                got = acache.load(p.key)
            out.append(got)
        return out

    def probe(self) -> list[dict[str, Any]]:
        self._need("data")
        layers = self._layers_for_probe()
        windows = sorted(set(self.cfg.probe.windows) | set(self.cfg.steer.windows))
        ch = self._tasks("train", ("changing",))
        inv = self._tasks("train", ("invariant",))
        ch_p = [(t, b, self._prompt(t, b)) for t in ch for b in (0, 1)]
        inv_p = [(t, b, self._prompt(t, b)) for t in inv for b in (0, 1)]
        ch_a = self._acts([p for *_, p in ch_p], layers, windows)
        inv_a = self._acts([p for *_, p in inv_p], layers, windows)
        rows = []
        for L in layers:
            for w in self.cfg.probe.windows:
                k = ActivationCache.key(L, w)
                try:
                    r = P.run_probe(
                        [a[k] for a in ch_a], [b for _, b, _ in ch_p], [t.task_id for t, _, _ in ch_p],
                        L, w, seed=self.cfg.data.seed,
                        invariant_activations=[a[k] for a in inv_a] or None,
                        invariant_labels=[b for _, b, _ in inv_p] or None,
                        invariant_task_ids=[t.task_id for t, _, _ in inv_p] or None)
                except ValueError as e:
                    self.log(f"probe L{L}/{w}: skipped ({e})")
                    continue
                rows.append({"layer": L, "window": w, "accuracy": r.accuracy, "auroc": r.auroc,
                             "majority": r.majority_baseline, "permuted_accuracy": r.permuted_accuracy,
                             "shortcut_auroc": r.shortcut_auroc, "beats_baseline": r.beats_baseline,
                             "suspected_shortcut": r.suspected_shortcut, "n": r.n})
        self.state["probe_rows"] = rows
        self.state["probe_layers"], self.state["probe_windows"] = layers, windows
        import pandas as pd

        pd.DataFrame(rows).to_csv(self.dir / "probe.csv", index=False)
        v = G.h1(rows)
        self.log(f"probe: {len(rows)} (layer, window); H1 {v.label}: {v.reason}")
        self._mark("probe")
        return rows

    def _steer_layers(self) -> list[int]:
        c = self.cfg.steer
        if c.layers != "auto":
            return list(c.layers)
        rows = self.state.get("probe_rows") or []
        if not rows:
            return auto_layers(self.backend.n_layers, c.top_k_layers)
        best: dict[int, float] = {}
        for r in rows:
            best[r["layer"]] = max(best.get(r["layer"], 0.0), r["auroc"])
        return sorted(sorted(best, key=best.get, reverse=True)[: c.top_k_layers])

    def vectors(self) -> list[dict[str, Any]]:
        """CAA только на train и только на меняющих парах: решения базы, где агент
        опирался на ответ (верно) vs где нет (неверно). Ветка как метка НЕ используется —
        иначе вмешательство знало бы правильный ответ."""
        self._need("data", "baseline")
        recs = self.state["records"]["train"]["baseline"]
        ok = {(r["task_id"], r["branch"]): r["ok"] for r in recs
              if r["pair_type"] == "changing" and r["trial"] == 0}
        tasks = {t.task_id: t for t in self._tasks("train", ("changing",))}
        prompts = [(k, self._prompt(tasks[k[0]], k[1])) for k in sorted(ok)]
        layers, windows = self._steer_layers(), list(self.cfg.steer.windows)
        acts = self._acts([p for _, p in prompts], layers, windows)
        n_ok = sum(ok[k] for k, _ in prompts)
        n_bad = len(prompts) - n_ok
        info = {"n_correct": n_ok, "n_incorrect": n_bad, "layers": layers, "windows": windows,
                "vectors": []}
        mn = self.cfg.steer.min_class_examples
        if n_ok < mn or n_bad < mn:
            info["reason"] = (f"мало материала для контраста: верных {n_ok}, неверных {n_bad} "
                              f"(нужно ≥ {mn} каждого) на train")
            self.log(f"vectors: {info['reason']}")
        else:
            vdir = self.dir / "vectors"
            vdir.mkdir(exist_ok=True)
            for L in layers:
                for w in windows:
                    k = ActivationCache.key(L, w)
                    good = [a[k] for (key, _), a in zip(prompts, acts) if ok[key]]
                    bad = [a[k] for (key, _), a in zip(prompts, acts) if not ok[key]]
                    v = ST.contrastive_vector(good, bad, L, w, {
                        "run_id": self.run_id, "split": "train", "contrast": "baseline correct vs incorrect",
                        "model": self.backend.model_key})
                    vid = sha(v.vector.tobytes(), 12)
                    self.vectors_by_id[vid] = v
                    v.save(str(vdir / f"L{L}_{w}_{vid}.npz"))
                    info["vectors"].append({"layer": L, "window": w, "vector_id": vid,
                                            "norm": v.norm})
            self.log(f"vectors: {len(info['vectors'])} (CAA на {n_ok} верных / {n_bad} неверных)")
        self.state["vectors"] = info
        self._save_json("vectors.json", info)
        self._mark("vectors")
        return info["vectors"]

    def sweep(self) -> list[dict[str, Any]]:
        """Подбор (слой, окно, alpha) — ТОЛЬКО на validation."""
        self._need("data", "baseline", "vectors")
        base = self.state["records"]["val"]["baseline"]
        bs = self.state["summaries"]["val"]["baseline"]
        rows, all_recs = [], []
        vecs = self.state["vectors"]["vectors"]
        for v in vecs:
            for a in self.cfg.steer.alphas:
                spec = SteerSpec(v["layer"], v["window"], float(a), v["vector_id"],
                                 self.cfg.steer.normalize)
                recs = self.evaluate("steering", "val", steer=spec, write=False,
                                     tag=f"L{v['layer']}/{v['window']}/a{a}")
                all_recs += [{**r, "layer": v["layer"], "window": v["window"], "alpha": a}
                             for r in recs]
                s = self.summarize(recs, base)
                rows.append({"layer": v["layer"], "window": v["window"], "alpha": float(a),
                             "vector_id": v["vector_id"], "success": s["success"],
                             "base_success": bs["success"], "sensitivity": s["sensitivity"],
                             "specificity": s["specificity"],
                             "spec_drop": bs["specificity"] - s["specificity"],
                             "diff": s["paired"]["mean_diff"], "ci_low": s["paired"]["ci_low"],
                             "ci_high": s["paired"]["ci_high"], "fixed": s["fixed"],
                             "broken": s["broken"]})
        write_jsonl(self.dir / "decisions" / "val__sweep.jsonl", all_recs)
        import pandas as pd

        pd.DataFrame(rows).to_csv(self.dir / "sweep.csv", index=False)
        tol = self.cfg.steer.max_specificity_drop
        ok = [r for r in rows if r["spec_drop"] <= tol + 1e-12]
        sel: dict[str, Any] = {"row": None, "tolerance": tol}
        if not rows:
            sel["reason"] = self.state["vectors"].get("reason", "нет векторов для подбора")
        elif not ok:
            sel["reason"] = f"все конфигурации роняют специфичность больше чем на {tol}"
        else:
            sel["row"] = max(ok, key=lambda r: (r["success"], -r["alpha"]))
        self.state["sweep_rows"], self.state["selection"] = rows, sel
        self._save_json("selection.json", sel)
        if sel["row"]:
            r = sel["row"]
            self.log(f"sweep: {len(rows)} конфигураций; выбрано L{r['layer']}/{r['window']}/α={r['alpha']}: "
                     f"успех {r['base_success']:.3f} → {r['success']:.3f}, Δспециф. {-r['spec_drop']:+.3f}")
        else:
            self.log(f"sweep: ничего не выбрано — {sel.get('reason')}")
        self._mark("sweep")
        return rows

    def freeze(self) -> str | None:
        self._need("sweep")
        r = self.state["selection"]["row"]
        if r is None:
            self.log("freeze: нечего замораживать — " + self.state["selection"].get("reason", ""))
            self._mark("freeze")
            return None
        # freeze_id идентифицирует МЕТОД (модель, данные, конфиг, выбранный вектор),
        # а не прогон: иначе повторный прогон того же метода обошёл бы test-lock.
        # Сюда НЕ входит хеш всего конфига: параметры отчёта (n_boot, список
        # сравнений, причина повторного открытия) не меняют метод и не должны
        # давать «новый» freeze_id в обход test-lock.
        payload = {"selection": {k: r[k] for k in ("layer", "window", "alpha", "vector_id")},
                   "normalize": self.cfg.steer.normalize,
                   "model_key": self.backend.model_key,
                   "data_hash": self.state["data_summary"]["data_hash"],
                   "decision": {"mode": self.cfg.decision.mode,
                                "temperature": self.cfg.decision.temperature,
                                "n_trials": self.cfg.decision.n_trials}}
        fid = self.store.save_frozen(payload, meta={"run_id": self.run_id, "git": self.git["commit"],
                                                    "val_row": r})
        v = self.vectors_by_id[r["vector_id"]]
        v.save(str(self.store.results / "frozen" / f"{fid}.npz"))
        self.state["frozen"] = {"freeze_id": fid, **payload}
        self.log(f"freeze: {fid}")
        self._mark("freeze")
        return fid

    def _sham_specs(self, spec: SteerSpec) -> dict[str, SteerSpec]:
        v = self.vectors_by_id[spec.vector_id]
        out: dict[str, SteerSpec] = {}
        seed = self.cfg.data.seed
        for s in self.cfg.eval.shams:
            if s == "random_direction":
                sv = ST.random_direction_like(v, seed)
            elif s == "permuted_coords":
                sv = ST.permuted_direction(v, seed)
            elif s == "layer_shift":
                n = self.backend.n_layers
                delta = 2 if spec.layer + 2 < n else -2
                sv = ST.shifted_layer(v, delta, n)
            else:  # window_shift
                out[f"sham:{s}"] = SteerSpec(spec.layer, spec.window, spec.alpha, spec.vector_id,
                                             spec.normalize, window_mode="before_tool")
                continue
            vid = sha(sv.vector.tobytes() + str(sv.layer).encode(), 12)
            self.vectors_by_id[vid] = sv
            out[f"sham:{s}"] = SteerSpec(sv.layer, spec.window, spec.alpha, vid, spec.normalize)
        return out

    def test(self) -> dict[str, Any]:
        """Test — один раз на замороженный метод."""
        self._need("freeze")
        fr = self.state.get("frozen")
        fid = fr["freeze_id"] if fr else ("nofreeze-" + sha([self.backend.model_key,
                                                              self.state["data_summary"]["data_hash"]], 12))
        prev = self.store.test_openings(fid)
        reason = self.cfg.eval.reopen_test_reason
        if prev and not reason:
            raise TestLockError(
                f"test для замороженного метода {fid} уже открывался ({prev[0]['run_id']}, "
                f"{prev[0]['time']}). Повторно — только с eval.reopen_test_reason='…', "
                "это будет записано в журнал как нарушение протокола."
            )
        if prev:
            self.flags.append(f"test ОТКРЫТ ПОВТОРНО для {fid}: {reason}")
        self.store.record_test_opening(fid, self.run_id, reason)

        base = self.evaluate("baseline", "test")
        res: dict[str, Any] = {"freeze_id": fid, "methods": {}}
        res["methods"]["baseline"] = self.summarize(base)
        for m in self.cfg.eval.methods:
            if m in ("baseline", "steering"):
                continue
            res["methods"][m] = self.summarize(self.evaluate(m, "test"), base)
        if fr and "steering" in self.cfg.eval.methods:
            r = fr["selection"]
            spec = SteerSpec(r["layer"], r["window"], r["alpha"], r["vector_id"],
                             self.cfg.steer.normalize)
            res["methods"]["steering"] = self.summarize(self.evaluate("steering", "test", spec), base)
            for name, sspec in self._sham_specs(spec).items():
                res["methods"][name] = self.summarize(self.evaluate(name, "test", sspec), base)
        self.state["test"] = res
        self.state["summaries"]["test"] = res["methods"]
        self._save_json("test.json", res)
        self.log("test: " + "; ".join(f"{k} {v['success']:.3f}" for k, v in res["methods"].items()))
        self._mark("test")
        return res

    def estimate(self, n_samples: int = 2) -> "Any":
        """Сколько займёт прогон: замер на нескольких промптах × план вызовов.

        На Mac это главное, что стоит посмотреть до запуска: sweep легко
        оказывается часами. Уменьшать: steer.alphas, steer.top_k_layers,
        steer.windows, data.n_tasks.
        """
        import pandas as pd

        if "data" not in self.done:
            self.data()
        tasks = self._tasks("train", ("changing",))[:n_samples]
        t0 = time.perf_counter()
        for t in tasks:
            self.backend.decide(self._prompt(t, 0), self.cfg.decision.mode, None,
                                self.cfg.decision.temperature, 0)
        per_decision = (time.perf_counter() - t0) / max(1, len(tasks))
        layers = self._layers_for_probe()
        windows = sorted(set(self.cfg.probe.windows) | set(self.cfg.steer.windows))
        t0 = time.perf_counter()
        self.backend.record(self._prompt(tasks[0], 1), layers, windows)
        per_record = time.perf_counter() - t0

        n = {k: len(v) for k, v in self.state["splits"].items()}
        tr = self.cfg.decision.n_trials
        n_vec = self.cfg.steer.top_k_layers * len(self.cfg.steer.windows)
        n_test_methods = len(self.cfg.eval.methods) + len(self.cfg.eval.shams)
        restate = 2.0 if "restate" in self.cfg.eval.methods else 1.0
        plan = [
            ("baseline", (n["train"] + n["val"]) * 2 * tr, per_decision),
            ("probe", n["train"] * 2, per_record),
            ("sweep", n_vec * len(self.cfg.steer.alphas) * n["val"] * 2 * tr, per_decision),
            ("test", n_test_methods * n["test"] * 2 * tr * restate, per_decision),
        ]
        df = pd.DataFrame([{"stage": s, "calls": int(c), "sec_per_call": round(t, 3),
                            "minutes": round(c * t / 60, 1)} for s, c, t in plan])
        self.log(f"estimate: ~{df['minutes'].sum():.0f} мин всего "
                 f"({per_decision:.2f} с на решение, {per_record:.2f} с на съём активаций); "
                 "повторные запуски берут готовое из кэша")
        return df

    def run(self, until: str = "test") -> "Experiment":
        if until not in STAGES:
            raise ValueError(f"until must be one of {STAGES}")
        for s in STAGES[: STAGES.index(until) + 1]:
            if s not in self.done:
                getattr(self, s)()
        return self

    # ------------------------------------------------------------------
    # отчёт
    # ------------------------------------------------------------------
    def verdicts(self) -> dict[str, G.Verdict]:
        return G.evaluate(self.state, self.cfg)

    def results_table(self, split: str | None = None):
        import pandas as pd

        split = split or ("test" if "test" in self.done else "val")
        rows = []
        for name, s in (self.state["summaries"].get(split) or {}).items():
            p = s.get("paired")
            rows.append({"method": name, "success": s["success"],
                         **{f"pass^{k}": v for k, v in s["pass_k"].items()},
                         "sensitivity": s["sensitivity"], "specificity": s["specificity"],
                         "diff_vs_base": p["mean_diff"] if p else None,
                         "ci": f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]" if p else "",
                         "fixed": s["fixed"], "broken": s["broken"],
                         "stress": s["stress_success"], "tokens": round(s["tokens_mean"]),
                         "calls": s["model_calls_mean"], "p50_ms": round(s["latency_p50_ms"], 1)})
        return pd.DataFrame(rows)

    def probe_table(self):
        import pandas as pd

        return pd.DataFrame(self.state.get("probe_rows") or [])

    def sweep_table(self):
        import pandas as pd

        return pd.DataFrame(self.state.get("sweep_rows") or [])

    def _md_table(self) -> str:
        df = self.results_table()
        if df.empty:
            return ""
        split = "test" if "test" in self.done else "val"
        cols = [c for c in ("method", "success", "sensitivity", "specificity", "diff_vs_base", "ci",
                            "fixed", "broken", "tokens", "calls", "p50_ms") if c in df.columns]
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "---|" * len(cols)
        lines = [f"**Результаты ({split}):**", "", head, sep]
        def fmt(x: Any) -> str:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)

        for _, r in df[cols].iterrows():
            lines.append("| " + " | ".join(fmt(x) for x in r) + " |")
        return "\n".join(lines)

    def _update_index(self) -> None:
        sm = self.state["summaries"]
        split = "test" if "test" in sm else "val"
        b = (sm.get(split) or {}).get("baseline")
        st = (sm.get("test") or {}).get("steering")
        sel = (self.state.get("selection") or {}).get("row")
        v = self.verdicts()
        self.store.update_index({
            "run_id": self.run_id, "time": time.strftime("%Y-%m-%d %H:%M"), "name": self.cfg.name,
            "hypothesis": self.cfg.hypothesis, "model": self.cfg.model.name,
            "mode": self.cfg.decision.mode, "stages": "+".join(self.done),
            "base_success": f"{b['success']:.3f}" if b else "",
            "steer_success": (f"{st['success']:.3f}" if st else
                              (f"{sel['success']:.3f} (val)" if sel else "")),
            "paired_diff": f"{st['paired']['mean_diff']:+.3f}" if st else "",
            "ci": (f"[{st['paired']['ci_low']:+.3f}, {st['paired']['ci_high']:+.3f}]" if st else ""),
            "spec_drop": (f"{sm['test']['baseline']['specificity'] - st['specificity']:+.3f}" if st
                          else (f"{sel['spec_drop']:+.3f} (val)" if sel else "")),
            "H1": v["H1"].status, "H2": v["H2"].status, "H3": v["H3"].label,
            "git": self.git["commit"], "dirty": self.git["dirty"], "notebook": self.cfg.notebook,
        })

    def report(self, conclusion: str = "", next_step: str = "", journal: bool = True) -> str:
        """Отчёт прогона: results/runs/<id>/report.md + запись в docs/JOURNAL.md."""
        v = self.verdicts()
        info = self.backend.info() if self._backend is not None else {}
        f = {"run_id": self.run_id, "hypothesis": self.cfg.hypothesis, "notebook": self.cfg.notebook,
             "model": f"{self.cfg.model.name} ({info.get('device', '?')}, {info.get('dtype', '?')})",
             "mode": self.cfg.decision.mode, "stages": self.done, "git": self.git["commit"],
             "dirty": self.git["dirty"], "notes": self.cfg.notes,
             "diff": self.cfg.diff_from_default(), "flags": list(dict.fromkeys(self.flags)),
             "table": self._md_table(), "verdicts": v, "conclusion": conclusion,
             "next_step": next_step}
        block = J.run_entry(self.store.root, f) if journal else ""
        md = block or "\n".join([f"# {self.run_id}", f["table"]])
        extra = []
        if self.state.get("probe_rows"):
            extra += ["", "## Probe (train)", "", self.probe_table().to_markdown(index=False)
                      if _has_tabulate() else self.probe_table().to_string(index=False)]
        if self.state.get("sweep_rows"):
            extra += ["", "## Sweep (val)", "", self.sweep_table().to_markdown(index=False)
                      if _has_tabulate() else self.sweep_table().to_string(index=False)]
        (self.dir / "report.md").write_text(md + "\n".join(extra) + "\n")
        self._save_json("verdicts.json", {k: x.to_dict() for k, x in v.items()})
        self._write_manifest()
        self._update_index()
        return md


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False
