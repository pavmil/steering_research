"""Пайплайн проверки гипотез: один класс Experiment, стадии строго по протоколу.

    exp = Experiment(cfg)
    exp.data()        # задачи, аудит, разбиение по задачам
    exp.baseline()    # база на train+val — материал для H2 и для вектора
    exp.probe()       # H1: различает ли модель ответы (train, CV по задачам)
    exp.vectors()     # CAA: верные vs неверные решения базы (train, меняющие пары)
    exp.sweep()       # слой × окно × alpha на VALIDATION, выбор с ограничением
    exp.fit()         # обучаемые конкуренты (гейт, TACT, LoRA): train -> выбор на val;
                      # сравнение всех методов на val (и эпизоды, если включены)
    exp.freeze()      # заморозка ВСЕХ выбранных методов одним freeze_id
    exp.test()        # test ОДИН раз: база, сравнения, steering, sham-контроли, эпизоды
    exp.report(conclusion="...", next_step="...")   # отчёт + запись в журнал

    exp.run(until="sweep")   # то же, по порядку, с пропуском сделанного

Что гарантирует пайплайн, чтобы в ноутбуке нельзя было случайно нарушить протокол:
  * вектор, гейт, оси TACT и адаптер LoRA строятся только на train, подбор —
    только на val, одним правилом (успех при падении специфичности ≤ допуска);
  * обучаемые методы получают одни и те же данные (меняющие пары train);
  * test открывается один раз на замороженный набор методов (results/test_lock.jsonl);
    повторно — только с eval.reopen_test_reason, и это попадает в журнал;
    открытие test той же модели и данных под другим freeze_id тоже помечается;
  * каждое вмешательство проверяет, что хук отработал ровно на len(окно) позициях;
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

from . import episodes as E
from . import gates as G
from . import journal as J
from . import metrics as M
from . import probe as P
from . import simulator as S
from . import steering as ST
from .backends import (Prompt, ResolvedAdapter, ResolvedSteer, ResolvedTact, auto_layers,
                       load_backend)
from .config import ExperimentConfig
from .methods import (TWO_STEP, AdapterSpec, GateSpec, MethodRun, SteerSpec, TactSpec,
                      build_messages, get_method)
from .store import ActivationCache, DecisionCache, Store, env_info, git_info, sha, write_jsonl
from .windows import PositionWindow

STAGES = ("data", "baseline", "probe", "vectors", "sweep", "fit", "freeze", "test")

# Методы, которые сравниваются со steering в вердикте ALT (CONTEXT.md §3:
# «если явное состояние или промпт дают то же качество дешевле — тезис отвергается»).
CHEAP_ALTERNATIVES = ("strong_prompt", "restate", "explicit_state")


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


def _select(rows: list[dict], tol: float, prefer: str = "alpha") -> dict | None:
    """Правило выбора на val, одно для всех методов: максимум успеха при падении
    специфичности не больше допуска; при равенстве — меньшая сила вмешательства."""
    ok = [r for r in rows if r["spec_drop"] <= tol + 1e-12]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["success"], -float(r.get(prefer, 0.0))))


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
        self.state: dict[str, Any] = {"summaries": defaultdict(dict), "episodes": defaultdict(dict)}
        self.done: list[str] = []
        self.flags: list[str] = []
        self.vectors_by_id: dict[str, ST.SteeringVector] = {}
        self.gates_by_id: dict[str, dict[str, Any]] = {}
        self.axes_by_id: dict[str, list[ST.DriftAxis]] = {}
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
                    "generator": self.cfg.data.generator,
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

    def _prompt_msgs(self, msgs: list[dict], essential: str | None, meta: dict) -> Prompt:
        """Промпт шага продолжения эпизода (после первого решения)."""
        k = ("msgs", sha([msgs, essential]), meta.get("variant"))
        if k not in self._prompts:
            tools = S.tools_schema() if self.cfg.data.use_tools_schema else None
            self._prompts[k] = self.backend.prompt(msgs, essential, tools, meta=meta,
                                                   k_post=self.cfg.data.k_post)
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

    def _resolve_tact(self, spec: TactSpec, p: Prompt) -> ResolvedTact:
        axes = [a for a in self.axes_by_id[spec.axes_id] if a.layer in set(spec.layers)]
        return ResolvedTact(axes, p.windows[spec.window], spec.k, spec.lam)

    def _gate_prob(self, gs: GateSpec, p: Prompt) -> float:
        g = self.gates_by_id[gs.gate_id]
        k = ActivationCache.key(g["layer"], g["window"])
        x = self._acts([p], [g["layer"]], [g["window"]])[0][k].astype(np.float64)
        z = (x - g["mean"]) / g["scale"]
        return float(1.0 / (1.0 + np.exp(-(z @ g["coef"] + g["intercept"]))))

    def _model_decision(self, mr: MethodRun, p: Prompt, trial: int,
                        with_args: bool = False) -> dict[str, Any]:
        """Ответ модели на промпт (кэшируется). Ничего про задачу — только модель."""
        if mr.gate is not None:
            # Гейт смотрит только на активации промпта; решение берётся из кэша
            # steering или базы — ровно то, что дал бы однопроходный хук с гейтом.
            prob = self._gate_prob(mr.gate, p)
            on = prob >= mr.gate.threshold
            sub = MethodRun(get_method("steering" if on else "baseline"),
                            steer=mr.gate.steer if on else None)
            out = dict(self._model_decision(sub, p, trial, with_args))
            out.update(gate_p=round(prob, 4), gate_on=bool(on))
            return out
        dcache, _ = self._caches()
        fmt = self.backend.info().get("action_format")
        dc = self.cfg.decision
        parts: list[Any] = [self.backend.model_key, mr.fingerprint(), p.key, dc.mode,
                            dc.temperature, trial, fmt]
        if with_args:
            parts += ["args", self.cfg.episode.arg_tokens]
        key = sha(parts)
        # В кэше — только ответ модели. Всё, что относится к задаче (id, ветка,
        # оценка), пересчитывается: одинаковый промпт может быть у разных задач.
        out = dcache.get(key)
        cached = out is not None
        if out is None:
            adapter = (ResolvedAdapter(mr.adapter.adapter_id, mr.adapter.path)
                       if mr.adapter is not None else None)
            kw = {"adapter": adapter, "with_args": with_args, "arg_tokens": self.cfg.episode.arg_tokens}
            if mr.spec.variant in TWO_STEP:
                ask, go, n_ask = TWO_STEP[mr.spec.variant]
                d = self.backend.decide_two_step(p, ask, go, n_ask, dc.mode, dc.temperature, trial, **kw)
            else:
                steer = (self._resolve(mr.steer, p) if mr.steer is not None else
                         self._resolve_tact(mr.tact, p) if mr.tact is not None else None)
                d = self.backend.decide(p, dc.mode, steer, dc.temperature, trial, **kw)
            out = {"key": key, "action": d.action, "scores": d.scores,
                   "text": (d.text or "")[:2000], "n_prompt_tokens": d.n_prompt_tokens,
                   "n_new_tokens": d.n_new_tokens, "n_model_calls": d.n_model_calls,
                   "latency_ms": round(d.latency_ms, 2), "hook_applied": d.hook_applied,
                   "positions_source": d.positions_source, "hook_changed": d.hook_changed,
                   "arguments": d.arguments}
            dcache.put(out)
        if out.get("positions_source") == "counter":
            self.flags.append("позиции хука взяты из счётчика, а не из модели — проверить окна")
        return {**out, "cached": cached}

    def _run(self, method: str, steer: SteerSpec | None = None, gate: GateSpec | None = None,
             tact: TactSpec | None = None, adapter: AdapterSpec | None = None) -> MethodRun:
        spec = get_method(method)
        mr = MethodRun(spec, strong_prompt=self.cfg.eval.strong_prompt, steer=steer, gate=gate,
                       tact=tact, adapter=adapter)
        need = {"steer": (spec.steer, steer), "gate": (spec.gate, gate), "tact": (spec.tact, tact),
                "adapter": (spec.adapter, adapter)}
        for k, (wanted, given) in need.items():
            if wanted and given is None:
                raise ValueError(f"method {method!r} needs a {k} spec")
        return mr

    def _decide(self, mr: MethodRun, task: S.PairedTask, branch: int, split: str,
                trial: int) -> dict[str, Any]:
        p = self._prompt(task, branch, mr.spec.variant)
        out = self._model_decision(mr, p, trial)
        w = task.branch(branch)
        return {**out, "task_id": task.task_id, "branch": branch, "trial": trial,
                "pair_type": task.pair_type, "domain": task.domain,
                "explicit": task.condition_explicit, "position": task.critical_step_position,
                "ok": S.score_action(out["action"], w), "allowed": sorted(w.allowed_actions),
                "split": split, "method": mr.spec.name}

    def _tasks(self, split: str, kinds=("changing", "invariant", "stress")) -> list[S.PairedTask]:
        self._need("data")
        return [t for t in self.state["splits"][split] if t.pair_type in kinds]

    def evaluate(self, method: str, split: str, steer: SteerSpec | None = None,
                 tag: str | None = None, write: bool = True, **specs: Any) -> list[dict[str, Any]]:
        """Все пары сплита (обе ветки × n_trials) одним методом. Кэшируется.

        specs — gate= / tact= / adapter= для обучаемых методов.
        """
        mr = self._run(method, steer=steer, **specs)
        tasks = self._tasks(split)
        jobs = [(t, b, k) for t in tasks for b in (0, 1) for k in range(self.cfg.decision.n_trials)]
        recs = [self._decide(mr, t, b, split, k)
                for t, b, k in _progress(jobs, f"{tag or method}/{split}", len(jobs),
                                         disable=not self.verbose)]
        if write:
            write_jsonl(self.dir / "decisions" / f"{split.replace(':', '-')}__{(tag or method).replace(':', '-')}.jsonl",
                        recs)
        return recs

    # ------------------------------------------------------------------
    # многошаговые эпизоды
    # ------------------------------------------------------------------
    def _episode(self, mr: MethodRun, task: S.PairedTask, branch: int, split: str,
                 trial: int) -> dict[str, Any]:
        base_msgs = build_messages(task, branch, mr.spec.variant, self.cfg.eval.strong_prompt)

        def agent(msgs, essential, expected, i):
            if i == 0:
                p = self._prompt(task, branch, mr.spec.variant)
            else:
                meta = {"task_id": task.task_id, "branch": branch, "pair_type": "continuation",
                        "allowed": sorted(expected), "other_allowed": [], "state": branch,
                        "variant": mr.spec.variant}
                p = self._prompt_msgs(msgs, essential, meta)
            out = self._model_decision(mr, p, trial, with_args=True)
            info = {"latency_ms": out["latency_ms"], "n_model_calls": out["n_model_calls"],
                    "n_tokens": out["n_prompt_tokens"] + out["n_new_tokens"], "cached": out["cached"]}
            if "gate_on" in out:
                info["gate_on"] = out["gate_on"]
            return out["action"], out.get("arguments") or {}, info

        ep = E.rollout(task, branch, base_msgs, agent, self.cfg.episode.max_steps).to_dict()
        ep.update(trial=trial, split=split, method=mr.spec.name, domain=task.domain,
                  explicit=task.condition_explicit, position=task.critical_step_position,
                  latency_ms=round(sum(s["latency_ms"] for s in ep["steps"]), 2),
                  tokens=sum(s["n_tokens"] for s in ep["steps"]),
                  model_calls=sum(s["n_model_calls"] for s in ep["steps"]))
        return ep

    def evaluate_episodes(self, method: str, split: str, steer: SteerSpec | None = None,
                          tag: str | None = None, write: bool = True, **specs: Any) -> list[dict]:
        """Эпизоды на меняющих и инвариантных парах сплита (стресс — только одношагово)."""
        mr = self._run(method, steer=steer, **specs)
        tasks = self._tasks(split, ("changing", "invariant"))
        jobs = [(t, b, k) for t in tasks for b in (0, 1) for k in range(self.cfg.decision.n_trials)]
        eps = [self._episode(mr, t, b, split, k)
               for t, b, k in _progress(jobs, f"episodes {tag or method}/{split}", len(jobs),
                                        disable=not self.verbose)]
        if write:
            name = (tag or method).replace(":", "-")
            write_jsonl(self.dir / "decisions" / f"{split.replace(':', '-')}__episodes__{name}.jsonl", eps)
        return eps

    def summarize_episodes(self, eps: list[dict], base: list[dict] | None = None) -> dict[str, Any]:
        by_task: dict[str, list[bool]] = defaultdict(list)
        by_ep: dict[tuple, list[bool]] = defaultdict(list)
        for e in eps:
            by_task[e["task_id"]].append(e["success"])
            by_ep[(e["task_id"], e["branch"])].append(e["success"])
        crit = [e for e in eps if e["critical_ok"]]
        lat = [e["latency_ms"] for e in eps]
        k_max = self.cfg.decision.n_trials
        out: dict[str, Any] = {
            "n": len(eps),
            "success": M.episode_success([e["success"] for e in eps]),
            "pass_k": {k: M.pass_hat_k(list(by_ep.values()), k) for k in sorted({1, k_max})},
            "critical_ok": M.episode_success([e["critical_ok"] for e in eps]),
            # доля эпизодов, доведённых до успеха, среди тех, где первый шаг верен
            "continuation_ok": (M.episode_success([e["success"] for e in crit]) if crit else None),
            "violations": float(np.mean([bool(e["violations"]) for e in eps])),
            "steps_mean": float(np.mean([e["n_steps"] for e in eps])),
            "tokens_mean": float(np.mean([e["tokens"] for e in eps])),
            "model_calls_mean": float(np.mean([e["model_calls"] for e in eps])),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "paired": None, "fixed": None, "broken": None,
        }
        for kind in ("changing", "invariant"):
            sub = [e["success"] for e in eps if e["pair_type"] == kind]
            out[f"success_{kind}"] = M.episode_success(sub) if sub else None
        if base is not None:
            b_task: dict[str, list[bool]] = defaultdict(list)
            for e in base:
                b_task[e["task_id"]].append(e["success"])
            pe = M.paired_bootstrap(b_task, by_task, n_boot=self.cfg.eval.n_boot,
                                    seed=self.cfg.data.seed)
            out["paired"] = _paired_dict(pe)
            ka = {(e["task_id"], e["branch"], e["trial"]): e["success"] for e in base}
            kb = {(e["task_id"], e["branch"], e["trial"]): e["success"] for e in eps}
            keys = sorted(set(ka) & set(kb))
            dlt = M.episode_delta([ka[k] for k in keys], [kb[k] for k in keys])
            out["fixed"], out["broken"] = dlt.fixed, dlt.broken
        return out

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
        gated = [r["gate_on"] for r in recs if "gate_on" in r]
        if gated:
            out["gate_on_rate"] = float(np.mean(gated))
        if base is not None:
            b_main = [r for r in base if r["pair_type"] in ("changing", "invariant")]
            b_task: dict[str, list[bool]] = defaultdict(list)
            for r in b_main:
                b_task[r["task_id"]].append(r["ok"])
            pe = M.paired_bootstrap(b_task, by_task, n_boot=self.cfg.eval.n_boot,
                                    seed=self.cfg.data.seed)
            out["paired"] = _paired_dict(pe)
            ka = {(r["task_id"], r["branch"], r["trial"]): r["ok"] for r in b_main}
            kb = {(r["task_id"], r["branch"], r["trial"]): r["ok"] for r in main}
            keys = sorted(set(ka) & set(kb))
            dlt = M.episode_delta([ka[k] for k in keys], [kb[k] for k in keys])
            out["fixed"], out["broken"] = dlt.fixed, dlt.broken
        return out

    def compare(self, recs_a: list[dict], recs_b: list[dict]) -> dict[str, Any]:
        """Метод A против метода B (не против базы): успех A − B, bootstrap по задачам."""
        def by_task(recs):
            d: dict[str, list[bool]] = defaultdict(list)
            for r in recs:
                if r["pair_type"] in ("changing", "invariant"):
                    d[r["task_id"]].append(r.get("ok", r.get("success")))
            return d
        pe = M.paired_bootstrap(by_task(recs_b), by_task(recs_a), n_boot=self.cfg.eval.n_boot,
                                seed=self.cfg.data.seed)
        return _paired_dict(pe)

    # ------------------------------------------------------------------
    # стадии
    # ------------------------------------------------------------------
    def data(self) -> dict[str, Any]:
        c = self.cfg.data
        tasks = S.build_pilot_set(c.n_tasks, seed=c.seed, version=c.generator)
        rep = S.audit(tasks)
        if not rep.ok:
            raise ValueError(f"аудит данных не пройден:\n{rep}")
        splits = S.split_tasks(tasks, tuple(c.split), seed=c.seed)
        self.state.update(tasks=tasks, splits=splits)
        by_pos: dict[str, int] = defaultdict(int)
        for t in tasks:
            by_pos[t.critical_step_position] += 1
        summary = {"n_tasks": len(tasks), "by_type": rep.by_pair_type, "by_domain": rep.by_domain,
                   "by_position": dict(by_pos), "generator": c.generator,
                   "splits": {k: len(v) for k, v in splits.items()},
                   "data_hash": sha([[t.task_id, t.history, t.world_0.observation(),
                                      t.world_1.observation()] for t in tasks])}
        self.state["data_summary"] = summary
        self._save_json("data.json", summary)
        self.log(f"data: {summary['splits']}  types={summary['by_type']}  "
                 f"positions={summary['by_position']}")
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
        for p in _progress(prompts, "activations", len(prompts),
                           disable=not self.verbose or len(prompts) < 4):
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
                        invariant_task_ids=[t.task_id for t, _, _ in inv_p] or None,
                        invariant_states=[S.semantic_state(t.branch(b)) for t, b, _ in inv_p] or None)
                except ValueError as e:
                    self.log(f"probe L{L}/{w}: skipped ({e})")
                    continue
                rows.append({"layer": L, "window": w, "accuracy": r.accuracy, "auroc": r.auroc,
                             "majority": r.majority_baseline, "permuted_accuracy": r.permuted_accuracy,
                             "shortcut_auroc": r.shortcut_auroc,
                             "invariant_transfer": r.invariant_transfer, "beats_baseline": r.beats_baseline,
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

    def _train_ok(self, kinds=("changing",)) -> dict[tuple[str, int], bool]:
        """Верно ли база решила каждую ветку train (trial 0) — метка для CAA, гейта, TACT."""
        recs = self.state["records"]["train"]["baseline"]
        return {(r["task_id"], r["branch"]): r["ok"] for r in recs
                if r["pair_type"] in kinds and r["trial"] == 0}

    def vectors(self) -> list[dict[str, Any]]:
        """CAA только на train и только на меняющих парах: решения базы, где агент
        опирался на ответ (верно) vs где нет (неверно). Ветка как метка НЕ используется —
        иначе вмешательство знало бы правильный ответ."""
        self._need("data", "baseline")
        ok = self._train_ok()
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

    def _row(self, s: dict, bs: dict, **params: Any) -> dict[str, Any]:
        return {**params, "success": s["success"], "base_success": bs["success"],
                "sensitivity": s["sensitivity"], "specificity": s["specificity"],
                "spec_drop": bs["specificity"] - s["specificity"],
                "diff": s["paired"]["mean_diff"], "ci_low": s["paired"]["ci_low"],
                "ci_high": s["paired"]["ci_high"], "fixed": s["fixed"], "broken": s["broken"]}

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
                rows.append(self._row(self.summarize(recs, base), bs, layer=v["layer"],
                                      window=v["window"], alpha=float(a), vector_id=v["vector_id"]))
        write_jsonl(self.dir / "decisions" / "val__sweep.jsonl", all_recs)
        import pandas as pd

        pd.DataFrame(rows).to_csv(self.dir / "sweep.csv", index=False)
        tol = self.cfg.steer.max_specificity_drop
        sel: dict[str, Any] = {"row": _select(rows, tol), "tolerance": tol}
        if not rows:
            sel["reason"] = self.state["vectors"].get("reason", "нет векторов для подбора")
        elif sel["row"] is None:
            sel["reason"] = f"все конфигурации роняют специфичность больше чем на {tol}"
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

    # --- fit: обучаемые конкуренты ------------------------------------------
    def _steer_spec(self) -> SteerSpec | None:
        r = (self.state.get("selection") or {}).get("row")
        if r is None:
            return None
        return SteerSpec(r["layer"], r["window"], r["alpha"], r["vector_id"], self.cfg.steer.normalize)

    def _fit_steering_gated(self) -> dict[str, Any]:
        """Гейт по активациям окна: учится на train предсказывать ошибку базы,
        порог (и вектор, alpha) — на val. Данные — те же train-пары."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        vecs = self.state["vectors"]["vectors"]
        res: dict[str, Any] = {"rows": [], "row": None}
        if not vecs:
            res["reason"] = "нет векторов steering — гейту нечего включать"
            return res
        ok = self._train_ok(("changing", "invariant"))
        tasks = {t.task_id: t for t in self._tasks("train", ("changing", "invariant"))}
        keys = sorted(ok)
        y = np.array([0 if ok[k] else 1 for k in keys])
        mn = self.cfg.steer.min_class_examples
        if y.sum() < mn or (len(y) - y.sum()) < mn:
            res["reason"] = f"мало ошибок/верных решений базы на train для гейта: {int(y.sum())}/{len(y)}"
            return res
        prompts = [self._prompt(tasks[t], b) for t, b in keys]
        base = self.state["records"]["val"]["baseline"]
        bs = self.state["summaries"]["val"]["baseline"]
        gdir = self.dir / "gates"
        gdir.mkdir(exist_ok=True)
        for v in vecs:
            L, w = v["layer"], v["window"]
            X = np.stack([a[ActivationCache.key(L, w)] for a in self._acts(prompts, [L], [w])])
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(C=self.cfg.gate.C, max_iter=2000,
                                     random_state=self.cfg.data.seed).fit(sc.transform(X), y)
            g = {"layer": L, "window": w, "mean": sc.mean_, "scale": sc.scale_,
                 "coef": clf.coef_[0], "intercept": float(clf.intercept_[0])}
            gid = sha(np.concatenate([g["mean"], g["scale"], g["coef"], [g["intercept"]]]).tobytes(), 12)
            self.gates_by_id[gid] = g
            np.savez(gdir / f"gate_{gid}.npz", **{k: np.asarray(x) for k, x in g.items()})
            for a in self.cfg.steer.alphas:
                st = SteerSpec(L, w, float(a), v["vector_id"], self.cfg.steer.normalize)
                for tau in self.cfg.gate.thresholds:
                    gs = GateSpec(st, gid, float(tau))
                    recs = self.evaluate("steering_gated", "val", write=False, gate=gs,
                                         tag=f"gated L{L}/{w}/a{a}/t{tau}")
                    s = self.summarize(recs, base)
                    res["rows"].append(self._row(s, bs, layer=L, window=w, alpha=float(a),
                                                 threshold=float(tau), vector_id=v["vector_id"],
                                                 gate_id=gid, gate_on_rate=s.get("gate_on_rate")))
        res["row"] = _select(res["rows"], self.cfg.steer.max_specificity_drop)
        if res["row"] is None:
            res["reason"] = "все конфигурации гейта роняют специфичность сверх допуска"
        return res

    def _gate_spec(self, row: dict) -> GateSpec:
        return GateSpec(SteerSpec(row["layer"], row["window"], row["alpha"], row["vector_id"],
                                  self.cfg.steer.normalize), row["gate_id"], row["threshold"])

    def _fit_tact(self) -> dict[str, Any]:
        """Адаптация TACT: ось «верно -> сбой» по train в точке действия на каждом слое,
        top-K слоёв по AUC на val, сила lam — на val (k фиксирован, как в статье)."""
        from sklearn.metrics import roc_auc_score

        c = self.cfg.tact
        res: dict[str, Any] = {"rows": [], "row": None}
        layers = list(range(self.backend.n_layers))
        ok_ch = self._train_ok(("changing",))
        ok_all = self._train_ok(("changing", "invariant"))
        n_fail = sum(not v for v in ok_ch.values())
        n_ok = sum(ok_ch.values())
        mn = self.cfg.steer.min_class_examples
        if n_fail < mn or n_ok < mn:
            res["reason"] = f"мало материала для оси: сбоев {n_fail}, верных {n_ok} на меняющих парах train"
            return res
        tasks = {t.task_id: t for t in self._tasks("train", ("changing", "invariant"))}
        keys = sorted(ok_all)
        acts = dict(zip(keys, self._acts([self._prompt(tasks[t], b) for t, b in keys], layers, [c.window])))
        val_tasks = self._tasks("val", ("changing", "invariant"))
        vkeys = [(t, b) for t in val_tasks for b in (0, 1)]
        vacts = self._acts([self._prompt(t, b) for t, b in vkeys], layers, [c.window])
        vbad = {(r["task_id"], r["branch"]): not r["ok"]
                for r in self.state["records"]["val"]["baseline"] if r["trial"] == 0}
        vy = np.array([int(vbad[(t.task_id, b)]) for t, b in vkeys])
        axes, aucs, infos = [], {}, {}
        for L in layers:
            k = ActivationCache.key(L, c.window)
            fail = [acts[x][k] for x in keys if x in ok_ch and not ok_ch[x]]
            good = [acts[x][k] for x in keys if x in ok_ch and ok_ch[x]]
            cal = [acts[x][k] for x in keys if ok_all[x]]
            ax, info = ST.drift_axis(fail, good, [x[0] for x in keys if x in ok_ch and not ok_ch[x]],
                                     [x[0] for x in keys if x in ok_ch and ok_ch[x]], L, cal)
            axes.append(ax)
            infos[L] = info
            score = np.array([(a[k] - ax.mu) @ ax.unit for a in vacts])
            aucs[L] = float(roc_auc_score(vy, score)) if 0 < vy.sum() < len(vy) else 0.5
        n_top = max(c.min_layers, int(round(c.layer_frac * len(layers))))
        top = tuple(sorted(sorted(aucs, key=aucs.get, reverse=True)[:n_top]))
        aid = sha(b"".join(a.unit.tobytes() + a.mu.tobytes() for a in axes), 12)
        self.axes_by_id[aid] = axes
        tdir = self.dir / "tact"
        tdir.mkdir(exist_ok=True)
        np.savez(tdir / f"axes_{aid}.npz", units=np.stack([a.unit for a in axes]),
                 mus=np.stack([a.mu for a in axes]), sigmas=np.array([a.sigma for a in axes]))
        res.update(axes_id=aid, layers=list(top), layer_auc=aucs, axis_mode=infos[top[0]]["mode"],
                   n_problems=infos[top[0]]["n_problems"])
        if vy.sum() in (0, len(vy)):
            res["note"] = "на val нет одного из классов — AUC слоёв не информативен (0.5)"
        base = self.state["records"]["val"]["baseline"]
        bs = self.state["summaries"]["val"]["baseline"]
        for lam in c.lambdas:
            spec = TactSpec(aid, top, c.window, c.k, float(lam))
            recs = self.evaluate("tact", "val", write=False, tact=spec, tag=f"tact lam{lam}")
            res["rows"].append(self._row(self.summarize(recs, base), bs, lam=float(lam), k=c.k,
                                         layers=list(top), axes_id=aid))
        res["row"] = _select(res["rows"], self.cfg.steer.max_specificity_drop, prefer="lam")
        if res["row"] is None:
            res["reason"] = "все lam роняют специфичность сверх допуска"
        return res

    def _tact_spec(self, row: dict) -> TactSpec:
        return TactSpec(row["axes_id"], tuple(row["layers"]), self.cfg.tact.window, row["k"], row["lam"])

    def _fit_lora(self) -> dict[str, Any]:
        """LoRA/SFT на тех же данных, что и вектор: меняющие пары train, обе ветки,
        цель — верный вызов. lr и эпохи — на val тем же правилом выбора."""
        c = self.cfg.lora
        res: dict[str, Any] = {"rows": [], "row": None}
        examples = []
        for t in self._tasks("train", ("changing",)):
            for b in (0, 1):
                gold = sorted(t.branch(b).allowed_actions)[0]
                args = E.Env(t, b).canonical_arguments(gold)
                examples.append(self.backend.sft_example(self._prompt(t, b), gold, args))
        if not examples:
            res["reason"] = "нет меняющих пар в train"
            return res
        data_id = sha(examples, 12)
        base = self.state["records"]["val"]["baseline"]
        bs = self.state["summaries"]["val"]["baseline"]
        lcfg = {k: getattr(c, k) for k in ("rank", "lora_alpha", "dropout", "target_modules",
                                           "grad_accum", "train_dtype", "seed")}
        for entry in c.grid:
            aid = sha([self.backend.model_key, data_id, lcfg, entry], 12)
            path = self.store.cache_dir(self.backend.model_key) / "lora" / aid
            info_path = path / "train_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
            else:
                self.log(f"lora: обучение {entry} на {len(examples)} примерах …")
                info = self.backend.train_adapter(examples, c, float(entry["lr"]), int(entry["epochs"]),
                                                  str(path))
                info_path.write_text(json.dumps(info, indent=2))
                self.log(f"lora: {aid} готов за {info['train_seconds']} с, loss по эпохам "
                         f"{info['epoch_losses']}")
            spec = AdapterSpec(aid, str(path))
            recs = self.evaluate("lora", "val", write=False, adapter=spec, tag=f"lora {aid}")
            res["rows"].append(self._row(self.summarize(recs, base), bs, lr=float(entry["lr"]),
                                         epochs=int(entry["epochs"]), adapter_id=aid, path=str(path),
                                         train_seconds=info["train_seconds"],
                                         n_examples=info["n_examples"], n_trainable=info["n_trainable"]))
        res["row"] = _select(res["rows"], self.cfg.steer.max_specificity_drop, prefer="lr")
        res["data_id"] = data_id
        if res["row"] is None:
            res["reason"] = "все конфигурации LoRA роняют специфичность сверх допуска"
        return res

    def _specs_for(self, method: str) -> dict[str, Any] | None:
        """Параметры выбранного (на val) метода для evaluate; None — метод не выбран."""
        if method == "steering":
            st = self._steer_spec()
            return None if st is None else {"steer": st}
        f = (self.state.get("fitted") or {}).get(method)
        if f is None or f.get("row") is None:
            return None
        r = f["row"]
        if method == "steering_gated":
            return {"gate": self._gate_spec(r)}
        if method == "tact":
            return {"tact": self._tact_spec(r)}
        if method == "lora":
            return {"adapter": AdapterSpec(r["adapter_id"], r["path"])}
        raise KeyError(method)

    def fit(self) -> dict[str, Any]:
        """Обучаемые конкуренты (train -> выбор на val) и сравнение всех методов на val."""
        self._need("sweep")
        fitted: dict[str, Any] = {}
        for m in self.cfg.eval.methods:
            if get_method(m).fitted:
                fitted[m] = getattr(self, f"_fit_{m}")()
                r = fitted[m]["row"]
                self.log(f"fit {m}: " + (f"выбрано, успех {r['base_success']:.3f} → {r['success']:.3f}, "
                                         f"Δспециф. {-r['spec_drop']:+.3f}" if r else
                                         f"ничего не выбрано — {fitted[m].get('reason')}"))
        self.state["fitted"] = fitted
        self._save_json("fitted.json", {m: {k: v for k, v in f.items() if k != "rows"}
                                        for m, f in fitted.items()})
        import pandas as pd

        for m, f in fitted.items():
            if f.get("rows"):
                pd.DataFrame(f["rows"]).to_csv(self.dir / f"fit_{m}.csv", index=False)
        self._val_compare()
        self._mark("fit")
        return fitted

    def _val_compare(self) -> None:
        """Все методы на val — картина ДО test. Для выбранных методов val оптимистичен:
        на нём же шёл подбор."""
        base = self.state["records"]["val"]["baseline"]
        recs_by: dict[str, list[dict]] = {"baseline": base}
        for m in self.cfg.eval.methods:
            if m == "baseline":
                continue
            specs = {} if not (get_method(m).steer or get_method(m).fitted) else self._specs_for(m)
            if specs is None:
                continue
            recs_by[m] = self.evaluate(m, "val", **specs)
            self.state["summaries"]["val"][m] = self.summarize(recs_by[m], base)
        self.state.setdefault("records", {})["val"].update(recs_by)
        self.state["val_comparisons"] = self._comparisons(recs_by)
        if self.cfg.episode.enabled and self.cfg.episode.on_val:
            eps = {"baseline": self.evaluate_episodes("baseline", "val")}
            for m in recs_by:
                if m != "baseline":
                    specs = {} if not (get_method(m).steer or get_method(m).fitted) else self._specs_for(m)
                    eps[m] = self.evaluate_episodes(m, "val", **(specs or {}))
            for m, e in eps.items():
                self.state["episodes"]["val"][m] = self.summarize_episodes(
                    e, None if m == "baseline" else eps["baseline"])
        self._save_json("val_summary.json", {"methods": self.state["summaries"]["val"],
                                             "episodes": self.state["episodes"]["val"],
                                             "comparisons": self.state["val_comparisons"]})
        self.log("val: " + "; ".join(f"{k} {v['success']:.3f}"
                                     for k, v in self.state["summaries"]["val"].items()))

    def transfer(self) -> dict[str, Any]:
        """Разведка переноса: методы, выбранные на val (стадия fit), без какого-либо подбора —
        на задачах со знакомыми типами требований (seen, свежий seed) и с новыми (unseen).

        Test-сплит не используется; набор генерируется отдельно и не должен пересекаться ни с
        одним сплитом по содержимому. Разница seen − unseen в приросте метода — его переносимость
        (один из критериев 2604.14090). Результат разведочный: подтверждать на отложенных данных.
        """
        self._need("fit")
        c = self.cfg.transfer
        sets = S.build_transfer_set(c.n_tasks, c.seed, tuple(c.seen), tuple(c.unseen))
        known = {S.signature(t) for k, v in self.state["splits"].items()
                 if not k.startswith("transfer:") for t in v}
        res: dict[str, Any] = {"cfg": {"n_tasks": c.n_tasks, "seed": c.seed, "seen": list(c.seen),
                                       "unseen": list(c.unseen)}}
        for name, tasks in sets.items():
            if {S.signature(t) for t in tasks} & known:
                raise ValueError("transfer set overlaps train/val/test by content")
            split = f"transfer:{name}"
            self.state["splits"][split] = tasks
            base = self.evaluate("baseline", split)
            recs_by: dict[str, list[dict]] = {"baseline": base}
            self.state["summaries"][split]["baseline"] = self.summarize(base)
            for m in self.cfg.eval.methods:
                if m == "baseline":
                    continue
                spec = get_method(m)
                specs = self._specs_for(m) if (spec.steer or spec.fitted) else {}
                if specs is None:
                    continue
                recs_by[m] = self.evaluate(m, split, **specs)
                self.state["summaries"][split][m] = self.summarize(recs_by[m], base)
            out = {"methods": dict(self.state["summaries"][split]),
                   "comparisons": self._comparisons(recs_by), "episodes": {}}
            if self.cfg.episode.enabled:
                eps = {}
                for m in c.episode_methods:
                    if m not in recs_by:
                        continue
                    spec = get_method(m)
                    specs = self._specs_for(m) if (spec.steer or spec.fitted) else {}
                    eps[m] = self.evaluate_episodes(m, split, **(specs or {}))
                for m, e in eps.items():
                    out["episodes"][m] = self.summarize_episodes(
                        e, None if m == "baseline" else eps.get("baseline"))
                self.state["episodes"][split] = out["episodes"]
            res[name] = out
            self.log(f"transfer {name}: " + "; ".join(
                f"{k} {v['success']:.3f}" for k, v in out["methods"].items()))
        self.state["transfer"] = res
        self._save_json("transfer.json", res)
        self._mark("transfer")
        return res

    def _comparisons(self, recs_by: dict[str, list[dict]]) -> dict[str, Any]:
        out = {}
        if "steering" in recs_by:
            for m in recs_by:
                if m not in ("baseline", "steering") and not m.startswith("sham:"):
                    out[f"steering_vs_{m}"] = self.compare(recs_by["steering"], recs_by[m])
        return out

    def freeze(self) -> str | None:
        self._need("fit")
        payload: dict[str, Any] = {
            "model_key": self.backend.model_key,
            "data_hash": self.state["data_summary"]["data_hash"],
            "decision": {"mode": self.cfg.decision.mode, "temperature": self.cfg.decision.temperature,
                         "n_trials": self.cfg.decision.n_trials}}
        methods: dict[str, Any] = {}
        r = self.state["selection"]["row"]
        if r is not None:
            # Только то, что определяет метод: параметры отчёта (n_boot, список
            # сравнений, причина повторного открытия) не должны давать «новый» freeze_id.
            methods["steering"] = {k: r[k] for k in ("layer", "window", "alpha", "vector_id")}
            methods["steering"]["normalize"] = self.cfg.steer.normalize
        for m, f in (self.state.get("fitted") or {}).items():
            fr = f.get("row")
            if fr is None:
                continue
            keep = {"steering_gated": ("layer", "window", "alpha", "vector_id", "gate_id", "threshold"),
                    "tact": ("axes_id", "layers", "k", "lam"),
                    "lora": ("adapter_id", "lr", "epochs")}[m]
            methods[m] = {k: fr[k] for k in keep}
        if not methods:
            self.log("freeze: нечего замораживать — " + self.state["selection"].get("reason", ""))
            self._mark("freeze")
            return None
        if set(methods) == {"steering"}:
            payload["selection"] = methods["steering"]          # формат v1: тот же freeze_id
            payload["normalize"] = payload["selection"].pop("normalize")
        else:
            payload["methods"] = methods
            if self.cfg.episode.enabled:
                payload["episode"] = {"max_steps": self.cfg.episode.max_steps,
                                      "arg_tokens": self.cfg.episode.arg_tokens}
        fid = self.store.save_frozen(payload, meta={"run_id": self.run_id, "git": self.git["commit"],
                                                    "val_row": r})
        fdir = self.store.results / "frozen"
        if "steering" in methods:
            self.vectors_by_id[methods["steering"]["vector_id"]].save(str(fdir / f"{fid}.npz"))
        if "steering_gated" in methods:
            g = self.gates_by_id[methods["steering_gated"]["gate_id"]]
            np.savez(fdir / f"{fid}_gate.npz", **{k: np.asarray(x) for k, x in g.items()})
        if "tact" in methods:
            axes = self.axes_by_id[methods["tact"]["axes_id"]]
            np.savez(fdir / f"{fid}_tact.npz", units=np.stack([a.unit for a in axes]),
                     mus=np.stack([a.mu for a in axes]), sigmas=np.array([a.sigma for a in axes]))
        self.state["frozen"] = {"freeze_id": fid, "methods": methods, **payload}
        self.log(f"freeze: {fid} ({', '.join(methods)})")
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
        """Test — один раз на замороженный набор методов."""
        self._need("freeze")
        fr = self.state.get("frozen")
        scope = sha([self.backend.model_key, self.state["data_summary"]["data_hash"]], 12)
        fid = fr["freeze_id"] if fr else f"nofreeze-{scope}"
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
        others = [r for r in self.store.test_openings_scope(scope) if r["freeze_id"] != fid]
        if others:
            self.flags.append(f"test этой модели и данных уже открывался под другим набором методов "
                              f"({others[0]['freeze_id']}, {others[0]['run_id']}): сравнение — "
                              "не первое открытие test")
        self.store.record_test_opening(fid, self.run_id, reason, scope)

        base = self.evaluate("baseline", "test")
        res: dict[str, Any] = {"freeze_id": fid, "methods": {}, "episodes": {}}
        res["methods"]["baseline"] = self.summarize(base)
        recs_by: dict[str, list[dict]] = {"baseline": base}
        frozen = (fr or {}).get("methods") or {}
        for m in self.cfg.eval.methods:
            if m == "baseline":
                continue
            spec = get_method(m)
            specs: dict[str, Any] | None = {}
            if spec.steer or spec.fitted:
                specs = self._specs_for(m) if m in frozen else None
            if specs is None:
                continue
            recs_by[m] = self.evaluate(m, "test", **specs)
            res["methods"][m] = self.summarize(recs_by[m], base)
            if m == "steering":
                for name, sspec in self._sham_specs(specs["steer"]).items():
                    recs_by[name] = self.evaluate(name, "test", steer=sspec)
                    res["methods"][name] = self.summarize(recs_by[name], base)
        res["comparisons"] = self._comparisons(recs_by)
        if self.cfg.episode.enabled:
            eps = {"baseline": self.evaluate_episodes("baseline", "test")}
            res["episodes"]["baseline"] = self.summarize_episodes(eps["baseline"])
            for m in recs_by:
                if m == "baseline" or m.startswith("sham:"):
                    continue
                spec = get_method(m)
                specs = self._specs_for(m) if (spec.steer or spec.fitted) else {}
                eps[m] = self.evaluate_episodes(m, "test", **(specs or {}))
                res["episodes"][m] = self.summarize_episodes(eps[m], eps["baseline"])
            if "steering" in eps:
                for m in eps:
                    if m not in ("baseline", "steering"):
                        res["comparisons"][f"episode:steering_vs_{m}"] = self.compare(eps["steering"], eps[m])
        self.state["test"] = res
        self.state["test_records"] = recs_by
        self.state["summaries"]["test"] = res["methods"]
        self.state["episodes"]["test"] = res["episodes"]
        self._save_json("test.json", res)
        self.log("test: " + "; ".join(f"{k} {v['success']:.3f}" for k, v in res["methods"].items()))
        self._mark("test")
        return res

    def estimate(self, n_samples: int = 2) -> "Any":
        """Сколько займёт прогон: замер на нескольких промптах × план вызовов.

        На Mac это главное, что стоит посмотреть до запуска: sweep легко
        оказывается часами. Уменьшать: steer.alphas, steer.top_k_layers,
        steer.windows, data.n_tasks. Обучение LoRA сюда не входит — его время
        печатается при обучении и пишется в train_info.json.
        """
        import pandas as pd

        if "data" not in self.done:
            self.data()
        tasks = self._tasks("train", ("changing",))[:n_samples]
        # Загрузка модели, построение промптов и прогрев (кэш системного хода) — вне замера.
        prompts = [self._prompt(t, 0) for t in tasks]
        self.backend.decide(self._prompt(tasks[0], 1), self.cfg.decision.mode, None,
                            self.cfg.decision.temperature, 0)
        t0 = time.perf_counter()
        for p in prompts:
            self.backend.decide(p, self.cfg.decision.mode, None, self.cfg.decision.temperature, 0)
        per_decision = (time.perf_counter() - t0) / max(1, len(tasks))
        t0 = time.perf_counter()
        self.backend.decide(self._prompt(tasks[0], 1), self.cfg.decision.mode, None,
                            self.cfg.decision.temperature, 0, with_args=True,
                            arg_tokens=self.cfg.episode.arg_tokens)
        per_step = time.perf_counter() - t0
        layers = self._layers_for_probe()
        windows = sorted(set(self.cfg.probe.windows) | set(self.cfg.steer.windows))
        t0 = time.perf_counter()
        self.backend.record(self._prompt(tasks[0], 1), layers, windows)
        per_record = time.perf_counter() - t0

        n = {k: len(v) for k, v in self.state["splits"].items()}
        nm = {k: len(self._tasks(k, ("changing", "invariant"))) for k in n}
        tr = self.cfg.decision.n_trials
        n_vec = self.cfg.steer.top_k_layers * len(self.cfg.steer.windows)
        methods = self.cfg.eval.methods
        two = 2.5 * sum(m in TWO_STEP for m in methods)
        n_test_methods = len(methods) + len(self.cfg.eval.shams) + two
        fit_calls = 0
        if "tact" in methods:
            fit_calls += len(self.cfg.tact.lambdas) * n["val"] * 2 * tr
        if "lora" in methods:
            fit_calls += len(self.cfg.lora.grid) * n["val"] * 2 * tr
        plan = [
            ("baseline", (n["train"] + n["val"]) * 2 * tr, per_decision),
            ("probe", n["train"] * 2, per_record),
            ("sweep", n_vec * len(self.cfg.steer.alphas) * n["val"] * 2 * tr, per_decision),
            ("fit", fit_calls + (len(methods) + two) * n["val"] * 2 * tr, per_decision),
            ("test", n_test_methods * n["test"] * 2 * tr, per_decision),
        ]
        if self.cfg.episode.enabled:
            # ~1.5 шага на эпизод (первый шаг + продолжение в «плохой» ветке)
            n_ep = (len(methods) + two) * 2 * tr * 1.5
            plan.append(("episodes", n_ep * (nm["test"] + (nm["val"] if self.cfg.episode.on_val else 0)),
                         per_step))
        df = pd.DataFrame([{"stage": s, "calls": int(c), "sec_per_call": round(t, 3),
                            "minutes": round(c * t / 60, 1)} for s, c, t in plan])
        self.log(f"estimate: ~{df['minutes'].sum():.0f} мин всего "
                 f"({per_decision:.2f} с на решение, {per_step:.2f} с на шаг эпизода, "
                 f"{per_record:.2f} с на съём активаций); повторные запуски берут готовое из кэша")
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
        eps = self.state["episodes"].get(split) or {}
        rows = []
        for name, s in (self.state["summaries"].get(split) or {}).items():
            p = s.get("paired")
            e = eps.get(name) or {}
            ep = e.get("paired")
            rows.append({"method": name, "success": s["success"],
                         **{f"pass^{k}": v for k, v in s["pass_k"].items()},
                         "sensitivity": s["sensitivity"], "specificity": s["specificity"],
                         "diff_vs_base": p["mean_diff"] if p else None,
                         "ci": f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]" if p else "",
                         "fixed": s["fixed"], "broken": s["broken"],
                         "episode": e.get("success"),
                         "ep_diff": ep["mean_diff"] if ep else None,
                         "ep_ci": f"[{ep['ci_low']:+.3f}, {ep['ci_high']:+.3f}]" if ep else "",
                         "stress": s["stress_success"], "tokens": round(s["tokens_mean"]),
                         "calls": s["model_calls_mean"], "p50_ms": round(s["latency_p50_ms"], 1)})
        return pd.DataFrame(rows)

    def probe_table(self):
        import pandas as pd

        return pd.DataFrame(self.state.get("probe_rows") or [])

    def sweep_table(self):
        import pandas as pd

        return pd.DataFrame(self.state.get("sweep_rows") or [])

    def fit_table(self, method: str):
        import pandas as pd

        return pd.DataFrame(((self.state.get("fitted") or {}).get(method) or {}).get("rows") or [])

    def _md_table(self, split: str | None = None) -> str:
        split = split or ("test" if "test" in self.done else "val")
        df = self.results_table(split)
        if df.empty:
            return ""
        cols = [c for c in ("method", "success", "sensitivity", "specificity", "diff_vs_base", "ci",
                            "fixed", "broken", "episode", "ep_diff", "tokens", "calls", "p50_ms")
                if c in df.columns and not (c in ("episode", "ep_diff") and df[c].isna().all())]
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "---|" * len(cols)
        title = (f"**Перенос, {split.split(':', 1)[1]} (разведка):**" if split.startswith("transfer:")
                 else f"**Результаты ({split}):**")
        lines = [title, "", head, sep]

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
        eb = (self.state["episodes"].get(split) or {}).get("baseline")
        es = (self.state["episodes"].get(split) or {}).get("steering")
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
            "episode_base": f"{eb['success']:.3f}" if eb else "",
            "episode_steer": f"{es['success']:.3f}" if es else "",
            "H1": v["H1"].status, "H2": v["H2"].status, "H3": v["H3"].label,
            "H4": v["H4"].label, "ALT": v["ALT"].label,
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
             "table": "\n\n".join(t for t in [self._md_table()] + [
                 self._md_table(k) for k in ("transfer:seen", "transfer:unseen")
                 if self.state["summaries"].get(k)] if t),
             "verdicts": v, "conclusion": conclusion,
             "next_step": next_step}
        block = J.run_entry(self.store.root, f) if journal else ""
        md = block or "\n".join([f"# {self.run_id}", f["table"]])
        extra = []
        tab = _has_tabulate()
        if self.state.get("probe_rows"):
            extra += ["", "## Probe (train)", "", self.probe_table().to_markdown(index=False)
                      if tab else self.probe_table().to_string(index=False)]
        if self.state.get("sweep_rows"):
            extra += ["", "## Sweep (val)", "", self.sweep_table().to_markdown(index=False)
                      if tab else self.sweep_table().to_string(index=False)]
        for m in (self.state.get("fitted") or {}):
            t = self.fit_table(m)
            if not t.empty:
                extra += ["", f"## Fit {m} (val)", "", t.to_markdown(index=False) if tab else t.to_string(index=False)]
        (self.dir / "report.md").write_text(md + "\n".join(extra) + "\n")
        self._save_json("verdicts.json", {k: x.to_dict() for k, x in v.items()})
        self._write_manifest()
        self._update_index()
        return md


def _paired_dict(pe: M.PairedEstimate) -> dict[str, Any]:
    return {"mean_diff": pe.mean_diff, "ci_low": pe.ci_low, "ci_high": pe.ci_high,
            "excludes_zero": pe.excludes_zero, "n_tasks": pe.n_tasks}


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False
