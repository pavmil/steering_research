"""Прогон трёх проверок H1 -> H2 -> H3 (docs/HYPOTHESES.md).

Два режима:

  --mock   без модели и torch. Синтетический агент с управляемым разрывом
           «различает, но не действует». Нужен, чтобы проверить саму обвязку
           и метрики до того, как тратить GPU. НЕ является результатом.

  --model  реальный прогон на HF-модели (требует torch + transformers).

Порядок проверок строгий: провал предыдущей обесценивает следующую.
Манифест пишется всегда — прогон без манифеста считается несуществующим
(AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M
import probe as P
import simulator as S


# --------------------------------------------------------------------------
# Манифест
# --------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class Manifest:
    """Всё, без чего прогон невоспроизводим."""

    mode: str
    model: str | None
    seed: int
    n_tasks: int
    chat_template_hash: str | None
    decoding: dict[str, Any]
    layer: int | None
    alpha: float | None
    window_kind: str | None
    git_commit: str
    python: str
    timestamp: str
    data_hash: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


def data_hash(tasks: list[S.PairedTask]) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda x: x.task_id):
        h.update(t.task_id.encode())
        h.update(t.pair_type.encode())
        h.update(t.world_0.observation().encode())
        h.update(t.world_1.observation().encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Синтетический агент (только для --mock)
# --------------------------------------------------------------------------


class MockAgent:
    """Воспроизводит целевой разрыв: активации несут состояние среды,
    но выбор действия от него зависит слабо.

    separability — насколько состояние читается из активаций (для H1).
    use_rate     — как часто агент реально пользуется различием (для H2).
    steer_gain   — прибавка к use_rate при включённом вмешательстве (для H3).
    steer_cost   — вероятность сломать инвариантную пару (ложное переключение).
    """

    def __init__(
        self,
        seed: int = 0,
        hidden: int = 64,
        separability: float = 2.0,
        use_rate: float = 0.35,
        steer_gain: float = 0.35,
        steer_cost: float = 0.10,
    ):
        self.rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.separability = separability
        self.use_rate = use_rate
        self.steer_gain = steer_gain
        self.steer_cost = steer_cost
        self._dir = self.rng.normal(size=hidden)
        self._dir /= np.linalg.norm(self._dir)

    def activation(self, state_label: int) -> np.ndarray:
        """Активация несёт состояние среды с заданной отделимостью."""
        base = self.rng.normal(size=self.hidden)
        return base + self.separability * state_label * self._dir

    def act(self, task: S.PairedTask, branch: int, steered: bool) -> str:
        w = task.branch(branch)
        p = self.use_rate + (self.steer_gain if steered else 0.0)

        if task.pair_type == "invariant":
            # Ложное переключение: вмешательство может «сломать» сохранение.
            if steered and self.rng.random() < self.steer_cost:
                wrong = [a for a in S.ACTIONS if a not in w.allowed_actions]
                return str(self.rng.choice(wrong))
            return sorted(w.allowed_actions)[0]

        if self.rng.random() < min(p, 1.0):
            return sorted(w.allowed_actions)[0]
        # Иначе — действие, верное в ДРУГОЙ ветке: ровно та ошибка, что изучаем.
        other = task.branch(1 - branch)
        return sorted(other.allowed_actions)[0]


# --------------------------------------------------------------------------
# Проверка 1: различает ли модель ответы
# --------------------------------------------------------------------------


def check_h1_mock(tasks: list[S.PairedTask], agent: MockAgent, seed: int) -> P.ProbeResult:
    acts, labels, ids = [], [], []
    inv_acts, inv_labels, inv_ids = [], [], []

    for t in tasks:
        for branch in (0, 1):
            if t.pair_type == "changing":
                acts.append(agent.activation(branch))
                labels.append(branch)
                ids.append(t.task_id)
            elif t.pair_type == "invariant":
                # Смысл один и тот же -> активация НЕ должна зависеть от ветки.
                inv_acts.append(agent.activation(0))
                inv_labels.append(branch)
                inv_ids.append(t.task_id)

    return P.run_probe(
        acts, labels, ids, layer=-1, window_kind="mock", seed=seed,
        invariant_activations=inv_acts or None,
        invariant_labels=inv_labels or None,
        invariant_task_ids=inv_ids or None,
    )


# --------------------------------------------------------------------------
# Проверки 2 и 3: использует ли различие / чинит ли steering
# --------------------------------------------------------------------------


def run_condition(
    tasks: list[S.PairedTask],
    agent: MockAgent,
    steered: bool,
    n_trials: int = 5,
) -> tuple[list[M.PairOutcome], dict[str, list[bool]]]:
    """Один метод на одном наборе. Возвращает исходы пар и успехи по задачам."""
    outcomes: list[M.PairOutcome] = []
    succ_by_task: dict[str, list[bool]] = {}

    for t in tasks:
        for _ in range(n_trials):
            a0 = agent.act(t, 0, steered)
            a1 = agent.act(t, 1, steered)
            ok0 = S.score_action(a0, t.world_0)
            ok1 = S.score_action(a1, t.world_1)
            outcomes.append(
                M.PairOutcome(
                    task_id=t.task_id,
                    pair_type=t.pair_type,
                    action_ok_0=ok0,
                    action_ok_1=ok1,
                    action_changed=(a0 != a1),
                    success_0=ok0,
                    success_1=ok1,
                )
            )
            succ_by_task.setdefault(t.task_id, []).extend([ok0, ok1])

    return outcomes, succ_by_task


def report_method(
    name: str,
    outcomes: list[M.PairOutcome],
    succ_by_task: dict[str, list[bool]],
    base_succ_by_task: dict[str, list[bool]] | None,
    seed: int,
) -> M.MethodReport:
    flat = [s for v in succ_by_task.values() for s in v]
    per_task = [succ_by_task[t] for t in sorted(succ_by_task)]

    paired = None
    delta = M.EpisodeDelta(0, 0, len(flat))
    if base_succ_by_task is not None:
        paired = M.paired_bootstrap(base_succ_by_task, succ_by_task, seed=seed)
        base_flat = [s for t in sorted(base_succ_by_task) for s in base_succ_by_task[t]]
        if len(base_flat) == len(flat):
            delta = M.episode_delta(base_flat, flat)

    return M.MethodReport(
        name=name,
        success=M.episode_success(flat),
        pass_k={1: M.pass_hat_k(per_task, 1), 3: M.pass_hat_k(per_task, 3)},
        effect=M.target_effect(outcomes),
        switch_rate=M.switch_rate(outcomes),
        delta=delta,
        paired=paired,
        cost=None,
    )


# --------------------------------------------------------------------------
# Реальный прогон
# --------------------------------------------------------------------------


def run_real(args: argparse.Namespace) -> int:
    """Скелет прогона на настоящей модели.

    Намеренно останавливается на явных TODO, а не делает вид, что работает:
    формат tool-call, парсер действия и границы окон зависят от конкретной
    модели и должны быть проверены руками (см. steering.locate_windows).
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: нужны torch и transformers. pip install torch transformers")
        return 2

    import steering as ST

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model.eval()
    print(f"loaded {args.model}: {ST.n_layers(model)} layers")

    tasks = S.build_pilot_set(args.n_tasks, seed=args.seed)
    rep = S.audit(tasks)
    print(rep)
    if not rep.ok:
        print("audit failed -> исправить данные до прогона")
        return 1

    splits = S.split_tasks(tasks, seed=args.seed)
    print({k: len(v) for k, v in splits.items()})

    raise NotImplementedError(
        "Перед реальным прогоном закрыть три пункта:\n"
        "  1) формат tool message и tool-call под выбранный chat template;\n"
        "  2) simulator.parse_action -> реальный парсер действий модели;\n"
        "  3) проверить окна из steering.locate_windows ГЛАЗАМИ: напечатать\n"
        "     токены по индексам окна и убедиться, что попали после ответа.\n"
        "Затем: собрать активации (ActivationRecorder) -> probe.sweep_layers на\n"
        "train -> contrastive_vector -> подбор alpha/слоя на VALIDATION ->\n"
        "заморозить -> прогон с sham-контролями -> test один раз."
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def run_mock(args: argparse.Namespace) -> int:
    t0 = time.time()
    tasks = S.build_pilot_set(args.n_tasks, seed=args.seed)

    print("=" * 72)
    print("MOCK RUN - проверка обвязки и метрик, НЕ результат исследования")
    print("=" * 72)

    rep = S.audit(tasks)
    print("\n[audit]")
    print(rep)
    if not rep.ok:
        return 1

    splits = S.split_tasks(tasks, seed=args.seed)
    print("\n[splits]", {k: len(v) for k, v in splits.items()})

    agent = MockAgent(
        seed=args.seed,
        separability=args.separability,
        use_rate=args.use_rate,
        steer_gain=args.steer_gain,
        steer_cost=args.steer_cost,
    )

    print("\n[H1] различает ли модель ответы (probe на train)")
    h1 = check_h1_mock(splits["train"], agent, args.seed)
    print("  " + str(h1))
    probe_ok = h1.beats_baseline and not h1.suspected_shortcut
    if not probe_ok:
        print("  -> " + P.gap_report(False, False, False))

    print("\n[H2] использует ли различие (база, без вмешательства)")
    base_out, base_succ = run_condition(splits["val"], agent, steered=False,
                                        n_trials=args.n_trials)
    base = report_method("baseline", base_out, base_succ, None, args.seed)
    print(base.summary())

    changing = [o for o in base_out if o.pair_type == "changing"]
    acts_on_diff = float(np.mean([o.behaved_correctly for o in changing]))
    print("\n  " + P.gap_report(probe_ok, True, acts_on_diff > 0.9))

    print("\n[H3] чинит ли steering (val)")
    st_out, st_succ = run_condition(splits["val"], agent, steered=True,
                                    n_trials=args.n_trials)
    steered = report_method("steering", st_out, st_succ, base_succ, args.seed)
    print(steered.summary())

    print("\n[H3 контроль] sham: вмешательство без целевого эффекта")
    sham_agent = MockAgent(seed=args.seed + 1, separability=args.separability,
                           use_rate=args.use_rate, steer_gain=0.0,
                           steer_cost=args.steer_cost)
    sh_out, sh_succ = run_condition(splits["val"], sham_agent, steered=True,
                                    n_trials=args.n_trials)
    sham = report_method("sham", sh_out, sh_succ, base_succ, args.seed)
    print(sham.summary())

    print("\n" + "=" * 72)
    print("ИТОГ (mock)")
    print("=" * 72)
    ng = M.normalized_gain(steered.success, base.success)
    print(f"  normalized_gain (steering vs base) = {ng:+.3f}")
    print(f"  sensitivity  {base.effect.sensitivity:.3f} -> {steered.effect.sensitivity:.3f}")
    print(f"  specificity  {base.effect.specificity:.3f} -> {steered.effect.specificity:.3f}")
    if steered.effect.specificity < base.effect.specificity - 0.05:
        print("  ВНИМАНИЕ: специфичность упала — чувствительность куплена "
              "ложными переключениями, это не улучшение")
    if steered.paired and sham.paired:
        print(f"  steering vs base {steered.paired}")
        print(f"  sham     vs base {sham.paired}")
        if not steered.paired.excludes_zero:
            print("  ВНИМАНИЕ: интервал включает ноль")
        if sham.paired.excludes_zero:
            print("  ВНИМАНИЕ: sham тоже даёт эффект -> вмешательство неспецифично")

    Manifest(
        mode="mock",
        model=None,
        seed=args.seed,
        n_tasks=len(tasks),
        chat_template_hash=None,
        decoding={"mock": True, "n_trials": args.n_trials},
        layer=None,
        alpha=None,
        window_kind=None,
        git_commit=git_commit(),
        python=platform.python_version(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        data_hash=data_hash(tasks),
    ).save(Path(args.out) / "manifest.json")

    print(f"\nmanifest -> {Path(args.out) / 'manifest.json'}  ({time.time() - t0:.1f}s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="прогон без модели")
    ap.add_argument("--model", type=str, default=None, help="HF model id")
    ap.add_argument("--n-tasks", type=int, default=120)
    ap.add_argument("--n-trials", type=int, default=5, help="повторов на задачу (для pass^k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/dev")
    # Параметры синтетического агента — только для --mock.
    ap.add_argument("--separability", type=float, default=2.0)
    ap.add_argument("--use-rate", type=float, default=0.35)
    ap.add_argument("--steer-gain", type=float, default=0.35)
    ap.add_argument("--steer-cost", type=float, default=0.10)
    args = ap.parse_args()

    if args.mock:
        return run_mock(args)
    if args.model:
        return run_real(args)
    ap.error("нужен --mock или --model")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
