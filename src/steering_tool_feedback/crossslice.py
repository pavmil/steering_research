"""H4′: сравнение условного и глобального срезов по двум прогонам (docs/HYPOTHESES.md H4′).

Δ_s(m) = успех_s(m) − успех_s(LoRA) — парная разность по задачам среза s; I(m) = Δ_усл − Δ_глоб,
интервал — из независимых bootstrap-выборок двух срезов, уровень по Холму на число методов
(docs/METRICS.md «Глобальный срез»). Ничего не подбирает и моделей не запускает: читает решения,
которые прогоны уже записали в results/runs/<run>/decisions/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import gates as G
from . import metrics as M


def load_decisions(run_dir: str | Path, split: str, method: str) -> list[dict[str, Any]]:
    """Решения метода на сплите (как их пишет Experiment.evaluate)."""
    path = Path(run_dir) / "decisions" / f"{split.replace(':', '-')}__{method.replace(':', '-')}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}: метод {method!r} не оценивался на {split!r} в этом прогоне")
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def by_task(recs: list[dict]) -> dict[str, list[bool]]:
    d: dict[str, list[bool]] = defaultdict(list)
    for r in recs:
        if r["pair_type"] in M.MAIN_TYPES:
            d[r["task_id"]].append(bool(r["ok"]))
    return d


def target_effect(recs: list[dict]) -> M.TargetEffect:
    pairs: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        if r["pair_type"] in M.MAIN_TYPES:
            pairs[(r["task_id"], r.get("trial", 0))][r["branch"]] = r
    out = [M.PairOutcome(t, br[0]["pair_type"], br[0]["ok"], br[1]["ok"], br[0]["action"] != br[1]["action"],
                         br[0]["ok"], br[1]["ok"]) for (t, _), br in sorted(pairs.items()) if 0 in br and 1 in br]
    return M.target_effect(out)


def compare(cond_run: str | Path, glob_run: str | Path, split: str = "transfer:seen",
            methods: dict[str, tuple[str, str]] | tuple[str, ...] = ("linmap", "linmap_prompt"),
            ref: str = "lora", tol: float = 0.05, margin: float = 0.05, n_boot: int = 10_000,
            alpha: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """Таблица H4′ и вердикт. methods — имена или {метка: (имя в условном прогоне, имя в глобальном)},
    например {"linmap_prompt": ("linmap_prompt@val", "linmap_prompt@val")}, если правило отбора — один val.
    tol — допуск падения специфичности / сохранности относительно базы (steer.max_specificity_drop)."""
    pairs = {m: (m, m) for m in methods} if not isinstance(methods, dict) else dict(methods)
    level = alpha / max(1, len(pairs))                         # Холм: первый шаг — alpha / k
    runs = {"cond": Path(cond_run), "glob": Path(glob_run)}
    base = {s: load_decisions(d, split, "baseline") for s, d in runs.items()}
    refr = {s: load_decisions(d, split, ref) for s, d in runs.items()}
    rows = []
    for label, names in pairs.items():
        rec = {s: load_decisions(runs[s], split, n) for s, n in zip(("cond", "glob"), names)}
        est = M.interaction_bootstrap(by_task(refr["cond"]), by_task(rec["cond"]), by_task(refr["glob"]),
                                      by_task(rec["glob"]), n_boot=n_boot, alpha=level, seed=seed)
        row: dict[str, Any] = {"method": label, "cond_name": names[0], "glob_name": names[1]}
        for s, d in (("cond", est.delta_a), ("glob", est.delta_b)):
            eb, em = target_effect(base[s]), target_effect(rec[s])
            row.update({f"delta_{s}": d.mean_diff, f"delta_{s}_ci": [d.ci_low, d.ci_high],
                        f"sens_{s}": em.sensitivity, f"spec_{s}": em.specificity,
                        f"spec_drop_{s}": eb.specificity - em.specificity,
                        f"eligible_{s}": eb.specificity - em.specificity <= tol + 1e-12, f"n_{s}": d.n_tasks})
        # строгая форма (как в H4): на условном не хуже LoRA, на глобальном значимо хуже
        row.update(I=est.mean, ci_low=est.ci_low, ci_high=est.ci_high,
                   strict=bool(est.delta_a.ci_low >= 0 and est.delta_b.ci_high < 0))
        rows.append(row)
    return {"split": split, "level": 1 - level, "ref": ref, "rows": rows,
            "verdict": G.h4_prime(rows, preliminary=split != "test", margin=margin)}
