"""Прогон эксперимента из командной строки — то же, что ноутбук, без Jupyter.

    stf-run --mock                                    # весь пайплайн на mock, секунды
    stf-run --config configs/h2_baseline.yaml --until baseline
    stf-run --model Qwen/Qwen3-1.7B --name h1-probe --until probe
    stf-run --config configs/x.yaml --set steer.alphas=[1,2] --set data.n_tasks=200

Основной путь — ноутбуки (notebooks/), там же пишется вывод в журнал.
Здесь отчёт тоже пишется в docs/JOURNAL.md, с выводом из --conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .pipeline import STAGES, Experiment


def _parse_set(items: list[str]) -> dict:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, help="YAML с конфигом (configs/*.yaml)")
    ap.add_argument("--mock", action="store_true", help="синтетический бэкенд, без модели")
    ap.add_argument("--model", type=str, help="HF id или локальная папка модели")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--hypothesis", type=str, default=None)
    ap.add_argument("--until", choices=STAGES, default="test")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="переопределить параметр, напр. steer.alphas=[1,2]")
    ap.add_argument("--results-dir", type=str, default=None, help="по умолчанию results/")
    ap.add_argument("--conclusion", type=str, default="", help="вывод для журнала")
    ap.add_argument("--no-journal", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    overrides = _parse_set(args.set)
    if args.mock:
        overrides["model.name"] = "mock"
    if args.model:
        overrides["model.name"] = args.model
    cfg = ExperimentConfig.make(
        name=args.name or (None if args.config else "cli"),
        hypothesis=args.hypothesis, overrides=overrides, base=args.config,
    )

    exp = Experiment(cfg, results_dir=Path(args.results_dir) if args.results_dir else None,
                     verbose=not args.quiet)
    exp.run(args.until)
    print()
    print(exp.results_table().round(3).to_string(index=False))
    md = exp.report(conclusion=args.conclusion, journal=not args.no_journal)
    if not args.quiet:
        print()
        print(md)
    print(f"\nrun dir: {exp.dir}")
    return 0


def note_main(argv: list[str] | None = None) -> int:
    """stf-note "текст" [--tag решение] — запись в docs/JOURNAL.md."""
    from . import journal

    ap = argparse.ArgumentParser(description="Добавить заметку в docs/JOURNAL.md")
    ap.add_argument("text")
    ap.add_argument("--tag", default="заметка")
    a = ap.parse_args(argv)
    print(journal.note(a.text, a.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
