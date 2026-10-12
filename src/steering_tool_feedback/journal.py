"""Журнал исследования: docs/JOURNAL.md — вся история работы в одном файле.

Записи только дописываются, старые не редактируются. Два вида записей:
  * прогон — пишется exp.report(...): что запускали, что поменяли, числа,
    вердикты, флаги протокола и вывод человека;
  * заметка — journal.note("...") или `stf-note "..."`: решения, идеи,
    обсуждения, всё, что не прогон.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

HEADER = """# Журнал исследования

Хронология всех прогонов, решений и заметок. Записи только дописываются —
старые не редактировать: если что-то оказалось неверным, добавить новую
запись с исправлением и ссылкой на старую.

Прогоны пишет `exp.report(...)` в конце ноутбука, заметки — `journal.note(...)`
или `stf-note "..."`. Подробности прогона — в `results/runs/<run_id>/`.
"""


def path(root: Path) -> Path:
    """docs/JOURNAL.md; STF_JOURNAL переопределяет (тесты, черновые прогоны)."""
    import os

    env = os.environ.get("STF_JOURNAL")
    return Path(env) if env else root / "docs" / "JOURNAL.md"


def _ensure(root: Path) -> Path:
    p = path(root)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(HEADER)
    return p


def _append(root: Path, block: str) -> None:
    p = _ensure(root)
    with p.open("a") as f:
        f.write("\n---\n\n" + block.rstrip() + "\n")


def note(text: str, tag: str = "заметка", root: Path | None = None) -> str:
    from .store import find_repo_root

    root = root or find_repo_root()
    stamp = time.strftime("%Y-%m-%d %H:%M")
    block = f"## {stamp} · {tag}\n\n{text.strip()}\n"
    _append(root, block)
    return block


def run_entry(root: Path, f: dict[str, Any]) -> str:
    """f — поля, собранные Experiment.report()."""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    L = [f"## {stamp} · прогон `{f['run_id']}`", ""]
    L.append(f"**Гипотеза:** {f['hypothesis']} · **Ноутбук:** {f['notebook'] or '—'} · "
             f"**Модель:** {f['model']} · **Режим решения:** {f['mode']}")
    L.append(f"**Стадии:** {', '.join(f['stages']) or '—'} · **git:** `{f['git']}`"
             + (" (есть незакоммиченные изменения)" if f["dirty"] else ""))
    if f.get("notes"):
        L += ["", f"**Зачем:** {f['notes']}"]
    if f["diff"]:
        L += ["", "**Отличия от конфига по умолчанию:** " +
              "; ".join(f"`{k}` = {v[1]!r}" for k, v in sorted(f["diff"].items()))]
    if f["flags"]:
        L += ["", "**⚠ Флаги протокола:**"] + [f"- {x}" for x in f["flags"]]
    if f["table"]:
        L += ["", f["table"]]
    L += ["", "**Вердикты (автоматически, по правилам HYPOTHESES.md):**"]
    L += [f"- {h}: {v.label} — {v.reason}" for h, v in f["verdicts"].items()]
    L += ["", f"**Вывод:** {f['conclusion'].strip() if f['conclusion'] else '_не записан_'}"]
    if f.get("next_step"):
        L += ["", f"**Следующий шаг:** {f['next_step'].strip()}"]
    block = "\n".join(L) + "\n"
    _append(root, block)
    return block
