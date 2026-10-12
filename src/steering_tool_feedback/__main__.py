"""Запуск без консольных команд — работает, даже если их папки нет в PATH.

    python3 -m steering_tool_feedback doctor
    python3 -m steering_tool_feedback windows --tokenizer <...>
    python3 -m steering_tool_feedback run --mock
    python3 -m steering_tool_feedback note "текст"

Эквивалентно stf-doctor / stf-windows / stf-run.
"""

from __future__ import annotations

import sys

COMMANDS = {
    "doctor": "steering_tool_feedback.doctor",
    "windows": "steering_tool_feedback.inspect_windows",
    "run": "steering_tool_feedback.run_experiment",
    "note": "steering_tool_feedback.run_experiment:note_main",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands:", ", ".join(COMMANDS))
        return 2
    import importlib

    target = COMMANDS[sys.argv[1]]
    modname, _, fn = target.partition(":")
    mod = importlib.import_module(modname)
    return int(getattr(mod, fn or "main")(sys.argv[2:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
