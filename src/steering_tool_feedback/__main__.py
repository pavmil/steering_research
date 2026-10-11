"""Запуск без консольных команд — работает, даже если их папки нет в PATH.

    python3 -m steering_tool_feedback doctor
    python3 -m steering_tool_feedback windows --tokenizer <...>
    python3 -m steering_tool_feedback run --mock

Эквивалентно stf-doctor / stf-windows / stf-run.
"""

from __future__ import annotations

import sys

COMMANDS = {
    "doctor": "steering_tool_feedback.doctor",
    "windows": "steering_tool_feedback.inspect_windows",
    "run": "steering_tool_feedback.run_experiment",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands:", ", ".join(COMMANDS))
        return 2
    import importlib

    mod = importlib.import_module(COMMANDS[sys.argv[1]])
    return int(mod.main(sys.argv[2:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
