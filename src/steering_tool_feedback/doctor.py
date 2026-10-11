"""Какой Python запущен и какие зависимости он видит.

    stf-doctor          # или: python -m steering_tool_feedback.doctor

Частая причина «модуль не найден»: пакет поставлен в один интерпретатор
(venv, conda, pyenv), а скрипт запускается другим. Здесь видно, какой именно
запущен, откуда импортируется каждый пакет и установлен ли сам проект.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import site
import sys

CORE = ("numpy", "scipy", "sklearn")
MODEL = ("torch", "transformers", "accelerate")
DEV = ("pytest",)
OPTIONAL = ("peft", "trl")


def _probe(name: str) -> tuple[bool, str, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "", ""
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        return False, "", f"import error: {e}"
    return True, str(getattr(mod, "__version__", "?")), str(getattr(mod, "__file__", "") or "")


def _check_scripts_on_path() -> None:
    """Консольные команды stf-* видны? При `pip install --user` они попадают в
    ~/.local/bin (Linux) или ~/Library/Python/3.x/bin (macOS) — часто не в PATH."""
    import os
    import shutil
    import sysconfig

    found = shutil.which("stf-doctor")
    if found:
        print(f"commands    on PATH ({os.path.dirname(found)})")
        return
    candidates = {sysconfig.get_path("scripts")}
    try:
        candidates.add(sysconfig.get_path("scripts", f"{os.name}_user"))
    except KeyError:
        pass
    if sys.platform == "darwin":
        try:
            candidates.add(sysconfig.get_path("scripts", "osx_framework_user"))
        except KeyError:
            pass
    where = [c for c in candidates if c and os.path.exists(os.path.join(c, "stf-doctor"))]
    print("commands    stf-* NOT on PATH")
    if where:
        print(f"            installed in: {where[0]}")
        print(f"            -> export PATH=\"{where[0]}:$PATH\"   (добавить в ~/.zshrc)")
    print("            -> или без PATH: python3 -m steering_tool_feedback doctor|windows|run")


def main(argv: list[str] | None = None) -> int:
    print(f"python      {platform.python_version()}  {sys.executable}")
    print(f"prefix      {sys.prefix}")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"venv        {'yes' if in_venv else 'no (global interpreter)'}")
    try:
        print(f"user site   {site.getusersitepackages()}  enabled={site.ENABLE_USER_SITE}")
    except Exception:  # noqa: BLE001
        pass

    ok, ver, path = _probe("steering_tool_feedback")
    print(f"\nproject     {'installed ' + ver if ok else 'NOT importable'}  {path}")
    if not ok:
        print("            -> pip install -e .   (из корня репозитория, ЭТИМ же python)")
    else:
        _check_scripts_on_path()

    missing_core = []
    for group, names in (("core", CORE), ("model", MODEL), ("dev", DEV), ("optional", OPTIONAL)):
        print(f"\n[{group}]")
        for n in names:
            ok, ver, where = _probe(n)
            status = f"{ver:<12} {where}" if ok else (where or "missing")
            print(f"  {n:<14} {status}")
            if group == "core" and not ok:
                missing_core.append(n)

    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
        print(f"\ntorch device  {dev}")
    except Exception:  # noqa: BLE001
        pass

    print()
    if missing_core:
        print(f"MISSING core deps: {missing_core} -> {sys.executable} -m pip install -e .")
        return 1
    print("core deps OK. Для моделей: pip install -e '.[model]'; для тестов: '.[dev]'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
