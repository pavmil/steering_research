"""Распечатать окна позиций на реальном токенизаторе — проверка глазами.

Обязательный шаг перед первым экспериментом на новой модели или новом шаблоне
(AGENTS.md). Скачивать веса модели не нужно, хватает папки с токенизатором:

    stf-windows --tokenizer ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit
    stf-windows --tokenizer Qwen/Qwen2.5-7B-Instruct --per-type 2 --show-prompt

Для каждой выбранной задачи печатаются обе ветки: индексы окон, токены внутри
окна (⟦ … ⟧) и по краям, текст окна. Ошибка построения окна — это ошибка, а не
предупреждение: на таком промпте вмешательство попадёт не туда.
"""

from __future__ import annotations

import argparse
import sys

from . import simulator as S


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="HF id или локальная папка с токенизатором")
    ap.add_argument("--per-type", type=int, default=1, help="сколько задач каждого типа пары")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator", default=S.GENERATOR_VERSION, choices=sorted(S.GENERATORS),
                    help="семейство задач (g1 — глобальный срез H4′)")
    ap.add_argument("--k-post", type=int, default=4)
    ap.add_argument("--context", type=int, default=3, help="токенов контекста по краям окна")
    ap.add_argument("--no-tools", action="store_true", help="не передавать схемы инструментов")
    ap.add_argument("--show-prompt", action="store_true", help="напечатать весь отрендеренный промпт")
    ap.add_argument("--all", action="store_true",
                    help="построить окна для ВСЕХ задач пилота и вывести только ошибки")
    args = ap.parse_args(argv)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("нужен transformers: pip install -e '.[model]'", file=sys.stderr)
        return 2

    from .windows import WindowError, describe_windows, prepare_prompt

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: {type(tok).__name__}  vocab={len(tok)}  fast={tok.is_fast}")
    if not getattr(tok, "chat_template", None):
        print("ERROR: у токенизатора нет chat_template", file=sys.stderr)
        return 1

    tasks = S.build_pilot_set(120, seed=args.seed, version=args.generator)
    tools = None if args.no_tools else S.tools_schema()

    if args.all:
        bad = 0
        for t in tasks:
            for b in (0, 1):
                try:
                    prepare_prompt(tok, t.prompt(b), essential_field=t.essential_field(b),
                                   tools=tools, k_post=args.k_post)
                except WindowError as e:
                    bad += 1
                    print(f"  FAIL {t.task_id}/{b} ({t.pair_type}): {e}")
        print(f"\n{2 * len(tasks) - bad}/{2 * len(tasks)} prompts OK")
        return 1 if bad else 0

    picked: list[S.PairedTask] = []
    for kind in (*S.MAIN_TYPES, "stress"):
        picked += [t for t in tasks if t.pair_type == kind][: args.per_type]

    failed = 0
    for t in picked:
        for b in (0, 1):
            w = t.branch(b)
            print("\n" + "=" * 78)
            print(f"{t.task_id}  {t.pair_type}  branch {b}  ({w.world_id})")
            print(f"allowed next actions: {sorted(w.allowed_actions)}")
            print("=" * 78)
            try:
                p = prepare_prompt(tok, t.prompt(b), essential_field=t.essential_field(b),
                                   tools=tools, k_post=args.k_post)
            except WindowError as e:
                failed += 1
                print(f"WINDOW ERROR: {e}")
                continue
            if args.show_prompt:
                print(p.text)
                print("-" * 78)
            print(describe_windows(tok, p, context=args.context))

    print()
    if failed:
        print(f"{failed} prompt(s) failed — исправить шаблон/данные до прогона")
        return 1
    print("Проверь глазами: tool_content = текст ответа, field_read = ключ+значение,")
    print("post_message = хвост ответа и закрывающая обёртка, pre_action = generation prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
