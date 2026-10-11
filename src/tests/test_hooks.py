"""Тесты механики хуков. ТРЕБУЕТ torch + transformers.

В облачном контейнере проекта torch не установился (прокси блокирует индекс
PyTorch, с PyPI не укладывается в таймаут), поэтому steering.py проверен только
на numpy-частях. ЭТОТ ФАЙЛ НАДО ПРОГНАТЬ ПЕРВЫМ на своей машине, до любых
экспериментов: молча не сработавший хук даёт "нулевой эффект", который легко
принять за опровержение H3.

    pip install torch transformers
    python3 src/tests/test_hooks.py

Используется крошечная случайная модель, вес ~2MB, скачивается с HF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"SKIP: {e}\nУстановить: pip install torch transformers")
        return 0

    import steering as ST

    print(f"loading {TINY}")
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()

    L = ST.n_layers(model)
    hidden = model.config.hidden_size
    print(f"  layers={L} hidden={hidden}")
    check("layers found", L > 0)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    batch = tok("the tool reported that the item is out of stock", return_tensors="pt")
    seq_len = batch["input_ids"].shape[1]
    print(f"  seq_len={seq_len}")

    layer = L // 2

    # ---------------------------------------------------------------
    print("\n[recorder] съём активаций")
    with ST.ActivationRecorder(model, layers=[layer]) as rec:
        with torch.no_grad():
            model(**batch)
        w = ST.PositionWindow(0, seq_len, "post_message")
        acts = rec.pooled(w)
    check("recorded requested layer", layer in acts)
    check("shape is [hidden]", acts[layer].shape == (hidden,), str(acts[layer].shape))
    check("finite", bool(np.isfinite(acts[layer]).all()))

    # Окно шире последовательности не должно падать — должно обрезаться.
    with ST.ActivationRecorder(model, layers=[layer]) as rec:
        with torch.no_grad():
            model(**batch)
        clipped = rec.pooled(ST.PositionWindow(0, seq_len + 50, "post_message"))
    check("window clipped to sequence", clipped[layer].shape == (hidden,))

    # ---------------------------------------------------------------
    print("\n[hook] вмешательство меняет выход")
    vec = ST.SteeringVector(
        layer=layer,
        vector=np.ones(hidden, dtype=np.float32),
        window_kind="post_message",
        provenance={"test": True},
    )
    window = ST.PositionWindow(max(0, seq_len - 3), seq_len, "post_message")

    with torch.no_grad():
        clean = model(**batch).logits.clone()

    cfg = ST.SteerConfig(alpha=5.0, window=window, normalize=True)
    with ST.steering(model, vec, cfg) as hook:
        with torch.no_grad():
            steered = model(**batch).logits.clone()
    check("hook fired", hook.n_applied > 0, f"n_applied={hook.n_applied}")
    check("applied exactly window size", hook.n_applied == len(window),
          f"{hook.n_applied} vs {len(window)}")
    check("logits changed", not torch.allclose(clean, steered))

    # ---------------------------------------------------------------
    print("\n[hook] alpha=0 — строгий no-op")
    with ST.steering(model, vec, ST.SteerConfig(alpha=0.0, window=window)) as hook0:
        with torch.no_grad():
            zero = model(**batch).logits.clone()
    check("alpha=0 identical", torch.allclose(clean, zero, atol=1e-6))
    check("alpha=0 applies nothing", hook0.n_applied == 0)

    print("\n[hook] enabled=False — no-op")
    with ST.steering(
        model, vec, ST.SteerConfig(alpha=5.0, window=window, enabled=False)
    ) as hoff:
        with torch.no_grad():
            off = model(**batch).logits.clone()
    check("disabled identical", torch.allclose(clean, off, atol=1e-6))
    check("disabled applies nothing", hoff.n_applied == 0)

    # ---------------------------------------------------------------
    print("\n[hook] позиции вне окна не затронуты")
    # Вмешиваемся только в последнюю позицию: логиты на РАННИХ позициях
    # не должны измениться (причинное внимание смотрит только назад).
    late = ST.PositionWindow(seq_len - 1, seq_len, "pre_action")
    with ST.steering(model, vec, ST.SteerConfig(alpha=5.0, window=late)):
        with torch.no_grad():
            tail = model(**batch).logits.clone()
    check("earlier positions unchanged",
          torch.allclose(clean[:, : seq_len - 1, :], tail[:, : seq_len - 1, :], atol=1e-5))
    check("last position changed",
          not torch.allclose(clean[:, -1, :], tail[:, -1, :]))

    # ---------------------------------------------------------------
    print("\n[hook] снятие хука восстанавливает модель")
    with torch.no_grad():
        after = model(**batch).logits.clone()
    check("restored after detach", torch.allclose(clean, after, atol=1e-6))

    # ---------------------------------------------------------------
    print("\n[hook] generate() с KV-кэшем")
    cfg_gen = ST.SteerConfig(alpha=3.0, window=window)
    with ST.steering(model, vec, cfg_gen) as hg:
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=5, do_sample=False)
    check("generate ran", out.shape[1] == seq_len + 5, str(out.shape))
    check("hook fired during prefill", hg.n_applied > 0, f"n_applied={hg.n_applied}")
    check("prefill-only by default", hg.n_applied == len(window),
          f"{hg.n_applied} vs {len(window)} — вмешались и в сгенерированные токены")

    print("\n[hook] apply_to_generated=True продолжает на декодировании")
    w_gen = ST.PositionWindow(window.start, window.end, "post_message",
                              apply_to_generated=True)
    with ST.steering(model, vec, ST.SteerConfig(alpha=3.0, window=w_gen)) as hg2:
        with torch.no_grad():
            model.generate(**batch, max_new_tokens=5, do_sample=False)
    check("applied to generated tokens too", hg2.n_applied > len(window),
          f"n_applied={hg2.n_applied}, window={len(window)}")

    # ---------------------------------------------------------------
    print("\n[hook] dose-response монотонен по |alpha|")
    dists = []
    for a in (0.0, 1.0, 3.0, 9.0):
        with ST.steering(model, vec, ST.SteerConfig(alpha=a, window=window)):
            with torch.no_grad():
                lg = model(**batch).logits
        dists.append(float((lg - clean).norm()))
    print(f"  distances: {[round(d, 4) for d in dists]}")
    check("monotone in alpha", all(dists[i] <= dists[i + 1] + 1e-4
                                  for i in range(len(dists) - 1)), str(dists))

    # ---------------------------------------------------------------
    print("\n[hook] sham того же размера даёт сопоставимое возмущение")
    sham = ST.random_direction_like(vec, seed=0)
    with ST.steering(model, sham, ST.SteerConfig(alpha=5.0, window=window)) as hs:
        with torch.no_grad():
            sham_lg = model(**batch).logits
    check("sham fired", hs.n_applied == len(window))
    d_real = float((steered - clean).norm())
    d_sham = float((sham_lg - clean).norm())
    print(f"  perturbation: real={d_real:.4f} sham={d_sham:.4f}")
    check("sham perturbs the model too (so it is a fair control)", d_sham > 0)

    # ---------------------------------------------------------------
    print("\n[windows] locate_windows на chat template")
    if getattr(tok, "chat_template", None):
        msgs = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Order a blue mug."},
            {"role": "assistant", "content": '{"tool":"check_stock"}'},
            {"role": "tool", "content": '{"item":"blue mug","in_stock":false}'},
        ]
        try:
            ws = ST.locate_windows(tok, msgs, essential_field="in_stock")
            print(f"  windows: { {k: (w.start, w.end) for k, w in ws.items()} }")
            check("post_message located", "post_message" in ws)
            check("windows are non-empty", all(len(w) > 0 for w in ws.values()))
            check("windows ordered sanely",
                  ws["post_message"].end >= ws["post_message"].start)
        except Exception as e:  # noqa: BLE001
            print(f"  note: locate_windows failed on this template ({e})")
            print("  -> ПРОВЕРИТЬ ГЛАЗАМИ на реальной модели, окна критичны")
    else:
        print("  skip: у tiny-модели нет chat_template")
        print("  -> на реальной модели ОБЯЗАТЕЛЬНО напечатать токены окна и проверить")

    print("\n" + "=" * 60)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
