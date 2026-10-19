"""LoRA/SFT — обучаемый baseline для H4 (docs/HYPOTHESES.md).

Правила честности (EXPERIMENT_PLAN, AGENTS.md):
  * данные те же, что у вектора steering: меняющие пары train, обе ветки;
    цель — верный вызов инструмента в формате шаблона модели;
  * гиперпараметры (lr, эпохи) выбираются на validation тем же правилом, что
    и steering: максимум успеха при падении специфичности ≤ допуска;
  * оценочная модель не меняется: обучается отдельная копия, адаптер
    сохраняется и подключается к оценочной модели только на время решения.

Цена обучения (секунды, число параметров, примеров) пишется в результат —
это слой 4 METRICS.md.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any


def train_lora(model: Any, device: str, examples: list[tuple[list[int], list[int]]], cfg: Any,
               lr: float, epochs: int, out_dir: str) -> dict[str, Any]:
    """SFT адаптера: loss только на токенах цели. Возвращает сведения о цене обучения."""
    import torch
    from peft import LoraConfig, get_peft_model

    if not examples:
        raise ValueError("no training examples")
    t0 = time.perf_counter()
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[cfg.train_dtype]
    torch.manual_seed(cfg.seed)
    base = copy.deepcopy(model).to(dtype=dtype)
    pm = get_peft_model(base, LoraConfig(r=cfg.rank, lora_alpha=cfg.lora_alpha,
                                         lora_dropout=cfg.dropout,
                                         target_modules=list(cfg.target_modules),
                                         task_type="CAUSAL_LM"))
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(cfg.seed)
    epoch_losses: list[float] = []
    n_tokens = 0
    with torch.enable_grad():
        for _ in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            losses = []
            for j, i in enumerate(order):
                prompt_ids, target = examples[i]
                ids = torch.tensor([prompt_ids + target], dtype=torch.long, device=device)
                # Логиты только там, где предсказываются токены цели (словари ~150–250k).
                out = pm(input_ids=ids, logits_to_keep=len(target) + 1)
                logits = out.logits[0, :-1].float()
                tgt = torch.tensor(target, dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits, tgt)
                (loss / cfg.grad_accum).backward()
                losses.append(float(loss.detach()))
                n_tokens += len(prompt_ids) + len(target)
                if (j + 1) % cfg.grad_accum == 0 or j == len(order) - 1:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            epoch_losses.append(float(sum(losses) / len(losses)))
    pm.save_pretrained(out_dir)
    n_trainable = int(sum(p.numel() for p in params))
    del pm, base, opt
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return {"train_seconds": round(time.perf_counter() - t0, 1), "n_examples": len(examples),
            "epochs": epochs, "lr": lr, "n_trainable": n_trainable, "train_tokens": n_tokens,
            "epoch_losses": [round(x, 4) for x in epoch_losses]}
