"""Автоматические вердикты по гипотезам — по правилам из docs/HYPOTHESES.md.

Вердикт — подсказка, а не вывод исследования: итоговую формулировку пишет
человек в exp.report(conclusion=...). Но вердикты считаются одинаково для всех
прогонов, поэтому их нельзя «подогнать» в конкретном ноутбуке.

Статусы: support | against | inconclusive | not_run. Суффикс «(val)» —
предварительно, по validation, до открытия test.

ALT — не гипотеза, а условие провала тезиса из CONTEXT.md §3: дешёвая альтернатива
(промпт, пересказ, явное состояние) даёт то же качество, что и steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Verdict:
    status: str
    reason: str
    preliminary: bool = False

    @property
    def label(self) -> str:
        icon = {"support": "✔", "against": "✘", "inconclusive": "?", "not_run": "·"}[self.status]
        return f"{icon} {self.status}" + (" (val)" if self.preliminary else "")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "preliminary": self.preliminary}


def h1(probe_rows: list[dict] | None) -> Verdict:
    if not probe_rows:
        return Verdict("not_run", "probe не запускался")
    clean = [r for r in probe_rows if r["beats_baseline"] and not r["suspected_shortcut"]]
    if clean:
        b = max(clean, key=lambda r: r["auroc"])
        return Verdict("support", f"лучший probe L{b['layer']}/{b['window']}: AUROC {b['auroc']:.3f}, "
                                  f"acc {b['accuracy']:.3f} при перестановке {b['permuted_accuracy']:.3f}")
    if any(r["beats_baseline"] for r in probe_rows):
        return Verdict("inconclusive", "сигнал есть только вместе с признаком shortcut "
                                       "(probe различает и инвариантные ветки — читает форму)")
    return Verdict("against", "ни один (слой, окно) не обходит базовую частоту и перестановку")


def h2(h1v: Verdict, base_val: dict | None, max_sens: float) -> Verdict:
    if not base_val:
        return Verdict("not_run", "baseline на val не считался")
    sens = base_val["sensitivity"]
    err = 1 - sens
    if sens >= max_sens:
        return Verdict("against", f"чувствительность базы {sens:.3f} ≥ {max_sens}: ошибок после "
                                  "ответа мало — пересмотреть постановку задач")
    if h1v.status != "support":
        return Verdict("inconclusive", f"агент ошибается в {err:.0%} меняющих пар, но H1 не "
                                       "подтверждена — нельзя сказать «различает, но не использует»")
    return Verdict("support", f"различает (H1), но ошибается в {err:.0%} меняющих пар: "
                              f"чувствительность {sens:.3f}, специфичность {base_val['specificity']:.3f}")


def h3(test: dict | None, sweep_sel: dict | None, max_drop: float) -> Verdict:
    """H3: следующее действие лучше базы, специфичность не упала, sham не значимы, и —
    если эпизоды оценивались — успех всего эпизода тоже лучше (HYPOTHESES.md: «повышает
    правильность следующего действия и успех всего эпизода»)."""
    if test is None:
        if not sweep_sel:
            return Verdict("not_run", "steering не подбирался")
        if sweep_sel.get("row") is None:
            return Verdict("against", sweep_sel.get("reason", "ни одна конфигурация не прошла "
                                                    "ограничение по специфичности"), True)
        r = sweep_sel["row"]
        return Verdict("inconclusive", f"на val: успех {r['base_success']:.3f} → {r['success']:.3f} "
                                       f"при L{r['layer']}/{r['window']}/α={r['alpha']}; "
                                       "решает только test", True)

    st = test["methods"].get("steering")
    if st is None:
        return Verdict("not_run", "steering не оценивался на test")
    pd = st["paired"]
    drop = test["methods"]["baseline"]["specificity"] - st["specificity"]
    shams_pos = [n for n, m in test["methods"].items()
                 if n.startswith("sham:") and m["paired"] and m["paired"]["excludes_zero"]
                 and m["paired"]["mean_diff"] > 0]
    ep = ((test.get("episodes") or {}).get("steering") or {}).get("paired")
    if pd["excludes_zero"] and pd["mean_diff"] < 0:
        return Verdict("against", f"steering хуже базы: {pd['mean_diff']:+.3f} [{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}]")
    if ep and ep["excludes_zero"] and ep["mean_diff"] < 0:
        return Verdict("against", f"успех эпизода хуже базы: {ep['mean_diff']:+.3f} "
                                  f"[{ep['ci_low']:+.3f}, {ep['ci_high']:+.3f}]")
    if drop > max_drop:
        return Verdict("against", f"специфичность упала на {drop:.3f} > {max_drop}: "
                                  "чувствительность куплена ложными переключениями")
    if shams_pos:
        return Verdict("inconclusive", f"sham тоже даёт значимый прирост ({', '.join(shams_pos)}) — "
                                       "эффект неспецифичен")
    if pd["excludes_zero"] and pd["mean_diff"] > 0:
        head = (f"следующее действие {pd['mean_diff']:+.3f} [{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}], "
                f"падение специфичности {drop:+.3f}, sham-контроли не значимы")
        if ep is None:
            return Verdict("support", head + "; эпизоды не оценивались")
        if ep["excludes_zero"] and ep["mean_diff"] > 0:
            return Verdict("support", head + f"; эпизод {ep['mean_diff']:+.3f} "
                                             f"[{ep['ci_low']:+.3f}, {ep['ci_high']:+.3f}]")
        return Verdict("inconclusive", head + f"; но на успех эпизода не переносится: "
                                              f"{ep['mean_diff']:+.3f} [{ep['ci_low']:+.3f}, {ep['ci_high']:+.3f}]")
    return Verdict("inconclusive", f"интервал включает ноль: {pd['mean_diff']:+.3f} "
                                   f"[{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}]")


def h4(test: dict | None, val_cmp: dict | None) -> Verdict:
    """H4 (HYPOTHESES.md): steering обходит LoRA на условном срезе и проигрывает на глобальном.
    Опровергнет: LoRA не хуже и на условном срезе при сопоставимой цене.

    Условный срез — парные задачи (этот протокол). Глобальный срез пока не определён,
    поэтому «support» недостижим: максимум — «inconclusive» при выигрыше steering.
    """
    cmp, prelim = (test or {}).get("comparisons") or {}, False
    if "steering_vs_lora" not in cmp:
        cmp, prelim = val_cmp or {}, True
    c = cmp.get("steering_vs_lora")
    if c is None:
        return Verdict("not_run", "нужны steering и LoRA, выбранные на val и оценённые на одних задачах")
    ci = f"{c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
    if c["excludes_zero"] and c["mean_diff"] > 0:
        return Verdict("inconclusive", f"условный срез за steering: steering − LoRA {ci}; глобальный "
                                       "срез не определён — без него H4 не подтверждается", prelim)
    return Verdict("against", f"LoRA не хуже steering на условном срезе: steering − LoRA {ci}", prelim)


def alternatives(test: dict | None, val_cmp: dict | None, summaries: dict | None,
                 cheap: tuple[str, ...] = ("strong_prompt", "restate", "explicit_state"),
                 cost_slack: float = 1.1) -> Verdict:
    """Условие провала тезиса (CONTEXT.md §3, HYPOTHESES «точки остановки»): если явное
    состояние, пересказ или промпт дают то же качество дешевле — steering как метод
    не нужен. «Не хуже» = steering − альтернатива не больше нуля значимо;
    «дешевле» = задержка p50 не больше cost_slack × задержки steering."""
    cmp, prelim = (test or {}).get("comparisons") or {}, False
    if not any(f"steering_vs_{m}" in cmp for m in cheap):
        cmp, prelim = val_cmp or {}, True
    rows = [(m, cmp[f"steering_vs_{m}"]) for m in cheap if f"steering_vs_{m}" in cmp]
    if not rows:
        return Verdict("not_run", "steering не сравнивался с промптом / пересказом / явным состоянием")
    lat = {m: (summaries or {}).get(m, {}).get("latency_p50_ms") for m in (*cheap, "steering")}
    tie_cheap, tie_costly = [], []
    for m, c in rows:
        if c["excludes_zero"] and c["mean_diff"] > 0:
            continue
        s = f"{m} (steering − {m} {c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}])"
        if lat.get(m) is not None and lat.get("steering") and lat[m] > cost_slack * lat["steering"]:
            tie_costly.append(s + f", но дороже: p50 {lat[m]:.0f} vs {lat['steering']:.0f} мс")
        else:
            tie_cheap.append(s)
    if tie_cheap:
        return Verdict("against", "не хуже steering и не дороже: " + "; ".join(tie_cheap), prelim)
    if tie_costly:
        return Verdict("inconclusive", "то же качество, что у steering, но дороже: "
                                       + "; ".join(tie_costly), prelim)
    return Verdict("support", "steering значимо лучше: " + ", ".join(
        f"{m} {c['mean_diff']:+.3f}" for m, c in rows), prelim)


def evaluate(state: dict[str, Any], cfg) -> dict[str, Verdict]:
    v1 = h1(state.get("probe_rows"))
    base_val = (state.get("summaries", {}).get("val", {}) or {}).get("baseline")
    v2 = h2(v1, base_val, cfg.eval.h2_max_sensitivity)
    test = state.get("test")
    v3 = h3(test, state.get("selection"), cfg.steer.max_specificity_drop)
    val_cmp = state.get("val_comparisons")
    v4 = h4(test, val_cmp)
    summaries = (test or {}).get("methods") or (state.get("summaries", {}).get("val") or {})
    alt = alternatives(test, val_cmp, summaries)
    return {"H1": v1, "H2": v2, "H3": v3, "H4": v4, "ALT": alt}
