"""Автоматические вердикты по гипотезам — по правилам из docs/HYPOTHESES.md.

Вердикт — подсказка, а не вывод исследования: итоговую формулировку пишет
человек в exp.report(conclusion=...). Но вердикты считаются одинаково для всех
прогонов, поэтому их нельзя «подогнать» в конкретном ноутбуке.

Статусы: support | against | inconclusive | not_run. Суффикс «(val)» —
предварительно, по validation, до открытия test.
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
    if pd["excludes_zero"] and pd["mean_diff"] < 0:
        return Verdict("against", f"steering хуже базы: {pd['mean_diff']:+.3f} [{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}]")
    if drop > max_drop:
        return Verdict("against", f"специфичность упала на {drop:.3f} > {max_drop}: "
                                  "чувствительность куплена ложными переключениями")
    if shams_pos:
        return Verdict("inconclusive", f"sham тоже даёт значимый прирост ({', '.join(shams_pos)}) — "
                                       "эффект неспецифичен")
    if pd["excludes_zero"] and pd["mean_diff"] > 0:
        return Verdict("support", f"успех {pd['mean_diff']:+.3f} [{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}], "
                                  f"падение специфичности {drop:+.3f}, sham-контроли не значимы")
    return Verdict("inconclusive", f"интервал включает ноль: {pd['mean_diff']:+.3f} "
                                   f"[{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}]")


def evaluate(state: dict[str, Any], cfg) -> dict[str, Verdict]:
    v1 = h1(state.get("probe_rows"))
    base_val = (state.get("summaries", {}).get("val", {}) or {}).get("baseline")
    v2 = h2(v1, base_val, cfg.eval.h2_max_sensitivity)
    v3 = h3(state.get("test"), state.get("selection"), cfg.steer.max_specificity_drop)
    v4 = Verdict("not_run", "нужен LoRA/SFT baseline на тех же данных (не реализован)")
    return {"H1": v1, "H2": v2, "H3": v3, "H4": v4}
