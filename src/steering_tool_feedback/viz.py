"""Графики для ноутбуков. Только matplotlib; на вход — таблицы из Experiment."""

from __future__ import annotations

from typing import Any


def _plt():
    import matplotlib.pyplot as plt

    return plt


def probe_heatmap(probe_df: Any, ax: Any = None):
    """AUROC probe по (слой × окно). Звёздочка — сигнал обходит обе нулевые модели;
    «s» — подозрение на shortcut (probe различает и инвариантные ветки)."""
    plt = _plt()
    if probe_df.empty:
        raise ValueError("probe table is empty — run exp.probe() first")
    pv = probe_df.pivot(index="window", columns="layer", values="auroc")
    ax = ax or plt.subplots(figsize=(1.0 + 0.7 * pv.shape[1], 1.2 + 0.5 * pv.shape[0]))[1]
    im = ax.imshow(pv.values, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    ax.set_xlabel("layer")
    ax.set_title("H1: probe AUROC (train, CV по задачам)")
    for _, r in probe_df.iterrows():
        i, j = list(pv.index).index(r["window"]), list(pv.columns).index(r["layer"])
        mark = "*" if r["beats_baseline"] and not r["suspected_shortcut"] else (
            "s" if r["suspected_shortcut"] else "")
        ax.text(j, i, f"{r['auroc']:.2f}{mark}", ha="center", va="center", fontsize=8,
                color="white" if r["auroc"] < 0.8 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04)
    return ax


def dose_response(sweep_df: Any, max_drop: float | None = None):
    """Успех и специфичность на val от alpha — по каждой (слой, окно)."""
    plt = _plt()
    if sweep_df.empty:
        raise ValueError("sweep table is empty — run exp.sweep() first")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for (L, w), g in sweep_df.groupby(["layer", "window"]):
        g = g.sort_values("alpha")
        axes[0].plot(g["alpha"], g["success"], marker="o", label=f"L{L}/{w}")
        axes[1].plot(g["alpha"], g["spec_drop"], marker="o", label=f"L{L}/{w}")
    axes[0].axhline(sweep_df["base_success"].iloc[0], color="grey", ls="--", lw=1, label="база")
    axes[0].set_title("успех эпизода (val)")
    axes[1].set_title("падение специфичности (val)")
    if max_drop is not None:
        axes[1].axhline(max_drop, color="red", ls="--", lw=1, label="допуск")
    for a in axes:
        a.set_xscale("log", base=2)
        a.set_xlabel("alpha")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def compare_methods(results_df: Any):
    """Разность успеха с базой и 95% интервал — по методам и sham-контролям."""
    plt = _plt()
    d = results_df[results_df["diff_vs_base"].notna()].copy()
    if d.empty:
        raise ValueError("no paired comparisons — run exp.test() (or evaluate vs baseline)")
    lo = d["ci"].str.extract(r"\[([-+0-9.]+),")[0].astype(float)
    hi = d["ci"].str.extract(r", ([-+0-9.]+)\]")[0].astype(float)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1))
    y = range(len(d))
    ax.errorbar(d["diff_vs_base"], y, xerr=[d["diff_vs_base"] - lo, hi - d["diff_vs_base"]],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(list(y), d["method"])
    ax.set_xlabel("успех: метод − база (95% bootstrap по задачам)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
