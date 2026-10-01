import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config

BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#555555"


def plot_forecasts(df: pd.DataFrame, fc: pd.DataFrame, model: str, out_path, source: str,
                   train_end: str = config.TRAIN_END):
    """Panel per region: actuals, median forecast, 80% (dark) and 95% (light) intervals.
    Held-out actuals outside the 95% interval are marked with a red cross."""
    regions = sorted(df["region"].unique())
    ncol = 4
    nrow = -(-len(regions) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), sharex=True)
    axes = axes.ravel()
    te = pd.Timestamp(train_end)
    for ax, region in zip(axes, regions):
        a = df[df["region"] == region].set_index("month")["rate"] * 100
        f = fc[fc["region"] == region].sort_values("target")
        ax.axvspan(a.index.min(), te, color="#eeeeee", zorder=0)
        ax.fill_between(f.target, f.q025 * 100, f.q975 * 100, color=BLUE, alpha=0.15, lw=0, label="95% interval")
        ax.fill_between(f.target, f.q100 * 100, f.q900 * 100, color=BLUE, alpha=0.30, lw=0, label="80% interval")
        ax.plot(f.target, f.q500 * 100, color=BLUE, lw=1.8, label="Forecast median")
        ax.plot(a.index, a.values, color="black", lw=1.2, marker="o", ms=3, label="Actual")
        ho = f.set_index("target")
        miss = ho[(a.reindex(ho.index) < ho.q025 * 100) | (a.reindex(ho.index) > ho.q975 * 100)].index
        if len(miss):
            ax.plot(miss, a.reindex(miss), "x", color=RED, ms=8, mew=2, label="Outside 95%")
        ax.set_title(region, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
    for ax in axes[len(regions):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    by = dict(zip(l, h))
    fig.legend(by.values(), by.keys(), loc="lower center", ncol=len(by), frameon=False, fontsize=9)
    tag = "  |  SYNTHETIC DATA - NOT NHS FIGURES" if source == "synthetic" else ""
    fig.suptitle(f"Long-wait appointment rate (%), held-out forecast from {te:%b %Y} - {model}{tag}",
                 fontsize=12, color=RED if tag else "black")
    fig.supylabel("% of appointments booked >14 days ahead", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
