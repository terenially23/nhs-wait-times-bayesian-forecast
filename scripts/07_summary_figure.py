"""One-page visual summary: achieved vs nominal interval coverage, and typical error, per model and test.

Reads outputs/tables/coverage_table.csv (2024 hold-out) and outputs/forward/coverage_table.csv (forward test).
Writes results/summary.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nhs_forecast import config

COL = {"uc": "#d98324", "pymc": "#1f5fa8", "naive": "#8c8c8c", "snaive": "#5aa469"}
LAB = {"uc": "UnobservedComponents", "pymc": "PyMC (Bayesian)", "naive": "Naive random walk", "snaive": "Seasonal naive"}
tests = [("2024 hold-out\n(train Oct 2022 - Dec 2023)", config.OUTPUT_DIR / "tables" / "coverage_table.csv", "fixed_origin"),
         ("Forward test\n(train to Oct 2024, forecast later months)", config.OUTPUT_DIR / "forward" / "coverage_table.csv", "forward_origin")]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw={"height_ratios": [3, 2]})
for j, (title, path, scheme) in enumerate(tests):
    t = pd.read_csv(path)
    t = t[(t.scheme == scheme) & (t.horizon == "all")].set_index("model")
    ax = axes[0, j]
    w = 0.21
    for i, m in enumerate(["uc", "pymc", "naive", "snaive"]):
        if m not in t.index:
            continue
        xs = [0 + (i - 1.5) * w, 1 + (i - 1.5) * w]
        vals = [t.loc[m, "cov80"] * 100, t.loc[m, "cov95"] * 100]
        bars = ax.bar(xs, vals, w * 0.92, color=COL[m], label=LAB[m])
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    for x, nom in ((0, 80), (1, 95)):
        ax.hlines(nom, x - 0.45, x + 0.45, color="black", lw=1.6, ls="--")
        ax.text(x + 0.46, nom, f"target {nom}%", va="center", fontsize=8)
    ax.set_xticks([0, 1], ["80% interval", "95% interval"])
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.55, 1.85)
    ax.set_ylabel("% of real values inside the interval (bar labels in %)" if j == 0 else "", fontsize=9)
    ax.set_title(f"{title}\nn = {int(t['n'].iloc[0])} region-month forecasts", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = axes[1, j]
    ms = [m for m in ["uc", "pymc", "naive", "snaive"] if m in t.index]
    ax2.barh([LAB[m] for m in ms], [t.loc[m, "mae_pp"] for m in ms], color=[COL[m] for m in ms], height=0.6)
    for k, m in enumerate(ms):
        ax2.text(t.loc[m, "mae_pp"] + 0.04, k, f"{t.loc[m, 'mae_pp']:.1f} pp", va="center", fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("typical error (mean absolute error, percentage points)")
    ax2.spines[["top", "right"]].set_visible(False)
src = pd.read_csv(config.REGIONAL_CSV, nrows=1)["source"].iloc[0]
tag = "   |   SYNTHETIC DATA" if src == "synthetic" else ""
fig.suptitle("Forecasting GP long-wait rates: how trustworthy are the prediction intervals?" + tag, fontsize=12, y=0.995)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
(config.ROOT / "results").mkdir(exist_ok=True)
fig.savefig(config.ROOT / "results" / "summary.png", dpi=150)
print("wrote results/summary.png")
