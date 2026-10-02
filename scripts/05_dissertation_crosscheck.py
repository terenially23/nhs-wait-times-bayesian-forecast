"""Compare descriptive patterns in the regional series with the dissertation's Table 1 rate ratios.

This is a SANITY CHECK, not a replication: Table 1 comes from a practice-level negative-binomial mixed
model adjusted for deprivation, rurality, staffing and ICB; here the comparison is raw aggregate rates.
"""
import numpy as np
import pandas as pd

from nhs_forecast import config
from nhs_forecast.data import load_regional

# Dissertation Table 1 (rate ratios)
DISS_REGION = {"London": 1.0, "South East": 1.510, "South West": 1.856, "East of England": 1.668,
               "North West": 1.499, "Midlands": 1.480, "North East and Yorkshire": 1.794}
DISS_MONTH = {1: 1.0, 2: 1.024, 3: 1.076, 4: 1.165, 5: 1.118, 6: 1.106, 7: 1.064, 8: 1.081,
              9: 1.209, 10: 1.361, 11: 1.123, 12: 1.015}

df = load_regional()
out = []

# Regions: pooled long-wait rate over the whole window, relative to London
reg = df.groupby("region").apply(lambda g: g.long_wait.sum() / g.total.sum(), include_groups=False)
rel = (reg / reg["London"]).rename("this_project").to_frame()
rel["dissertation_RR"] = pd.Series(DISS_REGION)
rel["rank_this"] = rel.this_project.rank(ascending=False).astype(int)
rel["rank_diss"] = rel.dissertation_RR.rank(ascending=False).astype(int)
print("REGION: rate relative to London (raw, unadjusted) vs dissertation adjusted RR\n", rel.round(3).to_string())
print("Spearman rank correlation:", round(rel.this_project.corr(rel.dissertation_RR, method="spearman"), 2))

# Months: national rate by calendar month relative to January (mean over years available)
nat = df.groupby("month")[["long_wait", "total"]].sum()
nat["rate"] = nat.long_wait / nat.total
m = nat.rate.groupby(nat.index.month).mean()
mo = pd.DataFrame({"this_project": m / m[1], "dissertation_RR": pd.Series(DISS_MONTH)})
mo["years_averaged"] = nat.rate.groupby(nat.index.month).size()
print("\nCALENDAR MONTH: national rate relative to January vs dissertation RR\n", mo.round(3).to_string())
print("Pearson correlation:", round(mo.this_project.corr(mo.dissertation_RR), 2),
      "| Spearman:", round(mo.this_project.corr(mo.dissertation_RR, method="spearman"), 2))
print("Top-4 months here:", m.sort_values(ascending=False).index[:4].tolist(), "| dissertation:", [10, 9, 4, 5])

lines = ["# Cross-check against the dissertation (Table 1)\n",
         "Raw aggregate rates from `data/processed/regional_monthly.csv` vs the dissertation's adjusted rate ratios. "
         "A sanity check on direction and ordering, not a replication.\n",
         "## Region (relative to London)\n", rel.round(3).to_markdown(),
         f"\nSpearman rank correlation: {rel.this_project.corr(rel.dissertation_RR, method='spearman'):.2f}\n",
         "## Calendar month (relative to January)\n", mo.round(3).to_markdown(),
         f"\nPearson r = {mo.this_project.corr(mo.dissertation_RR):.2f}, Spearman = {mo.this_project.corr(mo.dissertation_RR, method='spearman'):.2f}\n"]
(config.ROOT / "results" / "dissertation_crosscheck.md").write_text("\n".join(lines))
