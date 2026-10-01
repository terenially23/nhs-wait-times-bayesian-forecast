"""Backtest harness + interval-coverage metrics.

Two schemes, both training on <= TRAIN_END (Dec 2023) and scoring on held-out 2024 months:
  fixed_origin   fit once at TRAIN_END, forecast every held-out month (horizons 1..10)
  rolling_origin refit on an expanding window at TRAIN_END, +step, +2*step ... forecast <= max_horizon
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from . import config
from .data import expit, region_logit_series

Forecaster = Callable[[pd.Series, int, int], pd.DataFrame]
QCOLS = [config.qname(q) for q in config.QUANTILES]


def _score_rows(region, model, scheme, origin, fc: pd.DataFrame, actual_rate: pd.Series) -> list[dict]:
    rows = []
    for h, (month, r) in enumerate(fc.iterrows(), start=1):
        if month not in actual_rate.index:
            continue
        rows.append({"region": region, "model": model, "scheme": scheme, "origin": origin,
                     "target": month, "horizon": h, "actual": actual_rate[month],
                     **{c: expit(r[c]) for c in QCOLS}})
    return rows


def run_backtest(df: pd.DataFrame, forecasters: dict[str, Forecaster], scheme: str,
                 train_end: str = config.TRAIN_END, max_horizon: int = 6, origin_step: int = 3,
                 seed: int = config.SEED, regions: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    regions = regions or sorted(df["region"].unique())
    last = df["month"].max()
    train_end = pd.Timestamp(train_end)
    if scheme == "fixed_origin":
        origins = [train_end]
    elif scheme == "rolling_origin":
        origins = list(pd.date_range(train_end, last - pd.offsets.MonthBegin(1), freq=f"{origin_step}MS"))
    else:
        raise ValueError(scheme)
    rows = []
    for ri, region in enumerate(regions):
        y_all = region_logit_series(df, region)
        actual = df[df["region"] == region].set_index("month")["rate"]
        for oi, origin in enumerate(origins):
            y_tr = y_all[:origin]
            horizon = len(pd.date_range(origin, last, freq="MS")) - 1
            if scheme == "rolling_origin":
                horizon = min(horizon, max_horizon)
            for name, fn in forecasters.items():
                if verbose:
                    print(f"[{scheme}] {region} origin={origin:%Y-%m} model={name}", flush=True)
                fc = fn(y_tr, horizon, seed + 1000 * ri + oi)
                rows += _score_rows(region, name, scheme, origin, fc, actual)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ metrics
def _interval_score(y, lo, hi, level):
    a = 1 - level
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def add_flags(res: pd.DataFrame) -> pd.DataFrame:
    r = res.copy()
    r["in80"] = (r.actual >= r.q100) & (r.actual <= r.q900)
    r["in95"] = (r.actual >= r.q025) & (r.actual <= r.q975)
    r["w80"] = (r.q900 - r.q100) * 100  # percentage points
    r["w95"] = (r.q975 - r.q025) * 100
    r["is95"] = _interval_score(r.actual * 100, r.q025 * 100, r.q975 * 100, 0.95)
    r["err"] = (r.actual - r.q500) * 100
    r["hgroup"] = pd.cut(r.horizon, [0, 3, 6, 99], labels=["1-3", "4-6", "7+"])
    return r


def coverage_table(res: pd.DataFrame) -> pd.DataFrame:
    """Coverage (primary) + width/score/RMSE/MAE (reference), by scheme x model x horizon group."""
    r = add_flags(res)
    groups = []
    for (scheme, model), g in r.groupby(["scheme", "model"], observed=True):
        subsets = [("all", g)] + [(str(k), s) for k, s in g.groupby("hgroup", observed=True) if len(s)]
        for hg, s in subsets:
            n = len(s)
            row = {"scheme": scheme, "model": model, "horizon": hg, "n": n}
            for lvl, col in ((80, "in80"), (95, "in95")):
                k = int(s[col].sum())
                lo, hi = proportion_confint(k, n, method="wilson")
                row[f"cov{lvl}"] = k / n
                row[f"cov{lvl}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
            row.update({"width80_pp": s.w80.mean(), "width95_pp": s.w95.mean(),
                        "interval_score95": s.is95.mean(),
                        "rmse_pp": float(np.sqrt((s.err ** 2).mean())), "mae_pp": s.err.abs().mean()})
            groups.append(row)
    out = pd.DataFrame(groups)
    return out.sort_values(["scheme", "model", "horizon"]).reset_index(drop=True)


def to_markdown(tbl: pd.DataFrame) -> str:
    t = tbl.copy()
    for c in ("cov80", "cov95"):
        t[c] = (t[c] * 100).round(0).astype(int).astype(str) + "%"
    for c in ("width80_pp", "width95_pp", "interval_score95", "rmse_pp", "mae_pp"):
        t[c] = t[c].round(2)
    return t.to_markdown(index=False) if hasattr(t, "to_markdown") else t.to_string(index=False)
