"""Forward (true out-of-sample) test: train on everything up to Oct 2024, forecast the later months,
compare with the real figures. Needs the regional table built with --end beyond 2024-10.

    python scripts/06_forward_test.py --models uc pymc naive
Outputs: outputs/forward/{forecasts.csv, coverage_table.md, forecast_<model>.png, pymc_health.json}
"""
import os

# pin BLAS threads: with parallel NUTS chains, oversubscription made sampling ~20x slower
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from pathlib import Path

import pandas as pd

from nhs_forecast import config
from nhs_forecast.backtest import _score_rows, coverage_table, to_markdown
from nhs_forecast.data import load_regional, region_logit_series
from nhs_forecast.models import naive_rw_forecast, snaive_forecast
from nhs_forecast.models.uc import uc_forecast
from nhs_forecast.plotting import plot_forecasts

NAMES = {"uc": "UnobservedComponents", "pymc": "PyMC", "naive": "Naive random walk", "snaive": "Seasonal naive (same month last year)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["uc", "pymc", "naive"], choices=list(NAMES))
    ap.add_argument("--input", type=Path, default=config.REGIONAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR / "forward")
    ap.add_argument("--origin", default=config.FORWARD_ORIGIN)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()

    df = load_regional(a.input, end=None)
    origin = pd.Timestamp(a.origin)
    horizon = len(pd.date_range(origin, df["month"].max(), freq="MS")) - 1
    if horizon < 1:
        raise SystemExit(f"No months after {origin:%Y-%m} in {a.input}; build it with 01_prepare_data.py --end ...")
    print(f"origin {origin:%Y-%m}; scoring {horizon} later months ({origin + pd.offsets.MonthBegin(1):%Y-%m} to {df.month.max():%Y-%m})")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows, health = [], {}
    for ri, region in enumerate(sorted(df.region.unique())):
        y = region_logit_series(df, region)[:origin]
        actual = df[df.region == region].set_index("month")["rate"]
        for m in a.models:
            print(f"[forward] {region} {m} (n_train={len(y)}, horizon={horizon})", flush=True)
            if m == "uc":
                fc = uc_forecast(y, horizon)
            elif m == "naive":
                fc = naive_rw_forecast(y, horizon)
            elif m == "snaive":
                fc = snaive_forecast(y, horizon)
            else:
                from nhs_forecast.models.bayes import diagnostics, pymc_forecast
                fc, idata = pymc_forecast(y, horizon, seed=a.seed + ri, draws=a.draws, tune=a.tune, chains=a.chains)
                d = diagnostics(idata)
                health[region] = {k: v for k, v in d.items() if k != "summary"}
            rows += _score_rows(region, m, "forward_origin", origin, fc, actual)
    res = pd.DataFrame(rows)
    res["source"] = df.source.iloc[0]
    res.to_csv(a.out_dir / "forecasts.csv", index=False)
    tbl = coverage_table(res)
    tbl.to_csv(a.out_dir / "coverage_table.csv", index=False)
    (a.out_dir / "coverage_table.md").write_text(to_markdown(tbl) + "\n")
    print(to_markdown(tbl))
    if health:
        (a.out_dir / "pymc_health.json").write_text(json.dumps(health, indent=2))
        print(pd.DataFrame(health).T.to_string())
    for m in a.models:
        plot_forecasts(df, res[res.model == m], NAMES[m], a.out_dir / f"forecast_{m}.png", df.source.iloc[0],
                       train_end=a.origin)
        print("wrote", a.out_dir / f"forecast_{m}.png")


if __name__ == "__main__":
    main()
