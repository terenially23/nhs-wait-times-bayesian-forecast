"""Backtest harness -> outputs/backtest_results.csv, outputs/tables/coverage_table.{csv,md}."""
import os

# pin BLAS threads: with parallel NUTS chains, oversubscription made sampling ~20x slower
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import pandas as pd

from nhs_forecast import config
from nhs_forecast.backtest import coverage_table, run_backtest, to_markdown
from nhs_forecast.data import load_regional
from nhs_forecast.models import get_forecaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["uc", "naive"], choices=["uc", "pymc", "naive"])
    ap.add_argument("--schemes", nargs="+", default=["fixed_origin", "rolling_origin"])
    ap.add_argument("--origin-step", type=int, default=3, help="months between rolling origins")
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=750)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()

    df = load_regional()
    fcs = {}
    for m in a.models:
        kw = dict(draws=a.draws, tune=a.tune, chains=a.chains) if m == "pymc" else {}
        fcs[m] = get_forecaster(m, **kw)
    res = pd.concat([run_backtest(df, fcs, s, max_horizon=a.max_horizon, origin_step=a.origin_step, seed=a.seed)
                     for s in a.schemes], ignore_index=True)
    res["source"] = df.source.iloc[0]
    (config.OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUTPUT_DIR / "backtest_results.csv", index=False)
    tbl = coverage_table(res)
    tbl.to_csv(config.OUTPUT_DIR / "tables" / "coverage_table.csv", index=False)
    banner = "> **SYNTHETIC DATA - these are pipeline-test numbers, not NHS results.**\n\n" if res.source.iloc[0] == "synthetic" else ""
    md = banner + to_markdown(tbl) + "\n"
    (config.OUTPUT_DIR / "tables" / "coverage_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
