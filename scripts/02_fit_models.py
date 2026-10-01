"""Fit on the training window and forecast the held-out months, for UC and/or PyMC.

Writes outputs/forecasts_<model>.csv. For PyMC also writes ArviZ diagnostics per region
(summary table, trace plot, netcdf) to outputs/pymc_diagnostics/.
"""
import os

# pin BLAS threads: with parallel NUTS chains, oversubscription made sampling ~20x slower
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nhs_forecast import config
from nhs_forecast.backtest import _score_rows
from nhs_forecast.data import load_regional, region_logit_series
from nhs_forecast.models.uc import TRENDS, uc_forecast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["uc"], choices=["uc", "pymc"])
    ap.add_argument("--trend", default="lldtrend", choices=TRENDS, help="UC trend (PyMC is always lldtrend)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()

    df = load_regional()
    origin = pd.Timestamp(config.TRAIN_END)
    last = df["month"].max()
    horizon = len(pd.date_range(origin, last, freq="MS")) - 1
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    diag_dir = out_dir / "pymc_diagnostics"
    diag_dir.mkdir(exist_ok=True)

    for model in a.models:
        rows, diag = [], {}
        for ri, region in enumerate(sorted(df.region.unique())):
            y = region_logit_series(df, region)[:origin]
            actual = df[df.region == region].set_index("month")["rate"]
            print(f"[{model}] {region}: n_train={len(y)} horizon={horizon}", flush=True)
            if model == "uc":
                fc, res = uc_forecast(y, horizon, trend=a.trend, return_result=True)
            else:
                from nhs_forecast.models.bayes import diagnostics, pymc_forecast
                fc, idata = pymc_forecast(y, horizon, seed=a.seed + ri, draws=a.draws, tune=a.tune, chains=a.chains)
                d = diagnostics(idata)
                slug = region.replace(" ", "_")
                d["summary"].to_csv(diag_dir / f"{slug}_summary.csv")
                idata.to_netcdf(diag_dir / f"{slug}.nc")
                az.plot_trace(idata, var_names=["sigma_obs", "sigma_level", "drift", "beta"])
                plt.gcf().suptitle(region)
                plt.savefig(diag_dir / f"{slug}_trace.png", dpi=100)
                plt.close("all")
                diag[region] = {k: v for k, v in d.items() if k != "summary"}
            rows += _score_rows(region, model, "fixed_origin", origin, fc, actual)
        res_df = pd.DataFrame(rows)
        res_df["source"] = df.source.iloc[0]
        res_df.to_csv(out_dir / f"forecasts_{model}.csv", index=False)
        print(f"wrote forecasts_{model}.csv")
        if diag:
            (diag_dir / "sampler_health.json").write_text(json.dumps(diag, indent=2))
            print(pd.DataFrame(diag).T.to_string())


if __name__ == "__main__":
    main()
