"""Forecast plot(s) with 80/95% prediction intervals over actuals -> outputs/forecast_<model>.png."""
import os

# pin BLAS threads: with parallel NUTS chains, oversubscription made sampling ~20x slower
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import pandas as pd

from nhs_forecast import config
from nhs_forecast.data import load_regional
from nhs_forecast.plotting import plot_forecasts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["uc"])
    a = ap.parse_args()
    df = load_regional()
    for m in a.models:
        fc = pd.read_csv(config.OUTPUT_DIR / f"forecasts_{m}.csv", parse_dates=["target"])
        out = config.OUTPUT_DIR / f"forecast_{m}.png"
        plot_forecasts(df, fc, {"uc": "UnobservedComponents", "pymc": "PyMC"}.get(m, m), out, df.source.iloc[0])
        print("wrote", out)


if __name__ == "__main__":
    main()
