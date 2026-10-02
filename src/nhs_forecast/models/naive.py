"""Benchmark: logit random walk. Gives RMSE/coverage numbers something to be compared against."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import config


def naive_rw_forecast(y: pd.Series, horizon: int) -> pd.DataFrame:
    sigma = np.diff(y.values).std(ddof=1)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    h = np.arange(1, horizon + 1)
    out = {config.qname(q): y.values[-1] + norm.ppf(q) * sigma * np.sqrt(h) for q in config.QUANTILES}
    return pd.DataFrame(out, index=idx)


def snaive_forecast(y: pd.Series, horizon: int) -> pd.DataFrame:
    """Seasonal naive on the logit scale: forecast = the same calendar month one year earlier.

    Spread comes from the training year-on-year differences y_t - y_{t-12} (only ~3 of them with 15
    training months, so the interval width is itself very uncertain). Constant across horizons <= 12.
    """
    n = len(y)
    if n < 14:
        raise ValueError("seasonal naive needs at least 14 months")
    sigma = (y.values[12:] - y.values[:-12]).std(ddof=1)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    base = np.array([y.values[n - 12 + ((h - 1) % 12)] for h in range(1, horizon + 1)])
    return pd.DataFrame({config.qname(q): base + norm.ppf(q) * sigma for q in config.QUANTILES}, index=idx)
