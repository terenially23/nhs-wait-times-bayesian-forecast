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
