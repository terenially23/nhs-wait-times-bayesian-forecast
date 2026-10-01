"""Structural time series via statsmodels.tsa.UnobservedComponents (Kalman filter / MLE)."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .. import config

# Local-trend options. 'lldtrend' (random-walk level + constant drift) is the default: it has one
# fewer variance to estimate than 'lltrend', which matters with only ~15 training months.
TRENDS = ("lldtrend", "lltrend", "llevel")


def fit_uc(y: pd.Series, trend: str = "lldtrend", harmonics: int = config.SEASONAL_HARMONICS,
           stochastic_seasonal: bool = False):
    mod = UnobservedComponents(
        y, level=trend,
        freq_seasonal=[{"period": config.SEASONAL_PERIOD, "harmonics": harmonics}],
        stochastic_freq_seasonal=[stochastic_seasonal],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter on short series
        res = mod.fit(disp=False, maxiter=500)
    return res


def uc_forecast(y: pd.Series, horizon: int, trend: str = "lldtrend",
                harmonics: int = config.SEASONAL_HARMONICS, stochastic_seasonal: bool = False,
                return_result: bool = False):
    res = fit_uc(y, trend, harmonics, stochastic_seasonal)
    fc = res.get_forecast(horizon)
    mu = np.asarray(fc.predicted_mean)
    sd = np.sqrt(np.asarray(fc.var_pred_mean))  # includes the irregular (observation) variance
    out = pd.DataFrame({config.qname(q): mu + norm.ppf(q) * sd for q in config.QUANTILES},
                       index=fc.predicted_mean.index)
    return (out, res) if return_result else out
