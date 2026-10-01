"""Forecasters share one interface:

    forecaster(y_train: pd.Series, horizon: int, seed: int) -> pd.DataFrame

``y_train`` is logit(rate) on a monthly DatetimeIndex. The result is indexed by the forecast
months with one column per quantile in ``config.QUANTILES`` (names from ``config.qname``), on the
logit scale. Modelling on the logit scale keeps rates inside (0, 1); because the logit is monotone,
back-transforming quantiles gives the exact quantiles of the rate.
"""
from .naive import naive_rw_forecast
from .uc import uc_forecast

__all__ = ["naive_rw_forecast", "uc_forecast", "get_forecaster"]


def get_forecaster(name: str, **kwargs):
    if name == "uc":
        return lambda y, h, seed: uc_forecast(y, h, **kwargs)
    if name == "naive":
        return lambda y, h, seed: naive_rw_forecast(y, h)
    if name == "pymc":
        from .bayes import pymc_forecast  # lazy: importing pymc is slow
        return lambda y, h, seed: pymc_forecast(y, h, seed=seed, **kwargs)[0]
    raise ValueError(f"unknown model '{name}'")
