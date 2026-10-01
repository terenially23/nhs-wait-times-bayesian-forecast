import numpy as np
import pandas as pd
import pytest

from nhs_forecast import config
from nhs_forecast.backtest import add_flags, coverage_table
from nhs_forecast.data import (build_regional_series, expit, logit, parse_month,
                               region_logit_series, wait_lower_bound_days)
from nhs_forecast.models import naive_rw_forecast, uc_forecast
from nhs_forecast.synthetic import make_practice_level


@pytest.mark.parametrize("label,expected", [
    ("Same Day", 0), ("1 Day", 1), ("2 to 7 Days", 2), ("8 to 14 Days", 8),
    ("15 to 21 Days", 15), ("22 to 28 Days", 22), ("More than 28 Days", 29)])
def test_wait_bands(label, expected):
    assert wait_lower_bound_days(label) == expected


def test_unknown_band_is_nan():
    assert np.isnan(wait_lower_bound_days("Unknown / Data Quality"))


def test_parse_month_formats():
    assert parse_month(pd.Series(["OCT2022", "NOV2022"])).tolist() == [pd.Timestamp("2022-10-01"), pd.Timestamp("2022-11-01")]
    assert parse_month(pd.Series(["2022-10"]))[0] == pd.Timestamp("2022-10-01")


def test_logit_roundtrip():
    p = np.array([0.05, 0.3, 0.9])
    assert np.allclose(expit(logit(p)), p)


@pytest.fixture(scope="module")
def regional(tmp_path_factory):
    p = tmp_path_factory.mktemp("raw") / "Practice_Level_Crosstab_test.csv"
    make_practice_level(practices_per_region=2).to_csv(p, index=False)
    return build_regional_series([p], source="synthetic")


def test_aggregation_matches_manual(regional):
    raw = make_practice_level(practices_per_region=2)
    sub = raw[(raw.REGION_NAME == "London") & (raw.APPOINTMENT_MONTH == "OCT2022")]
    known = sub[~sub.TIME_BETWEEN_BOOK_AND_APPT.str.contains("Unknown")]
    long_ = known[known.TIME_BETWEEN_BOOK_AND_APPT.isin(["15 to 21 Days", "22 to 28 Days", "More than 28 Days"])]
    row = regional[(regional.region == "London") & (regional.month == "2022-10-01")].iloc[0]
    assert row.total == known.COUNT_OF_APPOINTMENTS.sum()
    assert row.long_wait == long_.COUNT_OF_APPOINTMENTS.sum()
    assert len(regional) == 7 * 25


def test_uc_forecast_shape_and_ordering(regional):
    y = region_logit_series(regional, "London")[: config.TRAIN_END]
    fc = uc_forecast(y, 6)
    assert len(fc) == 6 and fc.index[0] == pd.Timestamp("2024-01-01")
    q = [config.qname(x) for x in config.QUANTILES]
    assert (fc[q].diff(axis=1).iloc[:, 1:] > 0).all().all()  # quantiles strictly increasing


def test_naive_interval_widens(regional):
    y = region_logit_series(regional, "London")[: config.TRAIN_END]
    fc = naive_rw_forecast(y, 6)
    w = fc.q975 - fc.q025
    assert (w.diff().dropna() > 0).all()


def test_coverage_flags():
    res = pd.DataFrame({"region": "a", "model": "m", "scheme": "s", "origin": 0, "target": 0, "horizon": [1, 2],
                        "actual": [0.30, 0.50], "q025": 0.2, "q100": 0.25, "q500": 0.3, "q900": 0.35, "q975": 0.4})
    r = add_flags(res)
    assert r.in80.tolist() == [True, False] and r.in95.tolist() == [True, False]
    t = coverage_table(res)
    assert t[t.horizon == "all"].cov80.iloc[0] == 0.5
