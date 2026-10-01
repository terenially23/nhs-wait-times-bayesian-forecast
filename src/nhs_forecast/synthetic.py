"""Synthetic stand-in data in the *practice-level crosstab schema*.

Purpose: exercise ingestion -> model -> backtest end to end when the real NHS Digital files
are unavailable. The numbers are invented (seasonal peaks in October and April are baked in on
purpose) and say NOTHING about the real NHS. Every downstream artefact carries source='synthetic'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import expit

REGIONS = ["East of England", "London", "Midlands", "North East and Yorkshire",
           "North West", "South East", "South West"]
BANDS = ["Same Day", "1 Day", "2 to 7 Days", "8 to 14 Days", "15 to 21 Days",
         "22 to 28 Days", "More than 28 Days", "Unknown / Data Quality"]
# share of the *short* / *long* mass assigned to each band
SHORT_W = np.array([0.45, 0.20, 0.25, 0.10])
LONG_W = np.array([0.45, 0.30, 0.25])


def make_practice_level(seed: int = config.SEED, practices_per_region: int = 4,
                        start: str = config.DATA_START, end: str = config.DATA_END) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="MS")
    t = np.arange(len(months))
    moy = months.month.values
    rows = []
    for ri, region in enumerate(REGIONS):
        base = rng.normal(-0.9, 0.25)                       # logit of baseline long-wait rate (~30%)
        slope = rng.normal(0.006, 0.004)
        a1, b1 = rng.normal(0.0, 0.03, 2)
        # peaks: October (annual harmonic) and April (semi-annual harmonic gives spring peak)
        seas = (0.16 * np.cos(2 * np.pi * (moy - 10) / 12)
                + 0.10 * np.cos(2 * np.pi * 2 * (moy - 4) / 12) + a1 * np.cos(2 * np.pi * moy / 12))
        level = np.cumsum(rng.normal(0, 0.012, len(t)))     # slowly wandering level
        p_true = expit(base + slope * t + seas + level)
        for pi in range(practices_per_region):
            size = rng.integers(8_000, 20_000) * (1 + 0.1 * ri)
            for mi, m in enumerate(months):
                vol = size * (0.92 if m.month in (8, 12) else 1.0)  # fewer appointments in Aug/Dec
                n = rng.poisson(vol)
                p_p = np.clip(p_true[mi] + rng.normal(0, 0.01), 0.01, 0.99)  # practice noise
                n_long = rng.binomial(n, p_p)
                n_unknown = rng.binomial(n, 0.01)
                counts = np.concatenate([
                    rng.multinomial(n - n_long, SHORT_W), rng.multinomial(n_long, LONG_W), [n_unknown]])
                for band, c in zip(BANDS, counts):
                    rows.append((f"{region[:2].upper()}{pi:03d}", region, m.strftime("%b%Y").upper(),
                                 "Attended", band, int(c)))
    return pd.DataFrame(rows, columns=["GP_CODE", "REGION_NAME", "APPOINTMENT_MONTH", "APPT_STATUS",
                                       "TIME_BETWEEN_BOOK_AND_APPT", "COUNT_OF_APPOINTMENTS"])
