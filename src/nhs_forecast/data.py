"""Ingest NHS Digital 'Appointments in General Practice' practice-level files and
aggregate them to a regional monthly long-wait-rate time series.

Expected input: the practice-level crosstab CSVs (``Practice_Level_Crosstab_<Mon>_<YY>.csv``),
either loose or inside the monthly ``.zip`` downloads from
https://digital.nhs.uk/data-and-information/publications/statistical/appointments-in-general-practice

NOTE: column names below are written from knowledge of the published schema and have NOT
been checked against live files from this repo's build environment (NHS hosts were
unreachable). Column matching is therefore alias-based and case-insensitive, and fails loudly
with the columns it saw if something does not match.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COLUMN_ALIASES = {
    "region": ["REGION_NAME", "COMM_REGION_NAME", "NHSE_REGION_NAME"],
    "month": ["APPOINTMENT_MONTH", "APPT_MONTH", "APPOINTMENT_MONTH_START_DATE", "MONTH"],
    "status": ["APPT_STATUS"],
    "wait": ["TIME_BETWEEN_BOOK_AND_APPT"],
    "count": ["COUNT_OF_APPOINTMENTS"],
}
MONTH_FORMATS = ["%b%Y", "%B%Y", "%b %Y", "%B %Y", "%Y-%m", "%Y-%m-%d", "%d%b%Y", "%d/%m/%Y"]


# ---------------------------------------------------------------- parsing helpers
def wait_lower_bound_days(label: str) -> float:
    """Lower bound (days) of an NHS wait band label; NaN for unknown/data-quality bands.

    'Same Day'->0, '1 Day'->1, '2 to 7 Days'->2, '15 to 21 Days'->15, 'More than 28 Days'->29.
    """
    s = str(label).strip().lower()
    if "unknown" in s or "data quality" in s or s in ("", "nan"):
        return np.nan
    if "same day" in s:
        return 0.0
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    return n + 1 if ("more than" in s or "over" in s or ">" in s) else n


def parse_month(values: pd.Series) -> pd.Series:
    """Parse a month column to month-start timestamps, trying known formats."""
    v = values.astype(str).str.strip()
    for fmt in MONTH_FORMATS:
        parsed = pd.to_datetime(v, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed.dt.to_period("M").dt.to_timestamp()
    raise ValueError(f"Could not parse month values, e.g. {v.unique()[:5].tolist()}")


def _resolve_columns(header: list[str]) -> dict[str, str]:
    upper = {h.strip().upper(): h for h in header}
    out = {}
    for key, aliases in COLUMN_ALIASES.items():
        hit = next((upper[a] for a in aliases if a in upper), None)
        if hit is None:
            raise KeyError(f"No column for '{key}' (tried {aliases}). Columns seen: {header}")
        out[key] = hit
    return out


# ---------------------------------------------------------------- file reading
def _iter_csv_handles(path: Path):
    """Yield (name, binary handle) for a CSV, or for each practice-level CSV inside a zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv") and "crosstab" in n.lower()]
            if not names:
                raise FileNotFoundError(f"No *Crosstab*.csv in {path}; contents: {z.namelist()}")
            for n in names:
                with z.open(n) as fh:
                    yield n, io.BytesIO(fh.read())
    else:
        with open(path, "rb") as fh:
            yield path.name, fh


def aggregate_file(path: Path, statuses: tuple[str, ...] | None = ("Attended",),
                   chunksize: int = 500_000) -> pd.DataFrame:
    """Stream one raw file -> counts by (region, month, wait band). Practice rows are summed away."""
    parts = []
    for name, fh in _iter_csv_handles(path):
        header = list(pd.read_csv(fh, nrows=0).columns)
        fh.seek(0)
        cols = _resolve_columns(header)
        for chunk in pd.read_csv(fh, usecols=list(cols.values()), chunksize=chunksize,
                                 dtype={c: "string" for c in cols.values() if c != cols["count"]}):
            chunk = chunk.rename(columns={v: k for k, v in cols.items()})
            if statuses is not None:
                keep = {s.lower() for s in statuses}
                chunk = chunk[chunk["status"].str.strip().str.lower().isin(keep)]
            chunk["count"] = pd.to_numeric(chunk["count"], errors="coerce").fillna(0)
            parts.append(chunk.groupby(["region", "month", "wait"], as_index=False)["count"].sum())
    if not parts:
        raise ValueError(f"No rows read from {path}")
    out = pd.concat(parts).groupby(["region", "month", "wait"], as_index=False)["count"].sum()
    out["month"] = parse_month(out["month"])
    return out


def build_regional_series(paths: list[Path], long_wait_days: int = config.LONG_WAIT_DAYS,
                          statuses: tuple[str, ...] | None = ("Attended",),
                          source: str = "nhs_digital") -> pd.DataFrame:
    """Aggregate raw files to a tidy regional monthly table.

    rate = appointments booked > ``long_wait_days`` ahead / appointments with a known wait.
    If a month appears in several files (e.g. a later release restating it), the file that
    sorts last wins, so pass paths in release order.
    """
    by_month: dict[pd.Timestamp, pd.DataFrame] = {}
    for p in sorted(paths, key=lambda p: p.name):
        agg = aggregate_file(p, statuses)
        for m, g in agg.groupby("month"):
            by_month[m] = g
    df = pd.concat(by_month.values())
    df["lower"] = df["wait"].map(wait_lower_bound_days)
    df = df.dropna(subset=["lower"])  # drop 'Unknown / Data Quality' from numerator and denominator
    df["is_long"] = df["lower"] > long_wait_days
    g = df.groupby(["region", "month"])
    out = pd.DataFrame({
        "total": g["count"].sum(),
        "long_wait": df[df["is_long"]].groupby(["region", "month"])["count"].sum(),
    }).fillna(0).reset_index()
    out["rate"] = out["long_wait"] / out["total"]
    out["source"] = source
    return out.sort_values(["region", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------- series access
def validate_complete(df: pd.DataFrame, start: str = config.DATA_START, end: str = config.DATA_END) -> None:
    expected = pd.date_range(start, end, freq="MS")
    problems = []
    for region, g in df.groupby("region"):
        missing = expected.difference(pd.DatetimeIndex(g["month"]))
        if len(missing):
            problems.append(f"{region}: missing {[m.strftime('%Y-%m') for m in missing]}")
    if problems:
        raise ValueError("Incomplete regional series:\n  " + "\n  ".join(problems))


def load_regional(path: Path = config.REGIONAL_CSV, start: str = config.DATA_START,
                  end: str = config.DATA_END) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month"])
    df = df[(df["month"] >= start) & (df["month"] <= end)]
    validate_complete(df, start, end)
    return df


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def region_logit_series(df: pd.DataFrame, region: str) -> pd.Series:
    """Monthly logit(rate) for one region with an explicit month-start frequency."""
    g = df[df["region"] == region].set_index("month").sort_index()
    s = pd.Series(logit(g["rate"].values), index=pd.DatetimeIndex(g.index, freq="MS"), name=region)
    return s
