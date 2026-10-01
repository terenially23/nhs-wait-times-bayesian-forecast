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
    "month": ["APPOINTMENT_MONTH_START_DATE", "APPOINTMENT_MONTH", "APPT_MONTH", "MONTH"],
    "wait": ["TIME_BETWEEN_BOOK_AND_APPT"],
    "count": ["COUNT_OF_APPOINTMENTS"],
}
# Optional: present in some releases only (Dec 2022 / Jan 2023 releases have no APPT_STATUS).
OPTIONAL_ALIASES = {"status": ["APPT_STATUS"], "category": ["NATIONAL_CATEGORY"]}
# Series definition. The practice-level files carry sub-ICB but NOT region/ICB columns, so the
# default is a single national series; 'sub_icb' gives one series per sub-ICB location.
LEVEL_COLUMNS = {"national": None, "sub_icb": ["SUB_ICB_LOCATION_CODE"],
                 "region_lookup": ["SUB_ICB_LOCATION_CODE"],  # sub-ICB aggregated, then mapped via geography.apply_lookup
                 "region": ["REGION_NAME", "COMM_REGION_NAME", "NHSE_REGION_NAME"]}
NATIONAL_LABEL = "England"
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


def _resolve_columns(header: list[str], level: str = "national") -> dict[str, str | None]:
    upper = {h.strip().upper(): h for h in header}
    out: dict[str, str | None] = {}
    for key, aliases in COLUMN_ALIASES.items():
        hit = next((upper[a] for a in aliases if a in upper), None)
        if hit is None:
            raise KeyError(f"No column for '{key}' (tried {aliases}). Columns seen: {header}")
        out[key] = hit
    for key, aliases in OPTIONAL_ALIASES.items():
        out[key] = next((upper[a] for a in aliases if a in upper), None)
    group = LEVEL_COLUMNS[level]
    if group is None:
        out["region"] = None
    else:
        hit = next((upper[a] for a in group if a in upper), None)
        if hit is None:
            raise KeyError(f"level='{level}' needs one of {group}. Columns seen: {header}")
        out["region"] = hit
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
                   level: str = "national", chunksize: int = 500_000) -> pd.DataFrame:
    """Stream one raw release -> counts by (region, month, wait band). Practice rows are summed away.

    Every CSV in a zip is read and summed (large months are split over several CSVs).
    If APPT_STATUS is absent from a CSV the status filter cannot be applied; a warning is printed.
    """
    parts = []
    for name, fh in _iter_csv_handles(path):
        header = list(pd.read_csv(fh, nrows=0).columns)
        fh.seek(0)
        cols = _resolve_columns(header, level)
        use = {k: v for k, v in cols.items() if v is not None and k != "category"}
        apply_status = statuses is not None and cols["status"] is not None
        if statuses is not None and cols["status"] is None:
            print(f"WARNING: {path.name}/{name}: no APPT_STATUS column; status filter NOT applied")
        for chunk in pd.read_csv(fh, usecols=list(set(use.values())), chunksize=chunksize,
                                 dtype={c: "string" for c in set(use.values()) if c != cols["count"]}):
            chunk = chunk.rename(columns={v: k for k, v in use.items()})
            if apply_status:
                keep = {s.lower() for s in statuses}
                chunk = chunk[chunk["status"].str.strip().str.lower().isin(keep)]
            if cols["region"] is None:
                chunk["region"] = NATIONAL_LABEL
            chunk["count"] = pd.to_numeric(chunk["count"], errors="coerce").fillna(0)
            parts.append(chunk.groupby(["region", "month", "wait"], as_index=False)["count"].sum())
    if not parts:
        raise ValueError(f"No rows read from {path}")
    out = pd.concat(parts).groupby(["region", "month", "wait"], as_index=False)["count"].sum()
    out["month"] = parse_month(out["month"])
    return out


_MONTH_NAMES = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def release_month_from_name(name: str) -> pd.Timestamp | None:
    """Publication month from a file name, e.g. 'Appointments_in_General_Practice_December_2024.zip',
    'Practice_Level_Crosstab_Dec_24.csv' -> 2024-12-01. None if no month+year is recognisable."""
    s = name.lower()
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s_\-\.]*(20\d\d|\d\d)(?!\d)", s)
    if not m:
        return None
    yr = int(m.group(2))
    yr = yr + 2000 if yr < 100 else yr
    return pd.Timestamp(year=yr, month=_MONTH_NAMES[m.group(1)], day=1)


def select_final_months(per_release: dict[pd.Timestamp, pd.DataFrame], months: pd.DatetimeIndex,
                        final_lag: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick, for every target month, the release that makes it final.

    NHS Digital refreshes the latest ``final_lag`` months in each release, so month m is settled in
    the release published at m + final_lag. Rule: among available releases published >= m + final_lag
    that contain m, take the LATEST (so a later restatement wins, but a provisional figure from an
    earlier release never does). If no such release exists the month is flagged provisional and the
    latest release containing it is used.

    Returns (rows for the chosen figures, provenance table month -> release, provisional flag).
    """
    chosen, prov = [], []
    for m in months:
        eligible = [r for r, d in per_release.items() if r >= m + pd.DateOffset(months=final_lag)
                    and (d["month"] == m).any()]
        provisional = not eligible
        pool = eligible or [r for r, d in per_release.items() if (d["month"] == m).any()]
        if not pool:
            prov.append({"month": m, "release": pd.NaT, "provisional": True})
            continue
        rel = max(pool)
        chosen.append(per_release[rel][per_release[rel]["month"] == m])
        prov.append({"month": m, "release": rel, "provisional": provisional})
    return (pd.concat(chosen) if chosen else pd.DataFrame()), pd.DataFrame(prov)


def build_regional_series(paths: list[Path], long_wait_days: int = config.LONG_WAIT_DAYS,
                          statuses: tuple[str, ...] | None = ("Attended",),
                          source: str = "nhs_digital", final_lag: int = 2, level: str = "national",
                          start: str = config.DATA_START, end: str = config.DATA_END,
                          provenance_out: Path | None = None, lookup_path: Path | None = None,
                          overrides_path: Path | None = None) -> pd.DataFrame:
    """Aggregate raw release files to a tidy regional monthly table.

    rate = appointments booked > ``long_wait_days`` ahead / appointments with a known wait.
    Each file is a release; its publication month is read from the file name (fallback: the latest
    month it contains). Month-to-release assignment follows ``select_final_months``.
    """
    per_release: dict[pd.Timestamp, pd.DataFrame] = {}
    for p in sorted(paths, key=lambda p: p.name):
        agg = aggregate_file(p, statuses, level)
        rel = release_month_from_name(p.name) or agg["month"].max()
        per_release[rel] = pd.concat([per_release[rel], agg]) if rel in per_release else agg
    months = pd.date_range(start, end, freq="MS")
    df, prov = select_final_months(per_release, months, final_lag)
    if provenance_out is not None:
        prov.to_csv(provenance_out, index=False)
    bad = prov[prov.provisional]
    if len(bad):
        print("WARNING: months without a final release (provisional or missing):",
              [m.strftime("%Y-%m") for m in bad.month])
    if level == "region_lookup":
        from .geography import apply_lookup
        df, unmapped = apply_lookup(df, pd.read_csv(lookup_path, dtype=str),
                                    pd.read_csv(overrides_path, dtype=str) if overrides_path and Path(overrides_path).exists() else None)
        if len(unmapped):
            tot = df["count"].sum() + unmapped["appointments"].sum()
            print(f"WARNING: {len(unmapped)} sub-ICB codes not in lookup = "
                  f"{unmapped.appointments.sum() / tot:.2%} of appointments; add them to the overrides CSV:")
            print(unmapped.sort_values("appointments", ascending=False).head(15).to_string(index=False))
            unmapped.to_csv(Path(provenance_out).with_name("unmapped_sub_icbs.csv") if provenance_out else "unmapped_sub_icbs.csv", index=False)
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
