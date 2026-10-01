"""Build a sub-ICB -> NHS England region lookup from the ONS Postcode Directory (ONSPD).

Why this is needed: the appointments files carry SUB_ICB_LOCATION_CODE (3-character ODS code,
e.g. '15A') but no region. ONSPD tags every postcode with a sub-ICB (`sicbl`) and NHS England
region (`nhser`), but as ONS/GSS codes (E38..., E40...). The ODS<->GSS bridge is in the ONSPD
`Documents/` lookup CSVs ('SICBL names and codes ...' has a ...CDH column holding the ODS code).

Sub-ICBs nest inside regions, so each sub-ICB's region is unique; we take the majority region over
its live postcodes and report how pure that majority is as a sanity check.

Column names are matched by pattern (e.g. ^SICBL\\d*CDH$) because ONS versions the suffix yearly.
"""
from __future__ import annotations

import re
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def _members(source: Path, pred):
    """Yield list of (name, opener) for files in a zip or folder whose relative path satisfies pred."""
    if source.is_file() and source.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(source)
        try:
            yield [(n, (lambda n=n: zf.open(n))) for n in zf.namelist() if pred(n)]
        finally:
            zf.close()
    else:
        rels = [p for p in source.rglob("*") if p.is_file() and pred(str(p.relative_to(source)).replace("\\", "/"))]
        yield [(str(p), (lambda p=p: open(p, "rb"))) for p in rels]


def _candidates(source: Path) -> list[str]:
    with _members(source, lambda n: not n.lower().endswith("/") and "multi_csv" not in n.lower()
                  and n.lower().endswith(".csv") and re.search(r"icb|nhs", n.lower()) is not None) as ms:
        return sorted(n for n, _ in ms)


def _find_col(cols, pattern):
    hits = [c for c in cols if re.fullmatch(pattern, c.strip(), flags=re.I)]
    return hits[0] if hits else None


def _read_names(source: Path, kind: str) -> pd.DataFrame:
    """Read the Documents lookup for 'SICBL' or 'NHSER': returns gss, ods, name."""
    keys = {"SICBL": ("sicbl", "subicb"), "NHSER": ("nhser", "nhsengland")}[kind]
    with _members(source, lambda n: n.lower().endswith(".csv") and "documents" in n.lower()
                  and any(k in re.sub(r"[^a-z]", "", n.lower().rsplit("/", 1)[-1]) for k in keys)
                  and "names" in n.lower()) as ms:
        if not ms:
            raise FileNotFoundError(
                f"No Documents/*{kind}* names CSV found under {source}.\n"
                f"Candidate files there:\n  " + "\n  ".join(_candidates(source)))
        name, opener = sorted(ms, key=lambda m: m[0])[-1]  # latest-named file
        with opener() as fh:
            df = pd.read_csv(fh, dtype=str, encoding="utf-8-sig")
    cols = list(df.columns)
    # ONS suffixes the prefix with the vintage (SICBL23CD ...); match on the ending only.
    gss, ods, nm = (_find_col(cols, r"[A-Za-z_ ]*\d*CD"), _find_col(cols, r"[A-Za-z_ ]*\d*CDH"), _find_col(cols, r"[A-Za-z_ ]*\d*NM"))
    if not (gss and ods and nm):
        raise KeyError(f"{name}: expected {kind}..CD/CDH/NM columns, saw {cols}")
    return df[[gss, ods, nm]].rename(columns={gss: "gss", ods: "ods", nm: "name"}).dropna(subset=["gss"])


def _postcode_pairs(source: Path) -> pd.DataFrame:
    """Counts of live postcodes per (sicbl, nhser) from the ONSPD data CSV(s)."""
    def is_data(n):
        n = n.lower()
        return n.endswith(".csv") and "onspd" in n and "documents" not in n and "/data/" in "/" + n
    with _members(source, is_data) as ms:
        if not ms:
            raise FileNotFoundError(f"No ONSPD data CSV (…/Data/ONSPD_*.csv) found under {source}")
        multi = [m for m in ms if "multi_csv" in m[0].lower()]
        use = multi or ms[:1]  # per-area files if present, else the single big file
        parts = []
        for name, opener in use:
            with opener() as fh:
                header = list(pd.read_csv(fh, nrows=0).columns)
            c_s, c_r, c_d = (_find_col(header, "sicbl"), _find_col(header, "nhser"), _find_col(header, "doterm"))
            if not (c_s and c_r):
                raise KeyError(f"{name}: needs 'sicbl' and 'nhser' columns; saw {header[:60]}...")
            with opener() as fh:
                for ch in pd.read_csv(fh, usecols=[c for c in (c_s, c_r, c_d) if c], dtype=str, chunksize=500_000):
                    if c_d:
                        ch = ch[ch[c_d].isna()]  # live postcodes only (no termination date)
                    ch = ch.dropna(subset=[c_s, c_r])
                    parts.append(ch.groupby([c_s, c_r]).size().rename("n").reset_index()
                                 .rename(columns={c_s: "sicbl_gss", c_r: "nhser_gss"}))
    return pd.concat(parts).groupby(["sicbl_gss", "nhser_gss"], as_index=False)["n"].sum()


def build_lookup(onspd: Path) -> pd.DataFrame:
    pairs = _postcode_pairs(onspd)
    tot = pairs.groupby("sicbl_gss")["n"].transform("sum")
    pairs["share"] = pairs["n"] / tot
    top = pairs.sort_values("n", ascending=False).drop_duplicates("sicbl_gss")
    sicbl, nhser = _read_names(onspd, "SICBL"), _read_names(onspd, "NHSER")
    out = (top.merge(sicbl.rename(columns={"gss": "sicbl_gss", "ods": "sub_icb_code", "name": "sub_icb_name"}), on="sicbl_gss", how="left")
              .merge(nhser.rename(columns={"gss": "nhser_gss", "ods": "region_code", "name": "region"}), on="nhser_gss", how="left"))
    out = out.dropna(subset=["sub_icb_code", "region"])
    return out[["sub_icb_code", "sub_icb_name", "region_code", "region", "share", "n"]].sort_values("sub_icb_code").reset_index(drop=True)


def apply_lookup(df: pd.DataFrame, lookup: pd.DataFrame, overrides: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map df['region'] (holding sub-ICB ODS codes) to region names; re-sum counts.
    Returns (aggregated df, table of unmapped sub-ICB codes with their appointment totals)."""
    m = dict(zip(lookup.sub_icb_code.str.strip().str.upper(), lookup.region))
    if overrides is not None:
        m.update(dict(zip(overrides.sub_icb_code.str.strip().str.upper(), overrides.region)))
    code = df["region"].str.strip().str.upper()
    mapped = code.map(m)
    unmapped = (df.assign(code=code)[mapped.isna()].groupby("code")["count"].sum().rename("appointments").reset_index())
    out = df.assign(region=mapped).dropna(subset=["region"])
    out = out.groupby(["region", "month", "wait"], as_index=False)["count"].sum()
    return out, unmapped
