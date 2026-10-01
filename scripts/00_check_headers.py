"""Inspect each raw file: headers, parsed release month, and value counts of the key columns.

Run first on real data: python scripts/00_check_headers.py
Prints one compact block per CSV (first 500k rows only, so it is quick).
"""
import sys

import pandas as pd

from nhs_forecast import config
from nhs_forecast.data import _iter_csv_handles, _resolve_columns, release_month_from_name

files = sorted(list(config.RAW_DIR.glob("*.csv")) + list(config.RAW_DIR.glob("*.zip")))
if not files:
    sys.exit(f"No .csv/.zip files in {config.RAW_DIR}")
for f in files:
    rel = release_month_from_name(f.name)
    print(f"\n== {f.name}  release month: {rel:%Y-%m}" if rel is not None else f"\n== {f.name}  release month NOT parsed")
    try:
        for name, fh in _iter_csv_handles(f):
            head = pd.read_csv(fh, nrows=500_000, dtype=str)
            try:
                cols = _resolve_columns(list(head.columns))
            except KeyError as e:
                print(f"  -- {name}: NO MATCH {e}")
                continue
            months = sorted(head[cols["month"]].unique())
            status = sorted(head[cols["status"]].unique()) if cols["status"] else "NO APPT_STATUS COLUMN"
            print(f"  -- {name}: months={months[:4]} status={status}")
        if f is files[0] or f.name.endswith(("Oct_24.zip", "Feb_23.zip")):  # one-off: show the vocab
            print("     wait bands:", sorted(head[cols["wait"]].unique()))
            if cols["category"]:
                print("     national categories:", sorted(head[cols["category"]].unique()))
    except Exception as e:
        print("  ERROR:", e)
