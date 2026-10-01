"""Print the header row (and first rows) of each raw file, plus the release month parsed from its name.

Run this first on real data: python scripts/00_check_headers.py
"""
import sys
from pathlib import Path

import pandas as pd

from nhs_forecast import config
from nhs_forecast.data import _iter_csv_handles, _resolve_columns, release_month_from_name

files = sorted(list(config.RAW_DIR.glob("*.csv")) + list(config.RAW_DIR.glob("*.zip")))
if not files:
    sys.exit(f"No .csv/.zip files in {config.RAW_DIR}")
for f in files:
    rel = release_month_from_name(f.name)
    print(f"\n== {f.name}  (release month parsed: {rel:%Y-%m} )" if rel is not None
          else f"\n== {f.name}  (release month NOT parsed from name)")
    try:
        for name, fh in _iter_csv_handles(f):
            head = pd.read_csv(fh, nrows=3)
            print(f"  -- {name}\n  columns: {list(head.columns)}")
            try:
                print("  matched:", _resolve_columns(list(head.columns)))
            except KeyError as e:
                print("  NO MATCH:", e)
    except Exception as e:  # keep going so one odd file doesn't hide the rest
        print("  ERROR:", e)
