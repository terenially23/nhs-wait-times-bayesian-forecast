"""Derive data/processed/sub_icb_region_lookup.csv from the ONS Postcode Directory.

    python scripts/00b_build_region_lookup.py --onspd "C:\\path\\to\\ONSPD_FEB_2024_UK.zip"   (or the extracted folder)

Needs the WHOLE ONSPD download (Data/ and Documents/), not just the big postcode CSV: the
Documents/ lookups map ONS codes to the ODS codes used in the appointments files.
"""
import argparse
from pathlib import Path

from nhs_forecast import config
from nhs_forecast.geography import build_lookup

ap = argparse.ArgumentParser()
ap.add_argument("--onspd", type=Path, required=True)
ap.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "sub_icb_region_lookup.csv")
a = ap.parse_args()
lk = build_lookup(a.onspd)
lk.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(lk)} sub-ICBs -> {lk.region.nunique()} regions")
print(lk.groupby("region").size().to_string())
weak = lk[lk.share < 0.99]
print(f"\nsub-ICBs whose majority region covers <99% of postcodes (should be ~none): {len(weak)}")
if len(weak):
    print(weak[["sub_icb_code", "sub_icb_name", "region", "share"]].to_string(index=False))
