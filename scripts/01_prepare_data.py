"""Build the regional monthly long-wait series -> data/processed/regional_monthly.csv.

Real data:      python scripts/01_prepare_data.py            (reads data/raw/*.csv|*.zip)
Synthetic demo: python scripts/01_prepare_data.py --synthetic (NOT NHS data; labelled in 'source')
"""
import argparse
import sys
import tempfile
from pathlib import Path

from nhs_forecast import config
from nhs_forecast.data import build_regional_series, validate_complete
from nhs_forecast.synthetic import make_practice_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--long-wait-days", type=int, default=config.LONG_WAIT_DAYS)
    ap.add_argument("--all-statuses", action="store_true", help="include DNA/cancelled, not just Attended")
    ap.add_argument("--level", default="national", choices=["national", "sub_icb", "region_lookup"],
                    help="series: one national series (default) or one per sub-ICB location")
    ap.add_argument("--lookup", type=Path, default=config.PROCESSED_DIR / "sub_icb_region_lookup.csv")
    ap.add_argument("--overrides", type=Path, default=config.ROOT / "data" / "external" / "sub_icb_region_overrides.csv")
    ap.add_argument("--out", type=Path, default=config.REGIONAL_CSV)
    a = ap.parse_args()
    statuses = None if a.all_statuses else ("Attended",)

    if a.synthetic:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Practice_Level_Crosstab_synthetic.csv"
            make_practice_level().to_csv(p, index=False)
            out = build_regional_series([p], a.long_wait_days, statuses, source="synthetic", level="region")
    else:
        files = sorted(list(a.raw_dir.glob("*.csv")) + list(a.raw_dir.glob("*.zip")))
        if not files:
            sys.exit(f"No .csv/.zip files in {a.raw_dir}. Download the monthly 'Appointments in General "
                     "Practice' files (see README) or use --synthetic.")
        out = build_regional_series(files, a.long_wait_days, statuses, level=a.level, lookup_path=a.lookup, overrides_path=a.overrides,
                                     provenance_out=a.out.with_name("release_provenance.csv"))
    validate_complete(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {out.region.nunique()} regions x {out.month.nunique()} months, source={out.source.iloc[0]}")


if __name__ == "__main__":
    main()
