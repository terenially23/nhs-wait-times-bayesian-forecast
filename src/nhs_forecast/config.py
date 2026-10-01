"""Project-wide constants. Everything that defines the experiment lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
REGIONAL_CSV = PROCESSED_DIR / "regional_monthly.csv"

SEED = 20241001

# Study window (inclusive, month-start timestamps)
DATA_START = "2022-10-01"
DATA_END = "2024-10-01"
TRAIN_END = "2023-12-01"  # train on Oct 2022 - Dec 2023; hold out 2024

# "Long wait" = booked more than this many days before the appointment.
# NHS Digital bands are: Same Day, 1 Day, 2-7, 8-14, 15-21, 22-28, >28, Unknown.
# ASSUMPTION: the dissertation's exact cut-off is not available in this repo.
LONG_WAIT_DAYS = 14

# Central prediction intervals reported (nominal coverage)
LEVELS = (0.80, 0.95)
QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)  # = median + the 80% / 95% bounds
SEASONAL_PERIOD = 12
SEASONAL_HARMONICS = 2  # 1 = annual cycle, 2 adds a semi-annual cycle (autumn + spring peaks)


def qname(q: float) -> str:
    return f"q{int(round(q * 1000)):03d}"
