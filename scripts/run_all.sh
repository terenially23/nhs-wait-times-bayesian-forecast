#!/usr/bin/env bash
# Full pipeline. Usage: scripts/run_all.sh [--synthetic]
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${1:-}" = "--synthetic" ] && python scripts/01_prepare_data.py --synthetic || echo "using committed data/processed/regional_monthly.csv"
python scripts/02_fit_models.py --models uc pymc
python scripts/03_backtest.py --models uc pymc naive
python scripts/04_plot_forecast.py --models uc pymc
