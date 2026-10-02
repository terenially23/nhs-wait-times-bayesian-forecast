#!/usr/bin/env bash
# Full pipeline. Usage: scripts/run_all.sh [--synthetic]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--synthetic" ]; then python scripts/01_prepare_data.py --synthetic; else echo "using committed data/processed/regional_monthly.csv"; fi
python scripts/02_fit_models.py --models uc pymc            # train Oct22-Dec23, forecast 2024
python scripts/03_backtest.py --models uc pymc naive        # coverage table (2024 hold-out)
python scripts/04_plot_forecast.py --models uc pymc
python scripts/06_forward_test.py --models uc pymc naive    # train to Oct24, forecast later months (needs data beyond Oct 2024)
