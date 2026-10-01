# Probabilistic forecasting of NHS GP long-wait appointment rates

Short-term (3–6 month) forecasts of the share of GP appointments booked **more than 14 days ahead**,
by NHS England region, with prediction intervals judged on **calibration (interval coverage)** rather
than point accuracy. Two structural time-series models are compared: `statsmodels` `UnobservedComponents`
(Kalman filter / maximum likelihood) and a fully Bayesian `PyMC` model, plus a random-walk benchmark.

**Data used: real NHS Digital "Appointments in General Practice" practice-level crosstab releases,
Dec 2022 – Dec 2024 releases, covering Oct 2022 – Oct 2024.** (Nothing in `results/` is synthetic.)

## Headline results (held-out 2024, trained on Oct 2022 – Dec 2023)

Full table: [`results/coverage_table.md`](results/coverage_table.md). Nominal coverage is 80% / 95%.
Pooled over 7 regions; `n` is the number of region-month forecasts scored.

| Scheme | Model | n | 80% cov. | 95% cov. | 95% width (pp) | Interval score (95%) | RMSE (pp) |
|---|---|---|---|---|---|---|---|
| Fixed origin (Dec 2023, h=1..10) | **UC** | 70 | **49%** | **76%** | 4.2 | 12.9 | 1.87 |
| | **PyMC** | 70 | **83%** | **99%** | 9.6 | 9.6 | 1.76 |
| | Naive RW | 70 | 99% | 100% | 16.4 | 16.4 | 3.00 |
| Rolling origin (refit every 3 months, h≤6) | **UC** | 119 | 61% | 79% | 5.9 | 17.7 | 2.29 |
| | **PyMC** | 119 | 77% | 93% | 7.7 | 15.1 | 2.14 |
| | Naive RW | 119 | 89% | 94% | 11.8 | 18.4 | 2.47 |

What this says:

* **UC is over-confident.** Its intervals are too narrow (76% of actuals inside the 95% interval; 49% inside the 80%).
* **PyMC is close to calibrated and much sharper than the naive benchmark** (fixed-origin 80%: 83%; rolling 95%: 93%).
  Fixed-origin 95% coverage of 99% means it is somewhat conservative at long horizons.
* **The naive random walk "wins" coverage only by being very wide**; the interval score (which penalises width as
  well as misses) ranks PyMC best in both schemes. The random walk is better on point accuracy at h=1–2 (fixed origin,
  1-month-ahead RMSE well below the seasonal models), which is worth being straightforward about.
* **One event drives much of the miss.** October 2024 jumped in every region (e.g. East of England 17.7% → 25.3%; 22–29% outside London, London 9.3% → 11.8%),
  above anything in the 2022–23 training window. UC misses it in 6 of 7 regions. PyMC contains it at 95% in all 7 only
  because its intervals widen, and at 80% in just 2 of 7. **Excluding Oct 2024, fixed-origin coverage is PyMC 89% / 98%,
  UC 54% / 83%** (63 forecasts).
* **Both seasonal models under-forecast Jan 2024** by ~1.6–1.9 pp at h=1 (mean signed error, fixed origin). That is a
  start-point/seasonal-shape bias, not noise.

### Caveats on these numbers (please read before quoting them)

1. **Regions are strongly correlated.** All seven regions rise and fall together, so 70 pooled forecasts behave more like
   10 independent ones. The Wilson intervals in the table (which assume independence) are therefore too narrow, and the
   coverage percentages are noisy. Treat them as evidence about *direction* (UC over-confident), not precise rates.
2. **Only 15 training months** = 1.25 seasonal cycles. Seasonality is learned largely from two autumn peaks (Oct 2022,
   Sep–Oct 2023). Any 2024 behaviour not seen in 2022–23 (e.g. the larger Oct 2024 peak) is, by construction, a surprise.
3. **A single held-out year.** There is one 2024; rolling origins re-use the same months at different horizons.
4. Rolling-origin results use origins every 3 months (Dec 23, Mar 24, Jun 24, Sep 24); `--origin-step 1` is available.

## Sampler health (PyMC)

Fixed-origin fits, one per region ([`results/sampler_health.json`](results/sampler_health.json)):
max R-hat 1.00–1.02, min bulk ESS 448–859, **4–13 divergent transitions out of 2000 draws per region (0.2–0.65%)**.
R-hat/ESS are fine; the residual divergences are not zero, reflecting weak identification of the level-innovation scale
with so few observations (a `sigma → 0` funnel), so the tails of that parameter's posterior should be treated cautiously.
This is comparable to the one synthetic-data region I tested during development (9 divergences), so there is no sign that
real data behaves worse than the synthetic generator; there was no full synthetic run to compare against. Sampler
diagnostics were saved for the fixed-origin fits only, not for each rolling-origin refit.
Per-region ArviZ summaries, trace plots and `.nc` files are in `outputs/pymc_diagnostics/` after a run.

## Plots

* [`results/forecast_pymc.png`](results/forecast_pymc.png), [`results/forecast_uc.png`](results/forecast_uc.png): per-region
  forecast median with 80%/95% intervals over actuals; red crosses mark held-out actuals outside the 95% interval.

## Data and definitions

* **Source:** NHS Digital Appointments in General Practice, practice-level crosstab zips (`Practice_Level_Crosstab_<Mon>_<YY>.zip`).
* **Long wait:** `TIME_BETWEEN_BOOK_AND_APPT` band lower bound > 14 days (15–21, 22–28, >28 days).
  Rate = long-wait appointments / appointments with a known wait ("Unknown / Data Quality" excluded from both).
  The 14-day cut-off is an assumption; change with `--long-wait-days`.
* **Attended only** where the file has `APPT_STATUS`. **The Dec 2022 and Jan 2023 releases have no status column**, so
  **Oct and Nov 2022 are not filtered to "Attended"**. Those two months are slightly less comparable with the rest
  (rates in Oct 2022 and Oct 2023 are close, so the effect looks small, but it is untested).
* **Which release supplies which month (3-month refresh):** each release only finalises data from ~2 months earlier.
  For every month the figure comes from the latest release published **≥ 2 months later** that contains it
  (`select_final_months` in `src/nhs_forecast/data.py`; provenance written to `data/processed/release_provenance.csv`).
  All 25 months had a final release. Large months are split over several CSVs in a zip and are summed.
* **Regions:** the appointment files have only `SUB_ICB_LOCATION_CODE`. Sub-ICB → NHS England region was derived from the
  ONS Postcode Directory (Feb 2024): the majority `nhser` over each sub-ICB's live postcodes, bridged to ODS codes via the
  ONSPD `Documents/` lookups (`scripts/00b_build_region_lookup.py`; 106 sub-ICBs → 7 regions, none with a split majority).
  No sub-ICB codes in the data were left unmapped. The lookup uses April 2023 boundaries.
* **Not controlled for:** appointment *type*. The files include `NATIONAL_CATEGORY`; planned/vaccination clinics booked far
  ahead may drive the autumn peak. I have not tested this; it is an obvious next step and would change what the model is
  actually forecasting.

## Method and design choices

* **Logit scale.** Rates are modelled as `logit(rate)` so forecasts and intervals stay inside (0,1); back-transforming
  quantiles is exact because the logit is monotone. Regions are fitted separately.
* **Structure (identical in both models, for a fair comparison).** Local level with constant drift (`lldtrend`, random-walk
  level + drift) + **deterministic trigonometric seasonality with 2 harmonics** + observation noise.
  Two harmonics let the annual cycle carry an autumn peak and a semi-annual component for the spring bump, using 4
  parameters rather than 11 free monthly effects, which matters with 15 observations. Seasonality is fixed, not
  stochastic, for the same reason. The cost: two harmonics smooth a sharp ~2-month autumn peak, which likely contributes to
  under-predicting peak height. `--trend lltrend|llevel` is available for UC.
* **UC (`models/uc.py`).** `statsmodels.tsa.UnobservedComponents`, maximum-likelihood variances, Kalman-filter forecast
  variance. Intervals condition on the *estimated* variances, ignoring parameter uncertainty; with ~15 points the ML
  variances are noisy and tend to be under-estimated, which is the most likely reason for under-coverage here.
  ("Bayesian structural time series" is strictly the PyMC model; UC is its frequentist state-space cousin.)
* **PyMC (`models/bayes.py`).** Same model with weakly informative priors in logit units. The latent random walk is
  **marginalised analytically** (`y ~ MvNormal(level0 + drift·t + Xβ, σ_level²·min(s,s') + σ_obs²I)`), so NUTS samples only
  six parameters; an earlier version that sampled the latent states explicitly gave dozens of divergences and R-hat > 1.05.
  Forecasts are the exact Gaussian conditional given the data, drawn once per posterior sample, so they include
  parameter uncertainty. `target_accept=0.99`, 4 chains, seeded.
* **Naive benchmark (`models/naive.py`).** Logit random walk with variance estimated from training differences.
* **Validation.** Primary metric is empirical coverage of the 80% and 95% central intervals (Wilson CIs), with mean width
  and the Gneiting–Raftery interval score (rewards sharpness subject to calibration), and RMSE/MAE for reference.
  *Fixed origin:* fit once at Dec 2023, score Jan–Oct 2024 (h=1..10). *Rolling origin:* expanding-window refit, h≤6.
  Pooled over regions; reported by horizon bucket (1–3, 4–6, 7+).

## Things I would do next

1. More history (the NHS series is available from earlier than Oct 2022) so seasonality is learned from ≥3 cycles.
2. Break out or exclude planned/vaccination appointment categories and see whether the autumn peak is a category effect.
3. Hierarchical PyMC model sharing seasonal shape across regions (partial pooling), which also addresses the correlation
   caveat above; and a parameter-uncertainty-aware UC interval (bootstrap or Gaussian approximation) as a fairer UC baseline.
4. Calibrate the horizon-dependent width (PyMC is conservative at long horizons, over-confident at h=1–3: 71% at 80%).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -e .

# 1. (once, locally; raw files are large and git-ignored)
#    put the release zips in data/raw/, then:
python scripts/00_check_headers.py                      # inspect schema / release months
python scripts/00b_build_region_lookup.py --onspd path/to/ONSPD_FEB_2024_UK.zip
python scripts/01_prepare_data.py --level region_lookup # -> data/processed/regional_monthly.csv (committed)

# 2. everything downstream runs from the committed regional table (~15-25 min, mostly PyMC)
scripts/run_all.sh
#   or step by step:
python scripts/02_fit_models.py --models uc pymc        # fit on Oct22-Dec23, forecast 2024; ArviZ diagnostics
python scripts/03_backtest.py --models uc pymc naive    # coverage table -> outputs/tables/
python scripts/04_plot_forecast.py --models uc pymc
pytest                                                  # unit tests
```

`scripts/01_prepare_data.py --synthetic` generates clearly-labelled **synthetic** practice-level data in the same schema
for testing the pipeline without the NHS files; its outputs are tagged `source=synthetic` and are not NHS results.
Randomness is seeded (`config.SEED`; PyMC seed = SEED + region index). Layout: `src/nhs_forecast/` (library),
`scripts/` (numbered pipeline), `data/{raw,processed,external}`, `results/` (committed final outputs), `tests/`.
