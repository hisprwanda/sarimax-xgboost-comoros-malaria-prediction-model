# Comoros Malaria Climate-Health Forecasting

Probabilistic weekly malaria forecasting for the **Ngadjizi region, Comoros** (7 districts),
using climate drivers — rainfall, temperature, and humidity.

Built on the [CHAP](https://github.com/dhis2-chap) (Climate Health Analytics Platform) model
specification. Models are fitted independently per district and produce calibrated probabilistic
forecasts as sample distributions.

---

## The Problem

Malaria transmission in Comoros is strongly seasonal, driven by the Indian Ocean climate.
The **hot season is the wet season**: high temperatures heat the ocean surface, driving
evaporation and rainfall. This means temperature and rainfall are positively co-seasonal —
a warm week predicts a wet week 1–3 weeks later, and both predict a mosquito breeding surge
1–4 weeks after that.

Early, calibrated forecasts let district health teams pre-position rapid diagnostic tests,
antimalarials, and bed nets *before* a transmission peak — not after it.

---

## What This Project Does

An 8-experiment research series that systematically builds and improves a probabilistic
malaria forecasting system:

| Step | What was done |
|---|---|
| Baseline benchmark | 7 models evaluated across 78 weeks train / 26 weeks test |
| Exp 01 | Per-district climate–case cross-correlation & lag analysis |
| Exp 02 | SARIMAX auto-ARIMA order selection vs fixed (1,0,1) |
| Exp 03 | District-specific informed feature selection (\|r\| > 0.10 threshold) |
| Exp 04 | Prophet hyperparameter tuning (40-config grid search) |
| Exp 05 | XGBoost calibration fix — residual bootstrap → quantile regression |
| Exp 06 | Rebuilt S+P+X ensemble with all tuned components |
| Exp 07 | **S+X ensemble discovered as champion** — Prophet removed from pool |
| Exp 08 | **Operational robustness** — CRPS degradation under climate forecast noise |

---

## Champion Model — Ensemble S+X

| Metric | Value |
|---|---|
| **CRPS** | **25.91** |
| **RMSE** | **48.84 cases/week** |
| **80% PI Coverage** | **76.9%** |
| **95% PI Coverage** | **94.0%** |

Two model families combined via **sample concatenation** (50 + 50 = 100 samples):

- **SARIMAX(1,0,1) tuned** — linear state-space model with district-specific climate lag features.
  Captures week-to-week case momentum and responds to lagged climate signals.
- **XGBoost calibrated** — gradient-boosted trees with quantile regression uncertainty.
  Captures non-linear covariate interactions (e.g. cases spike only when rainfall AND temperature
  jointly exceed thresholds).

These two are structurally orthogonal — SARIMAX models *temporal autocorrelation*, XGBoost
models *covariate response* — which is what makes their combination genuinely additive.
Prophet was the weakest component (CRPS 37.4) and its removal improved ensemble accuracy by
1.4 CRPS points.

> Use **Ensemble S+P+X** (CRPS 27.1, cov80 80.8%) instead if an exact 80% PI calibration
> is a hard operational requirement.

---

## All Configurations — Final Benchmark

| Model | CRPS | RMSE | 80% Coverage | Notes |
|---|---|---|---|---|
| **Ensemble S+X** ⭐ | **25.91** | **48.84** | 76.9% | Champion — production recommended |
| SARIMAX tuned | 26.97 | 49.93 | 85.7% | Best standalone model |
| SARIMAX baseline | 27.01 | 49.57 | 81.9% | Simple and fast baseline |
| Ensemble S+P+X | 27.14 | 51.46 | 80.8% | Use when exact 80% coverage required |
| XGBoost calibrated | 27.83 | 50.93 | 53.3% | Good in ensemble; feature analysis |
| Prophet tuned | 37.42 | 64.83 | 61.5% | Seasonal decomposition only |

*Train: weeks 1–78 (2024-W01 → 2025-W26) · Test: weeks 79–104 (2025-W27 → 2025-W52)*
*7 districts · n_samples = 50 · CRPS = Continuous Ranked Probability Score (lower is better)*

---

## Project Structure

```
├── model_lib.py              # Core model implementations (SARIMAX, Prophet, XGBoost)
├── train.py                  # CHAP training entry point
├── predict.py                # CHAP prediction entry point
├── evaluate.py               # Metrics: CRPS, RMSE, MAE, R², PI coverage
├── run_benchmark.py          # 6-model benchmark runner
├── prepare_data.py           # Raw data → CHAP-compatible format
├── MLproject                 # CHAP model specification
├── config.yaml               # Runtime defaults
├── models-description.md     # Plain-language model descriptions for non-technical readers
│
├── experiments/
│   ├── exp01_feature_correlation.py
│   ├── exp02_sarimax_order_selection.py
│   ├── exp03_informed_feature_selection.py
│   ├── exp04_prophet_tuning.py
│   ├── exp05_xgboost_calibration.py
│   ├── exp06_improved_ensemble.py
│   ├── exp07_final_benchmark.py
│   └── exp08_forecast_uncertainty.py
│
├── output/
│   ├── benchmark_dashboard.html          # Interactive visual benchmark
│   ├── benchmark_results.csv
│   └── experiments/                      # Per-experiment HTML dashboards
│       ├── exp01_feature_correlation.html
│       ├── exp02_sarimax_order_selection.html
│       ├── exp03_informed_feature_selection.html
│       ├── exp04_prophet_tuning.html
│       ├── exp05_xgboost_calibration.html
│       ├── exp06_improved_ensemble.html
│       ├── exp07_final_benchmark.html
│       └── exp08_forecast_uncertainty.html
│
└── input/                    # gitignored — raw climate & health data
```

---

## Metrics Explained Simply

**CRPS (Continuous Ranked Probability Score)** — grades both accuracy and calibration together.
A model that says "expect 80 cases, range 50–120" and gets 110 scores better than one that
says "expect 80, range 78–82" and gets 110. Lower is better.

**80% PI Coverage** — if the model says "80% confident cases fall between X and Y", this
measures whether the truth actually lands in that band 80% of the time. A value of 76.9%
means the model is slightly conservative; 3.9% (old XGBoost) means the intervals are useless.

**RMSE** — average forecast error in cases per week. Easier to explain to district officers
("we're typically off by ~49 cases/week") but does not capture uncertainty quality.

---

## Climate Context

In Comoros, the **hot season = wet season**. The Indian Ocean heats up during the warm months,
driving evaporation and rainfall. Mosquito breeding peaks 1–4 weeks after heavy rain, and
adult mosquitoes survive longest in humid conditions. This creates measurable, forecastable
lag relationships between climate signals and malaria case counts — the biological foundation
this model exploits.

---

## CHAP Integration

```bash
# Train
python train.py input/training_data.csv output/model.pkl

# Predict
python predict.py output/model.pkl input/training_data.csv input/future_data.csv output/predictions.csv

# Environment variables
CHAP_MODEL_TYPE=sarimax      # sarimax | prophet | xgboost  (default: sarimax)
CHAP_N_SAMPLES=50            # probabilistic samples per location/week
CHAP_USE_FEATURES=0          # 1 = informed feature selection  (default: 0 for sarimax/prophet, 1 for xgboost)
```

---

## Re-run Trigger

When **3+ years of data** become available, re-run Exp 03 with threshold |r| > 0.08 and
re-run the full Exp 07 benchmark. SARIMAX + features (16 covariates) is expected to close
the gap or overtake the baseline as sample size grows — it was only held back by insufficient
data to fit 16 coefficients reliably on 78 training weeks.
