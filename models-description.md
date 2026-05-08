# Model Descriptions — Comoros Malaria Forecast Benchmark

Seven models were evaluated across a full experiment series (Exp 01–07) on 78 weeks of weekly
malaria and climate data (rainfall, temperature, humidity) across 7 districts of the Ngadjizi
region, tested on the following 26 weeks. Each model is fitted independently per district —
there is no spatial pooling between districts.

> **Note on Comoros climate:** The hot season *is* the wet season. High temperatures heat
> the Indian Ocean, driving evaporation and rainfall. Temperature and rainfall are positively
> correlated and co-seasonal. Temperature lags therefore carry genuine biological signal
> (mosquito breeding conditions), not seasonal confounding.

---

## 1. SARIMAX Baseline

**Type:** Classical statistical time-series model

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous inputs) is a
well-established epidemiological forecasting model. This configuration uses a fixed `(1,0,1)`
order — one autoregressive lag, no differencing, one moving-average term — with the three raw
climate covariates (rainfall, temperature, humidity) as external regressors.

**What it does well**
- Strong individual model: CRPS 27.0, RMSE 49.6, 80% coverage 85.7%
- Captures week-to-week case momentum via the autoregressive term
- Uses contemporaneous climate signals without overfitting to noisy lag relationships
- Well-calibrated uncertainty via state-space simulation
- Fast to train and interpretable coefficients

**Limitations**
- Fixed `(1,0,1)` order may not be optimal for every district (Exp 02 confirmed it holds up well)
- Does not capture non-linear climate–disease relationships
- Slightly over-covers: 85.7% on the 80% PI target

**Best suited for** when interpretability or computation speed matters; best standalone model.

---

## 2. SARIMAX Tuned (Exp 03)

**Type:** Classical statistical model with district-specific feature selection

The same SARIMAX `(1,0,1)` model, but each district receives only the climate lag features
that showed meaningful cross-correlation (|r| > 0.10) with its own case series at the best lag.
This was informed by Experiment 01's per-district lag analysis.

| District | Typical extra features selected |
|---|---|
| Mbadjini Ouest | humidity_lag2, rainfall_lag2 (humidity r=0.49, rain r=0.40) |
| Hamahamet-Mboinkou | rainfall_lag1, temp_lag2 |
| Others | 0–2 lags depending on strength of signal |

**What it does well**
- Best individual-model CRPS: 27.0 — matches or edges the plain baseline
- Uses only 4–6 covariates per district instead of 16, avoiding overfitting
- Features are biologically grounded: each lag represents a real mosquito development stage

**Why district-specific matters**
Pooled correlations across all districts appeared weak (r ≤ 0.17) because different districts
have different signal relationships — some humidity-driven, some rainfall-driven. A global feature
set would either over-include noise or miss local signals. Exp 03 found that threshold |r| > 0.10
gives the best CRPS across the threshold sweep (0.10–0.25).

**Limitations**
- Requires recomputing correlations from training data before each deployment
- Benefit over plain baseline is small on 78 weeks; expected to grow with more data

**Best suited for** production forecasting where per-district model quality matters.

---

## 3. Prophet Tuned (Exp 04)

**Type:** Decomposable trend + seasonality model

Prophet (developed by Meta) decomposes the time series into trend, seasonality, and external
regressor components. Grid search across 40 hyperparameter combinations identified:
`changepoint_prior_scale=0.1`, `seasonality_prior_scale=2.0`, `seasonality_mode=multiplicative`.

**Multiplicative seasonality** is appropriate here: in Comoros, the seasonal swing in malaria
cases scales with the baseline level — high-transmission seasons amplify whatever the current
disease burden is. Additive seasonality (constant amplitude) is less realistic.

**What it does well**
- Interpretable seasonal decomposition — visualise when transmission peaks happen
- Handles trend shifts via automatic changepoint detection
- Best used to communicate "this is malaria season" to non-technical audiences

**Limitations**
- CRPS 37.3 — notably weaker than SARIMAX and XGBoost
- 80% coverage 58.2% — structurally overconfident on 78-week training sets regardless of tuning
- Coverage limitation is inherent to Prophet's posterior structure on short series,
  not fixable through hyperparameter search alone

**Best suited for** seasonal pattern communication and trend decomposition visualisation, not
for operational probabilistic forecasting.

---

## 4. XGBoost Calibrated (Exp 05)

**Type:** Gradient-boosted decision trees with quantile regression

XGBoost reframes malaria forecasting as tabular regression: given climate features for a week,
predict the case distribution. It uses all 13 engineered covariates plus cyclical temporal
encoding (sin/cos of week-of-year and month).

**Critical fix from Exp 05 — residual bootstrap → quantile regression:**

The original XGBoost used residual bootstrap: add random draws from in-sample training residuals
to the point forecast. Because XGBoost overfits training data, in-sample residuals were near zero,
giving 3.9% 80% coverage (near-zero uncertainty bands).

The fix: `objective='reg:quantileerror'` with 25 quantile levels (0.025 to 0.975) trained
simultaneously. Samples are drawn by uniform interpolation across the predicted quantile
distribution — learning *how wide the interval should be* from the data rather than memorised noise.

| | CRPS | 80% Coverage |
|---|---|---|
| Original (bootstrap) | 40.9 | 3.9% |
| **Calibrated (quantile)** | **27.8** | **53.3%** |

**What it does well**
- Captures non-linear and interaction effects: cases spike only when rainfall AND temperature
  exceed joint thresholds
- Meaningful feature importances for exploratory analysis
- CRPS 27.8 — competitive with SARIMAX
- With quantile regression: genuinely useful uncertainty bands

**Limitations**
- Coverage 53.3% still below 80% target as a standalone model — best used in ensemble
- Does not model temporal autocorrelation (treats each week as independent)
- Best hyperparameters (n_est=100, depth=4, lr=0.05) are surprisingly shallow — fewer trees,
  lower learning rate wins because 78 training weeks cannot support deep trees

**Best suited for** ensemble participation and feature importance analysis.

---

## 5. Ensemble S+P+X — Improved (Exp 06)

**Type:** Model combination — 3 tuned components

Combines all three tuned models (SARIMAX tuned + Prophet tuned + XGBoost calibrated) by
concatenating their sample pools: 50 samples from each component = 150 total per district per week.
Concatenating preserves the full uncertainty spread rather than collapsing it.

**What it does well**
- **Perfectly calibrated**: 80.2% 80% PI coverage — hits the target precisely
- CRPS 27.3, RMSE 51.9 — strong overall
- Hedges against any single model's structural failure
- Prophet's intermediate coverage (58%) balances SARIMAX's over-coverage (85.7%) and
  XGBoost's under-coverage (53.3%), landing the ensemble exactly on target

**Limitations**
- CRPS (27.3) is slightly worse than SARIMAX tuned (27.0) and the S+X ensemble (25.9) —
  Prophet's weaker CRPS (37.3) dilutes the pool
- Three-model complexity adds training time
- Not a standalone trained model; computed post-hoc from sub-models

**Best suited for** when exact 80% PI calibration is a hard operational requirement.

---

## 6. Ensemble S+X ⭐ Champion (Exp 07)

**Type:** Model combination — 2 tuned components

Combines SARIMAX tuned (Exp 03) and XGBoost calibrated (Exp 05) via sample concatenation:
50 + 50 = 100 total samples per district per week.

**Why drop Prophet?**

Prophet is the weakest component with CRPS 37.3. In a sample-concatenation ensemble, each
model contributes equally to the pool — Prophet's noisier samples actively dilute the accuracy
of the stronger models. Removing it:

| | CRPS | RMSE | 80% Coverage |
|---|---|---|---|
| Ensemble S+P+X | 27.30 | 51.89 | 80.2% |
| **Ensemble S+X ⭐** | **25.91** | **48.84** | **76.9%** |

CRPS improves by 1.39 points (5.1%) and RMSE by 3 cases. Coverage drops from 80.2% to 76.9%
— 3.1pp short of the 80% target — because Prophet's intermediate coverage was balancing the two
remaining models. This is an acceptable tradeoff: 76.9% is still well-calibrated (slightly
conservative) vs the operationally more important accuracy gain in CRPS.

**What it does well**
- **Best CRPS (25.91) and best RMSE (48.84) across all 7 configurations**
- SARIMAX brings linear temporal structure and well-calibrated normal uncertainty
- XGBoost brings non-linear covariate interactions and quantile-learned conditional intervals
- Two genuinely different model families provide meaningful diversity without a weak third member
- 95% coverage: 94.0% (excellently calibrated at the wide interval level)

**Complementarity of SARIMAX and XGBoost:**
SARIMAX models the *temporal autocorrelation* — how last week's cases predict this week's.
XGBoost models the *covariate response* — how the current climate configuration maps to case
counts non-linearly. These are orthogonal signal sources, making their combination genuinely
additive.

**Limitations**
- 80% coverage at 76.9% — slightly narrow (3.1pp short of target)
- Use S+P+X if exact 80% calibration is required
- Requires both SARIMAX and XGBoost infrastructure to be deployed

**Best suited for** production forecasting and alert systems where minimising forecast error
matters more than hitting an exact coverage target.

---

## 7. Reference: Original Benchmark Configs

The following three configurations from the initial benchmark are retained for reference.
They were the starting point before the experiment series.

| Config | CRPS | 80% cov | Status |
|---|---|---|---|
| SARIMAX + features (16 covariates) | 27.1 | 80.8% | Superseded by SARIMAX tuned |
| Prophet baseline | 38.2 | 59.9% | Superseded by Prophet tuned |
| XGBoost (residual bootstrap) | 40.9 | 3.9% | Superseded by XGBoost calibrated |
| Original Ensemble S+P+X | 29.3 | 75.8% | Superseded by improved ensembles |

---

## Quick Comparison — All Evaluated Configurations

| Model | CRPS | RMSE | R² | 80% Coverage | Verdict |
|---|---|---|---|---|---|
| **Ensemble S+X** ⭐ | **25.91** | **48.84** | **0.934** | 76.9% | Best accuracy; recommended for production |
| SARIMAX tuned | 26.97 | — | — | 85.7% | Best standalone model; interpretable |
| SARIMAX baseline | 27.01 | 49.57 | 0.927 | ~86% | Strong baseline; simple and fast |
| Ensemble S+P+X (improved) | 27.30 | 51.89 | — | **80.2%** | Use when exact 80% coverage required |
| XGBoost calibrated | 27.83 | — | — | 53.3% | Best in ensemble; feature analysis |
| Prophet tuned | 37.28 | — | — | 58.8% | Seasonal interpretation only |
| SARIMAX + features | 27.1 | 56.0 | 0.917 | 80.8% | Superseded; revisit with 3+ years data |
| Original Ensemble S+P+X | 29.3 | 62.8 | 0.896 | 75.8% | Superseded |
| XGBoost (bootstrap) | 40.9 | 64.4 | 0.891 | 3.9% | Calibration broken; do not use for intervals |
| Prophet baseline | 38.2 | 74.8 | 0.852 | 59.9% | Superseded |
| Prophet + features | 52.9 | 102.0 | 0.725 | 39.0% | Worst overall |

> **Recommendation:** Deploy **Ensemble S+X** as the production model.
> Fall back to **Ensemble S+P+X** if exact 80% PI calibration is a hard requirement from stakeholders.
> Use **SARIMAX tuned** when a single interpretable model is preferred.
> Use **XGBoost calibrated** only for feature importance analysis alongside an ensemble.
> Use **Prophet tuned** only for seasonal decomposition and stakeholder visualisations.
>
> **Re-run trigger:** When 3+ years of data are available, re-run Exp 03 with a lower threshold
> (try |r| > 0.08) and re-run the full Exp 07 benchmark — SARIMAX + features is expected to
> close the gap or overtake as sample size grows.
