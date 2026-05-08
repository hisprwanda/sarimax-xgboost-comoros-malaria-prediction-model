"""
Shared model implementations for Comoros malaria-climate forecasting.

Three model families:
  SARIMAX  — classical time-series with climate regressors (statsmodels)
  Prophet  — decomposable trend+seasonality model (Meta)
  XGBoost  — gradient-boosted trees on engineered features + time dummies

All produce probabilistic forecasts as (n_periods × n_samples) numpy arrays.

Tuned defaults (from Exp 01–05 experiment series):
  SARIMAX  — fixed (1,0,1) order; district-specific features at |r| > 0.10
  Prophet  — cps=0.1, sps=2.0, multiplicative seasonality
  XGBoost  — quantile regression (25 levels); n_est=100, depth=4, lr=0.05
"""

import warnings
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

# CHAP-declared covariates (must be present in every input CSV)
DEFAULT_COVARIATES = ["rainfall", "mean_temperature", "humidity"]

# Number of trailing training rows kept for lag bridging at the train/predict boundary
N_LAG_ROWS = 4

# Extra features added by add_engineered_features()
EXTRA_COVARIATES = [
    "rainfall_lag1", "rainfall_lag2", "rainfall_lag3", "rainfall_lag4",
    "temp_lag1", "temp_lag2",
    "humidity_lag1", "humidity_lag2",
    "rainfall_roll4", "temp_roll4", "humidity_roll4",
    "rain_x_temp",
    "rain_x_humidity",
]

DEFAULT_SARIMAX_ORDER = (1, 0, 1)

# Exp 03: minimum |r| for a lag feature to be included per district
SARIMAX_FEATURE_THRESHOLD = 0.10

# Exp 04: tuned Prophet hyperparameters
PROPHET_CPS  = 0.1            # changepoint_prior_scale
PROPHET_SPS  = 2.0            # seasonality_prior_scale
PROPHET_MODE = "multiplicative"

# Exp 05: quantile levels for XGBoost uncertainty (replaces residual bootstrap)
XGB_QUANTILE_LEVELS = np.linspace(0.025, 0.975, 25)
XGB_N_ESTIMATORS    = 100
XGB_MAX_DEPTH       = 4
XGB_LEARNING_RATE   = 0.05

# Lag → column name mapping used by compute_district_feature_map()
_LAG_COL = {
    ("rainfall",         1): "rainfall_lag1",
    ("rainfall",         2): "rainfall_lag2",
    ("rainfall",         3): "rainfall_lag3",
    ("rainfall",         4): "rainfall_lag4",
    ("mean_temperature", 1): "temp_lag1",
    ("mean_temperature", 2): "temp_lag2",
    ("humidity",         1): "humidity_lag1",
    ("humidity",         2): "humidity_lag2",
}
_AVAILABLE_LAGS = {
    "rainfall":         [1, 2, 3, 4],
    "mean_temperature": [1, 2],
    "humidity":         [1, 2],
}


# ── Week-period parsing ────────────────────────────────────────────────────────

def isoweek_to_timestamp(s: str) -> pd.Timestamp:
    """Convert a week-period string to the Monday of that ISO week.

    Accepts:
      'YYYY-Www'               — e.g. '2024-W01'
      'YYYYWww'                — e.g. '2024W01'
      'YYYY-MM-DD/YYYY-MM-DD'  — CHAP range; uses the start date
    """
    s = str(s).strip()
    if "/" in s:
        s = s.split("/")[0]
    if "W" in s.upper():
        s = s.upper().replace("W", "-W") if "-W" not in s.upper() else s
        # pandas ISO week parsing
        try:
            return pd.to_datetime(s + "-1", format="%G-W%V-%u")
        except Exception:
            pass
    return pd.to_datetime(s)


# ── Feature engineering ────────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged, rolling, and interaction features per district.

    Biological rationale
    --------------------
    rainfall_lag1-4   : mosquito breeding cycle (rain → standing water → larvae → adults) takes 2-4 weeks
    temp_lag1-2       : temperature governs larval development speed with ~1-2 week lag
    humidity_lag1-2   : humidity affects adult mosquito survival and biting rates
    *_roll4           : 4-week rolling mean captures sustained conditions
    rain_x_temp       : warm AND wet = ideal breeding environment
    rain_x_humidity   : wet AND humid = prolonged mosquito survival
    """
    out = df.copy()

    for loc, grp in out.groupby("location", sort=False):
        idx = grp.index
        r = grp["rainfall"]
        t = grp["mean_temperature"]
        h = grp["humidity"]

        for k in range(1, 5):
            out.loc[idx, f"rainfall_lag{k}"] = r.shift(k)
        for k in range(1, 3):
            out.loc[idx, f"temp_lag{k}"] = t.shift(k)
        for k in range(1, 3):
            out.loc[idx, f"humidity_lag{k}"] = h.shift(k)

        out.loc[idx, "rainfall_roll4"] = r.rolling(4, min_periods=1).mean()
        out.loc[idx, "temp_roll4"] = t.rolling(4, min_periods=1).mean()
        out.loc[idx, "humidity_roll4"] = h.rolling(4, min_periods=1).mean()

        out.loc[idx, "rain_x_temp"] = r * t
        out.loc[idx, "rain_x_humidity"] = r * h

        # Fill NaN introduced by shifting the first rows
        for col in EXTRA_COVARIATES:
            out.loc[idx, col] = out.loc[idx, col].bfill().ffill()

    return out


# ── District-specific feature selection (Exp 03) ──────────────────────────────

def compute_district_feature_map(
    train_df: pd.DataFrame,
    threshold: float = SARIMAX_FEATURE_THRESHOLD,
) -> dict:
    """Return {district: [covariate_columns]} using per-district cross-correlation.

    For each district and each climate variable, finds the lag 0–8 with the
    highest |Pearson r| against disease_cases. Includes the corresponding lag
    column when |r| > threshold and the lag is available in the engineered set.
    Always includes the 3 base covariates regardless of correlation.

    Call this on the training split only to avoid data leakage.
    """
    from scipy import stats as sp_stats

    MAX_LAG = 8
    districts = sorted(train_df["location"].unique())
    feature_map = {}

    for loc in districts:
        grp = train_df[train_df["location"] == loc].sort_values("time_period")
        y   = grp["disease_cases"].values
        cols = list(DEFAULT_COVARIATES)

        for var in DEFAULT_COVARIATES:
            x = grp[var].values
            best_lag, best_r = 0, 0.0
            for lag in range(MAX_LAG + 1):
                xi = x[:-lag] if lag > 0 else x
                yi = y[lag:]  if lag > 0 else y
                if len(xi) < 10:
                    continue
                r, _ = sp_stats.pearsonr(xi, yi)
                if abs(r) > abs(best_r):
                    best_r, best_lag = r, lag

            if abs(best_r) > threshold and best_lag > 0:
                avail = _AVAILABLE_LAGS.get(var, [])
                chosen = next((l for l in sorted(avail, reverse=True)
                               if l <= best_lag), None)
                if chosen is None and avail:
                    chosen = min(avail)
                if chosen is not None:
                    col = _LAG_COL.get((var, chosen))
                    if col and col not in cols:
                        cols.append(col)

        feature_map[loc] = cols

    return feature_map


# ── SARIMAX ────────────────────────────────────────────────────────────────────

def fit_sarimax_one(y: pd.Series, X: pd.DataFrame) -> dict:
    """Fit a SARIMAX(1,0,1) model for a single district."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=y.values,
            exog=X.values,
            order=DEFAULT_SARIMAX_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)

    return {"order": DEFAULT_SARIMAX_ORDER, "fit": fit}


def predict_sarimax_one(
    payload: dict,
    future_X: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples probabilistic forecasts from a fitted SARIMAX.

    Returns (n_periods, n_samples) array, clipped at 0.
    """
    fit = payload["fit"]
    n_periods = len(future_X)
    samples = np.zeros((n_periods, n_samples))

    for i in range(n_samples):
        sim = fit.simulate(
            nsimulations=n_periods,
            exog=future_X.values,
            anchor="end",
            random_state=rng.integers(0, 2**31 - 1),
        )
        samples[:, i] = np.maximum(0, sim)

    return samples


# ── Prophet ────────────────────────────────────────────────────────────────────

def fit_prophet_one(y_df: pd.DataFrame, covariates: list) -> object:
    """Fit a Prophet model for a single district.

    y_df must have columns: ds (datetime), y (case count), + covariates.
    """
    from prophet import Prophet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=PROPHET_MODE,
            changepoint_prior_scale=PROPHET_CPS,
            seasonality_prior_scale=PROPHET_SPS,
        )
        for cov in covariates:
            m.add_regressor(cov, standardize=True)
        m.fit(y_df[["ds", "y"] + covariates])

    return m


def predict_prophet_one(
    model,
    future_df: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples from Prophet's posterior predictive distribution.

    Returns (n_periods, n_samples) array, clipped at 0.
    """
    raw = model.predictive_samples(future_df)
    pool = raw["yhat"]  # (n_periods, many_samples)

    n_periods = pool.shape[0]
    available = pool.shape[1]

    if available >= n_samples:
        chosen = rng.choice(available, size=n_samples, replace=False)
    else:
        chosen = rng.choice(available, size=n_samples, replace=True)

    return np.maximum(0, pool[:, chosen])


# ── XGBoost ────────────────────────────────────────────────────────────────────

def _xgb_features(X_df: pd.DataFrame, time_index: pd.Index) -> np.ndarray:
    """Build feature matrix for XGBoost: climate covariates + temporal encoding."""
    weeks = [isoweek_to_timestamp(t) for t in time_index]
    week_of_year = np.array([w.isocalendar()[1] for w in weeks], dtype=float)
    month = np.array([w.month for w in weeks], dtype=float)

    # Cyclical encoding prevents the model treating week 52→1 as a large jump
    sin_w = np.sin(2 * np.pi * week_of_year / 52)
    cos_w = np.cos(2 * np.pi * week_of_year / 52)
    sin_m = np.sin(2 * np.pi * month / 12)
    cos_m = np.cos(2 * np.pi * month / 12)

    temporal = np.column_stack([week_of_year, month, sin_w, cos_w, sin_m, cos_m])
    return np.hstack([X_df.values, temporal])


def fit_xgb_one(y: pd.Series, X: pd.DataFrame, time_index: pd.Index) -> dict:
    """Fit an XGBoost quantile regression model for a single district.

    Uses native multi-quantile objective (Exp 05 calibration fix).
    Replaces the old residual bootstrap which collapsed to near-zero intervals
    because XGBoost's in-sample residuals were too small.
    """
    import xgboost as xgb

    X_feat = _xgb_features(X, time_index)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=XGB_QUANTILE_LEVELS,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_feat, y.values)

    return {"model": model, "quantile_levels": XGB_QUANTILE_LEVELS}


def predict_xgb_one(
    payload: dict,
    future_X: pd.DataFrame,
    future_times: pd.Index,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples by interpolating across predicted quantile distribution.

    For each forecast step, draws uniform samples u ~ U(0,1) and interpolates
    across the predicted quantile function — correctly capturing covariate-
    conditional uncertainty without relying on in-sample residuals.

    Returns (n_periods, n_samples) array, clipped at 0.
    """
    X_feat  = _xgb_features(future_X, future_times)
    q_preds = payload["model"].predict(X_feat)   # (n_periods, n_quantiles)
    q_levels = payload["quantile_levels"]

    if q_preds.ndim == 1:
        q_preds = q_preds[:, None]
    q_preds = np.sort(q_preds, axis=1)           # enforce monotonicity

    n_periods = q_preds.shape[0]
    u = rng.uniform(0, 1, size=(n_periods, n_samples))
    samples = np.zeros((n_periods, n_samples))
    for t in range(n_periods):
        samples[t] = np.interp(u[t], q_levels, q_preds[t])

    return np.maximum(0, samples)
