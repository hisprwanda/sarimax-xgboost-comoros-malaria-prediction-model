"""
Shared model implementations for Comoros malaria-climate forecasting.

Three model families:
  SARIMAX  — classical time-series with climate regressors (statsmodels)
  Prophet  — decomposable trend+seasonality model (Meta)
  XGBoost  — gradient-boosted trees on engineered features + time dummies

All produce probabilistic forecasts as (n_periods × n_samples) numpy arrays.
"""

import warnings
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

# CHAP-declared covariates (must be present in every input CSV)
# Comoros uses humidity, rainfall, and mean_temperature
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
            seasonality_mode="additive",
            changepoint_prior_scale=0.1,
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
    """Fit an XGBoost regressor for a single district.

    Probabilistic forecasts are generated via residual bootstrap.
    """
    import xgboost as xgb

    X_feat = _xgb_features(X, time_index)
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_feat, y.values)
    residuals = y.values - model.predict(X_feat)

    return {"model": model, "residuals": residuals}


def predict_xgb_one(
    payload: dict,
    future_X: pd.DataFrame,
    future_times: pd.Index,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples via residual bootstrap from a fitted XGBoost model.

    Returns (n_periods, n_samples) array, clipped at 0.
    """
    X_feat = _xgb_features(future_X, future_times)
    point_pred = payload["model"].predict(X_feat)
    residuals = payload["residuals"]

    samples = np.array([
        np.maximum(0, point_pred + rng.choice(residuals, size=len(point_pred)))
        for _ in range(n_samples)
    ]).T  # (n_periods, n_samples)

    return samples
