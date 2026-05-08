"""
CHAP training entry point.

Usage:
    python train.py <train_data.csv> <model_output_path>

Environment variables:
    CHAP_MODEL_TYPE   : sarimax | prophet | xgboost  (default: sarimax)
    CHAP_N_SAMPLES    : int                          (default: 100)
    CHAP_USE_FEATURES : 1 | 0                        (default: 1 for sarimax/xgboost, 0 for prophet)
"""

import os
import sys
import pickle
import warnings
import pandas as pd

import model_lib as ml

warnings.filterwarnings("ignore")


def main(train_csv: str, model_path: str):
    model_type = os.environ.get("CHAP_MODEL_TYPE", "sarimax").lower()
    n_samples = int(os.environ.get("CHAP_N_SAMPLES", "50"))

    # Comoros benchmark result: baseline SARIMAX (no features) has better CRPS/RMSE/R²
    # than the engineered-features version (2 years of data insufficient for 16 extra coefficients).
    # XGBoost always needs features; Prophet is always better without them.
    _default_features = "1" if model_type == "xgboost" else "0"
    use_features = os.environ.get("CHAP_USE_FEATURES", _default_features) == "1"

    print(f"[train] model={model_type}  n_samples={n_samples}  use_features={use_features}")

    df = pd.read_csv(train_csv)
    _validate(df)

    # Resolve which covariates are present
    available = [c for c in ml.DEFAULT_COVARIATES if c in df.columns]

    if use_features:
        df = ml.add_engineered_features(df)
        extra_available = [c for c in ml.EXTRA_COVARIATES if c in df.columns]
        covariates = available + extra_available
        # Save tail rows so predict.py can bridge the lag boundary
        tail_data = df.groupby("location", sort=False).tail(ml.N_LAG_ROWS).reset_index(drop=True)
    else:
        covariates = available
        tail_data = None

    print(f"[train] covariates ({len(covariates)}): {covariates}")

    models = {}
    for loc, grp in df.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y = grp["disease_cases"].astype(float)

        if model_type == "sarimax":
            X = grp[covariates].astype(float)
            models[loc] = ml.fit_sarimax_one(y, X)

        elif model_type == "prophet":
            times = grp["time_period"].apply(ml.isoweek_to_timestamp)
            prophet_df = pd.DataFrame({"ds": times, "y": y.values})
            for c in covariates:
                prophet_df[c] = grp[c].values
            models[loc] = ml.fit_prophet_one(prophet_df, covariates)

        elif model_type == "xgboost":
            X = grp[covariates].astype(float)
            models[loc] = ml.fit_xgb_one(y, X, grp["time_period"])

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        print(f"[train]   fitted {loc}")

    bundle = {
        "model_type": model_type,
        "covariates": covariates,
        "n_samples": n_samples,
        "use_features": use_features,
        "models": models,
        "tail_data": tail_data,
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[train] saved → {model_path}")


def _validate(df: pd.DataFrame):
    required = {"time_period", "location", "disease_cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Null values found in required columns")
    dups = df.duplicated(subset=["location", "time_period"])
    if dups.any():
        raise ValueError(f"Duplicate (location, time_period) rows: {dups.sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python train.py <train_data.csv> <model_output_path>")
    main(sys.argv[1], sys.argv[2])
