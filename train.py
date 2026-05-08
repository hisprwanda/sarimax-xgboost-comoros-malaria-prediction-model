"""
CHAP training entry point.

Usage:
    python train.py <train_data.csv> <model_output_path>

Environment variables:
    CHAP_MODEL_TYPE   : sarimax | prophet | xgboost | ensemble_sx  (default: ensemble_sx)
    CHAP_N_SAMPLES    : int                                         (default: 50)
    CHAP_USE_FEATURES : 1 | 0   (default: 1 for sarimax/xgboost/ensemble_sx, 0 for prophet)

Model behaviour
---------------
ensemble_sx  — Trains SARIMAX(tuned) + XGBoost(calibrated) per district.
               SARIMAX uses district-specific features at |r| > 0.10 (Exp 03).
               XGBoost uses quantile regression calibration (Exp 05).
               Prediction concatenates 50+50 = 100 samples. Best overall CRPS.

sarimax      — SARIMAX(1,0,1). With CHAP_USE_FEATURES=1 uses district-specific
               informed feature selection; without features uses 3 base covariates.

prophet      — Tuned defaults: multiplicative seasonality, cps=0.1, sps=2.0 (Exp 04).

xgboost      — Quantile regression calibration, n_est=100, depth=4, lr=0.05 (Exp 05).
"""

import os
import sys
import pickle
import warnings
import pandas as pd

import model_lib as ml

warnings.filterwarnings("ignore")


def main(train_csv: str, model_path: str):
    model_type = os.environ.get("CHAP_MODEL_TYPE", "ensemble_sx").lower()
    n_samples  = int(os.environ.get("CHAP_N_SAMPLES", "50"))

    _default_features = "0" if model_type == "prophet" else "1"
    use_features = os.environ.get("CHAP_USE_FEATURES", _default_features) == "1"

    print(f"[train] model={model_type}  n_samples={n_samples}  use_features={use_features}")

    df = pd.read_csv(train_csv)
    _validate(df)

    if model_type == "ensemble_sx":
        _train_ensemble_sx(df, model_path, n_samples)
        return

    # ── Single-model path ──────────────────────────────────────────────────────
    available = [c for c in ml.DEFAULT_COVARIATES if c in df.columns]

    if use_features:
        df_feat = ml.add_engineered_features(df.copy())
        tail_data = df_feat.groupby("location", sort=False).tail(ml.N_LAG_ROWS).reset_index(drop=True)

        if model_type == "sarimax":
            # Per-district informed feature selection (Exp 03)
            feature_map = ml.compute_district_feature_map(df_feat)
            print(f"[train] using district-specific feature selection "
                  f"(threshold={ml.SARIMAX_FEATURE_THRESHOLD})")
            for loc, cols in feature_map.items():
                print(f"[train]   {loc}: {cols}")
        else:
            extra = [c for c in ml.EXTRA_COVARIATES if c in df_feat.columns]
            covariates  = available + extra
            feature_map = None
    else:
        df_feat    = df
        covariates = available
        tail_data  = None
        feature_map = None

    models = {}
    for loc, grp in df_feat.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y   = grp["disease_cases"].astype(float)

        if model_type == "sarimax":
            loc_covs = feature_map[loc] if feature_map else covariates
            X = grp[loc_covs].astype(float)
            models[loc] = {"payload": ml.fit_sarimax_one(y, X), "covariates": loc_covs}

        elif model_type == "prophet":
            times = grp["time_period"].apply(ml.isoweek_to_timestamp)
            pdf   = pd.DataFrame({"ds": times, "y": y.values})
            for c in covariates:
                pdf[c] = grp[c].values
            models[loc] = ml.fit_prophet_one(pdf, covariates)

        elif model_type == "xgboost":
            X = grp[covariates].astype(float)
            models[loc] = ml.fit_xgb_one(y, X, grp["time_period"])

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        print(f"[train]   fitted {loc}")

    # For SARIMAX with informed selection, covariates is stored per-district inside models
    global_covariates = None if (model_type == "sarimax" and feature_map) else covariates

    bundle = {
        "model_type":            model_type,
        "covariates":            global_covariates,
        "covariates_by_district": {loc: v["covariates"] for loc, v in models.items()}
                                   if (model_type == "sarimax" and feature_map) else None,
        "n_samples":             n_samples,
        "use_features":          use_features,
        "models":                {loc: v["payload"] if model_type == "sarimax" and feature_map
                                  else v for loc, v in models.items()},
        "tail_data":             tail_data,
    }

    _save(bundle, model_path)


def _train_ensemble_sx(df: pd.DataFrame, model_path: str, n_samples: int):
    """Train SARIMAX(tuned) + XGBoost(calibrated) and save as ensemble_sx bundle."""
    df_feat   = ml.add_engineered_features(df.copy())
    tail_data = df_feat.groupby("location", sort=False).tail(ml.N_LAG_ROWS).reset_index(drop=True)
    xgb_covs  = [c for c in ml.DEFAULT_COVARIATES + ml.EXTRA_COVARIATES if c in df_feat.columns]

    # SARIMAX per district with informed feature selection
    print("[train] ensemble_sx — fitting SARIMAX (informed features) ...")
    feature_map  = ml.compute_district_feature_map(df_feat)
    sarimax_models = {}
    for loc, grp in df_feat.groupby("location", sort=False):
        grp  = grp.sort_values("time_period")
        y    = grp["disease_cases"].astype(float)
        covs = feature_map[loc]
        X    = grp[covs].astype(float)
        sarimax_models[loc] = {"payload": ml.fit_sarimax_one(y, X), "covariates": covs}
        print(f"[train]   SARIMAX {loc}  covariates={covs}")

    # XGBoost per district with quantile regression
    print("[train] ensemble_sx — fitting XGBoost (quantile regression) ...")
    xgb_models = {}
    for loc, grp in df_feat.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y   = grp["disease_cases"].astype(float)
        X   = grp[xgb_covs].astype(float)
        xgb_models[loc] = ml.fit_xgb_one(y, X, grp["time_period"])
        print(f"[train]   XGBoost {loc}")

    bundle = {
        "model_type": "ensemble_sx",
        "n_samples":  n_samples,
        "sarimax": {
            "models":                {loc: v["payload"] for loc, v in sarimax_models.items()},
            "covariates_by_district": {loc: v["covariates"] for loc, v in sarimax_models.items()},
            "tail_data":             tail_data,
        },
        "xgboost": {
            "models":    xgb_models,
            "covariates": xgb_covs,
            "tail_data": tail_data,
        },
    }

    _save(bundle, model_path)


def _save(bundle: dict, model_path: str):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[train] saved → {model_path}")


def _validate(df: pd.DataFrame):
    required = {"time_period", "location", "disease_cases"}
    missing  = required - set(df.columns)
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
