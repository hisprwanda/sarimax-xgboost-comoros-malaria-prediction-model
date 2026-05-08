"""
CHAP prediction entry point.

Usage:
    python predict.py <model.pkl> <historic_data.csv> <future_data.csv> <out_file.csv>

Output CSV (CHAP standard):
    time_period, location, sample_0, sample_1, ..., sample_N
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

import model_lib as ml

warnings.filterwarnings("ignore")

RNG_SEED = 42


def main(model_path: str, historic_csv: str, future_csv: str, out_csv: str):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model_type = bundle["model_type"]
    n_samples  = bundle["n_samples"]
    rng        = np.random.default_rng(RNG_SEED)

    future_raw = pd.read_csv(future_csv)

    if model_type == "ensemble_sx":
        records = _predict_ensemble_sx(bundle, future_raw, n_samples, rng)
    else:
        records = _predict_single(bundle, future_raw, n_samples, rng)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[predict] wrote {len(out_df)} rows → {out_csv}")


# ── Single-model prediction ────────────────────────────────────────────────────

def _predict_single(bundle: dict, future_raw: pd.DataFrame,
                    n_samples: int, rng: np.random.Generator) -> list:
    model_type  = bundle["model_type"]
    use_features = bundle.get("use_features", False)
    models       = bundle["models"]
    tail_data    = bundle.get("tail_data")

    # Per-district covariates (SARIMAX with informed selection) or global list
    covariates_by_district = bundle.get("covariates_by_district")
    global_covariates      = bundle.get("covariates")

    future_df = _prepare_features(future_raw, use_features, tail_data)

    records = []
    for loc, grp in future_df.groupby("location", sort=False):
        if loc not in models:
            print(f"[predict] WARNING: no model for '{loc}', skipping")
            continue

        grp     = grp.sort_values("time_period")
        payload = models[loc]
        periods = grp["time_period"].values

        # Resolve covariates for this district
        covs = (covariates_by_district.get(loc) if covariates_by_district
                else global_covariates)

        # Fill any missing covariate columns with 0
        for c in covs:
            if c not in grp.columns:
                grp = grp.copy()
                grp[c] = 0.0

        X = grp[covs].astype(float)

        if model_type == "sarimax":
            samples = ml.predict_sarimax_one(payload, X, n_samples, rng)

        elif model_type == "prophet":
            times          = pd.Series(periods).apply(ml.isoweek_to_timestamp)
            future_prophet = pd.DataFrame({"ds": times})
            for c in covs:
                future_prophet[c] = X[c].values
            samples = ml.predict_prophet_one(payload, future_prophet, n_samples, rng)

        elif model_type == "xgboost":
            samples = ml.predict_xgb_one(payload, X, grp["time_period"], n_samples, rng)

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        records.extend(_rows(periods, loc, samples, n_samples))

    return records


# ── Ensemble S+X prediction ────────────────────────────────────────────────────

def _predict_ensemble_sx(bundle: dict, future_raw: pd.DataFrame,
                         n_samples: int, rng: np.random.Generator) -> list:
    sx_bundle  = bundle["sarimax"]
    xgb_bundle = bundle["xgboost"]

    future_feat = _prepare_features(future_raw, use_features=True,
                                    tail_data=sx_bundle["tail_data"])

    sarimax_covs = sx_bundle["covariates_by_district"]
    xgb_covs     = xgb_bundle["covariates"]

    records = []
    all_locs = sorted(future_feat["location"].unique())

    for loc in all_locs:
        s_grp = future_feat[future_feat["location"] == loc].sort_values("time_period")
        periods = s_grp["time_period"].values

        # ── SARIMAX half ──────────────────────────────────────────────────────
        if loc not in sx_bundle["models"]:
            print(f"[predict] WARNING: no SARIMAX model for '{loc}', skipping")
            continue

        s_covs   = sarimax_covs.get(loc, ml.DEFAULT_COVARIATES)
        for c in s_covs:
            if c not in s_grp.columns:
                s_grp = s_grp.copy()
                s_grp[c] = 0.0
        sX       = s_grp[s_covs].astype(float)
        s_samp   = ml.predict_sarimax_one(sx_bundle["models"][loc], sX, n_samples, rng)

        # ── XGBoost half ──────────────────────────────────────────────────────
        if loc not in xgb_bundle["models"]:
            print(f"[predict] WARNING: no XGBoost model for '{loc}', skipping")
            continue

        x_grp = future_feat[future_feat["location"] == loc].sort_values("time_period")
        for c in xgb_covs:
            if c not in x_grp.columns:
                x_grp = x_grp.copy()
                x_grp[c] = 0.0
        xX      = x_grp[xgb_covs].astype(float)
        x_samp  = ml.predict_xgb_one(xgb_bundle["models"][loc], xX,
                                      x_grp["time_period"], n_samples, rng)

        # ── Concatenate: 50 SARIMAX + 50 XGBoost = 100 samples ───────────────
        combined = np.hstack([s_samp, x_samp])   # (n_periods, 2*n_samples)
        records.extend(_rows(periods, loc, combined, 2 * n_samples))

    return records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare_features(future_raw: pd.DataFrame, use_features: bool,
                      tail_data) -> pd.DataFrame:
    if not use_features:
        return future_raw.copy()

    base = pd.concat([tail_data, future_raw], ignore_index=True) \
           if tail_data is not None else future_raw.copy()
    base = ml.add_engineered_features(base)
    future_times = set(future_raw["time_period"])
    return base[base["time_period"].isin(future_times)].reset_index(drop=True)


def _rows(periods, loc: str, samples: np.ndarray, n: int) -> list:
    rows = []
    for i, tp in enumerate(periods):
        row = {"time_period": tp, "location": loc}
        for j in range(n):
            row[f"sample_{j}"] = samples[i, j]
        rows.append(row)
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: python predict.py <model.pkl> <historic.csv> <future.csv> <out.csv>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
