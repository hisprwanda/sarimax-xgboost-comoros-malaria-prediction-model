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
    covariates = bundle["covariates"]
    n_samples = bundle["n_samples"]
    use_features = bundle.get("use_features", False)
    models = bundle["models"]
    tail_data = bundle.get("tail_data")

    rng = np.random.default_rng(RNG_SEED)

    future_df = pd.read_csv(future_csv)

    if use_features:
        # Prepend tail_data so lags can bridge the train/predict boundary
        if tail_data is not None:
            combined = pd.concat([tail_data, future_df], ignore_index=True)
        else:
            combined = future_df.copy()
        combined = ml.add_engineered_features(combined)
        # Keep only the rows that belong to the true forecast horizon
        future_times = set(future_df["time_period"])
        future_df = combined[combined["time_period"].isin(future_times)].reset_index(drop=True)

    # Fill any covariates missing from future_df with 0
    for c in covariates:
        if c not in future_df.columns:
            future_df[c] = 0.0

    records = []
    for loc, grp in future_df.groupby("location", sort=False):
        grp = grp.sort_values("time_period")

        if loc not in models:
            print(f"[predict] WARNING: no model for '{loc}', skipping")
            continue

        payload = models[loc]
        X = grp[covariates].astype(float)
        periods = grp["time_period"].values

        if model_type == "sarimax":
            samples = ml.predict_sarimax_one(payload, X, n_samples, rng)

        elif model_type == "prophet":
            times = pd.Series(periods).apply(ml.isoweek_to_timestamp)
            future_prophet = pd.DataFrame({"ds": times})
            for c in covariates:
                future_prophet[c] = X[c].values
            samples = ml.predict_prophet_one(payload, future_prophet, n_samples, rng)

        elif model_type == "xgboost":
            samples = ml.predict_xgb_one(payload, X, grp["time_period"], n_samples, rng)

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        for i, tp in enumerate(periods):
            row = {"time_period": tp, "location": loc}
            for j in range(n_samples):
                row[f"sample_{j}"] = samples[i, j]
            records.append(row)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[predict] wrote {len(out_df)} rows → {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: python predict.py <model.pkl> <historic.csv> <future.csv> <out.csv>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
