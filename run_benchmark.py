"""
6-model benchmark for Comoros malaria-climate forecasting.

Configurations
--------------
1. SARIMAX baseline       — SARIMAX(1,0,1) with 3 base climate covariates
2. Prophet baseline       — yearly seasonality + 3 base covariates as regressors
3. SARIMAX + features     — SARIMAX with lagged/rolling/interaction features (13 extra)
4. Prophet + features     — Prophet with same engineered feature set
5. XGBoost                — gradient boosting on engineered features + temporal encoding
6. Ensemble (S+P+X)       — concatenated samples from configs 3, 2, and 5

Split: train on first 78 weeks (2024-W01 → 2025-W26)
       test  on last  26 weeks (2025-W27 → 2025-W52)

Usage
-----
    python run_benchmark.py            # full run (n_samples=100)
    python run_benchmark.py --quick    # fast check (n_samples=20)
"""

import os
import sys
import time
import pickle
import warnings
import tempfile

import numpy as np
import pandas as pd

import model_lib as ml
from evaluate import evaluate, print_comparison, load_ground_truth

warnings.filterwarnings("ignore")

EVAL_CSV = os.path.join("input", "evaluation_data.csv")
TRAIN_WEEKS = 78
N_SAMPLES = 20 if "--quick" in sys.argv else 50
RNG_SEED = 42

os.makedirs("output", exist_ok=True)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_and_split(with_features: bool = False):
    """Return (train_df, future_df, test_truth) with optional feature engineering."""
    df = pd.read_csv(EVAL_CSV)
    all_weeks = sorted(df["time_period"].unique())
    train_weeks = set(all_weeks[:TRAIN_WEEKS])
    test_weeks = set(all_weeks[TRAIN_WEEKS:])

    if with_features:
        df = ml.add_engineered_features(df)

    train_df = df[df["time_period"].isin(train_weeks)].copy()
    future_df = df[df["time_period"].isin(test_weeks)].copy()

    test_truth = future_df[["time_period", "location", "disease_cases"]].copy()
    future_cov = future_df.drop(columns=["disease_cases"])

    return train_df, future_cov, test_truth


def covariates_for(df, with_features: bool) -> list:
    base = [c for c in ml.DEFAULT_COVARIATES if c in df.columns]
    if not with_features:
        return base
    extra = [c for c in ml.EXTRA_COVARIATES if c in df.columns]
    return base + extra


# ── Training + prediction helpers ─────────────────────────────────────────────

def _fit_predict_sarimax(train_df, future_df, covariates, rng) -> pd.DataFrame:
    """Train SARIMAX per district and return predictions DataFrame."""
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y = grp["disease_cases"].astype(float)
        X = grp[covariates].astype(float)
        payload = ml.fit_sarimax_one(y, X)

        fgrp = future_df[future_df["location"] == loc].sort_values("time_period")
        fX = fgrp[covariates].astype(float)
        samples = ml.predict_sarimax_one(payload, fX, N_SAMPLES, rng)

        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period": tp, "location": loc}
            for j in range(N_SAMPLES):
                row[f"sample_{j}"] = samples[i, j]
            records.append(row)
    return pd.DataFrame(records)


def _fit_predict_prophet(train_df, future_df, covariates, rng) -> pd.DataFrame:
    """Train Prophet per district and return predictions DataFrame."""
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        times = grp["time_period"].apply(ml.isoweek_to_timestamp)
        y = grp["disease_cases"].astype(float)
        prophet_df = pd.DataFrame({"ds": times, "y": y.values})
        for c in covariates:
            prophet_df[c] = grp[c].values
        model = ml.fit_prophet_one(prophet_df, covariates)

        fgrp = future_df[future_df["location"] == loc].sort_values("time_period")
        ftimes = fgrp["time_period"].apply(ml.isoweek_to_timestamp)
        future_prophet = pd.DataFrame({"ds": ftimes})
        for c in covariates:
            future_prophet[c] = fgrp[c].values
        samples = ml.predict_prophet_one(model, future_prophet, N_SAMPLES, rng)

        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period": tp, "location": loc}
            for j in range(N_SAMPLES):
                row[f"sample_{j}"] = samples[i, j]
            records.append(row)
    return pd.DataFrame(records)


def _fit_predict_xgboost(train_df, future_df, covariates, rng) -> pd.DataFrame:
    """Train XGBoost per district and return predictions DataFrame."""
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y = grp["disease_cases"].astype(float)
        X = grp[covariates].astype(float)
        payload = ml.fit_xgb_one(y, X, grp["time_period"])

        fgrp = future_df[future_df["location"] == loc].sort_values("time_period")
        fX = fgrp[covariates].astype(float)
        samples = ml.predict_xgb_one(payload, fX, fgrp["time_period"], N_SAMPLES, rng)

        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period": tp, "location": loc}
            for j in range(N_SAMPLES):
                row[f"sample_{j}"] = samples[i, j]
            records.append(row)
    return pd.DataFrame(records)


def _ensemble(pred_a: pd.DataFrame, pred_b: pd.DataFrame, pred_c: pd.DataFrame) -> pd.DataFrame:
    """Concatenate samples from three models to form an ensemble.

    Concatenating (vs averaging medians) preserves the full uncertainty spread
    and avoids the calibration collapse caused by sample averaging.
    """
    keys = ["time_period", "location"]

    def _relabel(df, offset):
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        rename = {c: f"sample_{i + offset}" for i, c in enumerate(sample_cols)}
        return df.rename(columns=rename)

    n = N_SAMPLES
    a = _relabel(pred_a, 0)
    b = _relabel(pred_b, n)
    c = _relabel(pred_c, 2 * n)

    merged = a.merge(b[keys + [f"sample_{i}" for i in range(n, 2*n)]], on=keys, how="inner")
    merged = merged.merge(c[keys + [f"sample_{i}" for i in range(2*n, 3*n)]], on=keys, how="inner")
    return merged


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*70}")
    print(f"  Comoros Malaria Forecast — 6-Model Benchmark")
    print(f"  Train: weeks 1-{TRAIN_WEEKS} | Test: weeks {TRAIN_WEEKS+1}-104 | n_samples={N_SAMPLES}")
    print(f"{'='*70}\n")

    truth_df = load_ground_truth()

    results = []
    pred_cache = {}   # reuse fitted predictions across configs

    # ── Config 1: SARIMAX baseline ─────────────────────────────────────────────
    print("Config 1/6 : SARIMAX baseline …")
    t0 = time.perf_counter()
    train_df, future_df, _ = load_and_split(with_features=False)
    covs = covariates_for(train_df, with_features=False)
    rng = np.random.default_rng(RNG_SEED)
    pred = _fit_predict_sarimax(train_df, future_df, covs, rng)
    elapsed = time.perf_counter() - t0
    r = evaluate(pred, truth_df, "SARIMAX baseline")
    r["elapsed"] = elapsed
    results.append(r)
    pred_cache["sarimax_base"] = pred
    print(f"  → CRPS={r['crps']:.0f}  RMSE={r['rmse']:.0f}  80%cov={r['cov80']*100:.1f}%  ({elapsed:.0f}s)")

    # ── Config 2: Prophet baseline ─────────────────────────────────────────────
    print("Config 2/6 : Prophet baseline …")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred = _fit_predict_prophet(train_df, future_df, covs, rng)
    elapsed = time.perf_counter() - t0
    r = evaluate(pred, truth_df, "Prophet baseline")
    r["elapsed"] = elapsed
    results.append(r)
    pred_cache["prophet_base"] = pred
    print(f"  → CRPS={r['crps']:.0f}  RMSE={r['rmse']:.0f}  80%cov={r['cov80']*100:.1f}%  ({elapsed:.0f}s)")

    # ── Config 3: SARIMAX + features ───────────────────────────────────────────
    print("Config 3/6 : SARIMAX + features …")
    t0 = time.perf_counter()
    train_df_f, future_df_f, _ = load_and_split(with_features=True)
    covs_f = covariates_for(train_df_f, with_features=True)
    rng = np.random.default_rng(RNG_SEED)
    pred = _fit_predict_sarimax(train_df_f, future_df_f, covs_f, rng)
    elapsed = time.perf_counter() - t0
    r = evaluate(pred, truth_df, "SARIMAX + features")
    r["elapsed"] = elapsed
    results.append(r)
    pred_cache["sarimax_feat"] = pred
    print(f"  → CRPS={r['crps']:.0f}  RMSE={r['rmse']:.0f}  80%cov={r['cov80']*100:.1f}%  ({elapsed:.0f}s)")

    # ── Config 4: Prophet + features ───────────────────────────────────────────
    print("Config 4/6 : Prophet + features …")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred = _fit_predict_prophet(train_df_f, future_df_f, covs_f, rng)
    elapsed = time.perf_counter() - t0
    r = evaluate(pred, truth_df, "Prophet + features")
    r["elapsed"] = elapsed
    results.append(r)
    pred_cache["prophet_feat"] = pred
    print(f"  → CRPS={r['crps']:.0f}  RMSE={r['rmse']:.0f}  80%cov={r['cov80']*100:.1f}%  ({elapsed:.0f}s)")

    # ── Config 5: XGBoost ──────────────────────────────────────────────────────
    print("Config 5/6 : XGBoost …")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred = _fit_predict_xgboost(train_df_f, future_df_f, covs_f, rng)
    elapsed = time.perf_counter() - t0
    r = evaluate(pred, truth_df, "XGBoost")
    r["elapsed"] = elapsed
    results.append(r)
    pred_cache["xgboost"] = pred
    print(f"  → CRPS={r['crps']:.0f}  RMSE={r['rmse']:.0f}  80%cov={r['cov80']*100:.1f}%  ({elapsed:.0f}s)")

    # ── Config 6: Ensemble (SARIMAX+feat + Prophet baseline + XGBoost) ─────────
    print("Config 6/6 : Ensemble (S+feat  P+base  X) …")
    t0 = time.perf_counter()
    ensemble_pred = _ensemble(
        pred_cache["sarimax_feat"],
        pred_cache["prophet_base"],
        pred_cache["xgboost"],
    )
    elapsed = time.perf_counter() - t0
    r = evaluate(ensemble_pred, truth_df, "Ensemble (S+P+X)")
    r["elapsed"] = elapsed
    results.append(r)
    pred_cache["ensemble"] = ensemble_pred
    print(f"  → CRPS={r['crps']:.0f}  RMSE={r['rmse']:.0f}  80%cov={r['cov80']*100:.1f}%  ({elapsed:.0f}s)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print_comparison(results)

    # Save CSV
    rows = []
    for r in results:
        rows.append({
            "config": r["label"],
            "rmse": r["rmse"],
            "mae": r["mae"],
            "mape_pct": r["mape"],
            "r2": r["r2"],
            "crps": r["crps"],
            "cov80_pct": r["cov80"] * 100,
            "cov95_pct": r["cov95"] * 100,
            "width80": r["width80"],
            "elapsed_s": r.get("elapsed", 0),
        })
    csv_path = os.path.join("output", "benchmark_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # Save best model's predictions as default
    best = min(results, key=lambda x: x.get("crps", float("inf")))
    cache_key = {
        "SARIMAX baseline": "sarimax_base",
        "Prophet baseline": "prophet_base",
        "SARIMAX + features": "sarimax_feat",
        "Prophet + features": "prophet_feat",
        "XGBoost": "xgboost",
        "Ensemble (S+P+X)": "ensemble",
    }.get(best["label"], "sarimax_feat")
    best_pred = pred_cache.get(cache_key)

    if best_pred is not None:
        best_pred.to_csv(os.path.join("output", "predictions.csv"), index=False)
        print(f"  Best model ({best['label']}) predictions saved → output/predictions.csv")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run()
