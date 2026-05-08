"""
Prepare CHAP-formatted input files from the Comoros source CSV.

Reads:  input/Ngadjizi_climate_health_data.csv
Writes:
  input/evaluation_data.csv   — full 104 weeks, CHAP columns, with disease_cases
  input/training_data.csv     — first 78 weeks (2024-W01 → 2025-W26), with disease_cases
  input/future_data.csv       — last 26 weeks (2025-W27 → 2025-W52), covariates only
  input/true_future_data.csv  — 2026-W01 row per district, covariates only (live forecast)

Column mapping (source → CHAP):
  avg_rainfall    → rainfall
  mean_temperature → mean_temperature  (unchanged)
  avg_humidity    → humidity
  malaria_cases   → disease_cases
  location        → location  (unchanged)
  time_period     → time_period (unchanged)
"""

import os
import sys
import pandas as pd

TRAIN_WEEKS = 78   # out of 104 total historical weeks
SRC = os.path.join("input", "Ngadjizi_climate_health_data.csv")
OUT_DIR = "input"


def main():
    df = pd.read_csv(SRC)

    # Rename to CHAP convention
    df = df.rename(columns={
        "avg_rainfall": "rainfall",
        "avg_humidity": "humidity",
        "malaria_cases": "disease_cases",
    })

    # Separate rows without cases (2026-W01 forward) → true future
    true_future = df[df["disease_cases"].isna()].copy()
    historical = df[df["disease_cases"].notna()].copy()
    historical["disease_cases"] = historical["disease_cases"].astype(int)

    # Sort by location then time_period (lexicographic works for YYYY-Www)
    historical = historical.sort_values(["location", "time_period"]).reset_index(drop=True)

    # Validate: every location must have the same set of weeks
    weeks_per_loc = historical.groupby("location")["time_period"].apply(set)
    reference = weeks_per_loc.iloc[0]
    for loc, weeks in weeks_per_loc.items():
        if weeks != reference:
            raise ValueError(f"Location '{loc}' has different time periods than others")

    all_weeks = sorted(reference)
    n_total = len(all_weeks)
    print(f"Districts : {sorted(historical['location'].unique())}")
    print(f"Total weeks : {n_total}  ({all_weeks[0]} → {all_weeks[-1]})")

    if n_total < TRAIN_WEEKS + 1:
        raise ValueError(f"Need at least {TRAIN_WEEKS + 1} weeks; found {n_total}")

    train_weeks = set(all_weeks[:TRAIN_WEEKS])
    future_weeks = set(all_weeks[TRAIN_WEEKS:])
    print(f"Train       : {TRAIN_WEEKS} weeks  ({all_weeks[0]} → {all_weeks[TRAIN_WEEKS-1]})")
    print(f"Test        : {n_total - TRAIN_WEEKS} weeks  ({all_weeks[TRAIN_WEEKS]} → {all_weeks[-1]})")

    # evaluation_data.csv — full history, has disease_cases
    eval_cols = ["time_period", "location", "disease_cases", "rainfall", "mean_temperature", "humidity"]
    evaluation = historical[eval_cols]
    evaluation.to_csv(os.path.join(OUT_DIR, "evaluation_data.csv"), index=False)
    print(f"\nWrote evaluation_data.csv  ({len(evaluation)} rows)")

    # training_data.csv — first 78 weeks, with disease_cases
    training = historical[historical["time_period"].isin(train_weeks)][eval_cols]
    training = training.sort_values(["location", "time_period"]).reset_index(drop=True)
    training.to_csv(os.path.join(OUT_DIR, "training_data.csv"), index=False)
    print(f"Wrote training_data.csv    ({len(training)} rows, {len(train_weeks)} weeks)")

    # future_data.csv — last 26 weeks, covariates only (no disease_cases)
    cov_cols = ["time_period", "location", "rainfall", "mean_temperature", "humidity"]
    future = historical[historical["time_period"].isin(future_weeks)][cov_cols]
    future = future.sort_values(["location", "time_period"]).reset_index(drop=True)
    future.to_csv(os.path.join(OUT_DIR, "future_data.csv"), index=False)
    print(f"Wrote future_data.csv      ({len(future)} rows, {len(future_weeks)} weeks)")

    # true_future_data.csv — 2026-W01 per district, for live forecasting
    if not true_future.empty:
        true_future = true_future[cov_cols].sort_values(["location", "time_period"]).reset_index(drop=True)
        true_future.to_csv(os.path.join(OUT_DIR, "true_future_data.csv"), index=False)
        print(f"Wrote true_future_data.csv ({len(true_future)} rows)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
