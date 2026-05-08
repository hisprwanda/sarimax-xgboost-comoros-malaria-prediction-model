"""
Standalone evaluation framework for Comoros malaria forecasts.

Metrics
-------
Point   : RMSE, MAE, MAPE, R²  (computed on per-week median prediction)
Probabilistic: CRPS, 50/80/95% prediction-interval coverage, mean interval width

Usage
-----
    python evaluate.py                            # default: output/predictions.csv
    python evaluate.py output/my_predictions.csv  # custom predictions file

Ground truth is taken from the last (100 - TRAIN_PCT)% of evaluation_data.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

EVAL_CSV = os.path.join("input", "evaluation_data.csv")
DEFAULT_PRED_CSV = os.path.join("output", "predictions.csv")

TRAIN_WEEKS = 78   # training split; test = weeks 79-104


# ── CRPS ──────────────────────────────────────────────────────────────────────

def crps_ensemble(obs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS for ensemble forecasts using the NRG estimator.

    crps = E|X - y| - 0.5 * E|X - X'|

    Parameters
    ----------
    obs     : (n,) observed values
    samples : (n, m) ensemble samples

    Returns
    -------
    (n,) per-observation CRPS values
    """
    m = samples.shape[1]
    sorted_s = np.sort(samples, axis=1)

    term1 = np.mean(np.abs(sorted_s - obs[:, None]), axis=1)

    # Efficient Gini mean difference via cumulative trick
    ranks = np.arange(1, m + 1)
    gini = np.sum((2 * ranks - m - 1) * sorted_s, axis=1) / (m * (m - 1))
    return term1 - gini


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ground_truth(eval_csv: str = EVAL_CSV) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)
    all_weeks = sorted(df["time_period"].unique())
    test_weeks = set(all_weeks[TRAIN_WEEKS:])
    return df[df["time_period"].isin(test_weeks)][["time_period", "location", "disease_cases"]]


def load_predictions(pred_csv: str) -> pd.DataFrame:
    return pd.read_csv(pred_csv)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(pred_df: pd.DataFrame, truth_df: pd.DataFrame, label: str = "") -> dict:
    merged = pred_df.merge(truth_df, on=["time_period", "location"], how="inner")
    if merged.empty:
        print(f"[evaluate] WARNING: no rows matched for '{label}'")
        return {}

    sample_cols = [c for c in merged.columns if c.startswith("sample_")]
    samples = merged[sample_cols].values.astype(float)   # (n, m)
    obs = merged["disease_cases"].values.astype(float)

    median = np.median(samples, axis=1)
    mean_pred = np.mean(samples, axis=1)

    # ── Point metrics ──────────────────────────────────────────────────────────
    rmse = float(np.sqrt(np.mean((median - obs) ** 2)))
    mae = float(np.mean(np.abs(median - obs)))
    nonzero = obs > 0
    mape = float(np.mean(np.abs((median[nonzero] - obs[nonzero]) / obs[nonzero])) * 100) if nonzero.any() else np.nan
    ss_res = np.sum((obs - median) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # ── Probabilistic metrics ──────────────────────────────────────────────────
    crps_vals = crps_ensemble(obs, samples)
    crps = float(np.mean(crps_vals))

    def pi_coverage(q_lo, q_hi):
        lo = np.quantile(samples, q_lo, axis=1)
        hi = np.quantile(samples, q_hi, axis=1)
        return float(np.mean((obs >= lo) & (obs <= hi)))

    def pi_width(q_lo, q_hi):
        lo = np.quantile(samples, q_lo, axis=1)
        hi = np.quantile(samples, q_hi, axis=1)
        return float(np.mean(hi - lo))

    cov50 = pi_coverage(0.25, 0.75)
    cov80 = pi_coverage(0.10, 0.90)
    cov95 = pi_coverage(0.025, 0.975)
    width80 = pi_width(0.10, 0.90)

    # ── Per-district RMSE ──────────────────────────────────────────────────────
    district_rmse = {}
    for loc, grp in merged.groupby("location"):
        s = grp[sample_cols].values.astype(float)
        o = grp["disease_cases"].values.astype(float)
        district_rmse[loc] = float(np.sqrt(np.mean((np.median(s, axis=1) - o) ** 2)))

    return {
        "label": label,
        "n_obs": int(len(obs)),
        "rmse": rmse, "mae": mae, "mape": mape, "r2": r2,
        "crps": crps,
        "cov50": cov50, "cov80": cov80, "cov95": cov95,
        "width80": width80,
        "district_rmse": district_rmse,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_results(r: dict):
    if not r:
        return
    print(f"\n{'─'*60}")
    print(f"  {r['label']}   (n={r['n_obs']})")
    print(f"{'─'*60}")
    print(f"  RMSE : {r['rmse']:>8.1f}")
    print(f"  MAE  : {r['mae']:>8.1f}")
    print(f"  MAPE : {r['mape']:>7.1f}%")
    print(f"  R²   : {r['r2']:>8.3f}")
    print(f"  CRPS : {r['crps']:>8.1f}")
    print(f"  50% PI coverage : {r['cov50']*100:.1f}%  (ideal 50%)")
    print(f"  80% PI coverage : {r['cov80']*100:.1f}%  (ideal 80%)")
    print(f"  95% PI coverage : {r['cov95']*100:.1f}%  (ideal 95%)")
    print(f"  80% PI width    : {r['width80']:.1f}")

    print("\n  Per-district RMSE:")
    for loc, v in sorted(r["district_rmse"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(int(v / 50), 20)
        print(f"    {loc:<25s}  {v:>7.1f}  {bar}")


def print_comparison(results: list):
    print("\n\n" + "═" * 90)
    print("  BENCHMARK COMPARISON")
    print("═" * 90)
    hdr = f"{'Configuration':<28} {'RMSE':>7} {'MAE':>7} {'MAPE':>7} {'R²':>6} {'CRPS':>7} {'80%Cov':>7}"
    print(hdr)
    print("─" * 90)
    for r in results:
        if not r:
            continue
        print(
            f"  {r['label']:<26} "
            f"{r['rmse']:>7.0f} "
            f"{r['mae']:>7.0f} "
            f"{r['mape']:>6.1f}% "
            f"{r['r2']:>6.3f} "
            f"{r['crps']:>7.0f} "
            f"{r['cov80']*100:>6.1f}%"
        )
    print("═" * 90)
    best_crps = min(results, key=lambda x: x.get("crps", float("inf")))
    best_rmse = min(results, key=lambda x: x.get("rmse", float("inf")))
    print(f"\n  Best CRPS (probabilistic champion) : {best_crps['label']}")
    print(f"  Best RMSE (point forecast champion) : {best_rmse['label']}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pred_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRED_CSV
    if not os.path.exists(pred_csv):
        sys.exit(f"Predictions file not found: {pred_csv}")

    truth_df = load_ground_truth()
    pred_df = load_predictions(pred_csv)
    r = evaluate(pred_df, truth_df, label=os.path.basename(pred_csv))
    print_results(r)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
