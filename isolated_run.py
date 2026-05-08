"""
End-to-end test without CHAP infrastructure.

Runs the full train → predict pipeline on the pre-split input files
and evaluates against the held-out test period.

Usage
-----
    python isolated_run.py                              # SARIMAX + features (default)
    CHAP_MODEL_TYPE=prophet python isolated_run.py
    CHAP_MODEL_TYPE=xgboost python isolated_run.py
"""

import os
import subprocess
import sys

import pandas as pd
from evaluate import evaluate, print_results, load_ground_truth

TRAIN_CSV = os.path.join("input", "training_data.csv")
FUTURE_CSV = os.path.join("input", "future_data.csv")
MODEL_PKL = os.path.join("output", "model.pkl")
PRED_CSV = os.path.join("output", "predictions.csv")


def run(cmd: list):
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main():
    os.makedirs("output", exist_ok=True)

    for f in [TRAIN_CSV, FUTURE_CSV]:
        if not os.path.exists(f):
            sys.exit(
                f"Missing {f}. Run:\n    python prepare_data.py\nbefore isolated_run.py."
            )

    python = sys.executable

    print("─" * 50)
    print("Step 1: train")
    run([python, "train.py", TRAIN_CSV, MODEL_PKL])

    print("─" * 50)
    print("Step 2: predict")
    run([python, "predict.py", MODEL_PKL, TRAIN_CSV, FUTURE_CSV, PRED_CSV])

    print("─" * 50)
    print("Step 3: evaluate")
    truth_df = load_ground_truth()
    pred_df = pd.read_csv(PRED_CSV)
    model_type = os.environ.get("CHAP_MODEL_TYPE", "sarimax")
    r = evaluate(pred_df, truth_df, label=f"{model_type} (isolated run)")
    print_results(r)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
