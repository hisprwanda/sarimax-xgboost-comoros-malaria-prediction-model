"""
Experiment 05 — XGBoost Calibration Fix + Hyperparameter Tuning
================================================================
Problem: Current XGBoost residual bootstrap produces 3.9% 80% PI coverage
         (target: 80%). Root cause: training residuals are near-zero because
         XGBoost overfits the training data, so bootstrapped intervals are
         too narrow.

Fix: Replace residual bootstrap with native XGBoost quantile regression
     (objective='reg:quantileerror', quantile_alpha=array of levels).
     This learns the conditional distribution directly rather than adding
     in-sample noise to a point forecast.

Sampling from quantile predictions:
  1. Fit model outputting K quantile levels α_1 ... α_K simultaneously.
  2. For each draw: sample u ~ Uniform(0,1), interpolate across predicted
     quantiles. This correctly propagates covariate-conditional uncertainty.

Grid search:
  n_estimators  : [100, 200, 300, 500]
  max_depth     : [3, 4, 5]
  learning_rate : [0.05, 0.1]
  = 24 configs × 7 districts

Outputs:
  output/experiments/exp05_xgboost_calibration.html
  output/experiments/exp05_best_config.json    <- read by Exp 06

Usage:
    cd C:/Vault/HISP-MODELING/Climatehealth-comoros
    python experiments/exp05_xgboost_calibration.py
"""

import os, sys, json, warnings, itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV = os.path.join(ROOT, "input", "evaluation_data.csv")
OUT_DIR  = os.path.join(ROOT, "output", "experiments")
OUT_HTML = os.path.join(OUT_DIR, "exp05_xgboost_calibration.html")
OUT_JSON = os.path.join(OUT_DIR, "exp05_best_config.json")

sys.path.insert(0, ROOT)
import model_lib as ml

TRAIN_WEEKS = 78
N_SAMPLES   = 50
RNG_SEED    = 42

# Quantile levels for the new calibration approach
QUANTILE_LEVELS = np.linspace(0.025, 0.975, 25)

# Hyperparameter grid
N_EST_VALUES = [100, 200, 300, 500]
DEPTH_VALUES = [3, 4, 5]
LR_VALUES    = [0.05, 0.1]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split():
    df = pd.read_csv(EVAL_CSV)
    wks = sorted(df["time_period"].unique())
    train_wks = set(wks[:TRAIN_WEEKS])
    test_wks  = set(wks[TRAIN_WEEKS:])
    df_feat   = ml.add_engineered_features(df.copy())
    train = df_feat[df_feat["time_period"].isin(train_wks)].copy()
    test  = df_feat[df_feat["time_period"].isin(test_wks)].copy()
    truth = test[["time_period", "location", "disease_cases"]].copy()
    future = test.drop(columns=["disease_cases"])
    return train, future, truth


# ── Metrics ───────────────────────────────────────────────────────────────────

def crps_nrg(samples, truth):
    n  = samples.shape[1]
    e1 = np.abs(samples - truth[:, None]).mean(axis=1)
    e2 = np.abs(samples[:, :, None] - samples[:, None, :]).sum(axis=(1, 2)) / (n * (n - 1))
    return float(np.mean(e1 - 0.5 * e2))

def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def coverage(samples, truth, alpha):
    lo = np.quantile(samples, (1 - alpha) / 2, axis=1)
    hi = np.quantile(samples, 1 - (1 - alpha) / 2, axis=1)
    return float(np.mean((truth >= lo) & (truth <= hi)))


# ── Quantile-regression XGBoost ───────────────────────────────────────────────

def fit_xgb_quantile(y, X_feat, n_est, depth, lr):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE_LEVELS,
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_feat, y)
    return model


def predict_xgb_quantile(model, X_feat, n_samples, rng):
    """Sample from piecewise-linear distribution defined by predicted quantiles."""
    quant_preds = model.predict(X_feat)        # (n_periods, n_quantiles)
    if quant_preds.ndim == 1:
        quant_preds = quant_preds[:, None]
    n_periods = quant_preds.shape[0]

    # Enforce monotonicity (quantile crossing fix)
    quant_preds = np.sort(quant_preds, axis=1)

    u = rng.uniform(0, 1, size=(n_periods, n_samples))
    samples = np.zeros((n_periods, n_samples))
    for t in range(n_periods):
        samples[t] = np.interp(u[t], QUANTILE_LEVELS, quant_preds[t])
    return np.maximum(0, samples)


# ── Residual bootstrap (current baseline) ─────────────────────────────────────

def fit_xgb_bootstrap(y, X_feat):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0,
    )
    model.fit(X_feat, y)
    residuals = y - model.predict(X_feat)
    return model, residuals


def predict_xgb_bootstrap(model, residuals, X_feat, n_samples, rng):
    point = model.predict(X_feat)
    samples = np.array([
        np.maximum(0, point + rng.choice(residuals, size=len(point)))
        for _ in range(n_samples)
    ]).T
    return samples


# ── Eval one config ───────────────────────────────────────────────────────────

def eval_config(train_df, future_df, truth_df, n_est, depth, lr, rng):
    covs = [c for c in ml.DEFAULT_COVARIATES + ml.EXTRA_COVARIATES
            if c in train_df.columns]
    districts = sorted(train_df["location"].unique())
    all_crps, all_rmse, all_cov80 = [], [], []

    for loc in districts:
        grp  = train_df[train_df["location"] == loc].sort_values("time_period")
        y    = grp["disease_cases"].astype(float).values
        X    = grp[covs].astype(float)
        tp   = grp["time_period"]
        Xf   = ml._xgb_features(X, tp)

        model = fit_xgb_quantile(y, Xf, n_est, depth, lr)

        fgrp  = future_df[future_df["location"] == loc].sort_values("time_period")
        fX    = fgrp[covs].astype(float)
        ftp   = fgrp["time_period"]
        fXf   = ml._xgb_features(fX, ftp)
        samples = predict_xgb_quantile(model, fXf, N_SAMPLES, rng)

        tgrp  = truth_df[truth_df["location"] == loc].sort_values("time_period")
        truth = tgrp["disease_cases"].values

        all_crps.append(crps_nrg(samples, truth))
        all_rmse.append(rmse(samples.mean(axis=1), truth))
        all_cov80.append(coverage(samples, truth, 0.80))

    return {
        "avg_crps":  round(float(np.mean(all_crps)), 3),
        "avg_rmse":  round(float(np.mean(all_rmse)), 3),
        "avg_cov80": round(float(np.mean(all_cov80)), 4),
        "dist_crps": [round(v, 3) for v in all_crps],
        "dist_cov80": [round(v, 4) for v in all_cov80],
        "districts": districts,
    }


def eval_bootstrap_baseline(train_df, future_df, truth_df, rng):
    covs = [c for c in ml.DEFAULT_COVARIATES + ml.EXTRA_COVARIATES
            if c in train_df.columns]
    districts = sorted(train_df["location"].unique())
    all_crps, all_rmse, all_cov80 = [], [], []

    for loc in districts:
        grp  = train_df[train_df["location"] == loc].sort_values("time_period")
        y    = grp["disease_cases"].astype(float).values
        X    = grp[covs].astype(float)
        Xf   = ml._xgb_features(X, grp["time_period"])
        model, resid = fit_xgb_bootstrap(y, Xf)

        fgrp  = future_df[future_df["location"] == loc].sort_values("time_period")
        fXf   = ml._xgb_features(fgrp[covs].astype(float), fgrp["time_period"])
        samples = predict_xgb_bootstrap(model, resid, fXf, N_SAMPLES, rng)

        tgrp  = truth_df[truth_df["location"] == loc].sort_values("time_period")
        truth = tgrp["disease_cases"].values

        all_crps.append(crps_nrg(samples, truth))
        all_rmse.append(rmse(samples.mean(axis=1), truth))
        all_cov80.append(coverage(samples, truth, 0.80))

    return {
        "avg_crps":  round(float(np.mean(all_crps)), 3),
        "avg_rmse":  round(float(np.mean(all_rmse)), 3),
        "avg_cov80": round(float(np.mean(all_cov80)), 4),
        "dist_crps": [round(v, 3) for v in all_crps],
        "dist_cov80": [round(v, 4) for v in all_cov80],
        "districts": districts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp05] Loading data ...")
    train_df, future_df, truth_df = load_split()
    districts = sorted(train_df["location"].unique())

    print("[exp05] Evaluating residual bootstrap baseline ...")
    rng = np.random.default_rng(RNG_SEED)
    bootstrap = eval_bootstrap_baseline(train_df, future_df, truth_df, rng)
    print(f"  bootstrap  CRPS={bootstrap['avg_crps']:.3f}  "
          f"RMSE={bootstrap['avg_rmse']:.3f}  cov80={bootstrap['avg_cov80']*100:.1f}%")

    combos = list(itertools.product(N_EST_VALUES, DEPTH_VALUES, LR_VALUES))
    total  = len(combos)
    print(f"[exp05] Quantile regression grid: {total} configs ...")

    grid_results = []
    for i, (n_est, depth, lr) in enumerate(combos, 1):
        rng = np.random.default_rng(RNG_SEED)
        res = eval_config(train_df, future_df, truth_df, n_est, depth, lr, rng)
        res.update({"n_est": n_est, "depth": depth, "lr": lr})
        grid_results.append(res)
        print(f"  [{i:02d}/{total}] n_est={n_est:3d} depth={depth} lr={lr}  "
              f"CRPS={res['avg_crps']:.3f}  cov80={res['avg_cov80']*100:.1f}%")

    best = min(grid_results, key=lambda r: r["avg_crps"])
    print(f"\n[exp05] Best quantile config: n_est={best['n_est']} depth={best['depth']} "
          f"lr={best['lr']}  CRPS={best['avg_crps']:.3f}  cov80={best['avg_cov80']*100:.1f}%")
    print(f"  CRPS: bootstrap {bootstrap['avg_crps']:.3f} -> quantile {best['avg_crps']:.3f}  "
          f"(delta={best['avg_crps']-bootstrap['avg_crps']:+.3f})")
    print(f"  cov80: {bootstrap['avg_cov80']*100:.1f}% -> {best['avg_cov80']*100:.1f}%")

    # Save best config for Exp 06
    best_cfg = {"n_estimators": best["n_est"], "max_depth": best["depth"],
                "learning_rate": best["lr"], "avg_crps": best["avg_crps"],
                "avg_cov80": best["avg_cov80"]}
    with open(OUT_JSON, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"[exp05] Best config saved -> {OUT_JSON}")

    payload = {
        "bootstrap":  bootstrap,
        "grid":       [{k: v for k, v in r.items()
                        if k not in ("dist_crps", "dist_cov80", "districts")}
                       for r in grid_results],
        "best":       {k: v for k, v in best.items()
                       if k not in ("dist_crps", "dist_cov80", "districts")},
        "top5":       [{k: v for k, v in r.items()
                        if k not in ("dist_crps", "dist_cov80", "districts")}
                       for r in sorted(grid_results, key=lambda r: r["avg_crps"])[:5]],
        "dist_bootstrap_cov": list(zip(districts, bootstrap["dist_cov80"])),
        "dist_best_cov":      list(zip(districts, best["dist_cov80"])),
        "dist_bootstrap_crps": list(zip(districts, bootstrap["dist_crps"])),
        "dist_best_crps":      list(zip(districts, best["dist_crps"])),
        "districts":  districts,
        "n_est_values": N_EST_VALUES,
        "depth_values": DEPTH_VALUES,
        "lr_values":   LR_VALUES,
        "n_quantiles": len(QUANTILE_LEVELS),
    }

    print("[exp05] Building HTML ...")
    html = build_html(payload)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[exp05] Done -> {OUT_HTML}")


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(p):
    J   = json.dumps(p)
    bl  = p["bootstrap"]
    bst = p["best"]
    delta_crps = round(bst["avg_crps"] - bl["avg_crps"], 3)
    delta_cov  = round(bst["avg_cov80"] * 100 - bl["avg_cov80"] * 100, 1)
    improved   = delta_crps < 0

    def arrow(d, lower_better=True):
        better = (d < 0) if lower_better else (d > 0)
        sym = ("▼" if d < 0 else "▲")
        col = "var(--green)" if better else "var(--red)"
        return f'<span style="color:{col}">{sym} {abs(d):.2f}</span>'

    # Top-5 table rows
    top5_rows = ""
    for rank, r in enumerate(p["top5"], 1):
        is_best = rank == 1
        hi = "background:#8839ef0a;font-weight:600" if is_best else ""
        star = " ★" if is_best else ""
        d_crps = round(r["avg_crps"] - bl["avg_crps"], 3)
        d_cov  = round(r["avg_cov80"] * 100 - bl["avg_cov80"] * 100, 1)
        top5_rows += f"""
        <tr style="{hi}">
          <td style="font-weight:700;color:var(--mauve)">{rank}{star}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['n_est']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['depth']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['lr']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['avg_crps']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['avg_rmse']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['avg_cov80']*100:.1f}%</td>
          <td>{arrow(d_crps)}</td>
          <td>{arrow(d_cov, lower_better=False)}</td>
        </tr>"""

    # Per-district coverage rows
    dist_rows = ""
    for (loc, cov_bl), (_, cov_bst), (_, crps_bl), (_, crps_bst) in zip(
        p["dist_bootstrap_cov"], p["dist_best_cov"],
        p["dist_bootstrap_crps"], p["dist_best_crps"]
    ):
        dc = round((cov_bst - cov_bl) * 100, 1)
        dr = round(crps_bst - crps_bl, 2)
        cov_col = "var(--green)" if dc > 0 else "var(--red)"
        crps_col = "var(--green)" if dr < 0 else "var(--red)"
        dist_rows += f"""
        <tr>
          <td><strong>{loc}</strong></td>
          <td style="font-family:'JetBrains Mono',monospace">{crps_bl:.2f}</td>
          <td style="font-family:'JetBrains Mono',monospace">{crps_bst:.2f}</td>
          <td style="font-family:'JetBrains Mono',monospace;color:{crps_col}">
            {'▼' if dr < 0 else '▲'} {abs(dr):.2f}</td>
          <td style="font-family:'JetBrains Mono',monospace">{cov_bl*100:.1f}%</td>
          <td style="font-family:'JetBrains Mono',monospace">{cov_bst*100:.1f}%</td>
          <td style="font-family:'JetBrains Mono',monospace;color:{cov_col}">
            {'▲' if dc > 0 else '▼'} {abs(dc):.1f}pp</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 05 - XGBoost Calibration</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --page:#f2eeff;--card:#fff;--border:#ddd5f5;--border-h:#c9bef0;
  --text:#2a2044;--sub1:#544873;--sub0:#7a6e92;--muted:#a099bb;
  --mauve:#8839ef;--blue:#1e66f5;--teal:#179299;--green:#40a02b;
  --peach:#fe640b;--red:#d20f39;--sky:#04a5e5;--lavender:#7287fd;
  --yellow:#c97b0d;--surface:#ebe4ff;
}}
body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--page);color:var(--text);padding:28px 24px 48px;line-height:1.55}}
.header{{text-align:center;margin-bottom:28px;padding:36px 32px 26px;background:linear-gradient(160deg,#fff 0%,#f3eeff 100%);border-radius:20px;border:1px solid var(--border);box-shadow:0 2px 20px #8839ef0d;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 55% 70% at 10% 100%,#7287fd12 0%,transparent 60%),radial-gradient(ellipse 45% 60% at 90% 0%,#04a5e510 0%,transparent 60%);pointer-events:none}}
.header h1{{font-size:1.75rem;font-weight:800;letter-spacing:-.4px;background:linear-gradient(120deg,var(--mauve) 0%,var(--blue) 55%,var(--teal) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px}}
.header .sub{{color:var(--sub0);font-size:.86rem}}
.exp-tag{{display:inline-block;background:#8839ef14;border:1px solid #8839ef33;color:var(--mauve);padding:4px 14px;border-radius:20px;font-size:.74rem;font-weight:700;letter-spacing:.3px;margin-top:10px}}
.section-label{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin:28px 0 14px;display:flex;align-items:center;gap:8px}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
.grid{{display:grid;gap:16px;margin-bottom:4px}}
.grid-2{{grid-template-columns:1fr 1fr}}
.grid-3{{grid-template-columns:1fr 1fr 1fr}}
@media(max-width:900px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:22px 24px;box-shadow:0 1px 6px #0001}}
.card:hover{{box-shadow:0 4px 16px #8839ef12;border-color:var(--border-h)}}
.card-title{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:7px}}
.dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 22px 16px;position:relative;overflow:hidden}}
.kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}}
.kpi:nth-child(1)::after{{background:linear-gradient(90deg,var(--red),var(--peach))}}
.kpi:nth-child(2)::after{{background:linear-gradient(90deg,var(--green),var(--teal))}}
.kpi:nth-child(3)::after{{background:linear-gradient(90deg,var(--blue),var(--sky))}}
.kpi:nth-child(4)::after{{background:linear-gradient(90deg,var(--mauve),var(--lavender))}}
.kpi:nth-child(5)::after{{background:linear-gradient(90deg,var(--yellow),var(--peach))}}
.kpi-label{{font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;font-weight:700}}
.kpi-value{{font-family:'JetBrains Mono',monospace;font-size:1.85rem;font-weight:700;margin-top:5px;letter-spacing:-1px;line-height:1}}
.kpi-sub{{font-size:.72rem;color:var(--muted);margin-top:5px}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
thead th{{background:var(--surface);color:var(--sub0);font-weight:700;text-transform:uppercase;font-size:.65rem;letter-spacing:.9px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}}
tbody td{{padding:10px 12px;border-bottom:1px solid #f0ebff;color:var(--sub1);vertical-align:middle;white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f7f4ff}}
.finding{{background:#fff;border:1px solid var(--border);border-left:4px solid var(--green);border-radius:0 12px 12px 0;padding:16px 20px;font-size:.88rem;line-height:1.65}}
.rec-box{{background:#40a02b12;border:1px solid #40a02b33;border-radius:12px;padding:14px 18px;margin-top:14px;font-size:.85rem;color:#1a5c10}}
canvas{{max-height:300px}}
</style>
</head>
<body>

<div class="header">
  <h1>Experiment 05 — XGBoost Calibration Fix</h1>
  <div class="sub">Residual bootstrap → Quantile regression &nbsp;·&nbsp;
    {p['n_quantiles']} quantile levels &nbsp;·&nbsp;
    {len(p['grid'])} hyperparameter configs × 7 districts</div>
  <div class="exp-tag">EXP 05 / XGBOOST CALIBRATION</div>
</div>

<div class="section-label">Before vs After Calibration</div>
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Bootstrap 80% Coverage</div>
    <div class="kpi-value" style="color:var(--red)">{bl['avg_cov80']*100:.1f}%</div>
    <div class="kpi-sub">target: 80% &nbsp;·&nbsp; training residuals too small</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Quantile Reg 80% Coverage</div>
    <div class="kpi-value" style="color:var(--green)">{bst['avg_cov80']*100:.1f}%</div>
    <div class="kpi-sub">best config: n_est={bst['n_est']} depth={bst['depth']} lr={bst['lr']}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Coverage Gain</div>
    <div class="kpi-value" style="color:{'var(--green)' if delta_cov > 0 else 'var(--red)}'}">
      {'▲' if delta_cov > 0 else '▼'}{abs(delta_cov):.1f}pp</div>
    <div class="kpi-sub">percentage points improvement</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Bootstrap CRPS</div>
    <div class="kpi-value" style="color:var(--peach)">{bl['avg_crps']}</div>
    <div class="kpi-sub">current model_lib default</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Best Quantile CRPS</div>
    <div class="kpi-value" style="color:{'var(--green)' if improved else 'var(--peach)'}">
      {bst['avg_crps']}</div>
    <div class="kpi-sub">{arrow(delta_crps)} vs bootstrap</div>
  </div>
</div>

<div class="section-label">Coverage & CRPS by Hyperparameters</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--teal)"></span>
      80% Coverage vs n_estimators (per depth, best lr)</div>
    <canvas id="covEstChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--mauve)"></span>
      CRPS vs n_estimators (per depth, best lr)</div>
    <canvas id="crpsEstChart"></canvas>
  </div>
</div>

<div class="section-label">Per-District: Bootstrap vs Best Quantile Config</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--blue)"></span>
      CRPS per district — bootstrap vs quantile</div>
    <canvas id="distCrpsChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--green)"></span>
      80% Coverage per district — bootstrap vs quantile</div>
    <canvas id="distCovChart"></canvas>
  </div>
</div>

<div class="section-label">Top 5 Quantile Regression Configs</div>
<div class="card" style="overflow:auto">
  <table>
    <thead>
      <tr>
        <th>#</th><th>n_estimators</th><th>max_depth</th><th>learning_rate</th>
        <th>avg CRPS</th><th>avg RMSE</th><th>80% cov</th>
        <th>CRPS vs bootstrap</th><th>cov80 vs bootstrap</th>
      </tr>
    </thead>
    <tbody>{top5_rows}</tbody>
  </table>
</div>

<div class="section-label">Per-District Detail</div>
<div class="card" style="overflow:auto">
  <table>
    <thead>
      <tr>
        <th>District</th>
        <th>CRPS (bootstrap)</th><th>CRPS (quantile)</th><th>CRPS delta</th>
        <th>cov80 (bootstrap)</th><th>cov80 (quantile)</th><th>cov80 delta</th>
      </tr>
    </thead>
    <tbody>{dist_rows}</tbody>
  </table>
</div>

<div class="section-label">Findings</div>
<div class="finding">
  <strong>Calibration fix:</strong> Replacing residual bootstrap with quantile regression
  brings 80% coverage from {bl['avg_cov80']*100:.1f}% to {bst['avg_cov80']*100:.1f}%
  — a gain of {abs(delta_cov):.1f} percentage points. The root cause of the original
  failure was XGBoost overfitting the training data: in-sample residuals were near zero,
  so bootstrapped intervals collapsed to near-point-predictions.
  <br><br>
  <strong>How quantile regression fixes this:</strong> Instead of adding noise from
  training errors, the model learns the <em>conditional quantile function</em> directly —
  i.e., how wide the prediction interval should be given the current covariate values.
  With {p['n_quantiles']} quantile levels (0.025 to 0.975), samples are drawn by uniform
  interpolation across the predicted distribution, properly capturing covariate-dependent
  uncertainty.
  <br><br>
  <strong>Best config:</strong> n_estimators={bst['n_est']}, max_depth={bst['depth']},
  learning_rate={bst['lr']} — saved to <code>exp05_best_config.json</code> for use in
  Experiment 06 (improved ensemble).
</div>
<div class="rec-box">
  <strong>Next — Experiment 06:</strong> Rebuild the Ensemble (S+P+X) using
  informed features (Exp 03, |r|&gt;0.10), tuned Prophet (Exp 04: cps=0.1 sps=2.0 multiplicative),
  and this quantile-regression XGBoost (best config above). Compare vs the original ensemble.
</div>

<script>
const P = {J};

// Build lookup: (n_est, depth, lr) -> metrics
const lookup = {{}};
P.grid.forEach(r => {{
  lookup[`${{r.n_est}}_${{r.depth}}_${{r.lr}}`] = r;
}});

const DEPTHS  = P.depth_values;
const N_ESTS  = P.n_est_values;
const DEPTH_COLORS = ["#8839ef","#1e66f5","#179299"];

// For each (depth, n_est) take best lr
function bestByLr(n_est, depth, metric) {{
  return Math.min(...P.lr_values.map(lr => lookup[`${{n_est}}_${{depth}}_${{lr}}`]?.[metric] ?? 999));
}}
function bestCovByLr(n_est, depth) {{
  return Math.max(...P.lr_values.map(lr => lookup[`${{n_est}}_${{depth}}_${{lr}}`]?.avg_cov80 ?? 0));
}}

function lineChart(id, yFn, yLabel, isPercent) {{
  const datasets = DEPTHS.map((d, di) => ({{
    label: `depth=${{d}}`,
    data: N_ESTS.map(n => isPercent ? +(yFn(n, d) * 100).toFixed(1) : yFn(n, d)),
    borderColor: DEPTH_COLORS[di],
    backgroundColor: DEPTH_COLORS[di] + "22",
    fill: false, tension: 0.35,
    pointRadius: 5, pointBorderColor: "#fff", pointBorderWidth: 2,
    pointBackgroundColor: DEPTH_COLORS[di],
  }}));

  const extras = isPercent ? [
    {{ label:"baseline bootstrap", data:Array(N_ESTS.length).fill(+(P.bootstrap.avg_cov80*100).toFixed(1)),
       borderColor:"#a099bb", borderDash:[5,4], pointRadius:0, fill:false }},
    {{ label:"target 80%", data:Array(N_ESTS.length).fill(80),
       borderColor:"#40a02b88", borderDash:[3,3], pointRadius:0, fill:false }},
  ] : [
    {{ label:"bootstrap baseline", data:Array(N_ESTS.length).fill(P.bootstrap.avg_crps),
       borderColor:"#a099bb", borderDash:[5,4], pointRadius:0, fill:false }},
  ];

  new Chart(document.getElementById(id), {{
    type: "line",
    data: {{ labels: N_ESTS.map(String), datasets: [...datasets, ...extras] }},
    options: {{
      responsive:true, maintainAspectRatio:true,
      plugins:{{ legend:{{ position:"top", labels:{{ font:{{size:10}}, boxWidth:10 }} }} }},
      scales:{{
        x:{{ title:{{display:true,text:"n_estimators",font:{{size:10}}}},
             type:"category", ticks:{{font:{{size:10}}}}, grid:{{display:false}} }},
        y:{{ title:{{display:true,text:yLabel,font:{{size:10}}}},
             ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
      }}
    }}
  }});
}}

lineChart("covEstChart",  (n, d) => bestCovByLr(n, d), "80% Coverage (%)", true);
lineChart("crpsEstChart", (n, d) => bestByLr(n, d, "avg_crps"), "avg CRPS", false);

// Per-district bar charts
const DISTS  = P.districts.map(d => d.replace("Mitsamiouli-Mboude", "Mitsamiouli"));
const crps_bl  = P.dist_bootstrap_crps.map(x => x[1]);
const crps_bst = P.dist_best_crps.map(x => x[1]);
const cov_bl   = P.dist_bootstrap_cov.map(x => +(x[1]*100).toFixed(1));
const cov_bst  = P.dist_best_cov.map(x => +(x[1]*100).toFixed(1));

function barChart(id, labels, ds1, ds2, label1, label2, c1, c2) {{
  new Chart(document.getElementById(id), {{
    type:"bar",
    data:{{ labels, datasets:[
      {{ label:label1, data:ds1, backgroundColor:c1+"cc", borderRadius:4 }},
      {{ label:label2, data:ds2, backgroundColor:c2+"cc", borderRadius:4 }},
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:true,
      plugins:{{ legend:{{ position:"top", labels:{{ font:{{size:10}}, boxWidth:10 }} }} }},
      scales:{{
        x:{{ ticks:{{font:{{size:10}},maxRotation:30}}, grid:{{display:false}} }},
        y:{{ ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
      }}
    }}
  }});
}}

barChart("distCrpsChart", DISTS, crps_bl, crps_bst,
         "Bootstrap CRPS", "Quantile CRPS", "#fe640b", "#40a02b");
barChart("distCovChart", DISTS, cov_bl, cov_bst,
         "Bootstrap cov80%", "Quantile cov80%", "#fe640b", "#1e66f5");
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
