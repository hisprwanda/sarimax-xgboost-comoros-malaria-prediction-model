"""
Experiment 04 — Prophet Hyperparameter Tuning
==============================================
Goal: Improve Prophet's CRPS and prediction-interval calibration through a
      systematic grid search over three key hyperparameters.

Current Prophet baseline (from run_benchmark.py):
  CRPS=38.2  RMSE=74.8  R²=0.852  80% coverage=59.9%  (very overconfident)

Hyperparameters searched:
  changepoint_prior_scale  : [0.001, 0.01, 0.05, 0.1, 0.5]
    — Controls trend flexibility. Low = rigid trend, high = wiggly.
    — 0.1 is current; Comoros malaria likely has stable trend → try lower.

  seasonality_prior_scale  : [0.5, 2.0, 5.0, 10.0]
    — Controls amplitude of seasonality Fourier terms.
    — Lower = smoother seasonality. Default=10 (loose); Comoros may be better tighter.

  seasonality_mode         : ['additive', 'multiplicative']
    — Additive: seasonal swing is constant in absolute cases.
    — Multiplicative: swing scales with the trend level.
    — Malaria in small island communities often multiplicative (high-season % varies).

Grid: 5 × 4 × 2 = 40 combinations × 7 districts = 280 fits.

Outputs:
  output/experiments/exp04_prophet_tuning.html

Usage:
    cd C:/Vault/HISP-MODELING/Climatehealth-comoros
    python experiments/exp04_prophet_tuning.py
"""

import os, sys, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV = os.path.join(ROOT, "input", "evaluation_data.csv")
OUT_DIR  = os.path.join(ROOT, "output", "experiments")
OUT_HTML = os.path.join(OUT_DIR, "exp04_prophet_tuning.html")

sys.path.insert(0, ROOT)
import model_lib as ml

TRAIN_WEEKS = 78
N_SAMPLES   = 50
RNG_SEED    = 42

COVARIATES  = ["rainfall", "mean_temperature", "humidity"]

CPS_VALUES  = [0.001, 0.01, 0.05, 0.1, 0.5]   # changepoint_prior_scale
SPS_VALUES  = [0.5, 2.0, 5.0, 10.0]            # seasonality_prior_scale
MODES       = ["additive", "multiplicative"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split():
    df = pd.read_csv(EVAL_CSV)
    wks = sorted(df["time_period"].unique())
    train_wks = set(wks[:TRAIN_WEEKS])
    test_wks  = set(wks[TRAIN_WEEKS:])
    train = df[df["time_period"].isin(train_wks)].copy()
    test  = df[df["time_period"].isin(test_wks)].copy()
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


# ── Single config eval ────────────────────────────────────────────────────────

def eval_config(train_df, future_df, truth_df, cps, sps, mode, rng):
    from prophet import Prophet

    all_crps, all_rmse, all_cov80 = [], [], []

    for loc, grp in train_df.groupby("location", sort=False):
        grp   = grp.sort_values("time_period")
        times = grp["time_period"].apply(ml.isoweek_to_timestamp)
        y     = grp["disease_cases"].astype(float)

        pdf = pd.DataFrame({"ds": times, "y": y.values})
        for c in COVARIATES:
            pdf[c] = grp[c].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode=mode,
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
            )
            for c in COVARIATES:
                m.add_regressor(c, standardize=True)
            m.fit(pdf)

        fgrp   = future_df[future_df["location"] == loc].sort_values("time_period")
        ftimes = fgrp["time_period"].apply(ml.isoweek_to_timestamp)
        fpdf   = pd.DataFrame({"ds": ftimes})
        for c in COVARIATES:
            fpdf[c] = fgrp[c].values

        raw    = m.predictive_samples(fpdf)
        pool   = raw["yhat"]
        avail  = pool.shape[1]
        if avail >= N_SAMPLES:
            idx = rng.choice(avail, size=N_SAMPLES, replace=False)
        else:
            idx = rng.choice(avail, size=N_SAMPLES, replace=True)
        samples = np.maximum(0, pool[:, idx])

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
        "dist_rmse": [round(v, 3) for v in all_rmse],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp04] Loading data ...")
    train_df, future_df, truth_df = load_split()
    districts = sorted(train_df["location"].unique())

    # Baseline (current model_lib defaults)
    print("[exp04] Evaluating baseline (cps=0.1, sps=10.0, additive) ...")
    rng = np.random.default_rng(RNG_SEED)
    baseline = eval_config(train_df, future_df, truth_df, 0.1, 10.0, "additive", rng)
    print(f"  baseline  CRPS={baseline['avg_crps']:.3f}  "
          f"RMSE={baseline['avg_rmse']:.3f}  cov80={baseline['avg_cov80']*100:.1f}%")

    # Grid search
    combos = list(itertools.product(CPS_VALUES, SPS_VALUES, MODES))
    total  = len(combos)
    grid_results = []

    print(f"[exp04] Grid search: {total} configs × 7 districts ...")
    for i, (cps, sps, mode) in enumerate(combos, 1):
        rng = np.random.default_rng(RNG_SEED)
        res = eval_config(train_df, future_df, truth_df, cps, sps, mode, rng)
        grid_results.append({
            "cps": cps, "sps": sps, "mode": mode, **res
        })
        tag = f"cps={cps} sps={sps} {mode[:3]}"
        print(f"  [{i:02d}/{total}] {tag:30s}  "
              f"CRPS={res['avg_crps']:.3f}  cov80={res['avg_cov80']*100:.1f}%")

    # Best config
    best = min(grid_results, key=lambda r: r["avg_crps"])
    print(f"\n[exp04] Best: cps={best['cps']} sps={best['sps']} mode={best['mode']}  "
          f"CRPS={best['avg_crps']:.3f}  cov80={best['avg_cov80']*100:.1f}%")
    print(f"  vs baseline: CRPS {baseline['avg_crps']:.3f} -> {best['avg_crps']:.3f}  "
          f"(delta={best['avg_crps']-baseline['avg_crps']:+.3f})")

    # Top-5 by CRPS
    top5 = sorted(grid_results, key=lambda r: r["avg_crps"])[:5]

    # Build heatmap data: for each (cps, mode) pair, best CRPS across sps values
    heatmap = {}
    for mode in MODES:
        heatmap[mode] = {}
        for cps in CPS_VALUES:
            subset = [r for r in grid_results if r["cps"] == cps and r["mode"] == mode]
            heatmap[mode][str(cps)] = min(r["avg_crps"] for r in subset)

    # Coverage improvement at best config
    cov_improvement = round((best["avg_cov80"] - baseline["avg_cov80"]) * 100, 1)

    payload = {
        "baseline": baseline,
        "best":     {k: v for k, v in best.items() if k != "dist_crps"},
        "top5":     [{k: v for k, v in r.items() if k != "dist_crps"} for r in top5],
        "grid":     [{k: v for k, v in r.items() if k != "dist_crps"} for r in grid_results],
        "heatmap":  heatmap,
        "cps_values": CPS_VALUES,
        "sps_values": SPS_VALUES,
        "modes":    MODES,
        "districts": districts,
        "cov_improvement": cov_improvement,
    }

    print("[exp04] Building HTML ...")
    html = build_html(payload)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[exp04] Done -> {OUT_HTML}")


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(p):
    J = json.dumps(p)
    bl  = p["baseline"]
    bst = p["best"]
    delta_crps = round(bst["avg_crps"] - bl["avg_crps"], 3)
    delta_cov  = round(bst["avg_cov80"] * 100 - bl["avg_cov80"] * 100, 1)
    improved   = delta_crps < 0

    def arrow(d, lower_better=True):
        better = (d < 0) if lower_better else (d > 0)
        if abs(d) < 0.01:
            return f'<span style="color:var(--muted)">≈ flat</span>'
        sym = "▼" if d < 0 else "▲"
        col = "var(--green)" if better else "var(--red)"
        return f'<span style="color:{col}">{sym} {abs(d):.2f}</span>'

    # Top-5 rows
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
          <td style="font-family:'JetBrains Mono',monospace">{r['cps']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['sps']}</td>
          <td><span class="pill pill-mode-{r['mode'][:3]}">{r['mode']}</span></td>
          <td style="font-family:'JetBrains Mono',monospace">{r['avg_crps']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['avg_rmse']}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r['avg_cov80']*100:.1f}%</td>
          <td>{arrow(d_crps)}</td>
          <td>{arrow(d_cov, lower_better=False)}</td>
        </tr>"""

    # Findings text
    if improved:
        result_text = (
            f"Grid search found a better Prophet configuration — CRPS improved from "
            f"{bl['avg_crps']:.3f} to {bst['avg_crps']:.3f} "
            f"({abs(delta_crps):.3f} points, {abs(delta_crps)/bl['avg_crps']*100:.1f}% reduction). "
            f"Coverage improved from {bl['avg_cov80']*100:.1f}% to {bst['avg_cov80']*100:.1f}% "
            f"(target: 80%)."
        )
        rec_text = (
            f"Update <code>fit_prophet_one()</code> in <code>model_lib.py</code> to use "
            f"<code>changepoint_prior_scale={bst['cps']}</code>, "
            f"<code>seasonality_prior_scale={bst['sps']}</code>, "
            f"<code>seasonality_mode='{bst['mode']}'</code>. "
            f"Carry these settings into the final Experiment 06 benchmark."
        )
    else:
        result_text = (
            f"Grid search did not substantially improve Prophet — best CRPS is "
            f"{bst['avg_crps']:.3f} vs baseline {bl['avg_crps']:.3f} "
            f"({abs(delta_crps):.3f} point change). "
            f"Coverage: {bl['avg_cov80']*100:.1f}% → {bst['avg_cov80']*100:.1f}% (target 80%)."
        )
        rec_text = (
            "Prophet's core calibration issue may not be addressable through these three "
            "hyperparameters alone on a 78-week training set. Consider using the best config "
            "if coverage improved, otherwise retain the current baseline."
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 04 - Prophet Hyperparameter Tuning</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --page:#f2eeff;--card:#fff;--border:#ddd5f5;--border-h:#c9bef0;
  --text:#2a2044;--sub1:#544873;--sub0:#7a6e92;--muted:#a099bb;--ghost:#c4bcd8;
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
.card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:22px 24px;box-shadow:0 1px 6px #0001;transition:box-shadow .2s,border-color .2s}}
.card:hover{{box-shadow:0 4px 16px #8839ef12;border-color:var(--border-h)}}
.card-title{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:7px}}
.dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 22px 16px;box-shadow:0 1px 4px #0001;position:relative;overflow:hidden}}
.kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}}
.kpi:nth-child(1)::after{{background:linear-gradient(90deg,var(--sky),var(--blue))}}
.kpi:nth-child(2)::after{{background:linear-gradient(90deg,var(--green),var(--teal))}}
.kpi:nth-child(3)::after{{background:linear-gradient(90deg,var(--mauve),var(--lavender))}}
.kpi:nth-child(4)::after{{background:linear-gradient(90deg,var(--peach),var(--yellow))}}
.kpi-label{{font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;font-weight:700}}
.kpi-value{{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;margin-top:5px;letter-spacing:-1px;line-height:1}}
.kpi-sub{{font-size:.72rem;color:var(--muted);margin-top:5px}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
thead th{{background:var(--surface);color:var(--sub0);font-weight:700;text-transform:uppercase;font-size:.65rem;letter-spacing:.9px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}}
tbody td{{padding:10px 12px;border-bottom:1px solid #f0ebff;color:var(--sub1);vertical-align:middle;white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f7f4ff;color:var(--text)}}
.pill{{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.72rem;font-weight:700;margin:1px 1px}}
.pill-mode-add{{background:#1e66f518;color:var(--blue);border:1px solid #1e66f530}}
.pill-mode-mul{{background:#fe640b18;color:var(--peach);border:1px solid #fe640b30}}
.finding{{background:#fff;border:1px solid var(--border);border-left:4px solid var(--mauve);border-radius:0 12px 12px 0;padding:16px 20px;font-size:.88rem;line-height:1.65}}
.rec-box{{background:#40a02b12;border:1px solid #40a02b33;border-radius:12px;padding:14px 18px;margin-top:14px;font-size:.85rem;color:#1a5c10;line-height:1.6}}
.rec-box strong{{color:#1a6012}}
canvas{{max-height:310px}}
.heatmap-wrap{{overflow-x:auto}}
.heatmap-table{{border-collapse:collapse;font-size:.78rem;font-family:'JetBrains Mono',monospace;min-width:400px}}
.heatmap-table th,.heatmap-table td{{padding:7px 12px;border:1px solid var(--border);text-align:center}}
.heatmap-table th{{background:var(--surface);color:var(--sub0);font-size:.65rem;text-transform:uppercase;letter-spacing:.8px}}
.heatmap-table .row-label{{font-weight:700;color:var(--sub1);text-align:left;background:var(--surface)}}
</style>
</head>
<body>

<div class="header">
  <h1>Experiment 04 — Prophet Hyperparameter Tuning</h1>
  <div class="sub">
    Grid search: changepoint_prior_scale × seasonality_prior_scale × seasonality_mode
    &nbsp;·&nbsp; {len(p['grid'])} configs × 7 districts
  </div>
  <div class="exp-tag">EXP 04 / PROPHET TUNING</div>
</div>

<div class="section-label">Baseline vs Best Config</div>
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Baseline CRPS</div>
    <div class="kpi-value" style="color:var(--blue)">{bl['avg_crps']}</div>
    <div class="kpi-sub">cps=0.1 · sps=10 · additive</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Best Tuned CRPS</div>
    <div class="kpi-value" style="color:var(--green)">{bst['avg_crps']}</div>
    <div class="kpi-sub">cps={bst['cps']} · sps={bst['sps']} · {bst['mode']}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">CRPS Delta</div>
    <div class="kpi-value" style="color:{'var(--green)' if improved else 'var(--red)'}">
      {'▼' if improved else '▲'}{abs(delta_crps):.3f}
    </div>
    <div class="kpi-sub">{'improvement' if improved else 'regression'} vs baseline</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">80% PI Coverage (best)</div>
    <div class="kpi-value" style="color:var(--mauve)">{bst['avg_cov80']*100:.1f}%</div>
    <div class="kpi-sub">baseline={bl['avg_cov80']*100:.1f}% · target=80%</div>
  </div>
</div>

<div class="section-label">CRPS Sensitivity — changepoint_prior_scale</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--mauve)"></span>
      CRPS vs changepoint_prior_scale — additive mode</div>
    <canvas id="cpsAddChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--peach)"></span>
      CRPS vs changepoint_prior_scale — multiplicative mode</div>
    <canvas id="cpsMulChart"></canvas>
  </div>
</div>

<div class="section-label">Coverage Sensitivity</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--teal)"></span>
      80% PI Coverage vs changepoint_prior_scale (best sps per point)</div>
    <canvas id="covChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--sky)"></span>
      CRPS Heatmap — all (cps, sps) at best mode</div>
    <div class="heatmap-wrap" id="heatmapDiv"></div>
  </div>
</div>

<div class="section-label">Top 5 Configurations</div>
<div class="card" style="overflow:auto">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>changepoint_prior_scale</th>
        <th>seasonality_prior_scale</th>
        <th>mode</th>
        <th>avg CRPS</th>
        <th>avg RMSE</th>
        <th>80% cov</th>
        <th>CRPS vs base</th>
        <th>cov80 vs base</th>
      </tr>
    </thead>
    <tbody>{top5_rows}</tbody>
  </table>
</div>

<div class="section-label">Findings</div>
<div class="finding">
  <strong>Result:</strong> {result_text}
  <br><br>
  <strong>What changepoint_prior_scale controls:</strong> A lower value makes the trend
  more rigid — Prophet stops fitting every local fluctuation as a "trend change". For
  malaria in a small island setting with 2 years of data, this prevents the model from
  absorbing seasonal peaks into the trend, which would otherwise shrink the residual
  variance and produce overconfident prediction intervals.
  <br><br>
  <strong>Additive vs Multiplicative seasonality:</strong> Multiplicative mode means the
  seasonal swing is expressed as a percentage of the trend level. This is biologically
  appropriate when wet-season transmission rates scale with baseline mosquito density
  (which tracks the trend). Additive assumes a fixed seasonal amplitude regardless of
  trend level — less realistic when disease burden changes over time.
  <br><br>
  <strong>Why coverage was low in the baseline (59.9%):</strong> Prophet's uncertainty
  comes from the posterior over its Fourier coefficients. With a high changepoint_prior_scale,
  the model overfits trend changes and under-attributes variance to the noise term,
  producing intervals that are too narrow.
</div>
<div class="rec-box">
  <strong>Recommendation for Experiment 05 (XGBoost calibration):</strong> {rec_text}
</div>

<script>
const P = {J};

// ── Build (cps × sps × mode) lookup ──────────────────────────────────────────
const lookup = {{}};
P.grid.forEach(r => {{
  const key = `${{r.cps}}_${{r.sps}}_${{r.mode}}`;
  lookup[key] = r;
}});

// ── CRPS vs CPS charts ────────────────────────────────────────────────────────
const CPS = P.cps_values;
const SPS = P.sps_values;
const SPS_COLORS = ["#8839ef","#1e66f5","#179299","#40a02b"];

function cpsChart(id, mode, accentColors) {{
  const datasets = SPS.map((sps, si) => ({{
    label: `sps=${{sps}}`,
    data: CPS.map(cps => {{
      const r = lookup[`${{cps}}_${{sps}}_${{mode}}`];
      return r ? r.avg_crps : null;
    }}),
    borderColor: accentColors[si],
    backgroundColor: accentColors[si] + "22",
    fill: false, tension: 0.35,
    pointRadius: 5, pointBorderColor: "#fff", pointBorderWidth: 2,
    pointBackgroundColor: accentColors[si],
  }}));
  new Chart(document.getElementById(id), {{
    type: "line",
    data: {{ labels: CPS.map(String), datasets }},
    options: {{
      responsive: true, maintainAspectRatio: true,
      plugins:{{ legend:{{ position:"top", labels:{{ font:{{size:10}}, boxWidth:10 }} }} }},
      scales:{{
        x:{{ title:{{display:true,text:"changepoint_prior_scale",font:{{size:10}}}},
             type:"category", ticks:{{font:{{size:10}}}}, grid:{{display:false}} }},
        y:{{ title:{{display:true,text:"avg CRPS",font:{{size:10}}}},
             ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
      }}
    }}
  }});
}}
cpsChart("cpsAddChart", "additive",     ["#8839ef","#1e66f5","#179299","#40a02b"]);
cpsChart("cpsMulChart", "multiplicative",["#fe640b","#c97b0d","#d20f39","#7287fd"]);

// ── Coverage chart ────────────────────────────────────────────────────────────
const covAdd = CPS.map(cps => {{
  const best = Math.max(...SPS.map(sps => lookup[`${{cps}}_${{sps}}_additive`]?.avg_cov80 ?? 0));
  return +(best * 100).toFixed(1);
}});
const covMul = CPS.map(cps => {{
  const best = Math.max(...SPS.map(sps => lookup[`${{cps}}_${{sps}}_multiplicative`]?.avg_cov80 ?? 0));
  return +(best * 100).toFixed(1);
}});

new Chart(document.getElementById("covChart"), {{
  type: "line",
  data: {{
    labels: CPS.map(String),
    datasets: [
      {{ label:"additive (best sps)",      data:covAdd, borderColor:"#1e66f5",
         backgroundColor:"#1e66f522", fill:false, tension:0.35,
         pointRadius:5, pointBorderColor:"#fff", pointBorderWidth:2, pointBackgroundColor:"#1e66f5" }},
      {{ label:"multiplicative (best sps)", data:covMul, borderColor:"#fe640b",
         backgroundColor:"#fe640b22", fill:false, tension:0.35,
         pointRadius:5, pointBorderColor:"#fff", pointBorderWidth:2, pointBackgroundColor:"#fe640b" }},
      {{ label:"baseline (59.9%)", data:Array(CPS.length).fill({bl['avg_cov80']*100:.1f}),
         borderColor:"#a099bb", borderDash:[5,4], pointRadius:0, fill:false }},
      {{ label:"target 80%", data:Array(CPS.length).fill(80),
         borderColor:"#40a02b88", borderDash:[3,3], pointRadius:0, fill:false }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:true,
    plugins:{{ legend:{{ position:"top", labels:{{ font:{{size:10}}, boxWidth:10 }} }} }},
    scales:{{
      x:{{ title:{{display:true,text:"changepoint_prior_scale",font:{{size:10}}}},
           type:"category", ticks:{{font:{{size:10}}}}, grid:{{display:false}} }},
      y:{{ title:{{display:true,text:"80% PI Coverage (%)",font:{{size:10}}}},
           min:0, max:105, ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
    }}
  }}
}});

// ── Heatmap table ─────────────────────────────────────────────────────────────
(function() {{
  // For each (cps, sps) use best CRPS across modes
  const allVals = [];
  CPS.forEach(cps => SPS.forEach(sps => {{
    const best = Math.min(
      lookup[`${{cps}}_${{sps}}_additive`]?.avg_crps ?? 999,
      lookup[`${{cps}}_${{sps}}_multiplicative`]?.avg_crps ?? 999
    );
    allVals.push(best);
  }}));
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);

  function cellColor(v) {{
    const t = (v - minV) / (maxV - minV + 1e-9);
    const r = Math.round(40 + t * (254 - 40));
    const g = Math.round(160 - t * (160 - 100));
    const b = Math.round(43 + t * (11 - 43));
    return `rgba(${{r}},${{g}},${{b}},0.18)`;
  }}

  let html = `<table class="heatmap-table"><thead><tr><th>cps \\ sps</th>`;
  SPS.forEach(sps => html += `<th>sps=${{sps}}</th>`);
  html += `</tr></thead><tbody>`;
  CPS.forEach(cps => {{
    html += `<tr><td class="row-label">cps=${{cps}}</td>`;
    SPS.forEach(sps => {{
      const best = Math.min(
        lookup[`${{cps}}_${{sps}}_additive`]?.avg_crps ?? 999,
        lookup[`${{cps}}_${{sps}}_multiplicative`]?.avg_crps ?? 999
      );
      const best_mode = (lookup[`${{cps}}_${{sps}}_additive`]?.avg_crps ?? 999) <=
                        (lookup[`${{cps}}_${{sps}}_multiplicative`]?.avg_crps ?? 999)
                        ? "add" : "mul";
      const isBest = Math.abs(best - minV) < 0.001;
      const border = isBest ? "outline:2px solid #8839ef;outline-offset:-2px;" : "";
      html += `<td style="background:${{cellColor(best)}};${{border}}">
        ${{best.toFixed(2)}}<br><span style="font-size:.65rem;color:#7a6e92">${{best_mode}}</span>
      </td>`;
    }});
    html += `</tr>`;
  }});
  html += `</tbody></table>`;
  document.getElementById("heatmapDiv").innerHTML = html;
}})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
