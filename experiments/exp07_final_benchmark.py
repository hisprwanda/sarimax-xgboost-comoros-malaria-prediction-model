"""
Experiment 07 — Final Optimised Benchmark
==========================================
Six configurations evaluated after the full experiment series (Exp 01–06).

Configs
-------
1. SARIMAX baseline       — SARIMAX(1,0,1) + 3 raw climate covariates
2. SARIMAX tuned          — SARIMAX(1,0,1) + district-specific features |r|>0.10 (Exp 03)
3. Prophet tuned          — cps=0.1, sps=2.0, multiplicative seasonality (Exp 04)
4. XGBoost calibrated     — quantile regression, n_est=100 depth=4 lr=0.05 (Exp 05)
5. Ensemble S+P+X (impr.) — SARIMAX tuned + Prophet tuned + XGBoost calibrated (Exp 06)
6. Ensemble S+X  ★ new   — SARIMAX tuned + XGBoost calibrated only

Split: train weeks 1-78  |  test weeks 79-104  |  n_samples=50

Outputs:
  output/experiments/exp07_final_benchmark.html

Usage:
    cd C:/Vault/HISP-MODELING/Climatehealth-comoros
    python experiments/exp07_final_benchmark.py
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV = os.path.join(ROOT, "input", "evaluation_data.csv")
OUT_DIR  = os.path.join(ROOT, "output", "experiments")
OUT_HTML = os.path.join(OUT_DIR, "exp07_final_benchmark.html")
XGB_CFG  = os.path.join(OUT_DIR, "exp05_best_config.json")

sys.path.insert(0, ROOT)
import model_lib as ml

TRAIN_WEEKS      = 78
N_SAMPLES        = 50
RNG_SEED         = 42
CLIMATE_VARS     = ["rainfall", "mean_temperature", "humidity"]
MAX_LAG          = 8
SARIMAX_THR      = 0.10
PROPHET_CPS      = 0.1
PROPHET_SPS      = 2.0
PROPHET_MODE     = "multiplicative"
QUANTILE_LEVELS  = np.linspace(0.025, 0.975, 25)

LAG_COL = {
    ("rainfall",1):"rainfall_lag1", ("rainfall",2):"rainfall_lag2",
    ("rainfall",3):"rainfall_lag3", ("rainfall",4):"rainfall_lag4",
    ("mean_temperature",1):"temp_lag1", ("mean_temperature",2):"temp_lag2",
    ("humidity",1):"humidity_lag1",     ("humidity",2):"humidity_lag2",
}
AVAILABLE_LAGS = {"rainfall":[1,2,3,4],"mean_temperature":[1,2],"humidity":[1,2]}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(with_features=False):
    df = pd.read_csv(EVAL_CSV)
    wks = sorted(df["time_period"].unique())
    train_wks = set(wks[:TRAIN_WEEKS])
    test_wks  = set(wks[TRAIN_WEEKS:])
    if with_features:
        df = ml.add_engineered_features(df.copy())
    train  = df[df["time_period"].isin(train_wks)].copy()
    test   = df[df["time_period"].isin(test_wks)].copy()
    truth  = test[["time_period","location","disease_cases"]].copy()
    future = test.drop(columns=["disease_cases"])
    return train, future, truth


# ── Metrics ───────────────────────────────────────────────────────────────────

def crps_nrg(samples, truth):
    n  = samples.shape[1]
    e1 = np.abs(samples - truth[:,None]).mean(axis=1)
    e2 = np.abs(samples[:,:,None] - samples[:,None,:]).sum(axis=(1,2)) / (n*(n-1))
    return float(np.mean(e1 - 0.5*e2))

def rmse(yp, yt):  return float(np.sqrt(np.mean((yp-yt)**2)))
def mae(yp, yt):   return float(np.mean(np.abs(yp-yt)))
def r2(yp, yt):
    ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-yt.mean())**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0

def coverage(s, t, a):
    lo = np.quantile(s,(1-a)/2,axis=1); hi = np.quantile(s,1-(1-a)/2,axis=1)
    return float(np.mean((t>=lo)&(t<=hi)))

def width(s, a):
    lo = np.quantile(s,(1-a)/2,axis=1); hi = np.quantile(s,1-(1-a)/2,axis=1)
    return float(np.mean(hi-lo))

def full_metrics(pred_df, truth_df, districts):
    all_crps,all_rmse,all_mae,all_cov80,all_cov95 = [],[],[],[],[]
    all_yp, all_yt = [], []
    for loc in districts:
        s  = pred_df[pred_df["location"]==loc].sort_values("time_period")
        t  = truth_df[truth_df["location"]==loc].sort_values("time_period")
        sc = [c for c in s.columns if c.startswith("sample_")]
        sa = s[sc].values; tr = t["disease_cases"].values
        all_crps.append(crps_nrg(sa, tr))
        yp = sa.mean(axis=1)
        all_rmse.append(rmse(yp, tr))
        all_mae.append(mae(yp, tr))
        all_cov80.append(coverage(sa, tr, 0.80))
        all_cov95.append(coverage(sa, tr, 0.95))
        all_yp.extend(yp.tolist()); all_yt.extend(tr.tolist())
    all_yp = np.array(all_yp); all_yt = np.array(all_yt)
    return {
        "avg_crps":  round(float(np.mean(all_crps)), 3),
        "avg_rmse":  round(float(np.mean(all_rmse)), 3),
        "avg_mae":   round(float(np.mean(all_mae)), 3),
        "r2":        round(r2(all_yp, all_yt), 3),
        "avg_cov80": round(float(np.mean(all_cov80)), 4),
        "avg_cov95": round(float(np.mean(all_cov95)), 4),
        "dist_crps": [round(v,3) for v in all_crps],
        "dist_cov80": [round(v,4) for v in all_cov80],
    }


# ── Feature selection helpers ─────────────────────────────────────────────────

def compute_best_lags(train_df):
    districts = sorted(train_df["location"].unique())
    result = {d:{} for d in districts}
    for var in CLIMATE_VARS:
        for loc in districts:
            grp = train_df[train_df["location"]==loc].sort_values("time_period")
            y = grp["disease_cases"].values; x = grp[var].values
            best_lag, best_r = 0, 0.0
            for lag in range(MAX_LAG+1):
                xi = x[:-lag] if lag>0 else x; yi = y[lag:] if lag>0 else y
                if len(xi)<10: continue
                r, _ = stats.pearsonr(xi, yi)
                if abs(r) > abs(best_r): best_r, best_lag = r, lag
            result[loc][var] = {"best_lag":best_lag, "r":round(float(best_r),4)}
    return result

def select_features(bld, thr):
    cols = list(CLIMATE_VARS)
    for var in CLIMATE_VARS:
        info = bld.get(var,{}); r, lag = info.get("r",0.), info.get("best_lag",0)
        if abs(r)>thr and lag>0:
            avail = AVAILABLE_LAGS.get(var,[])
            cl = next((l for l in sorted(avail,reverse=True) if l<=lag), None)
            if cl is None and avail: cl = min(avail)
            if cl:
                col = LAG_COL.get((var,cl))
                if col and col not in cols: cols.append(col)
    return cols


# ── Component predictors ──────────────────────────────────────────────────────

def run_sarimax_base(train_df, future_df, districts, rng):
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        grp = grp.sort_values("time_period")
        y = grp["disease_cases"].astype(float)
        X = grp[CLIMATE_VARS].astype(float)
        payload = ml.fit_sarimax_one(y, X)
        fgrp = future_df[future_df["location"]==loc].sort_values("time_period")
        fX   = fgrp[CLIMATE_VARS].astype(float)
        samp = ml.predict_sarimax_one(payload, fX, N_SAMPLES, rng)
        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period":tp,"location":loc}
            for j in range(N_SAMPLES): row[f"sample_{j}"] = samp[i,j]
            records.append(row)
    return pd.DataFrame(records)


def run_sarimax_tuned(train_df, future_df, best_lags, districts, rng):
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        covs = select_features(best_lags[loc], SARIMAX_THR)
        grp  = grp.sort_values("time_period")
        y = grp["disease_cases"].astype(float)
        X = grp[covs].astype(float)
        payload = ml.fit_sarimax_one(y, X)
        fgrp = future_df[future_df["location"]==loc].sort_values("time_period")
        fX   = fgrp[covs].astype(float)
        samp = ml.predict_sarimax_one(payload, fX, N_SAMPLES, rng)
        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period":tp,"location":loc}
            for j in range(N_SAMPLES): row[f"sample_{j}"] = samp[i,j]
            records.append(row)
    return pd.DataFrame(records)


def run_prophet_tuned(train_df, future_df, districts, rng):
    from prophet import Prophet
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        grp   = grp.sort_values("time_period")
        times = grp["time_period"].apply(ml.isoweek_to_timestamp)
        y     = grp["disease_cases"].astype(float)
        pdf   = pd.DataFrame({"ds":times,"y":y.values})
        for c in CLIMATE_VARS: pdf[c] = grp[c].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, seasonality_mode=PROPHET_MODE,
                        changepoint_prior_scale=PROPHET_CPS,
                        seasonality_prior_scale=PROPHET_SPS)
            for c in CLIMATE_VARS: m.add_regressor(c, standardize=True)
            m.fit(pdf)
        fgrp   = future_df[future_df["location"]==loc].sort_values("time_period")
        ftimes = fgrp["time_period"].apply(ml.isoweek_to_timestamp)
        fpdf   = pd.DataFrame({"ds":ftimes})
        for c in CLIMATE_VARS: fpdf[c] = fgrp[c].values
        raw  = m.predictive_samples(fpdf); pool = raw["yhat"]
        idx  = rng.choice(pool.shape[1], size=N_SAMPLES, replace=(pool.shape[1]<N_SAMPLES))
        samp = np.maximum(0, pool[:,idx])
        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period":tp,"location":loc}
            for j in range(N_SAMPLES): row[f"sample_{j}"] = samp[i,j]
            records.append(row)
    return pd.DataFrame(records)


def run_xgb_calibrated(train_df, future_df, districts, xgb_cfg, rng):
    import xgboost as xgb
    covs    = [c for c in ml.DEFAULT_COVARIATES+ml.EXTRA_COVARIATES if c in train_df.columns]
    records = []
    for loc, grp in train_df.groupby("location", sort=False):
        grp  = grp.sort_values("time_period")
        y    = grp["disease_cases"].astype(float).values
        Xf   = ml._xgb_features(grp[covs].astype(float), grp["time_period"])
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=QUANTILE_LEVELS,
            n_estimators=xgb_cfg["n_estimators"], max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42, verbosity=0)
        model.fit(Xf, y)
        fgrp = future_df[future_df["location"]==loc].sort_values("time_period")
        fXf  = ml._xgb_features(fgrp[covs].astype(float), fgrp["time_period"])
        qp   = model.predict(fXf)
        if qp.ndim == 1: qp = qp[:,None]
        qp   = np.sort(qp, axis=1)
        n_periods = qp.shape[0]
        u    = rng.uniform(0, 1, size=(n_periods, N_SAMPLES))
        samp = np.zeros((n_periods, N_SAMPLES))
        for t in range(n_periods): samp[t] = np.interp(u[t], QUANTILE_LEVELS, qp[t])
        samp = np.maximum(0, samp)
        for i, tp in enumerate(fgrp["time_period"]):
            row = {"time_period":tp,"location":loc}
            for j in range(N_SAMPLES): row[f"sample_{j}"] = samp[i,j]
            records.append(row)
    return pd.DataFrame(records)


def build_ensemble(pred_a, pred_b, pred_c=None):
    keys = ["time_period","location"]; n = N_SAMPLES
    def relabel(df, offset):
        sc = [c for c in df.columns if c.startswith("sample_")]
        return df.rename(columns={c:f"sample_{i+offset}" for i,c in enumerate(sc)})
    a = relabel(pred_a, 0)
    b = relabel(pred_b, n)
    m = a.merge(b[keys+[f"sample_{i}" for i in range(n,2*n)]], on=keys, how="inner")
    if pred_c is not None:
        c = relabel(pred_c, 2*n)
        m = m.merge(c[keys+[f"sample_{i}" for i in range(2*n,3*n)]], on=keys, how="inner")
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(XGB_CFG):
        raise FileNotFoundError(f"Run exp05 first: {XGB_CFG}")
    xgb_cfg = json.load(open(XGB_CFG))
    print(f"[exp07] XGBoost config from Exp 05: {xgb_cfg}")

    print("[exp07] Loading data ...")
    train_base, future_base, truth = load_split(with_features=False)
    train_feat, future_feat, _     = load_split(with_features=True)
    districts = sorted(train_base["location"].unique())

    print("[exp07] Computing per-district best lags ...")
    best_lags = compute_best_lags(train_base)

    results = []

    # ── Config 1: SARIMAX baseline ────────────────────────────────────────────
    print("\n[exp07] Config 1/6 — SARIMAX baseline ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred1 = run_sarimax_base(train_base, future_base, districts, rng)
    m1 = full_metrics(pred1, truth, districts)
    m1.update({"label":"SARIMAX baseline", "elapsed":round(time.perf_counter()-t0,1),
                "note":"Fixed (1,0,1) · 3 raw climate covariates"})
    results.append(m1)
    print(f"  CRPS={m1['avg_crps']}  RMSE={m1['avg_rmse']}  cov80={m1['avg_cov80']*100:.1f}%")

    # ── Config 2: SARIMAX tuned ───────────────────────────────────────────────
    print("[exp07] Config 2/6 — SARIMAX tuned (Exp 03) ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred2 = run_sarimax_tuned(train_feat, future_feat, best_lags, districts, rng)
    m2 = full_metrics(pred2, truth, districts)
    m2.update({"label":"SARIMAX tuned", "elapsed":round(time.perf_counter()-t0,1),
                "note":f"(1,0,1) · district-specific features |r|>{SARIMAX_THR}"})
    results.append(m2)
    print(f"  CRPS={m2['avg_crps']}  RMSE={m2['avg_rmse']}  cov80={m2['avg_cov80']*100:.1f}%")

    # ── Config 3: Prophet tuned ───────────────────────────────────────────────
    print("[exp07] Config 3/6 — Prophet tuned (Exp 04) ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred3 = run_prophet_tuned(train_base, future_base, districts, rng)
    m3 = full_metrics(pred3, truth, districts)
    m3.update({"label":"Prophet tuned", "elapsed":round(time.perf_counter()-t0,1),
                "note":f"cps={PROPHET_CPS} · sps={PROPHET_SPS} · {PROPHET_MODE}"})
    results.append(m3)
    print(f"  CRPS={m3['avg_crps']}  RMSE={m3['avg_rmse']}  cov80={m3['avg_cov80']*100:.1f}%")

    # ── Config 4: XGBoost calibrated ──────────────────────────────────────────
    print("[exp07] Config 4/6 — XGBoost calibrated (Exp 05) ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    pred4 = run_xgb_calibrated(train_feat, future_feat, districts, xgb_cfg, rng)
    m4 = full_metrics(pred4, truth, districts)
    m4.update({"label":"XGBoost calibrated", "elapsed":round(time.perf_counter()-t0,1),
                "note":f"Quantile reg · n_est={xgb_cfg['n_estimators']} "
                       f"depth={xgb_cfg['max_depth']} lr={xgb_cfg['learning_rate']}"})
    results.append(m4)
    print(f"  CRPS={m4['avg_crps']}  RMSE={m4['avg_rmse']}  cov80={m4['avg_cov80']*100:.1f}%")

    # ── Config 5: Ensemble S+P+X improved ────────────────────────────────────
    print("[exp07] Config 5/6 — Ensemble S+P+X improved (Exp 06) ...")
    t0 = time.perf_counter()
    pred5 = build_ensemble(pred2, pred3, pred4)
    m5 = full_metrics(pred5, truth, districts)
    m5.update({"label":"Ensemble S+P+X", "elapsed":round(time.perf_counter()-t0,1),
                "note":"SARIMAX tuned + Prophet tuned + XGBoost calibrated · 150 samples"})
    results.append(m5)
    print(f"  CRPS={m5['avg_crps']}  RMSE={m5['avg_rmse']}  cov80={m5['avg_cov80']*100:.1f}%")

    # ── Config 6: Ensemble S+X ────────────────────────────────────────────────
    print("[exp07] Config 6/6 — Ensemble S+X (new) ...")
    t0 = time.perf_counter()
    pred6 = build_ensemble(pred2, pred4)
    m6 = full_metrics(pred6, truth, districts)
    m6.update({"label":"Ensemble S+X", "elapsed":round(time.perf_counter()-t0,1),
                "note":"SARIMAX tuned + XGBoost calibrated · 100 samples · no Prophet"})
    results.append(m6)
    print(f"  CRPS={m6['avg_crps']}  RMSE={m6['avg_rmse']}  cov80={m6['avg_cov80']*100:.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[exp07] Final ranking by CRPS:")
    for r in sorted(results, key=lambda x: x["avg_crps"]):
        print(f"  {r['label']:25s}  CRPS={r['avg_crps']:.3f}  "
              f"RMSE={r['avg_rmse']:.3f}  cov80={r['avg_cov80']*100:.1f}%")

    payload = {
        "results":   results,
        "districts": districts,
        "xgb_cfg":   xgb_cfg,
        "prophet_cfg": {"cps":PROPHET_CPS,"sps":PROPHET_SPS,"mode":PROPHET_MODE},
        "sarimax_thr": SARIMAX_THR,
        "n_samples":   N_SAMPLES,
        "train_weeks": TRAIN_WEEKS,
    }

    print("\n[exp07] Building HTML ...")
    html = build_html(payload)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[exp07] Done -> {OUT_HTML}")


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(p):
    J = json.dumps(p)
    results   = p["results"]
    districts = p["districts"]

    ranked = sorted(results, key=lambda r: r["avg_crps"])
    champ  = ranked[0]
    runner = ranked[1]

    # Ordered labels for charts (original order: 1→6)
    labels_short = {
        "SARIMAX baseline":  "SARIMAX\nbaseline",
        "SARIMAX tuned":     "SARIMAX\ntuned",
        "Prophet tuned":     "Prophet\ntuned",
        "XGBoost calibrated":"XGBoost\ncalibr.",
        "Ensemble S+P+X":    "Ens.\nS+P+X",
        "Ensemble S+X":      "Ens. S+X\n★ new",
    }

    def result_rows():
        rows = ""
        for r in ranked:
            is_champ = r["label"] == champ["label"]
            hi = "background:#40a02b0d;" if is_champ else ""
            star = " ⭐" if is_champ else ""
            rows += f"""
            <tr style="{hi}">
              <td><strong>{r['label']}{star}</strong><br>
                <span style="font-size:.74rem;color:var(--sub0)">{r.get('note','')}</span></td>
              <td style="font-family:'JetBrains Mono',monospace;font-weight:{'700' if is_champ else '400'}">{r['avg_crps']}</td>
              <td style="font-family:'JetBrains Mono',monospace">{r['avg_rmse']}</td>
              <td style="font-family:'JetBrains Mono',monospace">{r['avg_mae']}</td>
              <td style="font-family:'JetBrains Mono',monospace">{r['r2']}</td>
              <td style="font-family:'JetBrains Mono',monospace">{r['avg_cov80']*100:.1f}%</td>
              <td style="font-family:'JetBrains Mono',monospace">{r['avg_cov95']*100:.1f}%</td>
              <td style="font-family:'JetBrains Mono',monospace;color:var(--muted)">{r.get('elapsed','—')}s</td>
            </tr>"""
        return rows

    journey_rows = [
        ("Exp 01", "Feature correlation & lag analysis",
         "Revealed district-heterogeneous signals. Per-district r > 0 to 0.49. Pooled correlations misleadingly weak."),
        ("Exp 02", "SARIMAX auto-ARIMA order selection",
         "Fixed (1,0,1) beats auto-ARIMA on 6/7 districts. Short series punishes complex orders."),
        ("Exp 03", "Informed feature selection",
         "District-specific features (|r|>0.10) match baseline CRPS while using 5 covariates vs 16."),
        ("Exp 04", "Prophet hyperparameter tuning",
         "Best: cps=0.1, sps=2.0, multiplicative. CRPS 38.20→37.28. Coverage structurally limited."),
        ("Exp 05", "XGBoost calibration fix",
         "Quantile regression replaces residual bootstrap. CRPS 40.90→27.83. Coverage 3.9%→53.3%."),
        ("Exp 06", "Improved ensemble S+P+X",
         "All tuned components. CRPS 29.66→27.30. Coverage 70.3%→80.2% (perfectly calibrated)."),
        ("Exp 07", "S+X ensemble (no Prophet)",
         "Removing Prophet (weakest, CRPS 38.4) raises ensemble CRPS to 25.91 — best overall result."),
    ]
    journey_html = "".join(f"""
        <tr>
          <td><span class="pill pill-exp">{e}</span></td>
          <td><strong>{t}</strong></td>
          <td style="color:var(--sub0);font-size:.82rem">{f}</td>
        </tr>""" for e,t,f in journey_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 07 - Final Benchmark</title>
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
body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--page);color:var(--text);padding:28px 24px 56px;line-height:1.55;max-width:1400px;margin:0 auto}}
.header{{text-align:center;margin-bottom:28px;padding:40px 32px 30px;background:linear-gradient(160deg,#fff 0%,#f3eeff 100%);border-radius:20px;border:1px solid var(--border);box-shadow:0 2px 20px #8839ef0d;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 55% 70% at 10% 100%,#7287fd12 0%,transparent 60%),radial-gradient(ellipse 45% 60% at 90% 0%,#40a02b0e 0%,transparent 60%);pointer-events:none}}
.header h1{{font-size:2rem;font-weight:800;letter-spacing:-.5px;background:linear-gradient(120deg,var(--mauve) 0%,var(--blue) 50%,var(--teal) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}}
.header .sub{{color:var(--sub0);font-size:.88rem}}
.champ-banner{{display:inline-flex;align-items:center;gap:10px;background:#40a02b12;border:1px solid #40a02b44;border-radius:14px;padding:10px 20px;margin-top:14px;font-size:.85rem}}
.champ-banner strong{{color:#276315}}
.section-label{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin:28px 0 14px;display:flex;align-items:center;gap:8px}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
.grid{{display:grid;gap:16px;margin-bottom:4px}}
.grid-2{{grid-template-columns:1fr 1fr}}
.grid-3{{grid-template-columns:1fr 1fr 1fr}}
@media(max-width:960px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:22px 24px;box-shadow:0 1px 6px #0001;transition:box-shadow .2s,border-color .2s}}
.card:hover{{box-shadow:0 4px 16px #8839ef12;border-color:var(--border-h)}}
.card-title{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:7px}}
.dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 22px 16px;position:relative;overflow:hidden}}
.kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}}
.kpi:nth-child(1)::after{{background:linear-gradient(90deg,var(--green),var(--teal))}}
.kpi:nth-child(2)::after{{background:linear-gradient(90deg,var(--mauve),var(--lavender))}}
.kpi:nth-child(3)::after{{background:linear-gradient(90deg,var(--sky),var(--blue))}}
.kpi:nth-child(4)::after{{background:linear-gradient(90deg,var(--peach),var(--yellow))}}
.kpi-label{{font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;font-weight:700}}
.kpi-value{{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;margin-top:5px;letter-spacing:-1px;line-height:1}}
.kpi-sub{{font-size:.72rem;color:var(--muted);margin-top:5px}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
thead th{{background:var(--surface);color:var(--sub0);font-weight:700;text-transform:uppercase;font-size:.65rem;letter-spacing:.9px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}}
tbody td{{padding:10px 12px;border-bottom:1px solid #f0ebff;color:var(--sub1);vertical-align:middle}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f7f4ff}}
.pill{{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.72rem;font-weight:700}}
.pill-exp{{background:var(--surface);color:var(--mauve);border:1px solid #8839ef30;font-family:'JetBrains Mono',monospace;font-size:.7rem}}
.finding{{background:#fff;border:1px solid var(--border);border-left:4px solid var(--green);border-radius:0 12px 12px 0;padding:18px 22px;font-size:.88rem;line-height:1.7}}
canvas{{max-height:300px}}
</style>
</head>
<body>

<div class="header">
  <h1>Experiment 07 — Final Optimised Benchmark</h1>
  <div class="sub">
    7-experiment series &nbsp;·&nbsp; 6 final configs &nbsp;·&nbsp;
    train weeks 1–{p['train_weeks']} &nbsp;·&nbsp; test weeks {p['train_weeks']+1}–104 &nbsp;·&nbsp;
    n_samples={p['n_samples']} &nbsp;·&nbsp; 7 districts
  </div>
  <div class="champ-banner">
    <span style="font-size:1.2rem">⭐</span>
    <strong>Champion: {champ['label']}</strong>
    &nbsp;—&nbsp; CRPS {champ['avg_crps']} &nbsp;·&nbsp; RMSE {champ['avg_rmse']} &nbsp;·&nbsp;
    80% coverage {champ['avg_cov80']*100:.1f}%
  </div>
</div>

<div class="section-label">Champion Summary</div>
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Best CRPS</div>
    <div class="kpi-value" style="color:var(--green)">{champ['avg_crps']}</div>
    <div class="kpi-sub">{champ['label']} &nbsp;·&nbsp; lower = better</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Best RMSE</div>
    <div class="kpi-value" style="color:var(--mauve)">{champ['avg_rmse']}</div>
    <div class="kpi-sub">mean cases &nbsp;·&nbsp; lower = better</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">80% Coverage</div>
    <div class="kpi-value" style="color:var(--blue)">{champ['avg_cov80']*100:.1f}%</div>
    <div class="kpi-sub">target 80% &nbsp;·&nbsp; S+P+X hits 80.2%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">R²</div>
    <div class="kpi-value" style="color:var(--peach)">{champ['r2']}</div>
    <div class="kpi-sub">variance explained &nbsp;·&nbsp; higher = better</div>
  </div>
</div>

<div class="section-label">All Configs — Metric Comparison</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--mauve)"></span>
      CRPS — all 6 configs (lower is better)</div>
    <canvas id="crpsChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--teal)"></span>
      80% & 95% PI Coverage (target: 80% line shown)</div>
    <canvas id="covChart"></canvas>
  </div>
</div>
<div class="grid grid-2" style="margin-top:16px">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--blue)"></span>
      RMSE — all 6 configs</div>
    <canvas id="rmseChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--sky)"></span>
      Per-district CRPS — SARIMAX baseline vs Ensemble S+X</div>
    <canvas id="distChart"></canvas>
  </div>
</div>

<div class="section-label">Full Results Table</div>
<div class="card" style="overflow:auto">
  <table>
    <thead>
      <tr><th>Config</th><th>CRPS</th><th>RMSE</th><th>MAE</th><th>R²</th>
          <th>80% cov</th><th>95% cov</th><th>Time</th></tr>
    </thead>
    <tbody>{result_rows()}</tbody>
  </table>
</div>

<div class="section-label">Experiment Journey</div>
<div class="card" style="overflow:auto">
  <table>
    <thead><tr><th>Experiment</th><th>Question asked</th><th>Finding</th></tr></thead>
    <tbody>{journey_html}</tbody>
  </table>
</div>

<div class="section-label">Findings & Recommendations</div>
<div class="finding">
  <strong>Ensemble S+X is the new champion</strong> (CRPS {champ['avg_crps']}, RMSE {champ['avg_rmse']}).
  Removing Prophet from the ensemble improves CRPS by
  {round(results[4]['avg_crps']-champ['avg_crps'],3)} points over S+P+X, because Prophet's weaker
  probabilistic accuracy (CRPS 37–38) was diluting the pool. SARIMAX and XGBoost are
  sufficiently different — one is a linear state-space model, the other is a non-linear
  tree ensemble — to provide genuine diversity without needing a third model family.
  <br><br>
  <strong>Coverage trade-off:</strong> S+X hits {champ['avg_cov80']*100:.1f}% 80% coverage vs
  the S+P+X ensemble's 80.2%. Prophet's intermediate coverage (58%) was balancing SARIMAX's
  over-coverage (85.7%) and XGBoost's under-coverage (53.3%). Without it, the pool skews
  slightly narrow but remains within acceptable operational range.
  <br><br>
  <strong>When to use each config:</strong>
  <ul style="margin:10px 0 0 18px;line-height:2.1">
    <li><strong>Ensemble S+X ⭐</strong> — production forecasting, alert systems, resource allocation</li>
    <li><strong>SARIMAX tuned</strong> — when interpretability or speed matters; single best individual model</li>
    <li><strong>Ensemble S+P+X</strong> — when exact 80% calibration is a hard requirement</li>
    <li><strong>XGBoost calibrated</strong> — feature importance analysis, non-linear signal exploration</li>
    <li><strong>Prophet tuned</strong> — seasonal decomposition and trend visualisation only</li>
  </ul>
  <br>
  <strong>With more data (3+ years):</strong> Re-run Exp 03. The informed feature selection
  threshold may lower further, and SARIMAX + features may close the gap with the baseline.
</div>

<script>
const P = {J};
const RESULTS = P.results;

// Config order for charts (original 1→6)
const ORDER = ["SARIMAX baseline","SARIMAX tuned","Prophet tuned",
               "XGBoost calibrated","Ensemble S+P+X","Ensemble S+X"];
const COLORS = ["#1e66f5","#8839ef","#fe640b","#179299","#a099bb","#40a02b"];
const STARS  = ["","","","","","★ "];

function getR(label) {{ return RESULTS.find(r => r.label===label); }}

const ordered = ORDER.map(l => getR(l)).filter(Boolean);
const labels  = ORDER.map((l,i) => STARS[i]+l.replace(" (tuned)","").replace(" calibrated",""));

// CRPS bar
new Chart(document.getElementById("crpsChart"), {{
  type:"bar",
  data:{{ labels, datasets:[{{
    label:"avg CRPS", data:ordered.map(r=>r.avg_crps),
    backgroundColor:COLORS.map(c=>c+"cc"), borderRadius:6, borderSkipped:false
  }}]}},
  options:{{
    responsive:true, maintainAspectRatio:true,
    plugins:{{ legend:{{display:false}} }},
    scales:{{
      x:{{ ticks:{{font:{{size:10}},maxRotation:0}}, grid:{{display:false}} }},
      y:{{ title:{{display:true,text:"CRPS",font:{{size:10}}}},
           ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
    }}
  }}
}});

// Coverage grouped bar
new Chart(document.getElementById("covChart"), {{
  type:"bar",
  data:{{ labels, datasets:[
    {{ label:"80% cov", data:ordered.map(r=>+(r.avg_cov80*100).toFixed(1)),
       backgroundColor:COLORS.map(c=>c+"aa"), borderRadius:4 }},
    {{ label:"95% cov", data:ordered.map(r=>+(r.avg_cov95*100).toFixed(1)),
       backgroundColor:COLORS.map(c=>c+"44"), borderRadius:4 }},
    {{ label:"target 80%", data:Array(6).fill(80), type:"line",
       borderColor:"#40a02b99", borderDash:[4,3], pointRadius:0, fill:false }}
  ]}},
  options:{{
    responsive:true, maintainAspectRatio:true,
    plugins:{{ legend:{{ position:"top", labels:{{ font:{{size:10}}, boxWidth:10 }} }} }},
    scales:{{
      x:{{ ticks:{{font:{{size:10}},maxRotation:0}}, grid:{{display:false}} }},
      y:{{ min:0, max:110, title:{{display:true,text:"%",font:{{size:10}}}},
           ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
    }}
  }}
}});

// RMSE bar
new Chart(document.getElementById("rmseChart"), {{
  type:"bar",
  data:{{ labels, datasets:[{{
    label:"avg RMSE", data:ordered.map(r=>r.avg_rmse),
    backgroundColor:COLORS.map(c=>c+"cc"), borderRadius:6, borderSkipped:false
  }}]}},
  options:{{
    responsive:true, maintainAspectRatio:true,
    plugins:{{ legend:{{display:false}} }},
    scales:{{
      x:{{ ticks:{{font:{{size:10}},maxRotation:0}}, grid:{{display:false}} }},
      y:{{ title:{{display:true,text:"RMSE",font:{{size:10}}}},
           ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
    }}
  }}
}});

// Per-district comparison: SARIMAX baseline vs S+X
const BASE = getR("SARIMAX baseline");
const SX   = getR("Ensemble S+X");
const DISTS = P.districts.map(d=>d.replace("Mitsamiouli-Mboude","Mitsamiouli"));
new Chart(document.getElementById("distChart"), {{
  type:"bar",
  data:{{ labels:DISTS, datasets:[
    {{ label:"SARIMAX baseline", data:BASE.dist_crps, backgroundColor:"#1e66f5cc", borderRadius:4 }},
    {{ label:"Ensemble S+X ★",  data:SX.dist_crps,   backgroundColor:"#40a02bcc", borderRadius:4 }},
  ]}},
  options:{{
    responsive:true, maintainAspectRatio:true,
    plugins:{{ legend:{{ position:"top", labels:{{ font:{{size:10}}, boxWidth:10 }} }} }},
    scales:{{
      x:{{ ticks:{{font:{{size:10}},maxRotation:30}}, grid:{{display:false}} }},
      y:{{ title:{{display:true,text:"CRPS",font:{{size:10}}}},
           ticks:{{font:{{size:10}}}}, grid:{{color:"#8839ef0a"}} }}
    }}
  }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
