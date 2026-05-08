"""
Experiment 02 — SARIMAX Order Selection via Auto-ARIMA
=======================================================
Goal: Find the optimal (p, d, q) ARIMA order for each district instead of
      using the fixed (1, 0, 1) assumption inherited from Rwanda.

Why this matters (from Exp 01):
  - Districts have different autocorrelation structures
  - A fixed order may under- or over-fit individual districts
  - Leonard: "optimise hyperparams for each model according to its requirements"

Method:
  - pmdarima auto_arima with stepwise search over p in [0..3], q in [0..3], d in [0..1]
  - No seasonal terms (only 104 weeks — too short for weekly seasonality estimation)
  - Base covariates only (rainfall, mean_temperature, humidity) — no engineered features yet
  - Evaluate each district's auto-selected order vs fixed (1,0,1) on the test split

Outputs:
  output/experiments/exp02_sarimax_order_selection.html

Usage:
    cd C:/Vault/HISP-MODELING/Climatehealth-comoros
    python experiments/exp02_sarimax_order_selection.py
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV = os.path.join(ROOT, "input", "evaluation_data.csv")
OUT_HTML = os.path.join(ROOT, "output", "experiments", "exp02_sarimax_order_selection.html")

TRAIN_WEEKS   = 78
COVARIATES    = ["rainfall", "mean_temperature", "humidity"]
FIXED_ORDER   = (1, 0, 1)
N_SAMPLES     = 50
RNG_SEED      = 42

sys.path.insert(0, ROOT)

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_split():
    df = pd.read_csv(EVAL_CSV)
    df = df[df["disease_cases"].notna()].copy()
    df["disease_cases"] = df["disease_cases"].astype(float)
    all_weeks  = sorted(df["time_period"].unique())
    train_wks  = set(all_weeks[:TRAIN_WEEKS])
    test_wks   = set(all_weeks[TRAIN_WEEKS:])
    train = df[df["time_period"].isin(train_wks)].sort_values(["location","time_period"])
    test  = df[df["time_period"].isin(test_wks)].sort_values(["location","time_period"])
    return train, test

# ── CRPS ───────────────────────────────────────────────────────────────────────
def crps_ensemble(obs, samples):
    m  = samples.shape[1]
    s  = np.sort(samples, axis=1)
    t1 = np.mean(np.abs(s - obs[:, None]), axis=1)
    ranks = np.arange(1, m + 1)
    gini  = np.sum((2*ranks - m - 1) * s, axis=1) / (m * (m-1))
    return float(np.mean(t1 - gini))

def rmse(obs, pred_median):
    return float(np.sqrt(np.mean((obs - pred_median)**2)))

# ── Fit & predict SARIMAX ──────────────────────────────────────────────────────
def fit_predict_sarimax(train_grp, test_grp, order, n_samples, rng):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_train = train_grp["disease_cases"].values
    X_train = train_grp[COVARIATES].values
    X_test  = test_grp[COVARIATES].values
    y_test  = test_grp["disease_cases"].values

    model = SARIMAX(endog=y_train, exog=X_train, order=order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=300)

    n_periods = len(X_test)
    samples   = np.zeros((n_periods, n_samples))
    for i in range(n_samples):
        sim = fit.simulate(nsimulations=n_periods, exog=X_test,
                           anchor="end",
                           random_state=int(rng.integers(0, 2**31-1)))
        samples[:, i] = np.maximum(0, sim)

    median  = np.median(samples, axis=1)
    c = crps_ensemble(y_test, samples)
    r = rmse(y_test, median)
    return {"crps": round(c, 2), "rmse": round(r, 2),
            "samples": samples, "y_test": y_test, "order": order}

# ── Auto-ARIMA order search ────────────────────────────────────────────────────
def find_best_order(y_train, X_train):
    try:
        from pmdarima import auto_arima
        model = auto_arima(
            y_train, X=X_train,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None, max_d=1,
            seasonal=False,
            stepwise=True,
            information_criterion="aic",
            error_action="ignore",
            suppress_warnings=True,
        )
        return model.order
    except Exception as e:
        print(f"    auto_arima failed ({e}), falling back to (1,0,1)")
        return FIXED_ORDER

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("[exp02] Loading data ...")
    train_df, test_df = load_split()
    districts = sorted(train_df["location"].unique())
    rng = np.random.default_rng(RNG_SEED)

    results = []
    for loc in districts:
        print(f"[exp02]   {loc} ...")
        tr = train_df[train_df["location"]==loc].sort_values("time_period")
        te = test_df[test_df["location"]==loc].sort_values("time_period")

        y_tr = tr["disease_cases"].values
        X_tr = tr[COVARIATES].values

        # Auto-ARIMA order search
        t0 = time.perf_counter()
        auto_order = find_best_order(y_tr, X_tr)
        search_time = time.perf_counter() - t0
        print(f"    auto order: {auto_order}  ({search_time:.1f}s)")

        # Evaluate fixed (1,0,1)
        r_fixed = fit_predict_sarimax(tr, te, FIXED_ORDER, N_SAMPLES,
                                       np.random.default_rng(RNG_SEED))

        # Evaluate auto order
        r_auto = fit_predict_sarimax(tr, te, auto_order, N_SAMPLES,
                                      np.random.default_rng(RNG_SEED))

        # Improvement
        crps_delta = round(r_auto["crps"] - r_fixed["crps"], 2)
        rmse_delta = round(r_auto["rmse"] - r_fixed["rmse"], 2)

        results.append({
            "district":    loc,
            "auto_order":  list(auto_order),
            "fixed_order": list(FIXED_ORDER),
            "auto_crps":   r_auto["crps"],
            "fixed_crps":  r_fixed["crps"],
            "auto_rmse":   r_auto["rmse"],
            "fixed_rmse":  r_fixed["rmse"],
            "crps_delta":  crps_delta,   # negative = auto is better
            "rmse_delta":  rmse_delta,   # negative = auto is better
            "search_secs": round(search_time, 1),
        })

    # Summary
    n_improved_crps = sum(1 for r in results if r["crps_delta"] < 0)
    avg_crps_fixed  = round(np.mean([r["fixed_crps"] for r in results]), 2)
    avg_crps_auto   = round(np.mean([r["auto_crps"]  for r in results]), 2)
    avg_rmse_fixed  = round(np.mean([r["fixed_rmse"] for r in results]), 2)
    avg_rmse_auto   = round(np.mean([r["auto_rmse"]  for r in results]), 2)

    print(f"\n[exp02] Results:")
    print(f"  Fixed (1,0,1): avg CRPS={avg_crps_fixed}  avg RMSE={avg_rmse_fixed}")
    print(f"  Auto order   : avg CRPS={avg_crps_auto}   avg RMSE={avg_rmse_auto}")
    print(f"  Districts improved (CRPS): {n_improved_crps}/{len(districts)}")

    payload = {
        "results":        results,
        "districts":      districts,
        "avg_crps_fixed": avg_crps_fixed,
        "avg_crps_auto":  avg_crps_auto,
        "avg_rmse_fixed": avg_rmse_fixed,
        "avg_rmse_auto":  avg_rmse_auto,
        "n_improved":     n_improved_crps,
        "n_total":        len(districts),
        "fixed_order":    list(FIXED_ORDER),
    }

    print("[exp02] Building HTML ...")
    html = build_html(payload)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[exp02] Done -> {OUT_HTML}")


# ── HTML ───────────────────────────────────────────────────────────────────────
def build_html(p):
    data_js = json.dumps(p, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 02 - SARIMAX Order Selection</title>
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
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:24px}}
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
tbody td{{padding:10px 12px;border-bottom:1px solid #f0ebff;color:var(--sub1);white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f7f4ff;color:var(--text)}}
.pill{{display:inline-block;padding:2px 10px;border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:.74rem;font-weight:700}}
.improved{{color:var(--green);font-weight:700;font-family:'JetBrains Mono',monospace}}
.worse{{color:var(--red);font-weight:700;font-family:'JetBrains Mono',monospace}}
.same{{color:var(--muted);font-family:'JetBrains Mono',monospace}}
.finding{{background:#f7f4ff;border-left:3px solid var(--mauve);border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:10px;font-size:.85rem}}
.finding strong{{color:var(--mauve);font-weight:700}}
.finding.good{{background:#f0fff5;border-left-color:var(--green)}}
.finding.good strong{{color:var(--green)}}
.finding.warn{{background:#fff8f0;border-left-color:var(--peach)}}
.finding.warn strong{{color:var(--peach)}}
canvas{{max-height:280px}}
.footer{{text-align:center;padding:28px 0 4px;color:var(--ghost);font-size:.72rem}}
</style>
</head>
<body>

<div class="header">
  <div class="exp-tag">Experiment 02 of 06</div>
  <h1>SARIMAX Order Selection — Auto-ARIMA per District</h1>
  <p class="sub">Does each district need its own (p,d,q) order, or is a fixed (1,0,1) good enough?</p>
</div>

<div class="kpi-row" id="kpis"></div>

<div class="section-label"><span>1 — Fixed (1,0,1) vs Auto-Selected Order — CRPS & RMSE per District</span></div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--sky)"></span>CRPS Comparison per District</div>
    <canvas id="crpsChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--green)"></span>RMSE Comparison per District</div>
    <canvas id="rmseChart"></canvas>
  </div>
</div>

<div class="section-label"><span>2 — Auto-Selected Orders &amp; Per-District Results</span></div>
<div class="card" style="margin-bottom:4px">
  <div class="card-title"><span class="dot" style="background:var(--mauve)"></span>Detailed Results Table</div>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>District</th>
          <th>Auto Order</th>
          <th>CRPS (fixed)</th>
          <th>CRPS (auto)</th>
          <th>CRPS delta</th>
          <th>RMSE (fixed)</th>
          <th>RMSE (auto)</th>
          <th>RMSE delta</th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>
</div>

<div class="section-label"><span>3 — Order Distribution: What Did Auto-ARIMA Choose?</span></div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--peach)"></span>Distribution of p (AR terms)</div>
    <canvas id="pChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--teal)"></span>Distribution of q (MA terms)</div>
    <canvas id="qChart"></canvas>
  </div>
</div>

<div class="section-label"><span>4 — Findings</span></div>
<div id="findings"></div>

<div class="footer">Experiment 02 · Comoros Malaria Climate-Health Modelling · Ngadjizi Region</div>

<script>
const DATA = {data_js};

Chart.defaults.color       = '#7a6e92';
Chart.defaults.borderColor = '#e8e2f5';
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.font.size   = 11;
Chart.defaults.plugins.tooltip.backgroundColor = '#ffffff';
Chart.defaults.plugins.tooltip.borderColor     = '#ddd5f5';
Chart.defaults.plugins.tooltip.borderWidth     = 1;
Chart.defaults.plugins.tooltip.titleColor      = '#2a2044';
Chart.defaults.plugins.tooltip.bodyColor       = '#544873';
Chart.defaults.plugins.tooltip.bodyFont        = {{family:"'JetBrains Mono',monospace",size:11}};
Chart.defaults.plugins.tooltip.padding         = 9;
const grid = '#ece7fa';

const shortNames = DATA.districts.map(d => d.length > 13 ? d.slice(0,13)+'...' : d);

// ── KPI cards ──────────────────────────────────────────────────────────────
const crpsDelta = (DATA.avg_crps_auto - DATA.avg_crps_fixed).toFixed(2);
const rmseDelta = (DATA.avg_rmse_auto - DATA.avg_rmse_fixed).toFixed(1);
const crpsSign  = crpsDelta < 0 ? '' : '+';
const rmseSign  = rmseDelta < 0 ? '' : '+';
document.getElementById('kpis').innerHTML = `
  <div class="kpi">
    <div class="kpi-label">Avg CRPS Fixed</div>
    <div class="kpi-value" style="color:var(--sky)">${{DATA.avg_crps_fixed}}</div>
    <div class="kpi-sub">SARIMAX (1,0,1)</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg CRPS Auto</div>
    <div class="kpi-value" style="color:var(--green)">${{DATA.avg_crps_auto}}</div>
    <div class="kpi-sub">Per-district optimal order</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">CRPS Change</div>
    <div class="kpi-value" style="color:${{crpsDelta<0?'var(--green)':'var(--red)'}}">${{crpsSign}}${{crpsDelta}}</div>
    <div class="kpi-sub">${{crpsDelta<0?'Auto is better':'Fixed is better'}}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Districts Improved</div>
    <div class="kpi-value" style="color:var(--mauve)">${{DATA.n_improved}}<span style="font-size:1rem;font-weight:500">/${{DATA.n_total}}</span></div>
    <div class="kpi-sub">on CRPS metric</div>
  </div>
`;

// ── CRPS bar chart ─────────────────────────────────────────────────────────
new Chart(document.getElementById('crpsChart'), {{
  type:'bar',
  data:{{
    labels: shortNames,
    datasets:[
      {{label:'Fixed (1,0,1)', data:DATA.results.map(r=>r.fixed_crps),
        backgroundColor:'#04a5e566',borderColor:'#04a5e5',borderWidth:1.5,borderRadius:4}},
      {{label:'Auto order', data:DATA.results.map(r=>r.auto_crps),
        backgroundColor:'#40a02baa',borderColor:'#40a02b',borderWidth:1.5,borderRadius:4}},
    ]
  }},
  options:{{responsive:true,
    plugins:{{legend:{{position:'top',labels:{{usePointStyle:true,pointStyle:'circle',padding:12}}}}}},
    scales:{{y:{{beginAtZero:true,grid:{{color:grid}}}},x:{{grid:{{display:false}},ticks:{{maxRotation:25}}}}}}
  }}
}});

// ── RMSE bar chart ─────────────────────────────────────────────────────────
new Chart(document.getElementById('rmseChart'), {{
  type:'bar',
  data:{{
    labels: shortNames,
    datasets:[
      {{label:'Fixed (1,0,1)', data:DATA.results.map(r=>r.fixed_rmse),
        backgroundColor:'#8839ef66',borderColor:'#8839ef',borderWidth:1.5,borderRadius:4}},
      {{label:'Auto order', data:DATA.results.map(r=>r.auto_rmse),
        backgroundColor:'#fe640baa',borderColor:'#fe640b',borderWidth:1.5,borderRadius:4}},
    ]
  }},
  options:{{responsive:true,
    plugins:{{legend:{{position:'top',labels:{{usePointStyle:true,pointStyle:'circle',padding:12}}}}}},
    scales:{{y:{{beginAtZero:true,grid:{{color:grid}}}},x:{{grid:{{display:false}},ticks:{{maxRotation:25}}}}}}
  }}
}});

// ── Table ──────────────────────────────────────────────────────────────────
const tbody = document.getElementById('tableBody');
DATA.results.forEach(r => {{
  const orderLabel = `(${{r.auto_order[0]}},​${{r.auto_order[1]}},​${{r.auto_order[2]}})`;
  const isFixed = r.auto_order[0]===1 && r.auto_order[1]===0 && r.auto_order[2]===1;
  const pillBg  = isFixed ? '#ebe4ff' : '#e8faff';
  const pillClr = isFixed ? 'var(--mauve)' : 'var(--teal)';
  const crpsCls = r.crps_delta < -0.5 ? 'improved' : r.crps_delta > 0.5 ? 'worse' : 'same';
  const rmseCls = r.rmse_delta < -1 ? 'improved' : r.rmse_delta > 1 ? 'worse' : 'same';
  const crpsArr = r.crps_delta < -0.5 ? ' v' : r.crps_delta > 0.5 ? ' ^' : '';
  const rmseArr = r.rmse_delta < -1   ? ' v' : r.rmse_delta > 1   ? ' ^' : '';
  tbody.innerHTML += `<tr>
    <td style="font-weight:600;color:var(--text)">${{r.district}}</td>
    <td><span class="pill" style="background:${{pillBg}};color:${{pillClr}}">${{orderLabel}}${{isFixed?' =fixed':''}}</span></td>
    <td><span class="pill" style="background:#e8faff;color:var(--teal)">${{r.fixed_crps}}</span></td>
    <td><span class="pill" style="background:#f0fff5;color:var(--green)">${{r.auto_crps}}</span></td>
    <td class="${{crpsCls}}">${{r.crps_delta > 0 ? '+' : ''}}​${{r.crps_delta}}${{crpsArr}}</td>
    <td>${{r.fixed_rmse}}</td>
    <td>${{r.auto_rmse}}</td>
    <td class="${{rmseCls}}">${{r.rmse_delta > 0 ? '+' : ''}}​${{r.rmse_delta}}${{rmseArr}}</td>
  </tr>`;
}});

// ── p and q distribution charts ────────────────────────────────────────────
function orderDist(key, canvasId, color) {{
  const vals  = DATA.results.map(r => r.auto_order[key==='p'?0:2]);
  const fixed = key==='p' ? 1 : 1;
  const counts = [0,0,0,0];
  vals.forEach(v => {{ if(v<=3) counts[v]++; }});
  new Chart(document.getElementById(canvasId), {{
    type:'bar',
    data:{{
      labels:['0','1','2','3'],
      datasets:[{{
        label: key==='p'?'AR order (p)':'MA order (q)',
        data: counts,
        backgroundColor: counts.map((_,i) => i===fixed ? color+'cc' : color+'44'),
        borderColor: color,
        borderWidth:1.5,borderRadius:5
      }}]
    }},
    options:{{responsive:true,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>`${{ctx.parsed.y}} district(s)`}}}}
      }},
      scales:{{
        y:{{beginAtZero:true,ticks:{{stepSize:1}},grid:{{color:grid}},
            title:{{display:true,text:'Number of districts',font:{{size:10}}}}}},
        x:{{grid:{{display:false}},title:{{display:true,
            text: key==='p'?'p value (AR terms)':'q value (MA terms)',font:{{size:10}}}}}}
      }}
    }}
  }});
}}
orderDist('p','pChart','#fe640b');
orderDist('q','qChart','#179299');

// ── Findings ───────────────────────────────────────────────────────────────
const findDiv = document.getElementById('findings');
const crpsImproved = DATA.n_improved;
const overallBetter = DATA.avg_crps_auto < DATA.avg_crps_fixed;

if (overallBetter) {{
  const saving = (DATA.avg_crps_fixed - DATA.avg_crps_auto).toFixed(2);
  findDiv.innerHTML += `<div class="finding good">
    <strong>Auto-ARIMA improves overall CRPS</strong> by ${{saving}} points on average
    (${{DATA.avg_crps_fixed}} to ${{DATA.avg_crps_auto}}). ${{crpsImproved}} out of
    ${{DATA.n_total}} districts benefited. Per-district order selection is worth the
    extra search time.
  </div>`;
}} else {{
  const cost = (DATA.avg_crps_auto - DATA.avg_crps_fixed).toFixed(2);
  findDiv.innerHTML += `<div class="finding warn">
    <strong>Auto-ARIMA does not consistently improve over fixed (1,0,1)</strong> — average CRPS
    is ${{cost}} points higher. Only ${{crpsImproved}}/${{DATA.n_total}} districts improved.
    The fixed order is surprisingly competitive for this dataset.
  </div>`;
}}

// Which districts most improved?
const sortedByDelta = [...DATA.results].sort((a,b) => a.crps_delta - b.crps_delta);
const bestGain = sortedByDelta[0];
const worstLoss = sortedByDelta[sortedByDelta.length-1];

findDiv.innerHTML += `<div class="finding">
  <strong>Largest CRPS improvement:</strong> ${{bestGain.district}}
  (${{bestGain.crps_delta > 0 ? '+' : ''}}${{bestGain.crps_delta}}) &mdash;
  auto order (${{bestGain.auto_order.join(',')}}) vs fixed (1,0,1).
</div>`;

if (worstLoss.crps_delta > 1) {{
  findDiv.innerHTML += `<div class="finding warn">
    <strong>Largest regression:</strong> ${{worstLoss.district}}
    (+${{worstLoss.crps_delta}} CRPS) &mdash; auto order (${{worstLoss.auto_order.join(',')}})
    overfit this district. The fixed (1,0,1) was better here.
  </div>`;
}}

// Implication for Exp 03
const useAuto = overallBetter;
findDiv.innerHTML += `<div class="finding">
  <strong>Decision for Experiment 03:</strong>
  ${{useAuto
    ? 'Use per-district auto-selected orders going forward. Carry these orders into Experiment 03 (informed feature selection) to avoid conflating order choice with feature choice.'
    : 'Retain fixed (1,0,1) as the ARIMA structure. The auto search did not justify its complexity on this dataset size. Focus Experiment 03 on feature selection with the stable (1,0,1) baseline.'
  }}
</div>`;
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
