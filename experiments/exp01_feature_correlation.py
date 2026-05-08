"""
Experiment 01 — Feature Correlation & Lag Analysis
====================================================
Goal: Understand which climate variables, and at which time lags, most strongly
      predict malaria cases in each district.

Why this comes first:
  - Tells us which lags genuinely carry signal vs noise
  - Informs feature selection for SARIMAX (Exp 03)
  - Informs XGBoost feature importance interpretation (Exp 05)
  - Checks whether the biological assumptions in model_lib.py hold for Comoros

Outputs:
  output/experiments/exp01_feature_correlation.html  — visual presentation

Usage:
    cd C:/Vault/HISP-MODELING/Climatehealth-comoros
    python experiments/exp01_feature_correlation.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV  = os.path.join(ROOT, "input", "evaluation_data.csv")
OUT_HTML  = os.path.join(ROOT, "output", "experiments", "exp01_feature_correlation.html")

CLIMATE_VARS = ["rainfall", "mean_temperature", "humidity"]
MAX_LAG      = 8   # weeks
MIN_OBS      = 20  # minimum paired observations to compute a correlation

# ── Load data ──────────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(EVAL_CSV)
    df = df[df["disease_cases"].notna()].copy()
    df["disease_cases"] = df["disease_cases"].astype(float)
    return df.sort_values(["location", "time_period"]).reset_index(drop=True)

# ── Cross-correlation at each lag ──────────────────────────────────────────────
def lagged_correlations(df):
    """
    For each climate variable and lag k (0–MAX_LAG), compute Pearson r between
    climate_var[t-k] and cases[t] across all districts.

    Returns:
      pooled  — dict {var: [r_lag0, r_lag1, ...]}
      by_dist — dict {var: {district: [r_lag0, ...]}}
      pvals   — dict {var: [p_lag0, p_lag1, ...]}
    """
    pooled  = {v: [] for v in CLIMATE_VARS}
    pvals   = {v: [] for v in CLIMATE_VARS}
    by_dist = {v: {} for v in CLIMATE_VARS}

    districts = sorted(df["location"].unique())

    for var in CLIMATE_VARS:
        for lag in range(MAX_LAG + 1):
            all_x, all_y = [], []
            dist_r = {}

            for loc in districts:
                grp = df[df["location"] == loc].sort_values("time_period")
                y = grp["disease_cases"].values
                x = grp[var].values

                if lag == 0:
                    xi, yi = x, y
                else:
                    xi = x[:-lag]
                    yi = y[lag:]

                if len(xi) >= MIN_OBS:
                    r, p = stats.pearsonr(xi, yi)
                    dist_r[loc] = round(float(r), 4)
                    all_x.extend(xi.tolist())
                    all_y.extend(yi.tolist())

            by_dist[var][lag] = dist_r

            if len(all_x) >= MIN_OBS:
                r, p = stats.pearsonr(all_x, all_y)
                pooled[var].append(round(float(r), 4))
                pvals[var].append(round(float(p), 6))
            else:
                pooled[var].append(0.0)
                pvals[var].append(1.0)

    return pooled, pvals, by_dist, districts

# ── Correlation matrix (lag=0) ─────────────────────────────────────────────────
def correlation_matrix(df):
    cols = CLIMATE_VARS + ["disease_cases"]
    sub  = df[cols].dropna()
    mat  = sub.corr(method="pearson")
    return mat.round(3)

# ── Scatter data (sampled for performance) ────────────────────────────────────
def scatter_data(df, n=300):
    sub = df[CLIMATE_VARS + ["disease_cases", "location"]].dropna()
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    out = {}
    for var in CLIMATE_VARS:
        out[var] = {
            "x": sub[var].round(2).tolist(),
            "y": sub["disease_cases"].astype(int).tolist(),
        }
    return out

# ── Per-district best lag ──────────────────────────────────────────────────────
def best_lags(by_dist):
    result = {}
    for var in CLIMATE_VARS:
        result[var] = {}
        for dist in by_dist[var].get(0, {}):
            corrs = [by_dist[var][lag].get(dist, 0) for lag in range(MAX_LAG + 1)]
            best_lag = int(np.argmax(np.abs(corrs)))
            best_r   = round(corrs[best_lag], 3)
            result[var][dist] = {"best_lag": best_lag, "r": best_r}
    return result

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("[exp01] Loading data …")
    df = load()
    print(f"[exp01] {len(df)} rows, {df['location'].nunique()} districts, "
          f"{df['time_period'].nunique()} weeks")

    print("[exp01] Computing lagged correlations …")
    pooled, pvals, by_dist, districts = lagged_correlations(df)

    print("[exp01] Building correlation matrix …")
    mat = correlation_matrix(df)

    print("[exp01] Preparing scatter data …")
    scatter = scatter_data(df)

    best = best_lags(by_dist)

    # Serialise by_dist with string keys for JSON
    by_dist_json = {}
    for var in CLIMATE_VARS:
        by_dist_json[var] = {}
        for lag in range(MAX_LAG + 1):
            for dist, r in by_dist[var][lag].items():
                if dist not in by_dist_json[var]:
                    by_dist_json[var][dist] = []
                by_dist_json[var][dist].append(r)

    payload = {
        "pooled":     pooled,
        "pvals":      pvals,
        "by_dist":    by_dist_json,
        "districts":  districts,
        "corr_matrix": {
            "labels": mat.columns.tolist(),
            "values": mat.values.tolist(),
        },
        "scatter":    scatter,
        "best_lags":  best,
        "max_lag":    MAX_LAG,
        "vars":       CLIMATE_VARS,
    }

    print("[exp01] Generating HTML …")
    html = build_html(payload)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[exp01] Done -> {OUT_HTML}")


# ── HTML builder ───────────────────────────────────────────────────────────────
def build_html(p):
    data_js = json.dumps(p, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 01 — Feature Correlation & Lag Analysis</title>
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

.header{{text-align:center;margin-bottom:28px;padding:36px 32px 26px;background:linear-gradient(160deg,#fff 0%,#f3eeff 100%);border-radius:20px;border:1px solid var(--border);box-shadow:0 2px 20px #8839ef0d,0 1px 4px #0002;position:relative;overflow:hidden}}
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

/* Findings box */
.finding{{background:#f7f4ff;border-left:3px solid var(--mauve);border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:10px;font-size:.85rem}}
.finding strong{{color:var(--mauve);font-weight:700}}
.finding.good{{background:#f0fff5;border-left-color:var(--green)}}
.finding.good strong{{color:var(--green)}}
.finding.warn{{background:#fff8f0;border-left-color:var(--peach)}}
.finding.warn strong{{color:var(--peach)}}
.finding.bad{{background:#fff1f3;border-left-color:var(--red)}}
.finding.bad strong{{color:var(--red)}}

/* Best-lag table */
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
thead th{{background:var(--surface);color:var(--sub0);font-weight:700;text-transform:uppercase;font-size:.65rem;letter-spacing:.9px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}}
tbody td{{padding:9px 12px;border-bottom:1px solid #f0ebff;color:var(--sub1)}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f7f4ff;color:var(--text)}}
.pill{{display:inline-block;padding:2px 10px;border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:.75rem;font-weight:700}}

canvas{{max-height:280px}}
.footer{{text-align:center;padding:28px 0 4px;color:var(--ghost);font-size:.72rem}}
</style>
</head>
<body>

<div class="header">
  <div class="exp-tag">Experiment 01 of 06</div>
  <h1>Feature Correlation &amp; Lag Analysis</h1>
  <p class="sub">How strongly does each climate variable predict malaria cases — and with how many weeks of delay?</p>
</div>

<!-- Section 1: Lag correlations -->
<div class="section-label"><span>1 — Cross-Correlation at Lags 0–8 Weeks (Pooled Across Districts)</span></div>
<div class="grid grid-3">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:#04a5e5"></span>Rainfall vs Cases</div>
    <canvas id="lagRain"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:#fe640b"></span>Temperature vs Cases</div>
    <canvas id="lagTemp"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:#8839ef"></span>Humidity vs Cases</div>
    <canvas id="lagHum"></canvas>
  </div>
</div>

<!-- Section 2: All vars on one chart -->
<div class="section-label"><span>2 — Combined Lag Profile (Peak Signal per Variable)</span></div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--mauve)"></span>All Variables — Pooled Cross-Correlation</div>
    <canvas id="lagAll"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--teal)"></span>Per-District Lag Profile (SARIMAX-optimal lag)</div>
    <canvas id="distLag"></canvas>
  </div>
</div>

<!-- Section 3: Correlation matrix -->
<div class="section-label"><span>3 — Correlation Matrix (Lag 0)</span></div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--green)"></span>Pearson Correlation Heatmap</div>
    <canvas id="heatmap"></canvas>
  </div>
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--yellow)"></span>Best Lag per District &amp; Variable</div>
    <div style="overflow-x:auto">
      <table id="bestLagTable">
        <thead><tr><th>District</th><th>Rainfall (best lag)</th><th>Temperature (best lag)</th><th>Humidity (best lag)</th></tr></thead>
        <tbody id="lagTableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- Section 4: Findings -->
<div class="section-label"><span>4 — Findings &amp; Implications for Modelling</span></div>
<div id="findings"></div>

<div class="footer">Experiment 01 · Comoros Malaria Climate-Health Modelling · Ngadjizi Region</div>

<script>
const DATA = {data_js};

// Colour scheme
const varColors = {{
  rainfall:        '#04a5e5',
  mean_temperature:'#fe640b',
  humidity:        '#8839ef',
}};
const varNames = {{
  rainfall:        'Rainfall',
  mean_temperature:'Temperature',
  humidity:        'Humidity',
}};
const lags = Array.from({{length: DATA.max_lag + 1}}, (_, i) => `Lag ${{i}}w`);

// ── Chart defaults ─────────────────────────────────────────────────────────
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

function lagBar(canvasId, variable, color) {{
  const corrs = DATA.pooled[variable];
  const ps    = DATA.pvals[variable];
  new Chart(document.getElementById(canvasId), {{
    type: 'bar',
    data: {{
      labels: lags,
      datasets: [{{
        label: 'Pearson r',
        data: corrs,
        backgroundColor: corrs.map((r, i) =>
          ps[i] < 0.05 ? color + 'cc' : color + '33'
        ),
        borderColor: corrs.map((r, i) =>
          ps[i] < 0.05 ? color : color + '66'
        ),
        borderWidth: 1.5,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{display: false}},
        tooltip: {{callbacks: {{
          label: (ctx) => {{
            const p = ps[ctx.dataIndex];
            const sig = p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : 'ns';
            return ` r = ${{ctx.parsed.y.toFixed(3)}}  p${{sig}}`;
          }}
        }}}}
      }},
      scales: {{
        y: {{min: -0.6, max: 0.6, grid: {{color: grid}},
             ticks: {{callback: v => v.toFixed(1)}}}},
        x: {{grid: {{display: false}}}}
      }}
    }}
  }});
}}

lagBar('lagRain', 'rainfall',         '#04a5e5');
lagBar('lagTemp', 'mean_temperature', '#fe640b');
lagBar('lagHum',  'humidity',         '#8839ef');

// ── Combined lag line chart ────────────────────────────────────────────────
new Chart(document.getElementById('lagAll'), {{
  type: 'line',
  data: {{
    labels: lags,
    datasets: DATA.vars.map(v => ({{
      label: varNames[v],
      data: DATA.pooled[v],
      borderColor: varColors[v],
      backgroundColor: varColors[v] + '18',
      borderWidth: 2.5,
      pointRadius: 4,
      pointBackgroundColor: varColors[v],
      tension: 0.35,
      fill: false,
    }}))
  }},
  options: {{
    responsive: true,
    plugins: {{legend: {{position: 'top', labels: {{usePointStyle:true,pointStyle:'circle',padding:14}}}}}},
    scales: {{
      y: {{min: -0.6, max: 0.6, grid: {{color: grid}},
           ticks: {{callback: v => v.toFixed(1)}}}},
      x: {{grid: {{display: false}}}}
    }}
  }}
}});

// ── Per-district best-lag bar chart ───────────────────────────────────────
const districts = DATA.districts;
const distColors = ['#04a5e5','#8839ef','#40a02b','#d20f39','#fe640b','#c97b0d','#7287fd'];
const distDatasets = DATA.vars.map((v, vi) => ({{
  label: varNames[v],
  data: districts.map(d => DATA.best_lags[v][d] ? DATA.best_lags[v][d].best_lag : 0),
  backgroundColor: Object.values(varColors)[vi] + 'aa',
  borderColor: Object.values(varColors)[vi],
  borderWidth: 1.5,
  borderRadius: 4,
}}));
new Chart(document.getElementById('distLag'), {{
  type: 'bar',
  data: {{labels: districts.map(d => d.length > 12 ? d.slice(0,12)+'…' : d), datasets: distDatasets}},
  options: {{
    responsive: true,
    plugins: {{legend: {{position:'top', labels:{{usePointStyle:true,pointStyle:'circle',padding:12}}}}}},
    scales: {{
      y: {{title:{{display:true,text:'Weeks lag',font:{{size:10}}}},
           min:0, max: DATA.max_lag, ticks:{{stepSize:1}}, grid:{{color:grid}}}},
      x: {{grid:{{display:false}}, ticks:{{maxRotation:25}}}}
    }}
  }}
}});

// ── Heatmap via grouped bar (Chart.js workaround) ─────────────────────────
const matLabels = DATA.corr_matrix.labels.map(l => ({{
  rainfall:'Rainfall', mean_temperature:'Temperature', humidity:'Humidity', disease_cases:'Cases'
}})[l] || l);
const matVals   = DATA.corr_matrix.values;
const heatColors = (v) => {{
  if (v >= 0.7)  return '#40a02bcc';
  if (v >= 0.4)  return '#40a02b55';
  if (v >= 0.2)  return '#04a5e555';
  if (v >= 0)    return '#04a5e522';
  if (v >= -0.2) return '#fe640b22';
  if (v >= -0.4) return '#fe640b55';
  return '#d20f39cc';
}};

// Render correlation matrix as an HTML table instead
(function() {{
  const mat  = DATA.corr_matrix.values;
  const labs = matLabels;
  const canvas = document.getElementById('heatmap');
  const container = canvas.parentElement;
  canvas.remove();

  let html = '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.8rem">';
  html += '<thead><tr><th style="padding:6px 8px;background:#ebe4ff;border:1px solid #ddd5f5"></th>';
  labs.forEach(l => {{
    html += `<th style="padding:6px 8px;background:#ebe4ff;border:1px solid #ddd5f5;font-size:.68rem;font-weight:700;color:#7a6e92;text-transform:uppercase;letter-spacing:.6px;white-space:nowrap">${{l}}</th>`;
  }});
  html += '</tr></thead><tbody>';
  mat.forEach((row, i) => {{
    html += `<tr><td style="padding:6px 8px;background:#ebe4ff;border:1px solid #ddd5f5;font-size:.68rem;font-weight:700;color:#7a6e92;text-transform:uppercase;letter-spacing:.6px;white-space:nowrap">${{labs[i]}}</td>`;
    row.forEach((val, j) => {{
      const bg = heatColors(val);
      const txt = Math.abs(val) > 0.3 ? '#2a2044' : '#7a6e92';
      const mono = val === 1 ? '—' : val.toFixed(3);
      html += `<td style="padding:8px 10px;border:1px solid #f0ebff;background:${{bg}};color:${{txt}};text-align:center;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:500">${{mono}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table></div>';
  container.insertAdjacentHTML('beforeend', html);
}})();

// ── Best-lag table ─────────────────────────────────────────────────────────
const lagColors = (lag) => {{
  const cs = ['#f0fff5','#e8faff','#eef0ff','#f5f0ff','#fff5ee','#fff0ee','#ffeef2','#fff8ee','#f5fff0'];
  return cs[lag] || '#f7f4ff';
}};
const tbody = document.getElementById('lagTableBody');
districts.forEach(d => {{
  const cells = DATA.vars.map(v => {{
    const info = DATA.best_lags[v][d];
    if (!info) return '<td>—</td>';
    const bg = lagColors(info.best_lag);
    const rSign = info.r >= 0 ? '+' : '';
    return `<td><span class="pill" style="background:${{bg}};color:${{info.r>0?'#2a5c3f':'#8b0000'}}">`
         + `lag ${{info.best_lag}}w &nbsp; r=${{rSign}}${{info.r}}</span></td>`;
  }});
  tbody.innerHTML += `<tr><td style="font-weight:600;color:#2a2044">${{d}}</td>${{cells.join('')}}</tr>`;
}});

// ── Auto-generate findings ─────────────────────────────────────────────────
const findDiv = document.getElementById('findings');

// Find strongest overall lag per variable
function peakLag(v) {{
  const c = DATA.pooled[v];
  const idx = c.reduce((bi,r,i) => Math.abs(r)>Math.abs(c[bi])?i:bi, 0);
  return {{lag: idx, r: c[idx]}};
}}
const peaks = {{}};
DATA.vars.forEach(v => peaks[v] = peakLag(v));

// Rainfall finding
const rf = peaks.rainfall;
const rfSign = rf.r >= 0 ? 'positively' : 'negatively';
const rfType = Math.abs(rf.r) > 0.3 ? 'good' : Math.abs(rf.r) > 0.15 ? 'warn' : 'bad';
findDiv.innerHTML += `<div class="finding ${{rfType}}">
  <strong>Rainfall:</strong> Strongest correlation with cases at <strong>lag ${{rf.lag}} weeks</strong>
  (r = ${{rf.r.toFixed(3)}}). Rainfall is <strong>${{rfSign}}</strong> correlated with future cases.
  ${{Math.abs(rf.r) > 0.3 ? 'This is a meaningful signal — include rainfall lags in models.' :
     'Signal is weak — rainfall lags may not add value in this dataset.'}}
</div>`;

// Temperature finding
const tf = peaks.mean_temperature;
const ttSign = tf.r >= 0 ? 'positively' : 'negatively';
const ttType = Math.abs(tf.r) > 0.3 ? 'good' : Math.abs(tf.r) > 0.15 ? 'warn' : 'bad';
findDiv.innerHTML += `<div class="finding ${{ttType}}">
  <strong>Temperature:</strong> Strongest correlation at <strong>lag ${{tf.lag}} weeks</strong>
  (r = ${{tf.r.toFixed(3)}}). Temperature is <strong>${{ttSign}}</strong> correlated with future cases.
  ${{Math.abs(tf.r) > 0.3 ? 'Meaningful signal — include temperature lags.' :
     'Weak signal — temperature lags may add noise rather than value.'}}
</div>`;

// Humidity finding
const hf = peaks.humidity;
const hhSign = hf.r >= 0 ? 'positively' : 'negatively';
const hhType = Math.abs(hf.r) > 0.3 ? 'good' : Math.abs(hf.r) > 0.15 ? 'warn' : 'bad';
findDiv.innerHTML += `<div class="finding ${{hhType}}">
  <strong>Humidity:</strong> Strongest correlation at <strong>lag ${{hf.lag}} weeks</strong>
  (r = ${{hf.r.toFixed(3)}}). Humidity is <strong>${{hhSign}}</strong> correlated with future cases.
  ${{Math.abs(hf.r) > 0.3 ? 'Include humidity lags in models.' :
     'Consider whether humidity adds independent signal beyond rainfall.'}}
</div>`;

// Model implication
const bestOverall = DATA.vars.reduce((bv, v) =>
  Math.abs(peaks[v].r) > Math.abs(peaks[bv].r) ? v : bv, DATA.vars[0]);
findDiv.innerHTML += `<div class="finding">
  <strong>Model implication:</strong> The strongest single predictor is
  <strong>${{varNames[bestOverall]}}</strong> at lag <strong>${{peaks[bestOverall].lag}} weeks</strong>
  (r = ${{peaks[bestOverall].r.toFixed(3)}}).
  For Experiment 03 (informed feature selection), only include lags where |r| &gt; 0.15
  to avoid adding noise. Shaded bars above show statistically significant correlations (p &lt; 0.05).
  Faded bars are not significant and should be excluded.
</div>`;
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
