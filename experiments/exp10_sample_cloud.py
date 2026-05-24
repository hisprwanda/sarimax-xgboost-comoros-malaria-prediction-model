"""
Exp 10 — Sample Cloud Visualization

The core question: where does the actual case count fall within the full
probabilistic forecast distribution? The Ensemble S+X model produces 100
samples per district per week. Exp 09 showed only summary statistics (PI bands
and median). This experiment exposes the raw sample cloud.

Key visualizations
──────────────────
① Sample-cloud timeline    All 100 sample trajectories drawn as a
                           semi-transparent cloud using a Chart.js Canvas2D
                           plugin hook (fast — not 100 separate datasets).
                           Actual cases are plotted on top in the district
                           colour. Predicted median as a dashed line.
                           Training period in grey. Hover shows rank of
                           actual among the 100 samples for that week.

② Per-week strip chart     For each of the 26 test weeks, 100 jittered
                           dots show the sample distribution as a vertical
                           column. Samples below the actual are in the
                           district colour; samples above are grey. The
                           actual is a bold diamond. Directly answers
                           "does the actual appear within the samples?"

③ Calibration rank strip   Bar chart of the percentile rank of the actual
                           among the 100 samples per test week. Green =
                           inside 80% PI, orange = 80–95%, red = outside.

④ PIT histogram            Histogram of the 26 percentile ranks. Expected
                           shape for a well-calibrated model: flat/uniform.
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd

from evaluate import crps_ensemble

# ── Paths ──────────────────────────────────────────────────────────────────────

EVAL_CSV = os.path.join("..", "input",  "evaluation_data.csv")
PRED_CSV = os.path.join("..", "output", "predictions.csv")
OUT_DIR  = os.path.join("..", "output", "experiments")
OUT_HTML = os.path.join(OUT_DIR, "exp10_sample_cloud.html")

TRAIN_WEEKS = 78

DISTRICT_COLORS = {
    "Hamahamet-Mboinkou": "#04a5e5",
    "Hambou":             "#8839ef",
    "Mbadjini Est":       "#40a02b",
    "Mbadjini Ouest":     "#d20f39",
    "Mitsamiouli-Mboudé": "#fe640b",
    "Moroni-Bambao":      "#df8e1d",
    "Oichili-Dimani":     "#179299",
}


# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_data():
    df   = pd.read_csv(EVAL_CSV)
    pred = pd.read_csv(PRED_CSV)

    all_weeks  = sorted(df["time_period"].unique())
    train_set  = set(all_weeks[:TRAIN_WEEKS])
    test_set   = set(all_weeks[TRAIN_WEEKS:])
    week_index = {w: i + 1 for i, w in enumerate(all_weeks)}

    sample_cols = [c for c in pred.columns if c.startswith("sample_")]
    n_samples   = len(sample_cols)

    districts = {}

    for loc in sorted(df["location"].unique()):
        loc_full = df[df["location"] == loc].sort_values("time_period")
        loc_pred = pred[pred["location"] == loc].sort_values("time_period")

        train_rows = []
        test_rows  = []

        for _, row in loc_full.iterrows():
            tp      = row["time_period"]
            is_test = tp in test_set
            entry   = {
                "week":  week_index[tp],
                "label": tp,
                "actual": float(row["disease_cases"]),
            }
            if not is_test:
                train_rows.append(entry)
            else:
                pr = loc_pred[loc_pred["time_period"] == tp]
                if not pr.empty:
                    samp = pr[sample_cols].values.astype(float).flatten()
                    obs  = float(row["disease_cases"])
                    rank = float(np.mean(samp < obs))
                    entry.update({
                        "samples": [round(float(v), 2) for v in samp],
                        "median":  round(float(np.median(samp)), 2),
                        "q10":     round(float(np.quantile(samp, 0.10)), 2),
                        "q90":     round(float(np.quantile(samp, 0.90)), 2),
                        "q025":    round(float(np.quantile(samp, 0.025)), 2),
                        "q975":    round(float(np.quantile(samp, 0.975)), 2),
                        "rank":    round(rank, 4),
                        "crps":    round(float(crps_ensemble(
                                       np.array([obs]), samp[None, :])[0]), 3),
                    })
                    test_rows.append(entry)

        # District-level metrics
        samps_all  = loc_pred[sample_cols].values.astype(float)
        obs_all    = (loc_full[loc_full["time_period"].isin(test_set)]
                      .sort_values("time_period")["disease_cases"].values.astype(float))
        median_all = np.median(samps_all, axis=1)
        crps_dist  = float(np.mean(crps_ensemble(obs_all, samps_all)))
        rmse       = float(np.sqrt(np.mean((median_all - obs_all) ** 2)))
        cov80      = float(np.mean(
            (obs_all >= np.quantile(samps_all, 0.10, axis=1)) &
            (obs_all <= np.quantile(samps_all, 0.90, axis=1))) * 100)
        cov95      = float(np.mean(
            (obs_all >= np.quantile(samps_all, 0.025, axis=1)) &
            (obs_all <= np.quantile(samps_all, 0.975, axis=1))) * 100)

        ranks = [r["rank"] for r in test_rows]
        pct_inside_80 = round(float(np.mean(
            [(r >= 0.10) and (r <= 0.90) for r in ranks])) * 100, 1)

        districts[loc] = {
            "color":   DISTRICT_COLORS.get(loc, "#8839ef"),
            "metrics": {
                "crps":          round(crps_dist, 2),
                "rmse":          round(rmse, 2),
                "cov80":         round(cov80, 1),
                "cov95":         round(cov95, 1),
                "pct_inside_80": pct_inside_80,
                "n_samples":     n_samples,
            },
            "train_rows": train_rows,
            "test_rows":  test_rows,
        }

    return districts


# ── HTML generation ────────────────────────────────────────────────────────────

def build_html(districts: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    ranked       = sorted(districts.items(), key=lambda x: x[1]["metrics"]["crps"])
    overall_crps = round(np.mean([v["metrics"]["crps"]  for _, v in ranked]), 2)
    overall_cov  = round(np.mean([v["metrics"]["cov80"] for _, v in ranked]), 1)

    # Tabs HTML
    tabs_html = ""
    for i, (loc, v) in enumerate(districts.items()):
        tid    = loc.replace(" ", "_").replace("-", "_")
        rank   = next(r + 1 for r, (n, _) in enumerate(ranked) if n == loc)
        active = "active" if i == 0 else ""
        tabs_html += f"""
        <button class="tab-btn {active}" onclick="selectDistrict('{loc}')"
                id="tab_{tid}" style="--dc:{v['color']}">
          <span class="tab-rank">#{rank}</span>
          <span class="tab-name">{loc}</span>
          <span class="tab-crps">CRPS {v['metrics']['crps']:.1f} · {v['metrics']['pct_inside_80']:.0f}% inside 80%PI</span>
        </button>"""

    dist_data_js = json.dumps(districts, ensure_ascii=False, separators=(",", ":"))

    # ── Main HTML ──────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Exp 10 — Sample Cloud · Ensemble S+X</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
    :root{{
      --page:#f2eeff;--surface:#ebe4ff;--card:#fff;
      --border:#ddd5f5;--border-h:#c9bef0;
      --text:#2a2044;--sub1:#544873;--sub0:#7a6e92;--muted:#a099bb;--ghost:#c4bcd8;
      --green:#40a02b;--orange:#fe640b;--red:#d20f39;--sky:#04a5e5;--teal:#179299;--mauve:#8839ef;
    }}
    body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--page);
          color:var(--text);padding:24px 20px 56px;line-height:1.55;min-width:900px}}
    .hdr{{text-align:center;padding:32px 32px 24px;background:linear-gradient(160deg,#fff,#f3eeff);
          border-radius:20px;border:1px solid var(--border);box-shadow:0 2px 20px #8839ef0d;margin-bottom:24px}}
    .hdr h1{{font-size:1.7rem;font-weight:800;letter-spacing:-.5px;
             background:linear-gradient(120deg,var(--mauve),var(--sky),var(--teal));
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin-bottom:6px}}
    .hdr .sub{{color:var(--sub0);font-size:.84rem;max-width:680px;margin:0 auto;line-height:1.65}}
    .kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
    .kpi{{background:var(--card);border:1px solid var(--border);border-radius:14px;
          padding:18px 20px 14px;box-shadow:0 1px 6px #0001;position:relative;overflow:hidden}}
    .kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
                border-radius:14px 14px 0 0;background:var(--kg,var(--sky))}}
    .kpi-lbl{{font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;font-weight:700}}
    .kpi-val{{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;
              letter-spacing:-1.5px;line-height:1;margin-top:5px;color:var(--kc,var(--text))}}
    .kpi-sub{{font-size:.72rem;color:var(--muted);margin-top:5px}}
    .sec{{background:var(--card);border:1px solid var(--border);border-radius:16px;
          padding:22px 24px;box-shadow:0 1px 6px #0001;margin-bottom:16px}}
    .sec-title{{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:1.1px;
                color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:8px}}
    .dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
    .tabs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}}
    .tab-btn{{display:flex;flex-direction:column;align-items:flex-start;gap:2px;padding:10px 16px;
              background:var(--card);border:1px solid var(--border);border-radius:12px;cursor:pointer;
              border-top:3px solid var(--dc,#888);transition:all .18s;min-width:138px}}
    .tab-btn:hover{{border-color:var(--border-h);box-shadow:0 3px 12px #8839ef12}}
    .tab-btn.active{{background:var(--surface);border-color:var(--dc,#888);box-shadow:0 3px 16px #8839ef18}}
    .tab-rank{{font-size:.59rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px}}
    .tab-name{{font-size:.82rem;font-weight:700;color:var(--text)}}
    .tab-crps{{font-size:.68rem;font-family:'JetBrains Mono',monospace;color:var(--sub0)}}
    .dist-panel{{display:none}}.dist-panel.active{{display:block}}
    .badge-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}}
    .badge{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
            padding:8px 16px;text-align:center}}
    .badge-lbl{{font-size:.6rem;color:var(--muted);text-transform:uppercase;
                letter-spacing:1px;font-weight:700}}
    .badge-val{{font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                font-weight:700;line-height:1.2;color:var(--text)}}
    .badge-unit{{font-size:.7rem;color:var(--muted)}}
    .chart-label{{font-size:.62rem;color:var(--muted);text-align:right;
                  margin-top:4px;margin-bottom:16px;letter-spacing:.3px}}
    .bottom-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}
    .legend-row{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:10px;
                 font-size:.72rem;color:var(--sub0);align-items:center}}
    .leg-item{{display:flex;align-items:center;gap:5px}}
    .insight-box{{background:linear-gradient(135deg,#f8f4ff,#f0eaff);
                  border:1px solid var(--border);border-radius:12px;
                  padding:14px 18px;margin-top:18px;font-size:.84rem;
                  color:var(--sub1);line-height:1.65}}
    .insight-box strong{{color:var(--text)}}
    .footer{{text-align:center;padding:24px 0 4px;color:var(--ghost);font-size:.7rem}}
  </style>
</head>
<body>

<div class="hdr">
  <h1>Exp 10 — Sample Cloud Visualization</h1>
  <p class="sub">
    Every one of the 100 probabilistic samples drawn by Ensemble S+X is shown as an individual
    trajectory. The actual case count is plotted on top. The insight: the actual is almost never
    the median — but it should appear <em>inside</em> the cloud most of the time.
  </p>
</div>

<div class="kpi-row">
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--sky),var(--teal));--kc:var(--sky)">
    <div class="kpi-lbl">Mean CRPS</div>
    <div class="kpi-val">{overall_crps:.2f}</div>
    <div class="kpi-sub">across 7 districts</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--green),var(--teal));--kc:var(--green)">
    <div class="kpi-lbl">Mean 80% Coverage</div>
    <div class="kpi-val">{overall_cov:.1f}%</div>
    <div class="kpi-sub">actual inside 80% PI</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--mauve),var(--sky));--kc:var(--mauve)">
    <div class="kpi-lbl">Samples per week</div>
    <div class="kpi-val">100</div>
    <div class="kpi-sub">50 SARIMAX + 50 XGBoost</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--orange),var(--red));--kc:var(--orange)">
    <div class="kpi-lbl">Test horizon</div>
    <div class="kpi-val">26 wk</div>
    <div class="kpi-sub">weeks 79 – 104</div>
  </div>
</div>

<div class="sec">
  <div class="sec-title">
    <span class="dot" style="background:var(--orange)"></span>
    District Deep-Dive — click a district to explore
  </div>
  <div class="tabs" id="district-tabs">{tabs_html}</div>
  <div id="district-panels"></div>
</div>

<div class="footer">Exp 10 · Ensemble S+X champion · 7 districts · Comoros Climate-Health Modeling</div>

<script>
const DISTRICTS = {dist_data_js};

// Chart.js globals
Chart.defaults.color = '#7a6e92';
Chart.defaults.borderColor = '#e8e2f5';
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
Chart.defaults.plugins.legend.labels.padding = 12;
Chart.defaults.plugins.tooltip.backgroundColor = '#fff';
Chart.defaults.plugins.tooltip.borderColor = '#ddd5f5';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.titleColor = '#2a2044';
Chart.defaults.plugins.tooltip.bodyColor  = '#544873';
Chart.defaults.plugins.tooltip.titleFont  = {{family:"'Plus Jakarta Sans'",weight:'700'}};
Chart.defaults.plugins.tooltip.bodyFont   = {{family:"'JetBrains Mono'",size:12}};
Chart.defaults.plugins.tooltip.padding    = 10;
const GC = '#ece7fa';

function hex2rgb(hex) {{
  const h = hex.replace('#','');
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];
}}
function rgba(hex,a) {{
  const [r,g,b] = hex2rgb(hex);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

// ── Panel HTML builder (structure only — charts built lazily on first show) ──
const chartRegistry = {{}};
const builtPanels   = new Set();

function buildDistrictPanel(loc) {{
  const d   = DISTRICTS[loc];
  const tid = loc.replace(/ /g,'_').replace(/-/g,'_');
  const c   = d.color;
  const m   = d.metrics;
  const crps_clr = m.crps <= 28 ? '#40a02b' : m.crps <= 40 ? '#fe640b' : '#d20f39';
  const cov_clr  = m.cov80 >= 75 ? '#40a02b' : m.cov80 >= 60 ? '#fe640b' : '#d20f39';
  const ins_clr  = m.pct_inside_80 >= 75 ? '#40a02b' : m.pct_inside_80 >= 60 ? '#fe640b' : '#d20f39';

  const panelHTML = `
<div class="badge-row">
  <div class="badge" style="border-top:3px solid ${{crps_clr}}">
    <div class="badge-lbl">CRPS</div>
    <div class="badge-val" style="color:${{crps_clr}}">${{m.crps}}</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">RMSE</div>
    <div class="badge-val">${{m.rmse}}</div><div class="badge-unit">cases/wk</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{cov_clr}}">
    <div class="badge-lbl">80% PI Coverage</div>
    <div class="badge-val" style="color:${{cov_clr}}">${{m.cov80}}%</div>
    <div class="badge-unit">target ≥ 80%</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">95% PI Coverage</div>
    <div class="badge-val">${{m.cov95}}%</div><div class="badge-unit">target ≥ 95%</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{ins_clr}}">
    <div class="badge-lbl">Actual inside 80%PI</div>
    <div class="badge-val" style="color:${{ins_clr}}">${{m.pct_inside_80}}%</div>
    <div class="badge-unit">of 26 test weeks</div>
  </div>
</div>

<div class="sec-title" style="margin-bottom:8px">
  <span class="dot" style="background:${{c}}"></span>
  ① Sample-Cloud Timeline — 100 trajectories + actual + median
</div>
<div class="legend-row">
  <span class="leg-item">
    <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="#b0a8c8" stroke-width="1.5" stroke-dasharray="3,2"/></svg>
    <span>Training actuals</span></span>
  <span class="leg-item">
    <svg width="24" height="8"><rect x="0" y="1" width="24" height="6" fill="${{c}}22" rx="2"/><line x1="0" y1="4" x2="24" y2="4" stroke="${{c}}88" stroke-width="1"/></svg>
    <span>100 sample paths</span></span>
  <span class="leg-item">
    <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="${{c}}" stroke-width="3"/></svg>
    <span>Actual cases</span></span>
  <span class="leg-item">
    <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4" stroke="${{c}}" stroke-width="1.5" stroke-dasharray="5,3"/></svg>
    <span>Predicted median</span></span>
</div>
<div style="position:relative;height:300px"><canvas id="cloud_${{tid}}"></canvas></div>
<div class="chart-label">Shaded = training period (wk 1–78) &nbsp;·&nbsp; Hover any test week for details &nbsp;·&nbsp; Dashed line = train/test split</div>

<div class="sec-title" style="margin-bottom:8px;margin-top:4px">
  <span class="dot" style="background:${{c}}"></span>
  ② Per-Week Sample Distribution — each dot is one sample
</div>
<div class="legend-row">
  <span class="leg-item">
    <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${{c}}bb"/></svg>
    <span>Sample below actual</span></span>
  <span class="leg-item">
    <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="#a099bb55"/></svg>
    <span>Sample above actual</span></span>
  <span class="leg-item">
    <svg width="22" height="10"><polygon points="11,1 21,9 11,17 1,9" fill="#d20f39" transform="scale(1,0.55)"/></svg>
    <span>Actual (diamond)</span></span>
</div>
<div style="position:relative;height:270px"><canvas id="strip_${{tid}}"></canvas></div>
<div class="chart-label">26 test weeks on x-axis &nbsp;·&nbsp; 100 jittered dots per column &nbsp;·&nbsp; Hover a dot to inspect</div>

<div class="bottom-row">
  <div>
    <div class="sec-title" style="margin-bottom:8px">
      <span class="dot" style="background:var(--teal)"></span>
      ③ Actual Rank Among 100 Samples — per test week
    </div>
    <div style="position:relative;height:210px"><canvas id="rank_${{tid}}"></canvas></div>
    <div class="chart-label">Green = inside 80%PI &nbsp;·&nbsp; Orange = 80–95% &nbsp;·&nbsp; Red = outside 95%PI</div>
  </div>
  <div>
    <div class="sec-title" style="margin-bottom:8px">
      <span class="dot" style="background:var(--mauve)"></span>
      ④ PIT Histogram — uniform = well-calibrated
    </div>
    <div style="position:relative;height:210px"><canvas id="pit_${{tid}}"></canvas></div>
    <div class="chart-label">Dashed line = expected count if uniform &nbsp;·&nbsp; 26 test weeks, 10 bins</div>
  </div>
</div>

<div class="insight-box" id="insight_${{tid}}"></div>`;

  document.getElementById('district-panels').innerHTML +=
    `<div class="dist-panel" id="panel_${{tid}}">${{panelHTML}}</div>`;

  chartRegistry[loc] = () => buildCharts(loc, tid, c, d.train_rows, d.test_rows, m);
}}

// ── Chart builder ─────────────────────────────────────────────────────────────
function buildCharts(loc, tid, color, trainRows, testRows, m) {{
  if (chartRegistry[loc + '_done']) return;
  chartRegistry[loc + '_done'] = true;

  const N_TRAIN   = trainRows.length;  // 78
  const N_SAMPLES = testRows[0].samples.length;  // 100
  const [cr, cg, cb] = hex2rgb(color);

  // Labels and series for Chart ①
  const allLabels  = [
    ...trainRows.map(w => w.label.replace('20','').replace('-W','W')),
    ...testRows .map(w => w.label.replace('20','').replace('-W','W')),
  ];
  const trainActual = [...trainRows.map(w => w.actual), ...new Array(testRows.length).fill(null)];
  const testActual  = [...new Array(N_TRAIN).fill(null), ...testRows.map(w => w.actual)];
  const medianArr   = [...new Array(N_TRAIN).fill(null), ...testRows.map(w => w.median)];
  const q025arr     = [...new Array(N_TRAIN).fill(null), ...testRows.map(w => w.q025)];
  const q975arr     = [...new Array(N_TRAIN).fill(null), ...testRows.map(w => w.q975)];
  const q10arr      = [...new Array(N_TRAIN).fill(null), ...testRows.map(w => w.q10)];
  const q90arr      = [...new Array(N_TRAIN).fill(null), ...testRows.map(w => w.q90)];

  // ── Plugin: shade training region ────────────────────────────────────────
  const splitBgPlugin = {{
    id: 'splitBg',
    beforeDraw(chart) {{
      const {{ctx, chartArea, scales}} = chart;
      if (!chartArea) return;
      const x1 = scales.x.getPixelForValue(N_TRAIN - 0.5);
      ctx.save();
      ctx.fillStyle = 'rgba(248,245,255,0.40)';
      ctx.fillRect(chartArea.left, chartArea.top,
                   x1 - chartArea.left, chartArea.bottom - chartArea.top);
      ctx.strokeStyle = 'rgba(160,153,187,0.55)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x1, chartArea.top);
      ctx.lineTo(x1, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }}
  }};

  // ── Plugin: draw 100 sample lines via raw Canvas2D paths ─────────────────
  // This is far faster than 100 separate Chart.js datasets and avoids
  // any array-spread or dataset-count limitations.
  const sampleLinesPlugin = {{
    id: 'sampleLines',
    afterDatasetsDraw(chart) {{
      const {{ctx, chartArea, scales}} = chart;
      if (!chartArea) return;
      ctx.save();
      // Clip to the chart area so samples don't bleed into axes
      ctx.beginPath();
      ctx.rect(chartArea.left, chartArea.top,
               chartArea.right  - chartArea.left,
               chartArea.bottom - chartArea.top);
      ctx.clip();

      // Two-pass rendering:
      // Pass 1: soft glow (wider, very transparent) for depth
      // Pass 2: crisp thin lines for individual trajectory visibility
      const passes = [
        {{ alpha: 0.035, lineWidth: 3.5 }},
        {{ alpha: 0.11,  lineWidth: 0.85 }},
      ];
      for (const pass of passes) {{
        ctx.strokeStyle = `rgba(${{cr}},${{cg}},${{cb}},${{pass.alpha}})`;
        ctx.lineWidth = pass.lineWidth;
        for (let s = 0; s < N_SAMPLES; s++) {{
          ctx.beginPath();
          let firstPt = true;
          for (let t = 0; t < testRows.length; t++) {{
            const px = scales.x.getPixelForValue(N_TRAIN + t);
            const py = scales.y.getPixelForValue(testRows[t].samples[s]);
            if (firstPt) {{ ctx.moveTo(px, py); firstPt = false; }}
            else         {{ ctx.lineTo(px, py); }}
          }}
          ctx.stroke();
        }}
      }}
      ctx.restore();
    }}
  }};

  // ── Chart ①: sample cloud timeline ──────────────────────────────────────
  new Chart(document.getElementById('cloud_' + tid), {{
    type: 'line',
    plugins: [splitBgPlugin, sampleLinesPlugin],
    data: {{
      labels: allLabels,
      datasets: [
        // 95% PI — fill from upper to lower band
        {{ label:'95% PI', data:q975arr, borderWidth:0, pointRadius:0,
           fill:'+1', backgroundColor:rgba(color,0.07), borderColor:'transparent', order:8 }},
        {{ label:'',       data:q025arr, borderWidth:0, pointRadius:0,
           fill:false,     borderColor:'transparent', order:8 }},
        // 80% PI
        {{ label:'80% PI', data:q90arr, borderWidth:0, pointRadius:0,
           fill:'+1', backgroundColor:rgba(color,0.13), borderColor:'transparent', order:7 }},
        {{ label:'',       data:q10arr, borderWidth:0, pointRadius:0,
           fill:false,     borderColor:'transparent', order:7 }},
        // Predicted median — drawn on top of the sample cloud
        {{ label:'Predicted median', data:medianArr,
           borderColor:rgba(color,0.72), borderWidth:1.8, borderDash:[6,3],
           pointRadius:0, fill:false, order:2 }},
        // Actual training period — grey dashed
        {{ label:'Actual (training)', data:trainActual,
           borderColor:'#a099bb', borderWidth:1.5, pointRadius:0,
           fill:false, borderDash:[3,2], order:1 }},
        // Actual test period — bold, always on top
        {{ label:'Actual (test)', data:testActual,
           borderColor:color, borderWidth:2.8,
           pointRadius:4, pointBackgroundColor:'#fff',
           pointBorderColor:color, pointBorderWidth:2,
           fill:false, order:0 }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      interaction:{{ mode:'index', intersect:false }},
      plugins:{{
        legend:{{
          position:'top',
          labels:{{
            filter: item => ['Actual (test)','Actual (training)',
                             'Predicted median','80% PI','95% PI'].includes(item.text),
            font:{{ size:11 }}
          }}
        }},
        tooltip:{{
          callbacks:{{
            title: ctx => ctx[0].label,
            label: ctx => {{
              const lbl = ctx.dataset.label;
              if (!lbl || lbl==='' || ctx.raw==null) return null;
              if (['80% PI','95% PI'].includes(lbl)) return null;
              return ' ' + lbl + ': ' + ctx.raw.toFixed(0);
            }},
            afterBody: ctx => {{
              const label = ctx[0].label;
              const tw = testRows.find(w =>
                w.label.replace('20','').replace('-W','W') === label);
              if (!tw) return [];
              const pct    = (tw.rank * 100).toFixed(1);
              const nBelow = tw.samples.filter(s => s < tw.actual).length;
              return [
                '',
                ' Actual rank: ' + pct + 'th pctile among 100 samples',
                ' (' + nBelow + ' samples below · ' + (100-nBelow) + ' above)',
                ' CRPS this week: ' + tw.crps,
              ];
            }}
          }}
        }}
      }},
      scales:{{
        x:{{ grid:{{display:false}},
              ticks:{{maxTicksLimit:14, font:{{size:9}}, maxRotation:30}} }},
        y:{{ beginAtZero:true, grid:{{color:GC}},
              title:{{display:true, text:'Cases / week', font:{{size:10}}}} }}
      }}
    }}
  }});

  // ── Chart ②: per-week sample strip ───────────────────────────────────────
  // Scatter with deterministic jitter per (week, sample) pair
  const belowPts  = [];
  const abovePts  = [];
  const actualPts = [];

  testRows.forEach((w, wi) => {{
    const actual = w.actual;
    w.samples.forEach((s, si) => {{
      // Deterministic jitter so chart looks stable across redraws
      const seed   = (wi * 137 + si * 31) % 1000;
      const jitter = (seed / 1000 - 0.5) * 0.52;
      const pt     = {{ x:wi + jitter, y:s, week:w.label, actual, si }};
      (s <= actual ? belowPts : abovePts).push(pt);
    }});
    actualPts.push({{ x:wi, y:actual, week:w.label }});
  }});

  new Chart(document.getElementById('strip_' + tid), {{
    type:'scatter',
    data:{{
      datasets:[
        {{
          label:'Samples ≤ actual',
          data:belowPts,
          backgroundColor:`rgba(${{cr}},${{cg}},${{cb}},0.55)`,
          borderColor:'transparent',
          pointRadius:3, pointHoverRadius:5, order:2,
        }},
        {{
          label:'Samples > actual',
          data:abovePts,
          backgroundColor:'rgba(160,153,187,0.32)',
          borderColor:'transparent',
          pointRadius:3, pointHoverRadius:5, order:2,
        }},
        {{
          label:'Actual cases',
          data:actualPts,
          backgroundColor:'#d20f39',
          borderColor:'#fff',
          borderWidth:1.5,
          pointRadius:8,
          pointStyle:'rectRot',
          pointHoverRadius:10,
          order:0,
        }},
      ]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      interaction:{{ mode:'nearest', intersect:true }},
      plugins:{{
        legend:{{ position:'top', labels:{{ font:{{size:10}} }} }},
        tooltip:{{
          callbacks:{{
            label: ctx => {{
              const d = ctx.raw;
              if (d.si !== undefined) {{
                const tw  = testRows[Math.round(ctx.raw.x)];
                const nb  = tw ? tw.samples.filter(s => s <= d.actual).length : '?';
                return ' ' + d.week + '  sample=' + d.y.toFixed(0) +
                       '  actual=' + d.actual.toFixed(0) +
                       '  (' + nb + '/100 below)';
              }}
              return ' ' + d.week + '  actual=' + d.y.toFixed(0);
            }}
          }}
        }}
      }},
      scales:{{
        x:{{
          min:-0.7, max:testRows.length - 0.3,
          grid:{{display:false}},
          ticks:{{
            stepSize:1, font:{{size:8}}, maxRotation:45,
            callback(v) {{
              const idx = Math.round(v);
              if (idx>=0 && idx<testRows.length && Math.abs(v-idx)<0.1)
                return testRows[idx].label.replace('20','').replace('-W','W');
              return '';
            }}
          }}
        }},
        y:{{ beginAtZero:true, grid:{{color:GC}},
              title:{{display:true, text:'Cases / week', font:{{size:10}}}} }}
      }}
    }}
  }});

  // ── Chart ③: calibration rank ────────────────────────────────────────────
  const ranks   = testRows.map(w => +(w.rank * 100).toFixed(1));
  const rLabels = testRows.map(w => w.label.replace('20','').replace('-W','W'));
  const rColors = ranks.map(r =>
    r>=10&&r<=90 ? rgba('#40a02b',0.75) : r>=2.5&&r<=97.5 ? rgba('#fe640b',0.75) : rgba('#d20f39',0.75));
  const rBorder = ranks.map(r =>
    r>=10&&r<=90 ? '#40a02b' : r>=2.5&&r<=97.5 ? '#fe640b' : '#d20f39');

  const rankBandsPlugin = {{
    id:'rankBands',
    beforeDraw(chart) {{
      const {{ctx, chartArea, scales}} = chart;
      if (!chartArea) return;
      const y = scales.y;
      ctx.save();
      const fill = (yHi, yLo, clr) => {{
        const py1 = y.getPixelForValue(yHi);
        const py2 = y.getPixelForValue(yLo);
        ctx.fillStyle = clr;
        ctx.fillRect(chartArea.left, py1, chartArea.right-chartArea.left, py2-py1);
      }};
      fill(90, 10,  'rgba(64,160,43,0.10)');   // 80% PI band
      fill(97.5, 90,'rgba(254,100,11,0.07)');   // 80–95% bands
      fill(10, 2.5, 'rgba(254,100,11,0.07)');
      // ideal 50% line
      const py50 = y.getPixelForValue(50);
      ctx.strokeStyle='rgba(160,153,187,0.55)';
      ctx.lineWidth=1; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.moveTo(chartArea.left,py50); ctx.lineTo(chartArea.right,py50); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }}
  }};

  new Chart(document.getElementById('rank_' + tid), {{
    type:'bar', plugins:[rankBandsPlugin],
    data:{{
      labels:rLabels,
      datasets:[{{ label:'Actual rank',
        data:ranks, backgroundColor:rColors, borderColor:rBorder,
        borderWidth:1, borderRadius:3 }}]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{
          label: ctx => {{
            const r = ctx.raw;
            const z = r>=10&&r<=90 ? 'inside 80% PI' : r>=2.5&&r<=97.5 ? 'inside 95% PI' : 'OUTSIDE 95% PI';
            return ' Rank: ' + r.toFixed(1) + 'th pctile  →  ' + z;
          }}
        }}}}
      }},
      scales:{{
        x:{{ grid:{{display:false}}, ticks:{{maxTicksLimit:10, font:{{size:8}}, maxRotation:45}} }},
        y:{{
          min:0, max:100, grid:{{color:GC}},
          title:{{display:true, text:'Actual rank among 100 samples', font:{{size:10}}}},
          ticks:{{ callback: v => v+'th', values:[0,10,25,50,75,90,100] }}
        }}
      }}
    }}
  }});

  // ── Chart ④: PIT histogram ────────────────────────────────────────────────
  const nBins  = 10;
  const bins   = new Array(nBins).fill(0);
  ranks.forEach(r => {{
    const i = Math.min(Math.floor(r / (100/nBins)), nBins-1);
    bins[i]++;
  }});
  const expected = ranks.length / nBins;
  const pitBinLabels = ['0–10','10–20','20–30','30–40','40–50',
                        '50–60','60–70','70–80','80–90','90–100'];

  const expectedLinePlugin = {{
    id:'expLine',
    afterDraw(chart) {{
      const {{ctx, chartArea, scales}} = chart;
      if (!chartArea) return;
      const py = scales.y.getPixelForValue(expected);
      ctx.save();
      ctx.strokeStyle='rgba(160,153,187,0.75)';
      ctx.lineWidth=1.5; ctx.setLineDash([5,4]);
      ctx.beginPath(); ctx.moveTo(chartArea.left,py); ctx.lineTo(chartArea.right,py); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#a099bb'; ctx.font='10px JetBrains Mono, monospace';
      ctx.textAlign='right';
      ctx.fillText('expected (' + expected.toFixed(1) + ')', chartArea.right-4, py-4);
      ctx.restore();
    }}
  }};

  new Chart(document.getElementById('pit_' + tid), {{
    type:'bar', plugins:[expectedLinePlugin],
    data:{{
      labels: pitBinLabels,
      datasets:[{{
        label:'Test weeks',
        data: bins,
        backgroundColor: bins.map(b =>
          Math.abs(b-expected) <= 1 ? rgba(color,0.70) :
          Math.abs(b-expected) <= 2 ? rgba('#fe640b',0.65) : rgba('#d20f39',0.65)),
        borderColor: color, borderWidth:1, borderRadius:3,
      }}]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{
          label: ctx => ' ' + ctx.raw + ' weeks (expected ' + expected.toFixed(1) + ')'
        }}}}
      }},
      scales:{{
        x:{{ grid:{{display:false}}, ticks:{{font:{{size:8}}}} }},
        y:{{ beginAtZero:true, grid:{{color:GC}},
              suggestedMax: Math.max(...bins)+2,
              title:{{display:true, text:'# of test weeks', font:{{size:10}}}} }}
      }}
    }}
  }});

  // ── Insight text ─────────────────────────────────────────────────────────
  const inside80   = ranks.filter(r => r>=10 && r<=90).length;
  const inside95   = ranks.filter(r => r>=2.5 && r<=97.5).length;
  const outside95  = ranks.filter(r => r<2.5 || r>97.5).length;
  const aboveMed   = ranks.filter(r => r>50).length;
  const meanRank   = (ranks.reduce((a,b)=>a+b,0)/ranks.length).toFixed(1);
  const pitUniform = Math.max(...bins) <= Math.ceil(expected)+2;

  let txt = '<strong>Actual rank analysis (26 test weeks):</strong> ';
  txt += 'The actual fell inside the <strong>80% PI in ' + inside80 + '/26 weeks (' +
         (inside80/26*100).toFixed(1) + '%)</strong> and inside the <strong>95% PI in ' +
         inside95 + '/26 weeks (' + (inside95/26*100).toFixed(1) + '%)</strong>. ';
  if (outside95>0) txt += '<strong>' + outside95 + ' week(s)</strong> had the actual entirely outside the 95% PI — genuine model surprises. ';
  txt += 'The actual was above the median in <strong>' + aboveMed + '/26</strong> weeks and below in <strong>' +
         (26-aboveMed) + '/26</strong> (mean rank: <strong>' + meanRank + 'th pctile</strong> — ';
  if      (parseFloat(meanRank)>58) txt += 'model tends to <em>underforecast</em>). ';
  else if (parseFloat(meanRank)<42) txt += 'model tends to <em>overforecast</em>). ';
  else                              txt += 'well-centred, no systematic bias). ';
  txt += 'PIT histogram: <strong>' + (pitUniform ? 'approximately uniform ✓' : 'non-uniform — some calibration gap') + '</strong>.';
  document.getElementById('insight_' + tid).innerHTML = txt;
}}

// ── Tab logic ─────────────────────────────────────────────────────────────────
function selectDistrict(loc) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.dist-panel').forEach(p => p.classList.remove('active'));
  const tid = loc.replace(/ /g,'_').replace(/-/g,'_');
  document.getElementById('tab_' + tid).classList.add('active');
  document.getElementById('panel_' + tid).classList.add('active');
  if (!builtPanels.has(loc)) {{
    builtPanels.add(loc);
    if (chartRegistry[loc]) chartRegistry[loc]();
  }}
}}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
Object.keys(DISTRICTS).forEach(loc => buildDistrictPanel(loc));
selectDistrict(Object.keys(DISTRICTS)[0]);
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def run():
    print("Exp 10 — Sample Cloud Visualization")
    print("Loading data ...")
    districts = prepare_data()

    print("\nDistrict summary:")
    for loc, d in sorted(districts.items(), key=lambda x: x[1]["metrics"]["crps"]):
        m  = d["metrics"]
        tr = d["test_rows"]
        inside80 = sum(1 for w in tr if w["rank"] >= 0.10 and w["rank"] <= 0.90)
        print(f"  {loc:<25s}  CRPS={m['crps']:.2f}  80%cov={m['cov80']:.1f}%  "
              f"inside80PI={inside80}/26 ({inside80/26*100:.0f}%)")

    print("\nBuilding HTML ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html(districts)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved -> {OUT_HTML}")


if __name__ == "__main__":
    run()
