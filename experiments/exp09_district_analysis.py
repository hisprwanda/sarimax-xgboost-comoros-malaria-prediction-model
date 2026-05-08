"""
Exp 09 — Per-District Forecast Analysis

Interactive HTML dashboard showing how the Ensemble S+X champion model
performs for each of the 7 Ngadjizi districts individually.

For every district the dashboard shows:
  ① Full 104-week forecast strip  — training actuals + test actuals vs
    predicted median, 80% PI, and 95% PI
  ② Climate context strip         — rainfall (bars) + temperature (line)
    on the same x-axis
  ③ Actual vs predicted scatter   — each test week as a dot, sized by
    error, coloured light-to-dark by forecast week
  ④ Per-week CRPS heat strip      — shows which weeks drove the score

Plus an overview section:
  • Ranked CRPS bar chart (which district is hardest?)
  • Multi-metric radar across all districts
  • Summary metrics table
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

EVAL_CSV  = os.path.join("..", "input",  "evaluation_data.csv")
PRED_CSV  = os.path.join("..", "output", "predictions.csv")
OUT_DIR   = os.path.join("..", "output", "experiments")
OUT_HTML  = os.path.join(OUT_DIR, "exp09_district_analysis.html")

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

    all_weeks   = sorted(df["time_period"].unique())
    train_set   = set(all_weeks[:TRAIN_WEEKS])
    test_set    = set(all_weeks[TRAIN_WEEKS:])
    week_index  = {w: i+1 for i, w in enumerate(all_weeks)}   # 1-based

    sample_cols = [c for c in pred.columns if c.startswith("sample_")]

    districts = {}

    for loc in sorted(df["location"].unique()):
        loc_full  = df[df["location"] == loc].sort_values("time_period")
        loc_pred  = pred[pred["location"] == loc].sort_values("time_period")

        weekly = []
        for _, row in loc_full.iterrows():
            tp    = row["time_period"]
            wnum  = week_index[tp]
            is_test = tp in test_set
            entry = {
                "week":     wnum,
                "label":    tp,
                "isTrain":  not is_test,
                "actual":   float(row["disease_cases"]),
                "rain":     round(float(row["rainfall"]),        2),
                "temp":     round(float(row["mean_temperature"]),2),
                "humidity": round(float(row["humidity"]),        2),
                "median":   None, "q10": None, "q90": None,
                "q025":     None, "q975": None, "crps": None,
            }
            if is_test:
                pr = loc_pred[loc_pred["time_period"] == tp]
                if not pr.empty:
                    samp  = pr[sample_cols].values.astype(float).flatten()
                    obs   = float(row["disease_cases"])
                    entry["median"] = round(float(np.median(samp)), 2)
                    entry["q10"]    = round(float(np.quantile(samp, 0.10)), 2)
                    entry["q90"]    = round(float(np.quantile(samp, 0.90)), 2)
                    entry["q025"]   = round(float(np.quantile(samp, 0.025)), 2)
                    entry["q975"]   = round(float(np.quantile(samp, 0.975)), 2)
                    entry["crps"]   = round(float(crps_ensemble(
                        np.array([obs]), samp[None, :])[0]), 3)
            weekly.append(entry)

        # --- district-level metrics over test period ---
        test_rows = loc_pred
        samps_all = test_rows[sample_cols].values.astype(float)
        obs_all   = (loc_full[loc_full["time_period"].isin(test_set)]
                     .sort_values("time_period")["disease_cases"].values.astype(float))

        median_all = np.median(samps_all, axis=1)
        rmse  = float(np.sqrt(np.mean((median_all - obs_all)**2)))
        mae   = float(np.mean(np.abs(median_all - obs_all)))
        nonzero = obs_all > 0
        mape  = float(np.mean(np.abs((median_all[nonzero] - obs_all[nonzero]) / obs_all[nonzero])) * 100) if nonzero.any() else 0.0
        r2    = float(1 - np.sum((obs_all - median_all)**2) / np.sum((obs_all - obs_all.mean())**2))
        crps_dist = float(np.mean(crps_ensemble(obs_all, samps_all)))
        cov80 = float(np.mean(
            (obs_all >= np.quantile(samps_all, 0.10, axis=1)) &
            (obs_all <= np.quantile(samps_all, 0.90, axis=1))
        ) * 100)
        cov95 = float(np.mean(
            (obs_all >= np.quantile(samps_all, 0.025, axis=1)) &
            (obs_all <= np.quantile(samps_all, 0.975, axis=1))
        ) * 100)

        train_weeks_data = [w for w in weekly if w["isTrain"]]
        test_weeks_data  = [w for w in weekly if not w["isTrain"]]
        train_mean = np.mean([w["actual"] for w in train_weeks_data])
        test_mean  = np.mean([w["actual"] for w in test_weeks_data])
        max_actual = max(w["actual"] for w in weekly)
        insight    = _make_insight(crps_dist, cov80, train_mean, test_mean, max_actual)

        districts[loc] = {
            "color":   DISTRICT_COLORS.get(loc, "#8839ef"),
            "metrics": {
                "crps":  round(crps_dist, 2),
                "rmse":  round(rmse,      2),
                "mae":   round(mae,       2),
                "mape":  round(mape,      1),
                "r2":    round(r2,        3),
                "cov80": round(cov80,     1),
                "cov95": round(cov95,     1),
            },
            "insight": insight,
            "weekly": weekly,
        }

    return districts


def _make_insight(crps, cov80, train_mean, test_mean, max_actual):
    crps_note = (
        f"<strong>Good calibration</strong> (CRPS {crps:.2f}) — model tracks seasonal patterns well."
        if crps <= 28 else
        f"<strong>Moderate calibration</strong> (CRPS {crps:.2f}) — captures trend but struggles with peak magnitude."
        if crps <= 45 else
        f"<strong>Challenging district</strong> (CRPS {crps:.2f}) — high case volume and non-linear dynamics make this district hard to forecast."
    )
    cov_note = (
        f"80% PI coverage of <strong>{cov80:.1f}%</strong> is near the ideal 80%, suggesting well-calibrated uncertainty."
        if cov80 >= 70 else
        f"80% PI coverage of <strong>{cov80:.1f}%</strong> is below the 80% target — prediction intervals are too narrow for this district."
    )
    drift_pct = abs(test_mean - train_mean) / train_mean * 100 if train_mean > 0 else 0
    drift_note = (
        f"Test-period mean (<strong>{test_mean:.0f}</strong>) differs from training mean (<strong>{train_mean:.0f}</strong>) "
        f"by &gt;20% — possible seasonal shift or trend."
        if drift_pct > 20 else
        f"Test-period mean (<strong>{test_mean:.0f}</strong>) is close to training mean (<strong>{train_mean:.0f}</strong>), "
        f"suggesting a stable seasonal pattern."
    )
    return f"{crps_note} {cov_note} {drift_note} Peak burden: <strong>{max_actual:.0f} cases/week</strong>."


# ── HTML generation ────────────────────────────────────────────────────────────

def build_html(districts: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Sorted by CRPS for ranking
    ranked = sorted(districts.items(), key=lambda x: x[1]["metrics"]["crps"])

    dist_names      = [d for d, _ in ranked]
    dist_colors     = [v["color"] for _, v in ranked]
    dist_crps       = [v["metrics"]["crps"]  for _, v in ranked]
    dist_rmse       = [v["metrics"]["rmse"]  for _, v in ranked]
    dist_cov80      = [v["metrics"]["cov80"] for _, v in ranked]
    dist_cov95      = [v["metrics"]["cov95"] for _, v in ranked]
    dist_r2         = [v["metrics"]["r2"]    for _, v in ranked]

    overall_crps = round(np.mean(dist_crps), 2)
    overall_rmse = round(np.mean(dist_rmse), 2)
    best_dist    = ranked[0][0]

    # Embed all district data as JSON
    dist_data_js = json.dumps(districts, ensure_ascii=False, separators=(",", ":"))

    # Tab list HTML
    tabs_html = ""
    panels_html = ""
    for i, (loc, v) in enumerate(districts.items()):
        tid  = loc.replace(" ", "_").replace("-", "_")
        rank = next(r+1 for r, (n, _) in enumerate(ranked) if n == loc)
        crps = v["metrics"]["crps"]
        is_active = "active" if i == 0 else ""
        tabs_html += f"""
        <button class="tab-btn {is_active}" onclick="selectDistrict('{loc}')" id="tab_{tid}"
                style="--dc:{v['color']}">
          <span class="tab-rank">#{rank}</span>
          <span class="tab-name">{loc}</span>
          <span class="tab-crps">CRPS {crps:.1f}</span>
        </button>"""

    # Table rows for overview
    table_rows_html = ""
    for rank, (loc, v) in enumerate(ranked, 1):
        m   = v["metrics"]
        clr = v["color"]
        bg  = "bg-best" if rank == 1 else ""
        cov_clr = "cval-green" if m["cov80"] >= 75 else ("cval-orange" if m["cov80"] >= 60 else "cval-red")
        crps_clr = "cval-green" if m["crps"] <= 28 else ("cval-orange" if m["crps"] <= 35 else "cval-red")
        table_rows_html += f"""
          <tr class="{bg}">
            <td><span class="rank-num" style="background:{clr}22;color:{clr};border:1px solid {clr}55">{rank}</span></td>
            <td><span class="dlabel" style="border-left:3px solid {clr}">&nbsp;{loc}</span></td>
            <td class="mono {crps_clr}">{m['crps']}</td>
            <td class="mono">{m['rmse']}</td>
            <td class="mono">{m['mae']}</td>
            <td class="mono">{m['mape']}%</td>
            <td class="mono">{m['r2']}</td>
            <td class="mono {cov_clr}">{m['cov80']}%</td>
            <td class="mono">{m['cov95']}%</td>
          </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Exp 09 — District Forecast Analysis · Ensemble S+X</title>
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
    body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--page);color:var(--text);padding:24px 20px 56px;line-height:1.55;min-width:900px}}

    /* ── header ── */
    .hdr{{text-align:center;padding:36px 32px 28px;background:linear-gradient(160deg,#fff 0%,#f3eeff 100%);
          border-radius:20px;border:1px solid var(--border);box-shadow:0 2px 20px #8839ef0d;margin-bottom:24px}}
    .hdr h1{{font-size:1.75rem;font-weight:800;letter-spacing:-0.5px;
             background:linear-gradient(120deg,var(--mauve),var(--sky),var(--teal));
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px}}
    .hdr .sub{{color:var(--sub0);font-size:0.85rem}}
    .kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
    .kpi{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 20px 14px;box-shadow:0 1px 6px #0001;position:relative;overflow:hidden}}
    .kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;background:var(--kg,var(--sky))}}
    .kpi-lbl{{font-size:0.64rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;font-weight:700}}
    .kpi-val{{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;letter-spacing:-1.5px;line-height:1;margin-top:5px;color:var(--kc,var(--text))}}
    .kpi-sub{{font-size:0.73rem;color:var(--muted);margin-top:5px}}

    /* ── section card ── */
    .sec{{background:var(--card);border:1px solid var(--border);border-radius:16px;
          padding:22px 24px;box-shadow:0 1px 6px #0001;margin-bottom:16px}}
    .sec-title{{font-size:0.67rem;font-weight:700;text-transform:uppercase;letter-spacing:1.1px;
                color:var(--muted);margin-bottom:18px;display:flex;align-items:center;gap:8px}}
    .dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
    .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
    @media(max-width:1000px){{.grid-2{{grid-template-columns:1fr}}}}

    /* ── overview table ── */
    table{{width:100%;border-collapse:collapse;font-size:0.83rem}}
    thead th{{background:var(--surface);color:var(--sub0);font-weight:700;text-transform:uppercase;
              font-size:0.63rem;letter-spacing:0.9px;padding:9px 13px;text-align:left;
              border-bottom:1px solid var(--border);white-space:nowrap}}
    tbody td{{padding:9px 13px;border-bottom:1px solid #f0ebff;white-space:nowrap;color:var(--sub1)}}
    tbody tr:last-child td{{border-bottom:none}}
    tbody tr:hover td{{background:#f7f4ff;color:var(--text)}}
    tbody tr.bg-best td{{background:#f0fff5}}
    tbody tr.bg-best td:first-child{{border-left:3px solid var(--green)}}
    .mono{{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--text)}}
    .cval-green{{color:var(--green)}}.cval-orange{{color:var(--orange)}}.cval-red{{color:var(--red)}}
    .rank-num{{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;
               border-radius:8px;font-size:0.7rem;font-weight:800;font-family:'JetBrains Mono',monospace}}
    .dlabel{{padding-left:8px;font-weight:600;color:var(--text)}}

    /* ── district tabs ── */
    .tabs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}
    .tab-btn{{display:flex;flex-direction:column;align-items:flex-start;gap:2px;padding:10px 16px;
              background:var(--card);border:1px solid var(--border);border-radius:12px;cursor:pointer;
              border-top:3px solid var(--dc,#888);transition:all 0.2s;min-width:130px}}
    .tab-btn:hover{{border-color:var(--border-h);box-shadow:0 3px 12px #8839ef12}}
    .tab-btn.active{{background:var(--surface);border-color:var(--dc,#888);box-shadow:0 3px 16px #8839ef18}}
    .tab-rank{{font-size:0.6rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:0.5px}}
    .tab-name{{font-size:0.82rem;font-weight:700;color:var(--text)}}
    .tab-crps{{font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:var(--sub0)}}

    /* ── district panel ── */
    .dist-panel{{display:none}}.dist-panel.active{{display:block}}
    .metric-badges{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
    .badge{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 16px;text-align:center}}
    .badge-lbl{{font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:700}}
    .badge-val{{font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700;line-height:1.2;color:var(--text)}}
    .badge-unit{{font-size:0.7rem;color:var(--muted)}}
    .chart-wrap{{position:relative;width:100%;height:260px;margin-bottom:6px}}
    .chart-wrap-sm{{position:relative;width:100%;height:200px;margin-bottom:6px}}
    .chart-label{{font-size:0.63rem;color:var(--muted);text-align:right;margin-bottom:12px;letter-spacing:0.3px}}
    .district-insight{{background:linear-gradient(135deg,#f8f4ff,#f0eaff);border:1px solid var(--border);
                       border-radius:12px;padding:14px 18px;margin-top:16px;font-size:0.84rem;color:var(--sub1);line-height:1.6}}
    .district-insight strong{{color:var(--text)}}
    canvas{{max-height:none!important}}

    .footer{{text-align:center;padding:24px 0 4px;color:var(--ghost);font-size:0.7rem}}
  </style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <h1>District-Level Forecast Analysis — Ensemble S+X Champion</h1>
  <p class="sub">Ngadjizi Region, Comoros &nbsp;·&nbsp; 7 districts &nbsp;·&nbsp; 26-week forecast horizon &nbsp;·&nbsp; 100 probabilistic samples per district/week</p>
</div>

<!-- KPI row -->
<div class="kpi-row">
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--sky),var(--teal));--kc:var(--sky)">
    <div class="kpi-lbl">Mean CRPS</div>
    <div class="kpi-val">{overall_crps:.2f}</div>
    <div class="kpi-sub">across 7 districts</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--green),var(--teal));--kc:var(--green)">
    <div class="kpi-lbl">Mean RMSE</div>
    <div class="kpi-val">{overall_rmse:.1f}</div>
    <div class="kpi-sub">cases / week</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--mauve),var(--sky));--kc:var(--mauve)">
    <div class="kpi-lbl">Best District</div>
    <div class="kpi-val" style="font-size:1.1rem;margin-top:8px">{best_dist.split('-')[0]}</div>
    <div class="kpi-sub">CRPS {ranked[0][1]['metrics']['crps']:.2f}</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,var(--orange),var(--red));--kc:var(--orange)">
    <div class="kpi-lbl">Hardest District</div>
    <div class="kpi-val" style="font-size:1.1rem;margin-top:8px">{ranked[-1][0].split('-')[0]}</div>
    <div class="kpi-sub">CRPS {ranked[-1][1]['metrics']['crps']:.2f}</div>
  </div>
</div>

<!-- Overview charts -->
<div class="grid-2">
  <div class="sec">
    <div class="sec-title"><span class="dot" style="background:var(--teal)"></span>CRPS by District — ranked best to worst</div>
    <canvas id="crpsBar" style="max-height:220px"></canvas>
  </div>
  <div class="sec">
    <div class="sec-title"><span class="dot" style="background:var(--mauve)"></span>Multi-Metric Profile — all districts</div>
    <canvas id="radarChart" style="max-height:220px"></canvas>
  </div>
</div>

<!-- Summary table -->
<div class="sec">
  <div class="sec-title"><span class="dot" style="background:var(--sky)"></span>Full Metrics Table — test period (weeks 79–104)</div>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr><th>Rank</th><th>District</th><th>CRPS ↓</th><th>RMSE ↓</th><th>MAE ↓</th><th>MAPE</th><th>R²</th><th>80% Coverage</th><th>95% Coverage</th></tr>
    </thead>
    <tbody>{table_rows_html}</tbody>
  </table>
  </div>
</div>

<!-- District deep-dive -->
<div class="sec">
  <div class="sec-title"><span class="dot" style="background:var(--peach,var(--orange))"></span>District Deep-Dive — forecast timeline + climate drivers</div>

  <!-- Tabs -->
  <div class="tabs" id="district-tabs">{tabs_html}
  </div>

  <!-- Panels — one per district -->
  <div id="district-panels"></div>
</div>

<div class="footer">Exp 09 · Ensemble S+X champion · 7 districts · Comoros Climate-Health Modeling</div>

<script>
// ── All district data ─────────────────────────────────────────────────────────
const DISTRICTS = {dist_data_js};

// ── Chart.js global defaults ──────────────────────────────────────────────────
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
Chart.defaults.plugins.tooltip.bodyColor = '#544873';
Chart.defaults.plugins.tooltip.titleFont = {{family:"'Plus Jakarta Sans'", weight:'700'}};
Chart.defaults.plugins.tooltip.bodyFont = {{family:"'JetBrains Mono'", size:12}};
Chart.defaults.plugins.tooltip.padding = 10;

const GC = '#ece7fa';

// ── Overview charts ───────────────────────────────────────────────────────────
const ranked = Object.entries(DISTRICTS).sort((a,b) => a[1].metrics.crps - b[1].metrics.crps);
const dNames  = ranked.map(d => d[0]);
const dColors = ranked.map(d => d[1].color);
const dCrps   = ranked.map(d => d[1].metrics.crps);
const dRmse   = ranked.map(d => d[1].metrics.rmse);
const dCov80  = ranked.map(d => d[1].metrics.cov80);
const dCov95  = ranked.map(d => d[1].metrics.cov95);
const dR2     = ranked.map(d => d[1].metrics.r2);
const minCrps = Math.min(...dCrps);

// Ranked CRPS bar
new Chart(document.getElementById('crpsBar'), {{
  type: 'bar',
  data: {{
    labels: dNames,
    datasets: [{{
      label: 'CRPS',
      data: dCrps,
      backgroundColor: dColors.map((c,i) => dCrps[i] === minCrps ? c+'ee' : c+'66'),
      borderColor: dColors,
      borderWidth: 1.5, borderRadius: 5,
    }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true,
    plugins: {{ legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ' CRPS: ' + ctx.raw.toFixed(2) }} }} }},
    scales: {{
      x: {{ beginAtZero: false, min: Math.min(...dCrps) - 5, grid: {{ color: GC }} }},
      y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});

// Radar
function norm(arr) {{ const mx = Math.max(...arr); return arr.map(v => +(v/mx*100).toFixed(1)); }}
const crpsN = norm(dCrps); const rmseN = norm(dRmse); const mapeN = norm(ranked.map(d => d[1].metrics.mape));
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: ['CRPS (inv)','RMSE (inv)','R² ×100','80% Coverage','95% Coverage'],
    datasets: ranked.map((d,i) => ({{
      label: d[0].split('-')[0],
      data: [
        +(100 - crpsN[i]).toFixed(1),
        +(100 - rmseN[i]).toFixed(1),
        +(d[1].metrics.r2 * 100).toFixed(1),
        +d[1].metrics.cov80.toFixed(1),
        +d[1].metrics.cov95.toFixed(1),
      ],
      borderColor: d[1].color,
      backgroundColor: d[1].color + '18',
      borderWidth: 2, pointRadius: 3,
      pointBackgroundColor: d[1].color,
    }}))
  }},
  options: {{
    responsive: true,
    scales: {{ r: {{
      beginAtZero: true, max: 100,
      grid: {{ color: GC }}, angleLines: {{ color: '#ddd5f5' }},
      pointLabels: {{ font: {{ size: 10, weight: '600' }}, color: '#7a6e92' }},
      ticks: {{ display: false }}
    }} }},
    plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }}, padding: 8 }} }} }}
  }}
}});

// ── District panel builder ────────────────────────────────────────────────────
const chartRegistry = {{}};

function buildDistrictPanel(loc) {{
  const d   = DISTRICTS[loc];
  const tid = loc.replace(/ /g,'_').replace(/-/g,'_');
  const c   = d.color;
  const m   = d.metrics;
  const weekly = d.weekly;

  const trainWeeks = weekly.filter(w => w.isTrain);
  const testWeeks  = weekly.filter(w => !w.isTrain);
  const allLabels  = weekly.map(w => w.label.replace('20','').replace('-W','W'));

  // Separate series
  const actualAll = weekly.map(w => w.actual);
  const medianAll = weekly.map(w => w.median);
  const q10All    = weekly.map(w => w.q10);
  const q90All    = weekly.map(w => w.q90);
  const q025All   = weekly.map(w => w.q025);
  const q975All   = weekly.map(w => w.q975);
  const rainAll   = weekly.map(w => w.rain);
  const tempAll   = weekly.map(w => w.temp);
  const crpsAll   = weekly.filter(w => !w.isTrain).map(w => w.crps);
  const trainBoundary = trainWeeks.length;  // index after last train week

  const cov_badge_clr = m.cov80 >= 75 ? '#40a02b' : (m.cov80 >= 60 ? '#fe640b' : '#d20f39');
  const crps_badge_clr = m.crps <= 28 ? '#40a02b' : (m.crps <= 40 ? '#fe640b' : '#d20f39');

  const maxActual = Math.max(...actualAll);
  const trainMean = +(trainWeeks.reduce((s,w) => s+w.actual, 0) / trainWeeks.length).toFixed(1);
  const testMean  = +(testWeeks.reduce((s,w) => s+w.actual, 0) / testWeeks.length).toFixed(1);
  const testMedianMean = +(testWeeks.reduce((s,w) => s+(w.median||0), 0) / testWeeks.length).toFixed(1);

  const insight = d.insight;

  const html = `
<div class="metric-badges">
  <div class="badge" style="border-top:3px solid ${{crps_badge_clr}}">
    <div class="badge-lbl">CRPS</div>
    <div class="badge-val" style="color:${{crps_badge_clr}}">${{m.crps}}</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">RMSE</div>
    <div class="badge-val">${{m.rmse}}</div>
    <div class="badge-unit">cases/wk</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">MAE</div>
    <div class="badge-val">${{m.mae}}</div>
    <div class="badge-unit">cases/wk</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">MAPE</div>
    <div class="badge-val">${{m.mape}}</div>
    <div class="badge-unit">%</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">R²</div>
    <div class="badge-val">${{m.r2}}</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{cov_badge_clr}}">
    <div class="badge-lbl">80% Coverage</div>
    <div class="badge-val" style="color:${{cov_badge_clr}}">${{m.cov80}}%</div>
  </div>
  <div class="badge" style="border-top:3px solid ${{c}}">
    <div class="badge-lbl">95% Coverage</div>
    <div class="badge-val">${{m.cov95}}%</div>
  </div>
</div>

<div class="chart-wrap">
  <canvas id="forecast_${{tid}}"></canvas>
</div>
<div class="chart-label">&#9632; Actual &nbsp; ---- Predicted median &nbsp; &#9632; 80% PI &nbsp; &#9632; 95% PI &nbsp;|&nbsp; Dashed line = train/test split</div>

<div class="chart-wrap-sm">
  <canvas id="climate_${{tid}}"></canvas>
</div>
<div class="chart-label">Rainfall (mm/wk, bars) &nbsp;·&nbsp; Mean Temperature (°C, line)</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px">
  <div>
    <div style="font-size:0.64rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:10px">Actual vs Predicted — test weeks</div>
    <div style="height:220px"><canvas id="scatter_${{tid}}"></canvas></div>
  </div>
  <div>
    <div style="font-size:0.64rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:10px">CRPS per forecast week</div>
    <div style="height:220px"><canvas id="crpsweek_${{tid}}"></canvas></div>
  </div>
</div>

<div class="district-insight">${{insight}}</div>`;

  document.getElementById('district-panels').innerHTML +=
    `<div class="dist-panel" id="panel_${{tid}}">${{html}}</div>`;

  // Deferred chart build (will be called after panel is shown)
  chartRegistry[loc] = () => buildDistrictCharts(loc, tid, c,
    allLabels, actualAll, medianAll, q10All, q90All, q025All, q975All,
    rainAll, tempAll, crpsAll, testWeeks, trainBoundary);
}}

function buildDistrictCharts(loc, tid, c,
    allLabels, actualAll, medianAll, q10All, q90All, q025All, q975All,
    rainAll, tempAll, crpsAll, testWeeks, trainBoundary) {{

  if (chartRegistry[loc + '_built']) return;
  chartRegistry[loc + '_built'] = true;

  const alpha = hex => hex + '33';
  const alpha2 = hex => hex + '18';

  // ── Forecast chart ────────────────────────────────────────────────────────
  // Use annotation-style background via chartArea plugin trick
  const splitPlugin = {{
    id: 'splitBg',
    beforeDraw(chart) {{
      const {{ctx, chartArea, scales}} = chart;
      if (!chartArea) return;
      const xScale = scales.x;
      // training area
      const x0 = chartArea.left;
      const x1 = xScale.getPixelForValue(trainBoundary - 0.5);
      ctx.save();
      ctx.fillStyle = '#f5f0ff44';
      ctx.fillRect(x0, chartArea.top, x1 - x0, chartArea.bottom - chartArea.top);
      // vertical boundary line
      ctx.strokeStyle = '#a099bb88';
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

  new Chart(document.getElementById(`forecast_${{tid}}`), {{
    type: 'line',
    plugins: [splitPlugin],
    data: {{
      labels: allLabels,
      datasets: [
        // 95% PI — upper (fills down to next dataset)
        {{ label: '95% PI', data: q975All, borderWidth:0, pointRadius:0,
           fill:'+1', backgroundColor: alpha2(c), borderColor:'transparent', order:4 }},
        // 95% PI — lower
        {{ label: '', data: q025All, borderWidth:0, pointRadius:0,
           fill:false, backgroundColor:'transparent', borderColor:'transparent', order:4 }},
        // 80% PI — upper
        {{ label: '80% PI', data: q90All, borderWidth:0, pointRadius:0,
           fill:'+1', backgroundColor: alpha(c), borderColor:'transparent', order:3 }},
        // 80% PI — lower
        {{ label: '', data: q10All, borderWidth:0, pointRadius:0,
           fill:false, backgroundColor:'transparent', borderColor:'transparent', order:3 }},
        // Predicted median
        {{ label: 'Predicted median', data: medianAll, borderColor: c,
           borderWidth:2, borderDash:[6,3], pointRadius:0,
           backgroundColor:'transparent', fill:false, order:2 }},
        // Actual
        {{ label: 'Actual cases', data: actualAll, borderColor: c,
           borderWidth:2.5, pointRadius:2, pointBackgroundColor: c,
           backgroundColor:'transparent', fill:false, order:1 }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      interaction: {{ mode:'index', intersect:false }},
      plugins: {{
        legend: {{ position:'top', labels: {{ filter: item => item.text !== '' }} }},
        tooltip: {{ callbacks: {{
          title: ctx => ctx[0].label,
          label: ctx => {{
            if(!ctx.dataset.label) return null;
            return ` ${{ctx.dataset.label}}: ${{ctx.raw != null ? ctx.raw.toFixed(0) : 'N/A'}}`;
          }}
        }} }}
      }},
      scales: {{
        x: {{ grid:{{display:false}}, ticks:{{ maxTicksLimit:13, font:{{size:9}}, maxRotation:30 }} }},
        y: {{ beginAtZero:true, grid:{{color:GC}}, title:{{display:true, text:'Cases / week', font:{{size:10}}}} }}
      }}
    }}
  }});

  // ── Climate chart ─────────────────────────────────────────────────────────
  new Chart(document.getElementById(`climate_${{tid}}`), {{
    type: 'bar',
    data: {{
      labels: allLabels,
      datasets: [
        {{ type:'bar', label:'Rainfall (mm)', data:rainAll,
           backgroundColor:'#04a5e533', borderColor:'#04a5e5', borderWidth:0.5,
           yAxisID:'yr', order:2 }},
        {{ type:'line', label:'Temperature (°C)', data:tempAll,
           borderColor:'#fe640b', borderWidth:2, pointRadius:0,
           backgroundColor:'transparent', fill:false, yAxisID:'yt', order:1 }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      interaction:{{mode:'index', intersect:false}},
      plugins:{{ legend:{{position:'top'}} }},
      scales:{{
        x: {{ grid:{{display:false}}, ticks:{{maxTicksLimit:13, font:{{size:9}}, maxRotation:30}} }},
        yr: {{ position:'left', beginAtZero:true, grid:{{color:GC}},
               title:{{display:true, text:'Rainfall (mm)', font:{{size:9}}}} }},
        yt: {{ position:'right', grid:{{display:false}},
               title:{{display:true, text:'Temp (°C)', font:{{size:9}}}},
               min:20, max:32 }},
      }}
    }}
  }});

  // ── Actual vs predicted scatter ───────────────────────────────────────────
  const scatterData = testWeeks
    .filter(w => w.median != null)
    .map((w, i) => ({{ x: w.median, y: w.actual, week: w.label, idx: i }}));
  const n = scatterData.length;
  const allActual   = testWeeks.map(w => w.actual);
  const allPred     = testWeeks.filter(w => w.median!=null).map(w => w.median);
  const maxVal = Math.max(...allActual, ...allPred) * 1.1;

  new Chart(document.getElementById(`scatter_${{tid}}`), {{
    type: 'scatter',
    data: {{
      datasets: [
        // Perfect prediction line
        {{ type:'line', label:'Perfect', data:[{{x:0,y:0}},{{x:maxVal,y:maxVal}}],
           borderColor:'#a099bb', borderWidth:1.5, borderDash:[4,4], pointRadius:0,
           backgroundColor:'transparent', fill:false }},
        // Actual dots
        {{ label: 'Test weeks', data: scatterData,
           backgroundColor: scatterData.map((_,i) => c + Math.round(80 + i/n*160).toString(16).padStart(2,'0')),
           borderColor: c, borderWidth:1, pointRadius:6, pointHoverRadius:8 }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins: {{
        legend: {{ display:false }},
        tooltip: {{ callbacks: {{
          label: ctx => ctx.raw.week
            ? ` ${{ctx.raw.week}}: actual=${{ctx.raw.y.toFixed(0)}} pred=${{ctx.raw.x.toFixed(0)}}`
            : null
        }} }}
      }},
      scales: {{
        x: {{ min:0, max:maxVal, grid:{{color:GC}}, title:{{display:true, text:'Predicted median', font:{{size:10}}}} }},
        y: {{ min:0, max:maxVal, grid:{{color:GC}}, title:{{display:true, text:'Actual cases', font:{{size:10}}}} }},
      }}
    }}
  }});

  // ── CRPS per test week ────────────────────────────────────────────────────
  const testLabels = testWeeks.map(w => w.label.replace('20','').replace('-W','W'));
  const minWcrps = Math.min(...crpsAll);
  new Chart(document.getElementById(`crpsweek_${{tid}}`), {{
    type: 'bar',
    data: {{
      labels: testLabels,
      datasets: [{{
        label: 'CRPS',
        data: crpsAll,
        backgroundColor: crpsAll.map(v => v === minWcrps ? c+'ee' : c+'55'),
        borderColor: c, borderWidth:1, borderRadius:3,
      }}]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins: {{ legend:{{display:false}} }},
      scales: {{
        x: {{ grid:{{display:false}}, ticks:{{maxTicksLimit:13, font:{{size:8}}, maxRotation:45}} }},
        y: {{ beginAtZero:true, grid:{{color:GC}}, title:{{display:true, text:'CRPS', font:{{size:10}}}} }}
      }}
    }}
  }});
}}

// ── Tab logic ─────────────────────────────────────────────────────────────────
let activeDistrict = null;
const builtPanels = new Set();

function selectDistrict(loc) {{
  // Deactivate old
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.dist-panel').forEach(p => p.classList.remove('active'));

  const tid = loc.replace(/ /g,'_').replace(/-/g,'_');
  document.getElementById('tab_' + tid).classList.add('active');
  document.getElementById('panel_' + tid).classList.add('active');

  // Build charts lazily the first time the panel is shown
  if (!builtPanels.has(loc)) {{
    builtPanels.add(loc);
    if (chartRegistry[loc]) chartRegistry[loc]();
  }}
  activeDistrict = loc;
}}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
// Build all panels (but not charts) immediately
Object.keys(DISTRICTS).forEach(loc => buildDistrictPanel(loc));

// Show first district
const firstDistrict = Object.keys(DISTRICTS)[0];
selectDistrict(firstDistrict);

</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    print("Exp 09 — District Forecast Analysis")
    print("Loading and computing per-district metrics ...")
    districts = prepare_data()

    for loc, d in sorted(districts.items(), key=lambda x: x[1]["metrics"]["crps"]):
        m = d["metrics"]
        print(f"  {loc:<25s}  CRPS={m['crps']:.2f}  RMSE={m['rmse']:.1f}  80%cov={m['cov80']:.1f}%")

    print("\nBuilding HTML dashboard ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html(districts)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved -> {OUT_HTML}")


if __name__ == "__main__":
    run()
