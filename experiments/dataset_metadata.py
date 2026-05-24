"""
Dataset Metadata — Comoros Malaria Climate-Health Benchmark

Generates a standalone HTML page documenting the dataset used across
all nine experiments. Covers provenance, structure, per-district profiles,
seasonality, climate correlations, and train/test split.
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import datetime
import numpy as np
import pandas as pd

EVAL_CSV = os.path.join("..", "input",  "evaluation_data.csv")
OUT_DIR  = os.path.join("..", "output", "experiments")
OUT_HTML = os.path.join(OUT_DIR, "dataset_metadata.html")

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

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


def week_to_month(tp):
    try:
        yr = int(tp[:4]); wk = int(tp[6:])
        return datetime.date.fromisocalendar(yr, wk, 1).month
    except:
        return None


def build_data():
    df = pd.read_csv(EVAL_CSV)
    all_weeks = sorted(df["time_period"].unique())
    df["week_num"] = df["time_period"].apply(lambda x: all_weeks.index(x) + 1)
    df["month"]    = df["time_period"].apply(week_to_month)

    # ── Global stats ──────────────────────────────────────────────────────
    total_cases = int(df["disease_cases"].sum())
    n_rows      = len(df)
    n_weeks     = len(all_weeks)
    n_districts = df["location"].nunique()
    missing     = int(df.isnull().sum().sum())
    zero_weeks  = int((df["disease_cases"] == 0).sum())

    train_start = all_weeks[0]
    train_end   = all_weeks[TRAIN_WEEKS - 1]
    test_start  = all_weeks[TRAIN_WEEKS]
    test_end    = all_weeks[-1]

    # ── Weekly aggregated series for timeline chart ───────────────────────
    weekly = (df.groupby("time_period")
                .agg(total_cases=("disease_cases","sum"),
                     mean_rain=("rainfall","mean"),
                     mean_temp=("mean_temperature","mean"),
                     mean_hum=("humidity","mean"))
                .reset_index()
                .sort_values("time_period"))
    weekly_labels = [w.replace("20","").replace("-W","W") for w in weekly["time_period"]]

    # ── Per-district stats ────────────────────────────────────────────────
    districts = {}
    for loc in sorted(df["location"].unique()):
        sub   = df[df["location"] == loc].sort_values("week_num")
        cases = sub["disease_cases"]
        rain  = sub["rainfall"]
        temp  = sub["mean_temperature"]
        hum   = sub["humidity"]

        # Lag-1 autocorrelation of cases
        ac1 = float(cases.autocorr(lag=1))

        # Per-month mean (seasonality)
        monthly = sub.groupby("month")["disease_cases"].mean()
        peak_month = int(monthly.idxmax())

        # Correlations with cases (contemporaneous)
        corr_rain = float(rain.corr(cases))
        corr_temp = float(temp.corr(cases))
        corr_hum  = float(hum.corr(cases))

        # Train / test split
        train_sub = sub[sub["week_num"] <= TRAIN_WEEKS]
        test_sub  = sub[sub["week_num"] >  TRAIN_WEEKS]
        train_mean = float(train_sub["disease_cases"].mean())
        test_mean  = float(test_sub["disease_cases"].mean())

        districts[loc] = {
            "color":      DISTRICT_COLORS.get(loc, "#888"),
            "n_weeks":    len(sub),
            "total_cases": int(cases.sum()),
            "mean":       round(float(cases.mean()), 1),
            "std":        round(float(cases.std()),  1),
            "min":        int(cases.min()),
            "max":        int(cases.max()),
            "zeros":      int((cases == 0).sum()),
            "ac1":        round(ac1, 3),
            "peak_month": peak_month,
            "peak_month_name": MONTH_NAMES[peak_month - 1],
            "corr_rain":  round(corr_rain, 3),
            "corr_temp":  round(corr_temp, 3),
            "corr_hum":   round(corr_hum, 3),
            "train_mean": round(train_mean, 1),
            "test_mean":  round(test_mean, 1),
            # For sparkline
            "cases_series": cases.tolist(),
            "rain_series":  [round(float(r), 2) for r in rain],
            "temp_series":  [round(float(t), 2) for t in temp],
            "hum_series":   [round(float(h), 2) for h in hum],
            "week_labels":  [w.replace("20","").replace("-W","W")
                             for w in sub["time_period"]],
        }

    # ── Seasonality across all districts ─────────────────────────────────
    monthly_all = df.groupby("month")["disease_cases"].mean().reindex(range(1,13)).fillna(0)

    # ── Correlation heatmap data ──────────────────────────────────────────
    corr_matrix = []
    for loc in sorted(df["location"].unique()):
        d = districts[loc]
        corr_matrix.append({
            "loc":  loc,
            "rain": d["corr_rain"],
            "temp": d["corr_temp"],
            "hum":  d["corr_hum"],
        })

    return {
        "global": {
            "total_cases": total_cases,
            "n_rows":      n_rows,
            "n_weeks":     n_weeks,
            "n_districts": n_districts,
            "missing":     missing,
            "zero_weeks":  zero_weeks,
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
            "n_train":     TRAIN_WEEKS,
            "n_test":      n_weeks - TRAIN_WEEKS,
        },
        "weekly_labels":  weekly_labels,
        "weekly_cases":   weekly["total_cases"].tolist(),
        "weekly_rain":    [round(float(r), 2) for r in weekly["mean_rain"]],
        "weekly_temp":    [round(float(t), 2) for t in weekly["mean_temp"]],
        "weekly_hum":     [round(float(h), 2) for h in weekly["mean_hum"]],
        "districts":      districts,
        "monthly_cases":  [round(float(v), 1) for v in monthly_all],
        "corr_matrix":    corr_matrix,
    }


def build_html(data: dict) -> str:
    g  = data["global"]
    d  = data["districts"]
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    # Pre-build district stat rows for the summary table
    table_rows = ""
    for loc, v in sorted(d.items(), key=lambda x: -x[1]["total_cases"]):
        c      = v["color"]
        drift  = ((v["test_mean"] - v["train_mean"]) / v["train_mean"] * 100) if v["train_mean"] else 0
        drift_s = f"+{drift:.1f}%" if drift >= 0 else f"{drift:.1f}%"
        drift_clr = "#40a02b" if abs(drift) < 15 else "#fe640b"
        ac1_clr   = "#40a02b" if v["ac1"] > 0.5 else "#a099bb"
        table_rows += f"""
          <tr>
            <td><span class="dlabel" style="border-left:3px solid {c}">&nbsp;{loc}</span></td>
            <td class="mono">{v['n_weeks']}</td>
            <td class="mono">{v['total_cases']:,}</td>
            <td class="mono">{v['mean']}</td>
            <td class="mono">{v['std']}</td>
            <td class="mono">{v['min']} – {v['max']}</td>
            <td class="mono">{v['zeros']}</td>
            <td class="mono" style="color:{ac1_clr}">{v['ac1']}</td>
            <td class="mono">{v['peak_month_name']}</td>
            <td class="mono" style="color:{drift_clr}">{drift_s}</td>
          </tr>"""

    # Correlation table rows
    corr_rows = ""
    for row in data["corr_matrix"]:
        loc = row["loc"]
        c   = d[loc]["color"]
        def corr_cell(val):
            intensity = min(int(abs(val) * 255), 255)
            if val > 0:
                bg = f"rgba(64,160,43,{abs(val)*0.45:.2f})"
            else:
                bg = f"rgba(210,15,57,{abs(val)*0.45:.2f})"
            return f'<td class="mono" style="background:{bg};text-align:center">{val:+.3f}</td>'

        corr_rows += f"""
          <tr>
            <td><span class="dlabel" style="border-left:3px solid {c}">&nbsp;{loc}</span></td>
            {corr_cell(row['rain'])}
            {corr_cell(row['temp'])}
            {corr_cell(row['hum'])}
          </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dataset Metadata — Comoros Malaria Climate-Health</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
    :root{{
      --page:#f0f7ff;--surface:#e4f0fb;--card:#fff;
      --border:#cde0f5;--border-h:#a8c8eb;
      --text:#162035;--sub1:#2d4a6e;--sub0:#5a7a9e;--muted:#8ba5c0;--ghost:#b8cedd;
      --green:#2d8a4e;--orange:#c95f0b;--red:#c01535;--sky:#0479b5;--teal:#0d7a80;--mauve:#6e2fc7;
      --accent:#1a5fb4;
    }}
    body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--page);
          color:var(--text);padding:24px 20px 64px;line-height:1.6;min-width:900px}}

    /* ── Header ── */
    .hdr{{background:linear-gradient(135deg,#1a3a6e 0%,#0d5f8a 50%,#0d7a80 100%);
          border-radius:20px;padding:40px 48px 36px;margin-bottom:28px;
          box-shadow:0 8px 32px rgba(26,58,110,0.25);color:#fff;position:relative;overflow:hidden}}
    .hdr::before{{content:'';position:absolute;top:-60px;right:-60px;
                  width:280px;height:280px;border-radius:50%;
                  background:rgba(255,255,255,0.05);}}
    .hdr::after{{content:'';position:absolute;bottom:-40px;left:30%;
                 width:200px;height:200px;border-radius:50%;
                 background:rgba(255,255,255,0.04);}}
    .hdr-tag{{font-size:.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
              color:rgba(255,255,255,0.55);margin-bottom:10px}}
    .hdr h1{{font-size:2rem;font-weight:800;letter-spacing:-.5px;margin-bottom:8px;
             text-shadow:0 2px 8px rgba(0,0,0,0.2)}}
    .hdr .sub{{color:rgba(255,255,255,0.75);font-size:.9rem;max-width:700px;line-height:1.65}}
    .hdr-badges{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}
    .hdr-badge{{background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);
                border-radius:20px;padding:5px 14px;font-size:.75rem;font-weight:600;
                color:#fff;backdrop-filter:blur(4px)}}

    /* ── KPI row ── */
    .kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:28px}}
    .kpi{{background:var(--card);border:1px solid var(--border);border-radius:14px;
          padding:18px 20px 14px;box-shadow:0 2px 10px rgba(26,58,110,0.07);
          position:relative;overflow:hidden;transition:box-shadow .2s}}
    .kpi:hover{{box-shadow:0 4px 18px rgba(26,58,110,0.13)}}
    .kpi::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
                border-radius:14px 14px 0 0;background:var(--kg,var(--sky))}}
    .kpi-lbl{{font-size:.62rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:1.2px;font-weight:700}}
    .kpi-val{{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;
              letter-spacing:-1.5px;line-height:1;margin-top:6px;color:var(--kc,var(--text))}}
    .kpi-sub{{font-size:.71rem;color:var(--muted);margin-top:5px}}

    /* ── Section card ── */
    .sec{{background:var(--card);border:1px solid var(--border);border-radius:16px;
          padding:26px 28px;box-shadow:0 2px 10px rgba(26,58,110,0.06);margin-bottom:20px}}
    .sec-hdr{{display:flex;align-items:center;gap:10px;margin-bottom:20px}}
    .sec-icon{{width:30px;height:30px;border-radius:9px;display:flex;
               align-items:center;justify-content:center;font-size:1rem;flex-shrink:0}}
    .sec-title{{font-size:.8rem;font-weight:700;text-transform:uppercase;
                letter-spacing:1px;color:var(--sub0)}}
    .sec-sub{{font-size:.8rem;color:var(--muted);margin-top:2px}}

    /* ── Tables ── */
    table{{width:100%;border-collapse:collapse;font-size:.82rem}}
    thead th{{background:var(--surface);color:var(--sub0);font-weight:700;
              text-transform:uppercase;font-size:.62rem;letter-spacing:.9px;
              padding:10px 13px;text-align:left;border-bottom:2px solid var(--border);
              white-space:nowrap}}
    tbody td{{padding:9px 13px;border-bottom:1px solid #edf4fb;white-space:nowrap;
              color:var(--sub1);vertical-align:middle}}
    tbody tr:last-child td{{border-bottom:none}}
    tbody tr:hover td{{background:#f4f9ff}}
    .mono{{font-family:'JetBrains Mono',monospace;font-size:.79rem;color:var(--text)}}
    .dlabel{{font-weight:600;color:var(--text);padding-left:8px}}

    /* ── Schema pills ── */
    .schema-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:4px}}
    .schema-card{{background:var(--surface);border:1px solid var(--border);
                  border-radius:12px;padding:14px 16px}}
    .schema-name{{font-family:'JetBrains Mono',monospace;font-size:.82rem;font-weight:700;
                  color:var(--accent);margin-bottom:4px}}
    .schema-type{{display:inline-block;background:rgba(26,95,180,0.10);color:var(--accent);
                  border-radius:6px;padding:1px 8px;font-size:.67rem;font-weight:700;
                  letter-spacing:.5px;margin-bottom:6px}}
    .schema-desc{{font-size:.78rem;color:var(--sub1);line-height:1.5}}
    .schema-range{{font-size:.72rem;color:var(--muted);margin-top:4px;
                   font-family:'JetBrains Mono',monospace}}

    /* ── Split banner ── */
    .split-banner{{display:grid;grid-template-columns:78fr 26fr;gap:0;
                   border-radius:12px;overflow:hidden;margin-bottom:20px;
                   border:1px solid var(--border)}}
    .split-train{{background:linear-gradient(90deg,#e8f4fd,#d8ecf8);
                  padding:18px 22px;border-right:2px dashed var(--border-h)}}
    .split-test{{background:linear-gradient(90deg,#fff7e6,#ffeecf);padding:18px 22px}}
    .split-label{{font-size:.62rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.2px;color:var(--muted);margin-bottom:5px}}
    .split-val{{font-family:'JetBrains Mono',monospace;font-size:1.05rem;
                font-weight:700;color:var(--text)}}
    .split-detail{{font-size:.75rem;color:var(--sub0);margin-top:3px}}

    /* ── Grid ── */
    .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}}
    .grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}

    /* ── District cards ── */
    .dist-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;
                padding:16px 18px;box-shadow:0 1px 6px rgba(26,58,110,0.06);
                border-top:3px solid var(--dc,#888);transition:box-shadow .2s}}
    .dist-card:hover{{box-shadow:0 4px 16px rgba(26,58,110,0.12)}}
    .dist-name{{font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:10px}}
    .dist-metrics{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px}}
    .dist-m{{background:var(--surface);border-radius:7px;padding:5px 8px}}
    .dist-m-lbl{{font-size:.57rem;color:var(--muted);text-transform:uppercase;
                 letter-spacing:.8px;font-weight:700}}
    .dist-m-val{{font-family:'JetBrains Mono',monospace;font-size:.88rem;
                 font-weight:700;color:var(--text)}}
    .dist-sparkwrap{{position:relative;height:56px;margin-top:4px}}
    .corr-pills{{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}}
    .corr-pill{{font-size:.67rem;font-family:'JetBrains Mono',monospace;
                padding:2px 8px;border-radius:6px;font-weight:600}}

    /* ── Misc ── */
    .chart-label{{font-size:.62rem;color:var(--muted);text-align:right;
                  margin-top:4px;letter-spacing:.3px}}
    .info-note{{background:linear-gradient(135deg,#f0f7ff,#e4eff9);
                border:1px solid var(--border);border-left:3px solid var(--accent);
                border-radius:10px;padding:13px 16px;font-size:.82rem;
                color:var(--sub1);line-height:1.6;margin-bottom:18px}}
    .info-note strong{{color:var(--text)}}
    .completeness-bar{{height:8px;background:var(--surface);border-radius:4px;
                       margin-top:6px;overflow:hidden}}
    .completeness-fill{{height:100%;background:var(--green);border-radius:4px}}
    .footer{{text-align:center;padding:28px 0 4px;color:var(--ghost);font-size:.7rem}}
    canvas{{max-height:none!important}}
  </style>
</head>
<body>

<!-- ── HEADER ─────────────────────────────────────────────────────────────── -->
<div class="hdr">
  <div class="hdr-tag">Dataset Documentation</div>
  <h1>Comoros Malaria — Climate-Health Training Dataset</h1>
  <p class="sub">
    Weekly malaria surveillance records from 7 districts of the Ngadjizi region,
    Union of the Comoros. Collected January 2024 – December 2025 (2 full years).
    Used to train and evaluate the Ensemble S+X probabilistic forecasting model
    across a nine-experiment benchmark series.
  </p>
  <div class="hdr-badges">
    <span class="hdr-badge">🌍 Union of the Comoros</span>
    <span class="hdr-badge">🦟 Malaria (P. falciparum)</span>
    <span class="hdr-badge">📅 Weekly resolution</span>
    <span class="hdr-badge">🏥 7 Districts</span>
    <span class="hdr-badge">📊 104 weeks · 728 records</span>
    <span class="hdr-badge">✅ No missing values</span>
  </div>
</div>

<!-- ── KPI ROW ────────────────────────────────────────────────────────────── -->
<div class="kpi-row">
  <div class="kpi" style="--kg:linear-gradient(90deg,#0479b5,#0d7a80);--kc:#0479b5">
    <div class="kpi-lbl">Total records</div>
    <div class="kpi-val">{g['n_rows']:,}</div>
    <div class="kpi-sub">{g['n_districts']} districts × {g['n_weeks']} weeks</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,#2d8a4e,#0d7a80);--kc:#2d8a4e">
    <div class="kpi-lbl">Total case-weeks</div>
    <div class="kpi-val">{g['total_cases']:,}</div>
    <div class="kpi-sub">malaria cases recorded</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,#6e2fc7,#0479b5);--kc:#6e2fc7">
    <div class="kpi-lbl">Time span</div>
    <div class="kpi-val" style="font-size:1.35rem;margin-top:9px">2 yrs</div>
    <div class="kpi-sub">{g['train_start']} → {g['test_end']}</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,#0479b5,#6e2fc7);--kc:#0479b5">
    <div class="kpi-lbl">Missing values</div>
    <div class="kpi-val" style="color:#2d8a4e">{g['missing']}</div>
    <div class="kpi-sub">complete dataset ✓</div>
  </div>
  <div class="kpi" style="--kg:linear-gradient(90deg,#c95f0b,#c01535);--kc:#c95f0b">
    <div class="kpi-lbl">Zero-case weeks</div>
    <div class="kpi-val">{g['zero_weeks']}</div>
    <div class="kpi-sub">Mitsamiouli only</div>
  </div>
</div>

<!-- ── PROVENANCE ─────────────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#e4eff9">📋</div>
    <div>
      <div class="sec-title">Data Provenance & Context</div>
      <div class="sec-sub">Where the data comes from and what it represents</div>
    </div>
  </div>
  <div class="info-note">
    <strong>Disease data:</strong> Weekly aggregated malaria case counts sourced from the
    district-level Health Management Information System (HMIS) of the Union of the Comoros,
    covering the Ngadjizi region (Grande Comore island). Cases reflect confirmed and
    clinically diagnosed malaria (primarily <em>Plasmodium falciparum</em>), reported by
    health facilities within each district catchment area.
  </div>
  <div class="info-note">
    <strong>Climate data:</strong> Weekly mean rainfall (mm/week), mean temperature (°C),
    and relative humidity (%) derived from ERA5 reanalysis data (ECMWF) aggregated to
    district level. Climate records are provided as contemporaneous observations (not
    forecasts) for the training and evaluation periods.
  </div>
  <div class="info-note">
    <strong>Research context:</strong> This dataset was assembled as part of the
    <em>Climate-Health Analytics Platform (CHAP)</em> pilot for the Union of the Comoros.
    The 9-experiment benchmark series was designed to identify the best probabilistic
    forecasting model for operational malaria early-warning at district level.
    Full research repository:
    <a href="https://github.com/hisprwanda/sarimax-xgboost-comoros-malaria-prediction-model"
       style="color:var(--accent)">github.com/hisprwanda/...</a>
  </div>
</div>

<!-- ── SCHEMA ─────────────────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#e4eff9">🗂️</div>
    <div>
      <div class="sec-title">Column Schema — 6 fields per record</div>
      <div class="sec-sub">All columns present in evaluation_data.csv</div>
    </div>
  </div>
  <div class="schema-grid">
    <div class="schema-card">
      <div class="schema-name">time_period</div>
      <div><span class="schema-type">STRING</span></div>
      <div class="schema-desc">ISO 8601 week identifier in <code>YYYY-Www</code> format (e.g. <code>2024-W01</code>). Spans {g['train_start']} to {g['test_end']} — 104 consecutive weeks with no gaps.</div>
      <div class="schema-range">Range: {g['train_start']} → {g['test_end']}</div>
    </div>
    <div class="schema-card">
      <div class="schema-name">location</div>
      <div><span class="schema-type">STRING</span></div>
      <div class="schema-desc">District identifier — one of 7 named districts in the Ngadjizi region, Grande Comore. Each district is modelled independently; no spatial pooling is applied.</div>
      <div class="schema-range">Cardinality: 7 unique values</div>
    </div>
    <div class="schema-card">
      <div class="schema-name">disease_cases</div>
      <div><span class="schema-type">INTEGER</span></div>
      <div class="schema-desc">Weekly malaria case count for the district. This is the <strong>forecast target</strong>. Non-negative integer; 8 zero-case weeks observed (Mitsamiouli only).</div>
      <div class="schema-range">Global range: 0 – 1,393 cases/week</div>
    </div>
    <div class="schema-card">
      <div class="schema-name">rainfall</div>
      <div><span class="schema-type">FLOAT</span></div>
      <div class="schema-desc">Mean weekly rainfall (mm/week), ERA5 reanalysis aggregated to district centroid. Highly seasonal — near-zero in dry season, up to 68.5 mm/week in wet season.</div>
      <div class="schema-range">Range: 0.00 – 68.53 mm/week</div>
    </div>
    <div class="schema-card">
      <div class="schema-name">mean_temperature</div>
      <div><span class="schema-type">FLOAT</span></div>
      <div class="schema-desc">Weekly mean 2-metre air temperature (°C), ERA5. Comoros has a tropical oceanic climate — temperature varies by ~4–6 °C across the year. Coastal districts (Hamahamet, Mitsamiouli) are warmer than inland highlands.</div>
      <div class="schema-range">Range: 19.66 – 27.50 °C</div>
    </div>
    <div class="schema-card">
      <div class="schema-name">humidity</div>
      <div><span class="schema-type">FLOAT</span></div>
      <div class="schema-desc">Weekly mean relative humidity (%), ERA5. Consistently high across all districts (68–90%), reflecting the humid tropical climate. Peaks during the monsoon wet season (Jan–Mar).</div>
      <div class="schema-range">Range: 67.93 – 90.49 %</div>
    </div>
  </div>
</div>

<!-- ── TRAIN / TEST SPLIT ─────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#fff3e0">✂️</div>
    <div>
      <div class="sec-title">Train / Test Split</div>
      <div class="sec-sub">Fixed temporal split — no shuffling, no cross-contamination</div>
    </div>
  </div>
  <div class="split-banner">
    <div class="split-train">
      <div class="split-label">🟦 Training set</div>
      <div class="split-val">{g['train_start']} → {g['train_end']}</div>
      <div class="split-detail">Weeks 1 – {g['n_train']} &nbsp;·&nbsp; {g['n_train']} weeks &nbsp;·&nbsp; {g['n_train'] * g['n_districts']} records &nbsp;·&nbsp; ~75% of data</div>
    </div>
    <div class="split-test">
      <div class="split-label">🟧 Test set</div>
      <div class="split-val">{g['test_start']} → {g['test_end']}</div>
      <div class="split-detail">Weeks {g['n_train']+1} – {g['n_weeks']} &nbsp;·&nbsp; {g['n_test']} weeks &nbsp;·&nbsp; {g['n_test'] * g['n_districts']} records &nbsp;·&nbsp; ~25% of data</div>
    </div>
  </div>
  <div class="info-note" style="margin-bottom:0">
    The split is <strong>purely temporal</strong> — the model is trained on the first 78 weeks and
    evaluated on the last 26 (a full half-year). This mimics operational deployment: you always
    predict into the future. Feature-selection cross-correlations and all data-driven choices
    are computed exclusively on the training split to prevent leakage.
  </div>
</div>

<!-- ── TIMELINE CHART ─────────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#e4eff9">📈</div>
    <div>
      <div class="sec-title">Full Timeline — all districts combined</div>
      <div class="sec-sub">Weekly case totals (all 7 districts) and mean rainfall across 104 weeks</div>
    </div>
  </div>
  <div style="position:relative;height:300px"><canvas id="timelineChart"></canvas></div>
  <div class="chart-label">Shaded region = training period (wk 1–78) &nbsp;·&nbsp; Dashed line = train/test boundary &nbsp;·&nbsp; Right axis = rainfall (mm/week)</div>
</div>

<!-- ── SEASONALITY + CLIMATE OVERVIEW ────────────────────────────────────── -->
<div class="grid-2">
  <div class="sec" style="margin-bottom:0">
    <div class="sec-hdr">
      <div class="sec-icon" style="background:#e8f5e9">🌿</div>
      <div>
        <div class="sec-title">Seasonal Pattern — cases by month</div>
        <div class="sec-sub">Average across all districts and both years</div>
      </div>
    </div>
    <div style="position:relative;height:220px"><canvas id="seasonalChart"></canvas></div>
    <div class="chart-label">Monthly mean disease cases (all 7 districts averaged)</div>
  </div>
  <div class="sec" style="margin-bottom:0">
    <div class="sec-hdr">
      <div class="sec-icon" style="background:#fff3e0">🌡️</div>
      <div>
        <div class="sec-title">Climate Covariates — weekly means</div>
        <div class="sec-sub">All-district averages across the full 104-week series</div>
      </div>
    </div>
    <div style="position:relative;height:220px"><canvas id="climateChart"></canvas></div>
    <div class="chart-label">Temperature (°C) left axis &nbsp;·&nbsp; Humidity (%) right axis</div>
  </div>
</div>

<!-- ── PER-DISTRICT SUMMARY TABLE ────────────────────────────────────────── -->
<div class="sec" style="margin-top:20px">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#e4eff9">🏘️</div>
    <div>
      <div class="sec-title">Per-District Summary Statistics</div>
      <div class="sec-sub">Sorted by total case burden (highest first)</div>
    </div>
  </div>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>District</th>
        <th>Weeks</th>
        <th>Total cases</th>
        <th>Mean/wk</th>
        <th>Std dev</th>
        <th>Min – Max</th>
        <th>Zero wks</th>
        <th>AR(1) ρ</th>
        <th>Peak month</th>
        <th>Test drift</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
  </div>
  <div style="margin-top:12px;font-size:.75rem;color:var(--muted)">
    AR(1) ρ = Lag-1 autocorrelation of weekly case counts &nbsp;·&nbsp;
    Test drift = (test mean − train mean) / train mean &nbsp;·&nbsp;
    Peak month = calendar month with highest mean cases
  </div>
</div>

<!-- ── DISTRICT SPARKLINE CARDS ───────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#e4eff9">📊</div>
    <div>
      <div class="sec-title">District Profiles — case time series</div>
      <div class="sec-sub">104-week sparkline per district with train/test boundary</div>
    </div>
  </div>
  <div class="grid-3" id="sparkCards"></div>
</div>

<!-- ── CLIMATE–CASE CORRELATIONS ─────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#fce4e4">🔗</div>
    <div>
      <div class="sec-title">Climate–Case Correlations (Pearson r, lag 0)</div>
      <div class="sec-sub">Contemporaneous correlation between each climate variable and disease_cases per district</div>
    </div>
  </div>
  <div class="info-note">
    These are <strong>lag-0 Pearson correlations</strong>. The per-location feature-selection
    procedure in Exp 03 finds the optimal lag (0–8 weeks) for each variable — which is why
    weak lag-0 correlations don't prevent a variable from being selected as a useful feature.
    Cells are shaded green (positive) or red (negative) proportional to |r|.
  </div>
  <div style="overflow-x:auto;max-width:600px">
  <table>
    <thead>
      <tr>
        <th>District</th>
        <th style="text-align:center">Rainfall</th>
        <th style="text-align:center">Temperature</th>
        <th style="text-align:center">Humidity</th>
      </tr>
    </thead>
    <tbody>{corr_rows}</tbody>
  </table>
  </div>
  <div style="margin-top:12px;font-size:.75rem;color:var(--muted)">
    Green = positive correlation &nbsp;·&nbsp; Red = negative correlation &nbsp;·&nbsp;
    Intensity proportional to |r| &nbsp;·&nbsp; Weak correlations at lag 0 are expected —
    mosquito breeding takes 2–4 weeks after rainfall
  </div>
</div>

<!-- ── COMPLETENESS ───────────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-hdr">
    <div class="sec-icon" style="background:#e8f5e9">✅</div>
    <div>
      <div class="sec-title">Data Completeness &amp; Quality</div>
      <div class="sec-sub">Record-level completeness across all columns and districts</div>
    </div>
  </div>
  <div class="grid-3">
    <div>
      <div style="font-size:.75rem;font-weight:600;color:var(--sub0);margin-bottom:6px">disease_cases — completeness</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#2d8a4e">100%</div>
      <div class="completeness-bar"><div class="completeness-fill" style="width:100%"></div></div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:4px">728/728 records · 0 missing</div>
    </div>
    <div>
      <div style="font-size:.75rem;font-weight:600;color:var(--sub0);margin-bottom:6px">rainfall — completeness</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#2d8a4e">100%</div>
      <div class="completeness-bar"><div class="completeness-fill" style="width:100%"></div></div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:4px">728/728 records · ERA5 reanalysis</div>
    </div>
    <div>
      <div style="font-size:.75rem;font-weight:600;color:var(--sub0);margin-bottom:6px">temperature + humidity — completeness</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#2d8a4e">100%</div>
      <div class="completeness-bar"><div class="completeness-fill" style="width:100%"></div></div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:4px">728/728 records each · ERA5 reanalysis</div>
    </div>
  </div>
  <div style="margin-top:18px">
    <div style="font-size:.75rem;font-weight:600;color:var(--sub0);margin-bottom:10px">Zero-case weeks by district</div>
    <div id="zeroBar"></div>
  </div>
</div>

<div class="footer">
  Dataset Metadata · Comoros Climate-Health Forecasting · HISP Rwanda 2026 ·
  <a href="https://github.com/hisprwanda/sarimax-xgboost-comoros-malaria-prediction-model"
     style="color:var(--muted)">Research repository</a>
</div>

<script>
const DATA = {js};
const GC   = '#d8e8f5';
const TRAIN_WEEKS = {g['n_train']};

Chart.defaults.color = '#5a7a9e';
Chart.defaults.borderColor = GC;
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.tooltip.backgroundColor = '#fff';
Chart.defaults.plugins.tooltip.borderColor = '#cde0f5';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.titleColor = '#162035';
Chart.defaults.plugins.tooltip.bodyColor  = '#2d4a6e';
Chart.defaults.plugins.tooltip.titleFont  = {{family:"'Plus Jakarta Sans'",weight:'700'}};
Chart.defaults.plugins.tooltip.bodyFont   = {{family:"'JetBrains Mono'",size:12}};
Chart.defaults.plugins.tooltip.padding    = 10;

// ── Train/test split plugin ───────────────────────────────────────────────
function makeSplitPlugin(trainIdx, color) {{
  return {{
    id: 'splitBg_' + Math.random(),
    beforeDraw(chart) {{
      const {{ctx, chartArea, scales}} = chart;
      if (!chartArea) return;
      const x1 = scales.x.getPixelForValue(trainIdx - 0.5);
      ctx.save();
      ctx.fillStyle = 'rgba(230,240,255,0.45)';
      ctx.fillRect(chartArea.left, chartArea.top,
                   x1-chartArea.left, chartArea.bottom-chartArea.top);
      ctx.strokeStyle = 'rgba(90,122,158,0.5)';
      ctx.lineWidth=1.5; ctx.setLineDash([5,4]);
      ctx.beginPath(); ctx.moveTo(x1,chartArea.top); ctx.lineTo(x1,chartArea.bottom);
      ctx.stroke(); ctx.setLineDash([]); ctx.restore();
    }}
  }};
}}

// ── ① Timeline chart ─────────────────────────────────────────────────────
new Chart(document.getElementById('timelineChart'), {{
  type: 'line',
  plugins: [makeSplitPlugin(TRAIN_WEEKS, '#0479b5')],
  data: {{
    labels: DATA.weekly_labels,
    datasets: [
      {{
        label: 'Total cases (all districts)', type: 'line',
        data: DATA.weekly_cases,
        borderColor: '#1a5fb4', borderWidth: 2,
        backgroundColor: 'rgba(26,95,180,0.08)',
        fill: true, pointRadius: 0, yAxisID: 'y', order: 1,
      }},
      {{
        label: 'Mean rainfall (mm/wk)', type: 'bar',
        data: DATA.weekly_rain,
        backgroundColor: 'rgba(4,121,181,0.22)',
        borderColor: 'rgba(4,121,181,0.55)',
        borderWidth: 0.5, yAxisID: 'yr', order: 2,
      }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode:'index', intersect:false }},
    plugins: {{
      legend: {{ position:'top' }},
      tooltip: {{
        callbacks: {{
          title: ctx => ctx[0].label,
          label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.raw.toFixed(1),
        }}
      }}
    }},
    scales: {{
      x: {{ grid:{{display:false}}, ticks:{{maxTicksLimit:14,font:{{size:9}},maxRotation:30}} }},
      y: {{ beginAtZero:true, grid:{{color:GC}},
             title:{{display:true,text:'Cases / week',font:{{size:10}}}} }},
      yr: {{ position:'right', beginAtZero:true, grid:{{display:false}},
              title:{{display:true,text:'Rainfall (mm)',font:{{size:10}}}} }},
    }}
  }}
}});

// ── ② Seasonal chart ─────────────────────────────────────────────────────
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
new Chart(document.getElementById('seasonalChart'), {{
  type: 'bar',
  data: {{
    labels: MONTHS,
    datasets: [{{
      label: 'Mean cases',
      data: DATA.monthly_cases,
      backgroundColor: DATA.monthly_cases.map(v => {{
        const norm = v / Math.max(...DATA.monthly_cases);
        return `rgba(26,95,180,${{(0.25 + norm*0.65).toFixed(2)}})`;
      }}),
      borderColor: '#1a5fb4', borderWidth: 1, borderRadius: 4,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      legend:{{display:false}},
      tooltip:{{callbacks:{{label: ctx => ' Mean: ' + ctx.raw.toFixed(1) + ' cases/wk'}}}}
    }},
    scales:{{
      x:{{ grid:{{display:false}} }},
      y:{{ beginAtZero:true, grid:{{color:GC}},
            title:{{display:true,text:'Mean cases/week',font:{{size:10}}}} }}
    }}
  }}
}});

// ── ③ Climate chart ───────────────────────────────────────────────────────
new Chart(document.getElementById('climateChart'), {{
  type: 'line',
  data: {{
    labels: DATA.weekly_labels,
    datasets: [
      {{
        label: 'Mean temperature (°C)',
        data: DATA.weekly_temp,
        borderColor: '#e64d3d', borderWidth: 1.5, pointRadius: 0,
        fill: false, yAxisID: 'yt',
      }},
      {{
        label: 'Mean humidity (%)',
        data: DATA.weekly_hum,
        borderColor: '#0d7a80', borderWidth: 1.5, pointRadius: 0,
        fill: false, yAxisID: 'yh', borderDash:[4,2],
      }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{position:'top'}},
      tooltip:{{callbacks:{{label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.raw.toFixed(1)}}}}
    }},
    scales:{{
      x:{{ grid:{{display:false}}, ticks:{{maxTicksLimit:14,font:{{size:9}},maxRotation:30}} }},
      yt:{{ position:'left', grid:{{color:GC}}, min:18, max:30,
             title:{{display:true,text:'Temperature (°C)',font:{{size:10}}}} }},
      yh:{{ position:'right', grid:{{display:false}}, min:60, max:95,
             title:{{display:true,text:'Humidity (%)',font:{{size:10}}}} }},
    }}
  }}
}});

// ── ④ District sparkline cards ────────────────────────────────────────────
const sparkContainer = document.getElementById('sparkCards');
const locOrder = Object.keys(DATA.districts).sort((a,b) =>
  DATA.districts[b].total_cases - DATA.districts[a].total_cases);

locOrder.forEach(loc => {{
  const d = DATA.districts[loc];
  const c = d.color;
  const corrPill = (label, val) => {{
    const bg = val > 0.15 ? 'rgba(45,138,78,0.15)' : val < -0.15 ? 'rgba(192,21,53,0.15)' : 'rgba(90,122,158,0.12)';
    const tc = val > 0.15 ? '#2d8a4e' : val < -0.15 ? '#c01535' : '#5a7a9e';
    return `<span class="corr-pill" style="background:${{bg}};color:${{tc}}">${{label}} r=${{val.toFixed(2)}}</span>`;
  }};

  const card = document.createElement('div');
  card.className = 'dist-card';
  card.style.setProperty('--dc', c);
  card.innerHTML = `
    <div class="dist-name" style="color:${{c}}">${{loc}}</div>
    <div class="dist-metrics">
      <div class="dist-m">
        <div class="dist-m-lbl">Mean/wk</div>
        <div class="dist-m-val">${{d.mean}}</div>
      </div>
      <div class="dist-m">
        <div class="dist-m-lbl">Total cases</div>
        <div class="dist-m-val">${{d.total_cases.toLocaleString()}}</div>
      </div>
      <div class="dist-m">
        <div class="dist-m-lbl">Range</div>
        <div class="dist-m-val">${{d.min}}–${{d.max}}</div>
      </div>
      <div class="dist-m">
        <div class="dist-m-lbl">Peak month</div>
        <div class="dist-m-val">${{d.peak_month_name}}</div>
      </div>
    </div>
    <div class="dist-sparkwrap"><canvas id="spark_${{loc.replace(/ /g,'_').replace(/-/g,'_')}}"></canvas></div>
    <div class="corr-pills">
      ${{corrPill('🌧 Rain', d.corr_rain)}}
      ${{corrPill('🌡 Temp', d.corr_temp)}}
      ${{corrPill('💧 Hum', d.corr_hum)}}
    </div>`;
  sparkContainer.appendChild(card);

  // Build sparkline after card is in DOM
  requestAnimationFrame(() => {{
    const canvasId = 'spark_' + loc.replace(/ /g,'_').replace(/-/g,'_');
    const cvs = document.getElementById(canvasId);
    if (!cvs) return;
    new Chart(cvs, {{
      type: 'line',
      plugins: [makeSplitPlugin(TRAIN_WEEKS, c)],
      data: {{
        labels: d.week_labels,
        datasets: [{{
          data: d.cases_series,
          borderColor: c, borderWidth: 1.2,
          backgroundColor: c + '18',
          fill: true, pointRadius: 0, tension: 0.3,
        }}]
      }},
      options: {{
        responsive:true, maintainAspectRatio:false,
        plugins:{{ legend:{{display:false}}, tooltip:{{enabled:false}} }},
        scales:{{
          x:{{display:false}},
          y:{{display:false, beginAtZero:true}}
        }},
        animation:{{ duration:0 }}
      }}
    }});
  }});
}});

// ── ⑤ Zero-case bar (inline HTML) ────────────────────────────────────────
const zeroContainer = document.getElementById('zeroBar');
zeroContainer.style.display = 'grid';
zeroContainer.style.gridTemplateColumns = 'repeat(7,1fr)';
zeroContainer.style.gap = '10px';
locOrder.forEach(loc => {{
  const d   = DATA.districts[loc];
  const pct = d.zeros / d.n_weeks * 100;
  const clr = d.zeros === 0 ? '#2d8a4e' : '#c95f0b';
  zeroContainer.innerHTML += `
    <div>
      <div style="font-size:.65rem;font-weight:600;color:var(--sub0);margin-bottom:4px;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${{loc.split('-')[0]}}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;
                  color:${{clr}}">${{d.zeros}}</div>
      <div style="font-size:.68rem;color:var(--muted)">${{pct.toFixed(1)}}% of weeks</div>
    </div>`;
}});
</script>
</body>
</html>"""


def run():
    print("Dataset Metadata page")
    print("Loading data ...")
    data = build_data()

    g = data["global"]
    print(f"  {g['n_rows']} rows · {g['n_weeks']} weeks · {g['n_districts']} districts")
    print(f"  Train: {g['train_start']} → {g['train_end']}")
    print(f"  Test:  {g['test_start']} → {g['test_end']}")
    print(f"  Total cases: {g['total_cases']:,}")
    print(f"  Missing: {g['missing']}")

    print("\nBuilding HTML ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html(data)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved -> {OUT_HTML}")


if __name__ == "__main__":
    run()
