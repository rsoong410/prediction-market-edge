import pandas as pd
import numpy as np
import math

SERIES_NAMES = {
    'KXMLBHIT':       'Hits',
    'KXMLBHR':        'HR',
    'KXMLBHRR':       'H+R+RBI',
    'KXMLBTB':        'Total Bases',
    'KXMLBKS':        'Ks',
    'KXMLBGAME':      'Win',
    'KXMLBSPREAD':    'Spread',
    'KXMLBTOTAL':     'O/U Total',
    'KXMLBTEAMTOTAL': 'Team Total',
    'KXMLBF5SPREAD':  'F5 Spread',
}

def clean_label(label):
    if not isinstance(label, str) or label.strip() in ('', 'nan'):
        return ''
    for series, name in SERIES_NAMES.items():
        if f' {series} ' in label:
            player, rest = label.split(f' {series} ', 1)
            thresh = rest.rstrip('+')
            return f"{player.strip()} &nbsp;<span class='prop-tag'>{thresh}+ {name}</span>"
        if label.startswith(f'{series} '):
            rest = label[len(series):].strip().rstrip('+')
            return f"<span class='prop-tag'>{rest}+ {name}</span>"
    return label

def edge_class(ratio):
    if ratio >= 4:   return 'edge-5'
    if ratio >= 3:   return 'edge-4'
    if ratio >= 2:   return 'edge-3'
    if ratio >= 1.5: return 'edge-2'
    return 'edge-1'

def fmt_pct(v):
    try:
        f = float(v)
        if math.isnan(f): return '—'
        return f"{f*100:.1f}%"
    except: return '—'

def fmt_r(v):
    try: return f"{float(v):.3f}"
    except: return '—'

def fmt_num(v, decimals=2):
    try:
        f = float(v)
        if math.isnan(f): return '—'
        return f"{f:.{decimals}f}"
    except: return '—'

def build_rows(df, show_score=False):
    rows = []
    for _, r in df.iterrows():
        legs = [clean_label(r['leg_1']), clean_label(r['leg_2'])]
        if r['n_legs'] == 3 and isinstance(r.get('leg_3'), str) and r['leg_3'].strip():
            legs.append(clean_label(r['leg_3']))

        leg_html = '<br>'.join(f"<span class='leg'>{l}</span>" for l in legs if l)

        probs = [fmt_pct(r['avg_p_leg1']), fmt_pct(r['avg_p_leg2'])]
        if r['n_legs'] == 3 and not (isinstance(r.get('avg_p_leg3'), float) and math.isnan(r.get('avg_p_leg3', float('nan')))):
            probs.append(fmt_pct(r['avg_p_leg3']))
        prob_html = ' · '.join(probs)

        ec = edge_class(r['edge_ratio'])
        score_td = f"<td>{fmt_num(r.get('credibility_score', float('nan')), 2)}</td>" if show_score else ''

        rows.append(f"""
        <tr>
          <td><span class='team-badge'>{r['team']}</span></td>
          <td class='legs-cell'>{leg_html}</td>
          <td class='num'>{r['n_shared']}</td>
          <td class='num'>{fmt_r(r['r_max'])}</td>
          <td class='num'>{fmt_r(r['ci_lo'])}</td>
          <td class='num prob-col'>{prob_html}</td>
          <td class='num'>{fmt_pct(r['p_joint_observed'])}</td>
          <td class='num'>{fmt_pct(r['p_independent'])}</td>
          <td class='num {ec}'><strong>{fmt_num(r['edge_ratio'], 2)}×</strong></td>
          {score_td}
        </tr>""")
    return '\n'.join(rows)

def build_table(df, title, description, show_score=False):
    score_th = "<th>Score</th>" if show_score else ''
    rows = build_rows(df, show_score)
    return f"""
    <section>
      <h2>{title}</h2>
      <p class='desc'>{description}</p>
      <div class='table-wrap'>
      <table>
        <thead>
          <tr>
            <th>Team</th>
            <th>Props</th>
            <th title="Shared games">n</th>
            <th title="Max pairwise Pearson r on surprise series">r</th>
            <th title="Fisher 95% CI lower bound on r">CI&nbsp;lo</th>
            <th title="Average pregame implied probability per leg">Avg&nbsp;Prob</th>
            <th title="Observed joint hit rate in shared games">Joint&nbsp;Hit&nbsp;%</th>
            <th title="Expected joint if legs were independent">Indep&nbsp;%</th>
            <th title="Joint Hit % / Indep % — how much more likely than independence">Edge&nbsp;Ratio</th>
            {score_th}
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
      </div>
    </section>"""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; padding: 32px 24px; }
h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; color: #f8fafc; }
.subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 40px; }
section { margin-bottom: 52px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #f1f5f9;
     border-left: 3px solid #3b82f6; padding-left: 12px; margin-bottom: 6px; }
.desc { font-size: 0.8rem; color: #64748b; margin-bottom: 14px; padding-left: 15px; }
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid #1e293b; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
thead th { background: #1e293b; color: #94a3b8; font-weight: 600;
           padding: 10px 12px; text-align: left; white-space: nowrap;
           border-bottom: 1px solid #334155; cursor: help; }
tbody tr { border-bottom: 1px solid #1e293b; }
tbody tr:hover { background: #1a2235; }
td { padding: 9px 12px; vertical-align: middle; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.prob-col { color: #94a3b8; font-size: 0.78rem; }
.legs-cell { line-height: 1.7; }
.leg { display: block; }
.prop-tag { background: #1e3a5f; color: #7dd3fc; border-radius: 4px;
            padding: 1px 6px; font-size: 0.74rem; font-weight: 600;
            white-space: nowrap; }
.team-badge { background: #1e293b; color: #cbd5e1; border-radius: 4px;
              padding: 3px 8px; font-size: 0.78rem; font-weight: 700;
              letter-spacing: 0.05em; white-space: nowrap; }
.edge-1 { color: #fbbf24; }
.edge-2 { color: #fb923c; }
.edge-3 { color: #f87171; }
.edge-4 { color: #a78bfa; }
.edge-5 { color: #34d399; }
.legend { display: flex; gap: 18px; font-size: 0.78rem; margin-bottom: 32px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.glossary { background: #1e293b; border-radius: 8px; padding: 18px 22px;
            font-size: 0.8rem; color: #94a3b8; margin-bottom: 40px; }
.glossary h3 { font-size: 0.85rem; color: #cbd5e1; margin-bottom: 10px; }
.glossary dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
.glossary dt { color: #7dd3fc; font-weight: 600; }
"""

LEGEND_HTML = """
<div class='legend'>
  <span style='color:#94a3b8; font-size:0.78rem; align-self:center;'>Edge Ratio:</span>
  <div class='legend-item'><div class='legend-dot' style='background:#fbbf24'></div><span class='edge-1'>1.2–1.5×</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#fb923c'></div><span class='edge-2'>1.5–2×</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#f87171'></div><span class='edge-3'>2–3×</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#a78bfa'></div><span class='edge-4'>3–4×</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#34d399'></div><span class='edge-5'>4×+</span></div>
</div>"""

GLOSSARY_HTML = """
<div class='glossary'>
  <h3>Column guide</h3>
  <dl>
    <dt>n</dt><dd>Shared games — how many games both players had resolved markets</dd>
    <dt>r</dt><dd>Pearson correlation of surprise (outcome − implied prob) series</dd>
    <dt>CI lo</dt><dd>Fisher 95% CI lower bound on r — a conservative floor on the true correlation</dd>
    <dt>Avg Prob</dt><dd>Average Kalshi pregame implied probability per leg (market's estimate)</dd>
    <dt>Joint Hit %</dt><dd>Fraction of shared games where ALL legs hit — observed in-sample</dd>
    <dt>Indep %</dt><dd>Product of Avg Probs — what joint hit rate would be if legs were independent</dd>
    <dt>Edge Ratio</dt><dd>Joint Hit % ÷ Indep % — how much more likely than independence. Sportsbook parlays assume 1.0× (independence), so &gt;1.2× clears typical vig.</dd>
  </dl>
</div>"""

df_cred    = pd.read_csv('parlays_best_credibility.csv')
df_genuine = pd.read_csv('parlays_genuine.csv')
df_pract   = pd.read_csv('parlays_practical.csv')

# cap display at 30 rows each
cred_table    = build_table(df_cred.head(30),    "Best Credibility",    "Ranked by composite score: log(edge) × √n × CI-lower. Filters: r &lt; 0.95, n ≥ 8, p_joint &gt; 0, edge ≥ 1.2×.", show_score=True)
genuine_table = build_table(df_genuine.head(30), "Genuine Correlations", "r &lt; 0.95 (not a binary-probability artifact), n ≥ 10, joint event actually occurred, edge ≥ 1.0×. Sorted by CI lower bound.")
pract_table   = build_table(df_pract.head(30),   "Most Practical",       "Sorted by Joint Hit % — combinations that fire most often. Filter: edge ≥ 1.2×. Higher frequency = more betting opportunities.")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Parlay Correlations</title>
<style>{CSS}</style>
</head>
<body>
<h1>MLB Parlay Correlation Report</h1>
<p class='subtitle'>2026 season · Kalshi market data · BH-FDR q=10% · in-sample estimates</p>
{GLOSSARY_HTML}
{LEGEND_HTML}
{cred_table}
{genuine_table}
{pract_table}
</body>
</html>"""

with open('parlays_report.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Saved parlays_report.html")
