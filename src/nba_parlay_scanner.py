import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
import itertools
import os
import json
import datetime
import math

BASE_URL     = "https://api.elections.kalshi.com/trade-api/v2"
SEASON_START = "2025-10-01"   # NBA 2025-26 tip-off
SEASON_END   = datetime.date.today().isoformat()
CACHE_DIR    = os.path.join(os.path.dirname(__file__), ".cache_nba")
os.makedirs(CACHE_DIR, exist_ok=True)

# Confirmed via Kalshi API: KXNBAPTS-26JUN08SASNYK-SASVWEMBANYAMA1-35
# parts[1] = '26JUN08SASNYK'  (date + matchup, no time component)
# parts[2] = 'SASVWEMBANYAMA1' (team prefix + player code)
# parts[3] = '35'              (threshold)
NBA_PLAYER_SERIES = ['KXNBAPTS', 'KXNBAREB', 'KXNBAAST', 'KXNBA3PT']
NBA_TEAM_SERIES   = ['KXNBAGAME', 'KXNBASPREAD']

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
          "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

SERIES_DISPLAY = {
    'KXNBAPTS':    'Pts',
    'KXNBAREB':    'Reb',
    'KXNBAAST':    'Ast',
    'KXNBA3PT':    '3PM',
    'KXNBAGAME':   'Win',
    'KXNBASPREAD': 'Spread',
}

# All 30 NBA teams (standard Kalshi 3-char abbreviations)
TEAMS_TO_ANALYZE = [
    # East
    'NYK', 'BOS', 'MIL', 'CLE', 'IND', 'PHI', 'MIA', 'ATL',
    'CHI', 'TOR', 'BKN', 'ORL', 'DET', 'CHA', 'WAS',
    # West
    'SAS', 'OKC', 'DEN', 'LAL', 'DAL', 'HOU', 'GSW', 'LAC',
    'SAC', 'MIN', 'POR', 'MEM', 'NOP', 'PHX', 'UTA',
]

# ─── API HELPERS ──────────────────────────────────────────────────────────────

def get_markets(series, ticker_prefix=None, max_pages=100):
    params = {"limit": 1000, "series_ticker": series}
    if ticker_prefix:
        params["ticker_prefix"] = ticker_prefix
    markets = []
    for _ in range(max_pages):
        resp = requests.get(f"{BASE_URL}/markets", params=params)
        if resp.status_code != 200:
            print(f"  get_markets {series} HTTP {resp.status_code}")
            break
        data = resp.json()
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
    return markets


_markets_cache: dict = {}

def get_markets_cached(series):
    """Fetch NBA markets for a series, covering both 2025-26 season years."""
    if series in _markets_cache:
        return _markets_cache[series]
    today = datetime.date.today().isoformat()
    path  = os.path.join(CACHE_DIR, f"nba_{series}_{today}.json")
    if os.path.exists(path):
        with open(path) as f:
            markets = json.load(f)
    else:
        print(f"    fetching {series} (2025 prefix)...")
        m25 = get_markets(series, ticker_prefix=f"{series}-25", max_pages=100)
        print(f"    fetching {series} (2026 prefix)...")
        m26 = get_markets(series, ticker_prefix=f"{series}-26", max_pages=100)
        markets = m25 + m26
        with open(path, 'w') as f:
            json.dump(markets, f)
    _markets_cache[series] = markets
    return markets


# ─── NBA TICKER PARSING ───────────────────────────────────────────────────────

def nba_ticker_parse(ticker):
    """
    Returns (date_str YYYY-MM-DD, matchup str).
    Format: KXNBAPTS-26JUN08SASNYK-SASVWEMBANYAMA1-35
    parts[1] = '26JUN08SASNYK' -> year=2026, month=JUN, day=08, matchup=SASNYK
    """
    seg     = ticker.split('-')[1]   # '26JUN08SASNYK'
    year    = int(seg[0:2]) + 2000
    month   = MONTHS[seg[2:5]]
    day     = int(seg[5:7])
    matchup = seg[7:]                # 'SASNYK'
    return f"{year}-{month:02d}-{day:02d}", matchup


# ─── BUILD PROP DICT ──────────────────────────────────────────────────────────

def build_prop_dict(team_abbr, series_list=None):
    """
    Build prop dict from Kalshi NBA market results.
    Only fetches player series (no team win/spread) — team-level excluded from
    correlation analysis and not needed for player parlay discovery.
    """
    if series_list is None:
        series_list = NBA_PLAYER_SERIES

    all_markets = []
    for series in series_list:
        print(f"  fetching {series}...")
        for m in get_markets_cached(series):
            if m.get('result') not in ('yes', 'no'):
                continue
            parts = m['ticker'].split('-')
            # Player prop: parts[2] = 'SASVWEMBANYAMA1' — starts with team abbr
            if len(parts) < 4 or not parts[2].startswith(team_abbr):
                continue
            all_markets.append(m)

    print(f"  {len(all_markets)} resolved markets")
    if not all_markets:
        print("  0 resolved markets")
        return {}, {}

    groups  = {}
    skipped = 0
    for m in all_markets:
        ticker  = m['ticker']
        last_p  = m.get('last_price_dollars') or '0'
        price   = float(last_p)
        if price <= 0.01:
            skipped += 1
            continue

        outcome  = 1 if m['result'] == 'yes' else 0
        surprise = outcome - price
        parts    = ticker.split('-')
        series   = parts[0]
        title    = m.get('title', '')
        threshold = parts[3]

        date_str, matchup = nba_ticker_parse(ticker)
        game_key = f"{date_str}-{matchup}"

        # Player name from market title: "Victor Wembanyama: 35+ points" -> "Victor Wembanyama"
        name  = title.split(':')[0].strip() if ':' in title else parts[2]
        label = f"{name} {series} {threshold}+"

        groups.setdefault(label, []).append({
            'game_key': game_key, 'surprise': surprise,
            'outcome': outcome, 'price': price
        })

    used = len(all_markets) - skipped
    print(f"  {used}/{len(all_markets)} markets had last_price > $0.01 "
          f"({used/len(all_markets):.0%})")

    prop_dict = {}
    raw_dict  = {}
    for label, rows in groups.items():
        df = pd.DataFrame(rows).drop_duplicates('game_key').set_index('game_key')
        if len(df) >= 3:
            prop_dict[label] = df['surprise']
            raw_dict[label]  = df[['outcome', 'price']]

    print(f"  Built {len(prop_dict)} props total.")
    return prop_dict, raw_dict


# ─── CORRELATION & FISHER CI ──────────────────────────────────────────────────

def _label_entity(label):
    """
    Returns the 'thing being measured': player name for player props,
    series code for team-level props.
    Prevents same-player or same-team-series pairs.
    """
    for s in NBA_PLAYER_SERIES + NBA_TEAM_SERIES:
        if f' {s} ' in label:
            return label.split(f' {s} ')[0].strip()
        if label.startswith(f'{s} '):
            return s
    return label


def _corr_pvalue(r, n, one_sided=False):
    if n <= 3 or np.isnan(r):
        return 1.0
    z = np.arctanh(r) * np.sqrt(n - 3)
    if one_sided:
        return float(norm.sf(z))
    return float(2 * (1 - norm.cdf(abs(z))))


def fisher_ci(r, n, alpha=0.05):
    if n <= 3 or np.isnan(r):
        return (np.nan, np.nan)
    z     = np.arctanh(r)
    se    = 1 / np.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    return (round(np.tanh(z - zcrit * se), 3), round(np.tanh(z + zcrit * se), 3))


# ─── PARLAY FINDER ────────────────────────────────────────────────────────────

def parlay_finder(prop_dict, n_legs=2, top_n=10, min_n=20, fdr_q=0.10):
    """
    BH-FDR corrected search for positively-correlated player prop pairs.
    Filters: n >= min_n, one-sided p-values (r > 0), BH-FDR at fdr_q.
    """
    labels           = list(prop_dict.keys())
    _team_series_set = set(NBA_TEAM_SERIES)

    pair_cache = {}
    for a, b in itertools.combinations(labels, 2):
        ea, eb = _label_entity(a), _label_entity(b)
        if ea == eb:
            continue
        if ea in _team_series_set or eb in _team_series_set:
            continue
        shared = pd.concat([prop_dict[a], prop_dict[b]], axis=1, join='inner').dropna()
        n = len(shared)
        if n < min_n:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            r = shared.iloc[:, 0].corr(shared.iloc[:, 1]) if n > 2 else np.nan
        lo, hi = fisher_ci(r, n)
        p = _corr_pvalue(r, n, one_sided=True)
        pair_cache[(a, b)] = {'r': r, 'n': n, 'ci': (lo, hi), 'p': p}

    # BH-FDR across all valid pairs
    if pair_cache:
        pair_keys = list(pair_cache.keys())
        pvals     = np.array([pair_cache[k]['p'] for k in pair_keys])
        m_pairs   = len(pvals)
        order     = np.argsort(pvals)
        sorted_p  = pvals[order]
        below     = sorted_p <= fdr_q * np.arange(1, m_pairs + 1) / m_pairs
        if below.any():
            k_max    = int(np.max(np.where(below)[0])) + 1
            fdr_pass = {pair_keys[order[i]] for i in range(k_max)}
        else:
            fdr_pass = set()
    else:
        fdr_pass = set()

    def _lookup(a, b):
        return pair_cache.get((a, b)) or pair_cache.get((b, a))

    def _key(a, b):
        return (a, b) if (a, b) in pair_cache else (b, a)

    results = []
    for combo in itertools.combinations(labels, n_legs):
        pair_stats = []
        valid = True
        for a, b in itertools.combinations(combo, 2):
            ps = _lookup(a, b)
            if ps is None or _key(a, b) not in fdr_pass:
                valid = False
                break
            pair_stats.append({'pair': (a, b), **ps})
        if not valid:
            continue
        min_lo = min(ps['ci'][0] for ps in pair_stats)
        results.append({'combo': combo, 'min_lo': min_lo, 'pairs': pair_stats})

    results.sort(key=lambda x: x['min_lo'], reverse=True)

    print(f"\nTop {top_n} {n_legs}-leg parlays (BH-FDR q<={fdr_q:.0%}, min_n={min_n}):")
    if not results:
        print("  (none)")
    for i, res in enumerate(results[:top_n], 1):
        print(f"\n#{i}  min_CI_lo={res['min_lo']:.3f}")
        for leg in res['combo']:
            print(f"  * {leg}")
        for p in res['pairs']:
            a, b = p['pair']
            print(f"    [{a}] x [{b}]  r={p['r']:.3f}  n={p['n']}  p={p['p']:.4f}")

    return results


def parlay_edge_rows(results, raw_dict, team):
    rows = []
    for res in results:
        combo  = res['combo']
        n_legs = len(combo)
        frames = []
        for i, label in enumerate(combo):
            if label not in raw_dict:
                break
            frames.append(raw_dict[label][['outcome', 'price']].rename(
                columns={'outcome': f'out{i}', 'price': f'p{i}'}))
        if len(frames) != n_legs:
            continue
        shared = frames[0].join(frames[1:], how='inner').dropna()
        if shared.empty:
            continue
        n_shared    = len(shared)
        mean_prices = [shared[f'p{i}'].mean() for i in range(n_legs)]
        all_yes     = (shared[[f'out{i}' for i in range(n_legs)]] == 1).all(axis=1)
        p_joint     = all_yes.mean()
        p_indep     = float(np.prod(mean_prices))
        ratio       = p_joint / p_indep if p_indep > 0 else float('nan')
        rows.append({
            'team':             team,
            'n_legs':           n_legs,
            'leg_1':            combo[0],
            'leg_2':            combo[1],
            'leg_3':            combo[2] if n_legs > 2 else '',
            'r_min':            round(min(p['r'] for p in res['pairs']), 3),
            'r_max':            round(max(p['r'] for p in res['pairs']), 3),
            'ci_lo':            round(res['min_lo'], 3),
            'n_shared':         n_shared,
            'avg_p_leg1':       round(mean_prices[0], 3),
            'avg_p_leg2':       round(mean_prices[1], 3),
            'avg_p_leg3':       round(mean_prices[2], 3) if n_legs > 2 else '',
            'p_joint_observed': round(p_joint, 4),
            'p_independent':    round(p_indep, 4),
            'edge_ratio':       round(ratio, 2),
        })
    return rows


# ─── VALIDATION HELPERS ───────────────────────────────────────────────────────

def find_season_midpoint(team_raw_dicts):
    all_keys = []
    for raw_dict in team_raw_dicts.values():
        for df in raw_dict.values():
            all_keys.extend(df.index.tolist())
    sorted_keys = sorted(set(all_keys))
    mid = sorted_keys[len(sorted_keys) // 2]
    print(f"  first: {sorted_keys[0]}  mid: {mid}  last: {sorted_keys[-1]}")
    print(f"  ({len(sorted_keys)} unique game slots)")
    return mid


def split_raw_dict(raw_dict, cutoff, min_n=3):
    train, test = {}, {}
    for label, df in raw_dict.items():
        before = df[df.index <= cutoff]
        after  = df[df.index >  cutoff]
        if len(before) >= min_n:
            train[label] = before
        if len(after)  >= min_n:
            test[label]  = after
    return train, test


def raw_to_prop_dict(raw_dict):
    return {lbl: df['outcome'] - df['price'] for lbl, df in raw_dict.items()}


def pair_edge_on(label_a, label_b, raw_dict, min_n=3):
    if label_a not in raw_dict or label_b not in raw_dict:
        return np.nan, 0
    da = raw_dict[label_a].rename(columns={'outcome': 'out_a', 'price': 'p_a'})
    db = raw_dict[label_b].rename(columns={'outcome': 'out_b', 'price': 'p_b'})
    shared = da.join(db, how='inner').dropna()
    n = len(shared)
    if n < min_n:
        return np.nan, n
    p_joint = ((shared['out_a'] == 1) & (shared['out_b'] == 1)).mean()
    p_indep = shared['p_a'].mean() * shared['p_b'].mean()
    return (p_joint / p_indep if p_indep > 0 else np.nan), n


# ─── HTML HELPERS ─────────────────────────────────────────────────────────────

def clean_label(label):
    if not isinstance(label, str) or label.strip() in ('', 'nan'):
        return ''
    for series, name in SERIES_DISPLAY.items():
        if f' {series} ' in label:
            player, rest = label.split(f' {series} ', 1)
            thresh = rest.rstrip('+')
            return (f"{player.strip()} &nbsp;"
                    f"<span class='prop-tag'>{thresh}+ {name}</span>")
        if label.startswith(f'{series} '):
            rest = label[len(series):].strip().rstrip('+')
            return f"<span class='prop-tag'>{rest} {name}</span>"
    return label


def edge_class(ratio):
    try:
        r = float(ratio)
        if r >= 4:   return 'edge-5'
        if r >= 3:   return 'edge-4'
        if r >= 2:   return 'edge-3'
        if r >= 1.5: return 'edge-2'
        return 'edge-1'
    except:
        return 'edge-1'


def fmt_pct(v):
    try:
        f = float(v)
        if math.isnan(f): return '--'
        return f"{f*100:.1f}%"
    except:
        return '--'


def fmt_r(v):
    try:    return f"{float(v):.3f}"
    except: return '--'


def fmt_num(v, d=2):
    try:
        f = float(v)
        if math.isnan(f): return '--'
        return f"{f:.{d}f}"
    except:
        return '--'


# ─── HTML TEMPLATES ───────────────────────────────────────────────────────────

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; padding: 32px 24px; }
h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; color: #f8fafc; }
.subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 40px; }
section { margin-bottom: 52px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #f1f5f9;
     border-left: 3px solid #8b5cf6; padding-left: 12px; margin-bottom: 6px; }
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
.prop-tag { background: #2d1b69; color: #c4b5fd; border-radius: 4px;
            padding: 1px 6px; font-size: 0.74rem; font-weight: 600; white-space: nowrap; }
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
.glossary dt { color: #c4b5fd; font-weight: 600; }
"""

LEGEND_HTML = """<div class='legend'>
  <span style='color:#94a3b8;font-size:0.78rem;align-self:center;'>Edge Ratio:</span>
  <div class='legend-item'><div class='legend-dot' style='background:#fbbf24'></div><span class='edge-1'>1.2-1.5x</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#fb923c'></div><span class='edge-2'>1.5-2x</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#f87171'></div><span class='edge-3'>2-3x</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#a78bfa'></div><span class='edge-4'>3-4x</span></div>
  <div class='legend-item'><div class='legend-dot' style='background:#34d399'></div><span class='edge-5'>4x+</span></div>
</div>"""

GLOSSARY_HTML = """<div class='glossary'>
  <h3>Column guide</h3>
  <dl>
    <dt>n</dt><dd>Shared games -- games where both players had resolved Kalshi markets</dd>
    <dt>r</dt><dd>Pearson correlation of surprise series (outcome - implied prob)</dd>
    <dt>CI lo</dt><dd>Fisher 95% CI lower bound on r (conservative floor on true correlation)</dd>
    <dt>Avg Prob</dt><dd>Average Kalshi pregame implied probability per leg</dd>
    <dt>Joint Hit %</dt><dd>Fraction of shared games where ALL legs hit -- observed in-sample</dd>
    <dt>Indep %</dt><dd>Product of Avg Probs -- expected joint hit rate under independence</dd>
    <dt>Edge Ratio</dt><dd>Joint Hit % / Indep % -- above 1.2x clears typical parlay vig</dd>
  </dl>
</div>"""


def _table_rows(df, show_score=False):
    rows = []
    for _, r in df.iterrows():
        legs = [clean_label(r['leg_1']), clean_label(r['leg_2'])]
        if r['n_legs'] == 3 and isinstance(r.get('leg_3'), str) and r['leg_3'].strip():
            legs.append(clean_label(r['leg_3']))
        leg_html = '<br>'.join(f"<span class='leg'>{l}</span>" for l in legs if l)

        probs = [fmt_pct(r['avg_p_leg1']), fmt_pct(r['avg_p_leg2'])]
        try:
            v3 = float(r.get('avg_p_leg3', float('nan')))
            if not math.isnan(v3) and r['n_legs'] == 3:
                probs.append(fmt_pct(v3))
        except:
            pass

        ec       = edge_class(r['edge_ratio'])
        score_td = (f"<td class='num'>{fmt_num(r.get('credibility_score', float('nan')))}</td>"
                    if show_score else '')
        rows.append(f"""
        <tr>
          <td><span class='team-badge'>{r['team']}</span></td>
          <td class='legs-cell'>{leg_html}</td>
          <td class='num'>{r['n_shared']}</td>
          <td class='num'>{fmt_r(r['r_max'])}</td>
          <td class='num'>{fmt_r(r['ci_lo'])}</td>
          <td class='num prob-col'>{' · '.join(probs)}</td>
          <td class='num'>{fmt_pct(r['p_joint_observed'])}</td>
          <td class='num'>{fmt_pct(r['p_independent'])}</td>
          <td class='num {ec}'><strong>{fmt_num(r['edge_ratio'])}x</strong></td>
          {score_td}
        </tr>""")
    return '\n'.join(rows)


def _build_section(df, title, desc, show_score=False):
    score_th = "<th>Score</th>" if show_score else ''
    return f"""
    <section>
      <h2>{title}</h2>
      <p class='desc'>{desc}</p>
      <div class='table-wrap'>
      <table>
        <thead><tr>
          <th>Team</th><th>Props</th>
          <th title='Shared games'>n</th>
          <th title='Max pairwise r'>r</th>
          <th title='Fisher 95% CI lower'>CI lo</th>
          <th title='Avg pregame implied prob per leg'>Avg Prob</th>
          <th title='Observed joint hit rate'>Joint Hit %</th>
          <th title='Expected under independence'>Indep %</th>
          <th title='Joint / Indep'>Edge Ratio</th>
          {score_th}
        </tr></thead>
        <tbody>{_table_rows(df, show_score)}</tbody>
      </table></div>
    </section>"""


def make_full_season_html(df_cred, df_genuine, df_pract):
    t1 = _build_section(
        df_cred.head(30), "Best Credibility",
        "Ranked by log(edge) x sqrt(n) x CI-lower. "
        "Filters: r &lt; 0.95, n &ge; 8, p_joint &gt; 0, edge &ge; 1.2x.",
        show_score=True)
    t2 = _build_section(
        df_genuine.head(30), "Genuine Correlations",
        "r &lt; 0.95, n &ge; 10, joint event occurred, edge &ge; 1.0x. Sorted by CI lower bound.")
    t3 = _build_section(
        df_pract.head(30), "Most Practical",
        "Sorted by Joint Hit % -- combinations that fire most often. Filter: edge &ge; 1.2x.")
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>NBA Parlay Correlations</title>
<style>{BASE_CSS}</style>
</head>
<body>
<h1>NBA Parlay Correlation Report</h1>
<p class='subtitle'>2025-26 season &middot; Kalshi market data &middot; BH-FDR q=10% &middot; in-sample estimates &middot; through {SEASON_END}</p>
{GLOSSARY_HTML}
{LEGEND_HTML}
{t1}
{t2}
{t3}
</body>
</html>"""


def make_validation_html(val_df, cutoff):
    valid   = val_df.dropna(subset=['test_edge']) if len(val_df) > 0 else val_df
    n_total = len(val_df)
    n_test  = len(valid)
    n_held  = int(valid['edge_held'].sum()) if n_test > 0 else 0

    med_train_raw = val_df['train_edge'].median() if n_total > 0 else float('nan')
    med_test_raw  = valid['test_edge'].median()    if n_test  > 0 else float('nan')
    med_train = fmt_num(med_train_raw) if not (isinstance(med_train_raw, float) and math.isnan(med_train_raw)) else '--'
    med_test  = fmt_num(med_test_raw)  if not (isinstance(med_test_raw,  float) and math.isnan(med_test_raw))  else '--'

    extra_css = """
.stat-card { display: inline-block; background: #1e293b; border-radius: 8px;
             padding: 16px 24px; margin: 0 12px 16px 0; text-align: center; min-width: 120px; }
.stat-card .val { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
.stat-card .lbl { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
.ec { color: #f87171; }
.es { color: #fbbf24; }
.eh { color: #34d399; }
.ex { color: #a78bfa; }
"""

    def row_class(te):
        try:
            t = float(te)
            if math.isnan(t): return 'ec'
            if t >= 2.0: return 'ex'
            if t >= 1.2: return 'eh'
            if t >= 0.5: return 'es'
            return 'ec'
        except:
            return 'ec'

    tbody = []
    if n_test > 0:
        for _, r in valid.sort_values('test_edge', ascending=False).iterrows():
            rc = row_class(r['test_edge'])
            tbody.append(f"""
        <tr>
          <td><span class='team-badge'>{r['team']}</span></td>
          <td style='font-size:0.78rem'>{r['leg_1']}</td>
          <td style='font-size:0.78rem'>{r['leg_2']}</td>
          <td class='num'>{fmt_r(r['train_r'])}</td>
          <td class='num'>{r['train_n']}</td>
          <td class='num'>{fmt_num(r['train_edge'])}x</td>
          <td class='num'>{r['test_n']}</td>
          <td class='num {rc}'><strong>{fmt_num(r['test_edge'])}x</strong></td>
        </tr>""")
    else:
        tbody.append("<tr><td colspan='8' style='text-align:center;color:#64748b;"
                     "padding:32px'>No pairs selected in training</td></tr>")

    buckets = []
    if n_test > 0:
        te = valid['test_edge']
        for lbl, mask in [
            ('collapsed (<0.5x)',  te < 0.5),
            ('shrunk (0.5-1.0x)',  (te >= 0.5) & (te < 1.0)),
            ('flat (1.0-1.2x)',    (te >= 1.0) & (te < 1.2)),
            ('held (1.2-2.0x)',    (te >= 1.2) & (te < 2.0)),
            ('strong (2.0x+)',     te >= 2.0),
        ]:
            n = mask.sum()
            buckets.append(f"<li>{lbl}: <strong>{n}</strong> "
                           f"({n/n_test:.0%})</li>")
    bucket_html = f"<ul style='color:#94a3b8;font-size:0.82rem;margin-top:10px;padding-left:18px'>{''.join(buckets)}</ul>" if buckets else ''

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>NBA Half-Season Validation</title>
<style>{BASE_CSS}{extra_css}</style>
</head>
<body>
<h1>NBA Parlay Correlation -- Half-Season Validation</h1>
<p class='subtitle'>Train on games up to <strong>{cutoff}</strong> &rarr; measure edge on untouched second-half games</p>

<div style='margin: 32px 0'>
  <div class='stat-card'><div class='val'>{n_total}</div><div class='lbl'>Pairs selected (train)</div></div>
  <div class='stat-card'><div class='val'>{n_test}</div><div class='lbl'>Pairs with test data</div></div>
  <div class='stat-card'><div class='val'>{n_held}</div><div class='lbl'>Edge held (&ge;1.2x)</div></div>
  <div class='stat-card'><div class='val'>{med_train}x</div><div class='lbl'>Median train edge</div></div>
  <div class='stat-card'><div class='val'>{med_test}x</div><div class='lbl'>Median test edge</div></div>
</div>

{bucket_html}

<section style='margin-top:32px'>
  <h2>Validated Pairs (sorted by test edge)</h2>
  <p class='desc'>Each pair was discovered using only first-half games. Edge ratio then measured on held-out second-half games it never saw.</p>
  <div class='table-wrap'>
  <table>
    <thead><tr>
      <th>Team</th><th>Leg 1</th><th>Leg 2</th>
      <th title='In-sample r'>Train r</th>
      <th title='Training games n'>Train n</th>
      <th title='Edge ratio in training'>Train Edge</th>
      <th title='Test games n'>Test n</th>
      <th title='Edge ratio on held-out games'>Test Edge</th>
    </tr></thead>
    <tbody>{''.join(tbody)}</tbody>
  </table></div>
</section>

<div class='glossary' style='margin-top:40px'>
  <h3>What does collapse mean?</h3>
  <dl>
    <dt>&lt;0.5x (red)</dt><dd>Edge collapsed -- the correlation was in-sample noise</dd>
    <dt>0.5-1.0x (yellow)</dt><dd>Shrunk but positive -- might be partial signal, might be noise</dd>
    <dt>1.0-1.2x</dt><dd>Flat -- held its ground but not enough to beat vig</dd>
    <dt>1.2x+ (green)</dt><dd>Edge held -- this is what genuine correlation looks like out-of-sample</dd>
    <dt>2.0x+ (purple)</dt><dd>Strong hold -- rare but indicates real structural correlation</dd>
  </dl>
</div>

</body>
</html>"""


# ─── MAIN RUN ─────────────────────────────────────────────────────────────────

significant:  dict = {}
all_csv_rows: list = []
all_team_raw: dict = {}

for team in TEAMS_TO_ANALYZE:
    print(f"\n{'='*60}")
    print(f"  {team}")
    print('='*60)

    prop_dict, raw_dict = build_prop_dict(team)
    if len(prop_dict) < 5:
        print(f"  skipping -- only {len(prop_dict)} props")
        continue

    all_team_raw[team] = raw_dict

    print(f"\n  {len(prop_dict)} props")
    two_leg   = parlay_finder(prop_dict, n_legs=2, top_n=5, min_n=10)
    three_leg = parlay_finder(prop_dict, n_legs=3, top_n=3, min_n=10)

    if two_leg or three_leg:
        significant[team] = {'2-leg': two_leg, '3-leg': three_leg}
        all_csv_rows.extend(parlay_edge_rows(two_leg,   raw_dict, team))
        all_csv_rows.extend(parlay_edge_rows(three_leg, raw_dict, team))

print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
if significant:
    for team, res in significant.items():
        print(f"  {team}: {len(res['2-leg'])} two-leg, {len(res['3-leg'])} three-leg")
else:
    print("  No significant combinations found.")

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

# ─── CSV EXPORT ───────────────────────────────────────────────────────────────

if all_csv_rows:
    df_all = pd.DataFrame(all_csv_rows)
    df_all.to_csv(os.path.join(OUT_DIR, 'nba_parlays.csv'), index=False)
    print(f"\nSaved {len(df_all)} rows to nba_parlays.csv")

    # Best credibility: log(edge) * sqrt(n) * CI_lower
    df_cred = df_all[
        (df_all['r_max'] < 0.95) &
        (df_all['p_joint_observed'] > 0) &
        (df_all['edge_ratio'] >= 1.2) &
        (df_all['n_shared'] >= 8)
    ].copy()
    df_cred['credibility_score'] = (
        np.log(df_cred['edge_ratio'].clip(lower=1.001)) *
        np.sqrt(df_cred['n_shared']) *
        df_cred['ci_lo'].clip(lower=0)
    )
    df_cred = df_cred.sort_values('credibility_score', ascending=False)
    df_cred.to_csv(os.path.join(OUT_DIR, 'nba_parlays_best_credibility.csv'), index=False)
    print(f"Saved {len(df_cred)} rows to nba_parlays_best_credibility.csv")

    # Genuine: r < 0.95, n >= 10, joint event occurred, edge >= 1.0
    df_genuine = df_all[
        (df_all['r_max'] < 0.95) &
        (df_all['n_shared'] >= 10) &
        (df_all['p_joint_observed'] > 0) &
        (df_all['edge_ratio'] >= 1.0)
    ].copy()
    df_genuine = df_genuine.sort_values('ci_lo', ascending=False)
    df_genuine.to_csv(os.path.join(OUT_DIR, 'nba_parlays_genuine.csv'), index=False)
    print(f"Saved {len(df_genuine)} rows to nba_parlays_genuine.csv")

    # Practical: highest joint hit rate, edge >= 1.2
    df_pract = df_all[
        (df_all['p_joint_observed'] > 0) &
        (df_all['edge_ratio'] >= 1.2)
    ].copy()
    df_pract = df_pract.sort_values('p_joint_observed', ascending=False)
    df_pract.to_csv(os.path.join(OUT_DIR, 'nba_parlays_practical.csv'), index=False)
    print(f"Saved {len(df_pract)} rows to nba_parlays_practical.csv")

    # Full season HTML
    html_full = make_full_season_html(df_cred, df_genuine, df_pract)
    with open(os.path.join(OUT_DIR, 'nba_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_full)
    print("Saved nba_report.html")
else:
    print("\nNo significant combinations found -- skipping CSVs and HTML.")
    df_cred = df_genuine = df_pract = pd.DataFrame()

# ─── TRAIN / TEST VALIDATION ──────────────────────────────────────────────────

print("\n\n" + "="*60)
print("TRAIN/TEST VALIDATION")
print("="*60)

if not all_team_raw:
    print("No team data -- skipping validation.")
else:
    cutoff = find_season_midpoint(all_team_raw)
    print(f"Cutoff: {cutoff}\n")

    val_rows = []
    for team, raw_dict in all_team_raw.items():
        train_raw, test_raw = split_raw_dict(raw_dict, cutoff, min_n=5)
        train_prop = raw_to_prop_dict(train_raw)
        if len(train_prop) < 5:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            train_results = parlay_finder(
                train_prop, n_legs=2, top_n=5, min_n=8, fdr_q=0.10)
        for res in train_results:
            for ps in res['pairs']:
                a, b = ps['pair']
                train_edge, train_n = pair_edge_on(a, b, train_raw)
                test_edge,  test_n  = pair_edge_on(a, b, test_raw)
                val_rows.append({
                    'team':       team,
                    'leg_1':      a,
                    'leg_2':      b,
                    'train_r':    round(ps['r'], 3),
                    'train_n':    train_n,
                    'train_edge': round(train_edge, 2) if not np.isnan(train_edge) else np.nan,
                    'test_n':     test_n,
                    'test_edge':  round(test_edge, 2)  if not np.isnan(test_edge)  else np.nan,
                    'edge_held':  (not np.isnan(test_edge)) and test_edge >= 1.2,
                })

    empty_cols = ['team','leg_1','leg_2','train_r','train_n',
                  'train_edge','test_n','test_edge','edge_held']
    val_df = (pd.DataFrame(val_rows) if val_rows
              else pd.DataFrame(columns=empty_cols))
    val_df = val_df.drop_duplicates(subset=['team','leg_1','leg_2'])
    val_df.to_csv(os.path.join(OUT_DIR, 'nba_validation.csv'), index=False)

    valid = val_df.dropna(subset=['test_edge'])
    print(f"Pairs selected in training:  {len(val_df)}")
    print(f"Pairs with test data:        {len(valid)}")
    print(f"Pairs where test_edge>=1.2:  {valid['edge_held'].sum()}")
    if len(valid) > 0:
        print(f"\nMedian train edge:  {val_df['train_edge'].median():.2f}x")
        print(f"Median test  edge:  {valid['test_edge'].median():.2f}x")
        print("\nEdge ratio: train vs test")
        te = valid['test_edge']
        for lbl, mask in [
            ('collapsed  (<0.5x)',  te < 0.5),
            ('shrunk  (0.5-1.0x)', (te >= 0.5) & (te < 1.0)),
            ('flat    (1.0-1.2x)', (te >= 1.0) & (te < 1.2)),
            ('held    (1.2-2.0x)', (te >= 1.2) & (te < 2.0)),
            ('strong  (2.0x+)',    te >= 2.0),
        ]:
            n = mask.sum()
            print(f"  {lbl}: {n} ({n/len(valid):.0%})")

    # Validation HTML
    val_html = make_validation_html(val_df, cutoff)
    with open(os.path.join(OUT_DIR, 'nba_validation_report.html'), 'w', encoding='utf-8') as f:
        f.write(val_html)
    print("\nSaved nba_validation_report.html")
    print("Saved nba_validation.csv")
