import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, json, math, datetime, itertools, time
import pandas as pd
import numpy as np
from scipy.stats import norm
from nba_api.stats.endpoints import leaguegamelog
import requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BASE_KALSHI   = "https://api.elections.kalshi.com/trade-api/v2"
SRC_DIR       = os.path.dirname(__file__)
CACHE_NBA_API = os.path.join(SRC_DIR, ".cache_nba_api")
CACHE_KALSHI  = os.path.join(SRC_DIR, ".cache_nba")
OUT_DIR       = os.path.join(SRC_DIR, "..")
NBA_SEASON    = "2025-26"
FINALS_TEAMS  = {'NYK', 'SAS'}
MIN_N         = 30   # minimum shared regular-season games

os.makedirs(CACHE_NBA_API, exist_ok=True)

STAT_COLS = ['PTS', 'REB', 'AST', 'FG3M']
STAT_LABEL = {'PTS': 'Pts', 'REB': 'Reb', 'AST': 'Ast', 'FG3M': '3PM'}

NBA_PLAYER_SERIES = ['KXNBAPTS', 'KXNBAREB', 'KXNBAAST', 'KXNBA3PT']
SERIES_TO_STAT    = {'KXNBAPTS': 'PTS', 'KXNBAREB': 'REB', 'KXNBAAST': 'AST', 'KXNBA3PT': 'FG3M'}
MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
          "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

# ─── REGULAR SEASON DATA (nba_api) ───────────────────────────────────────────

def fetch_regular_season_logs() -> pd.DataFrame:
    existing = sorted(f for f in os.listdir(CACHE_NBA_API) if f.startswith(f"logs_{NBA_SEASON}_"))
    if existing:
        path = os.path.join(CACHE_NBA_API, existing[-1])
        print(f"  Loading cached game logs: {existing[-1]}")
        return pd.read_csv(path)
    today = datetime.date.today().isoformat()
    path  = os.path.join(CACHE_NBA_API, f"logs_{NBA_SEASON}_{today}.csv")
    print("  Fetching regular season game logs from stats.nba.com...")
    log = leaguegamelog.LeagueGameLog(
        season=NBA_SEASON,
        season_type_all_star='Regular Season',
        player_or_team_abbreviation='P',
        timeout=60,
    )
    df = log.get_data_frames()[0]
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {os.path.basename(path)}")
    return df


def build_player_game_dict(df: pd.DataFrame) -> dict:
    """Returns {player_name: {'team': str, 'games': DataFrame(index=date, cols=STAT_COLS)}}"""
    players = {}
    for _, row in df.iterrows():
        name = str(row['PLAYER_NAME'])
        date = str(row['GAME_DATE'])[:10]
        team = str(row['TEAM_ABBREVIATION'])
        stats = {s: int(row.get(s) or 0) for s in STAT_COLS}
        if name not in players:
            players[name] = []
        players[name].append({'date': date, 'team': team, **stats})

    result = {}
    for name, rows in players.items():
        sub = pd.DataFrame(rows)
        primary_team = sub['team'].value_counts().index[0]
        games = (sub[sub['team'] == primary_team]
                 .drop_duplicates('date')
                 .set_index('date')
                 [STAT_COLS])
        result[name] = {'team': primary_team, 'games': games}
    return result


# ─── KALSHI CACHE LOADER ─────────────────────────────────────────────────────

def nba_ticker_parse(ticker):
    seg  = ticker.split('-')[1]
    year = int(seg[0:2]) + 2000
    mon  = MONTHS[seg[2:5]]
    day  = int(seg[5:7])
    return f"{year}-{mon:02d}-{day:02d}", seg[7:]


def load_kalshi_markets() -> dict:
    """Returns {player_name: {series: {threshold_int: {live_price, live_ticker, ...}}}}"""
    out = {}
    for series in NBA_PLAYER_SERIES:
        files = sorted(f for f in os.listdir(CACHE_KALSHI) if f.startswith(f"nba_{series}_"))
        if not files:
            continue
        with open(os.path.join(CACHE_KALSHI, files[-1])) as f:
            markets = json.load(f)

        for m in markets:
            ticker = m.get('ticker', '')
            parts  = ticker.split('-')
            if len(parts) < 4:
                continue
            title = m.get('title', '')
            name  = title.split(':')[0].strip() if ':' in title else parts[2]
            try:
                thresh = int(parts[3])
            except ValueError:
                continue
            lp = m.get('last_price_dollars')
            if not lp:
                continue
            try:
                price = float(lp)
            except (TypeError, ValueError):
                continue
            if price <= 0.01:
                continue

            result  = m.get('result', '')
            is_open = result not in ('yes', 'no', 'scalar')

            bucket = (out
                      .setdefault(name, {})
                      .setdefault(series, {})
                      .setdefault(thresh, {'live_price': None, 'live_ticker': None}))
            if is_open:
                bucket['live_price']  = price
                bucket['live_ticker'] = ticker

    return out


# ─── ORDERBOOK / BID-ASK ─────────────────────────────────────────────────────

def fetch_bid_ask(ticker: str) -> tuple:
    """Returns (bid, ask) for yes side from live API."""
    try:
        r = requests.get(f"{BASE_KALSHI}/markets/{ticker}", timeout=10)
        if r.status_code != 200:
            return None, None
        m = r.json().get('market', {})
        bid = m.get('yes_bid_dollars')
        ask = m.get('yes_ask_dollars')
        if bid is not None and ask is not None:
            return float(bid), float(ask)
        lp = m.get('last_price_dollars')
        if lp:
            v = float(lp)
            return v, v
        return None, None
    except Exception:
        return None, None


# ─── MATH HELPERS ────────────────────────────────────────────────────────────

def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    if n <= 3 or np.isnan(r):
        return (np.nan, np.nan)
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return (round(np.tanh(z - zc * se), 3), round(np.tanh(z + zc * se), 3))


def _corr_pvalue(r: float, n: int) -> float:
    if n <= 3 or np.isnan(r):
        return 1.0
    return float(norm.sf(np.arctanh(r) * np.sqrt(n - 3)))


def bivariate_bernoulli_edge(r: float, p_a: float, p_b: float) -> float:
    if np.isnan(r) or p_a <= 0 or p_b <= 0:
        return np.nan
    p_joint = p_a * p_b + r * np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    return float(np.clip(p_joint, 0.0, 1.0)) / (p_a * p_b)


def _build_binary(player_dict, player_name, stat, thresh) -> pd.Series:
    d = player_dict.get(player_name)
    if not d:
        return None
    g = d['games']
    if stat not in g.columns:
        return None
    return (g[stat].fillna(0).astype(int) >= thresh).astype(int)


# ─── TASK 3: STABILITY CHECK ─────────────────────────────────────────────────

def _stability_check(shared: pd.DataFrame, r_full: float) -> dict:
    """
    First half / second half / last-20 split.
    Flags unstable if r flips sign or half-CIs don't overlap.
    """
    sh  = shared.sort_index()
    n   = len(sh)
    mid = n // 2

    def _r(sub):
        if len(sub) < 10:
            return np.nan
        v = float(sub['a'].corr(sub['b']))
        return np.nan if np.isnan(v) else v

    r_h1  = _r(sh.iloc[:mid])
    r_h2  = _r(sh.iloc[mid:])
    r_l20 = _r(sh.iloc[-20:]) if n >= 20 else np.nan

    lo_h1, hi_h1 = fisher_ci(r_h1, mid)          if not np.isnan(r_h1) else (np.nan, np.nan)
    lo_h2, hi_h2 = fisher_ci(r_h2, n - mid)      if not np.isnan(r_h2) else (np.nan, np.nan)

    reasons = []
    if not (np.isnan(r_h1) or np.isnan(r_h2)):
        if np.sign(r_h1) != np.sign(r_h2):
            reasons.append(f"sign_flip H1={r_h1:.2f}/H2={r_h2:.2f}")
        elif not any(np.isnan(v) for v in [lo_h1, hi_h1, lo_h2, hi_h2]):
            if lo_h1 > hi_h2 or lo_h2 > hi_h1:
                reasons.append(f"CI_no_overlap H1=[{lo_h1:.2f},{hi_h1:.2f}] H2=[{lo_h2:.2f},{hi_h2:.2f}]")

    if not (np.isnan(r_l20) or np.isnan(r_full)):
        if (r_full > 0 and r_l20 < -0.05) or (r_full < 0 and r_l20 > 0.05):
            reasons.append(f"L20_sign_flip r_l20={r_l20:.2f}")

    return {
        'r_h1':   round(r_h1, 3)  if not np.isnan(r_h1)  else None,
        'r_h2':   round(r_h2, 3)  if not np.isnan(r_h2)  else None,
        'r_l20':  round(r_l20, 3) if not np.isnan(r_l20) else None,
        'stable': len(reasons) == 0,
        'stability_note': '; '.join(reasons) if reasons else 'OK',
    }


# ─── TASK 2: THRESHOLD-MATCHED CORRELATION ───────────────────────────────────

def compute_finals_pairs(player_dict: dict, kalshi_markets: dict) -> list:
    """
    For Finals teams only, compute correlation at the EXACT Kalshi threshold.
    Returns rows with CI lo > 0 only (pre-filter; net-of-cost gate applied later).
    """
    # Build live Kalshi entries per player: {name: [(stat, thresh, last_price, ticker)]}
    player_kalshi = {}
    for pname, series_dict in kalshi_markets.items():
        entries = []
        for series, thresh_dict in series_dict.items():
            stat = SERIES_TO_STAT.get(series)
            if not stat:
                continue
            for thresh, info in thresh_dict.items():
                lp  = info.get('live_price')
                tkt = info.get('live_ticker')
                if lp is not None and tkt is not None:
                    entries.append((stat, thresh, lp, tkt))
        if entries:
            player_kalshi[pname] = entries

    # Group Finals players by team (must have Kalshi live price)
    team_players: dict = {}
    for pname, pdata in player_dict.items():
        team = pdata['team']
        if team in FINALS_TEAMS and pname in player_kalshi:
            team_players.setdefault(team, []).append(pname)

    rows = []
    for team, players in team_players.items():
        for pa, pb in itertools.combinations(sorted(players), 2):
            for (stat_a, thresh_a, price_a, ticker_a) in player_kalshi[pa]:
                sa = _build_binary(player_dict, pa, stat_a, thresh_a)
                if sa is None:
                    continue
                for (stat_b, thresh_b, price_b, ticker_b) in player_kalshi[pb]:
                    sb = _build_binary(player_dict, pb, stat_b, thresh_b)
                    if sb is None:
                        continue

                    shared = pd.concat(
                        [sa.rename('a'), sb.rename('b')],
                        axis=1, join='inner'
                    ).dropna()
                    n = len(shared)
                    if n < MIN_N:
                        continue
                    # Both outcomes must each appear ≥ 3 times
                    for col in ('a', 'b'):
                        s = int(shared[col].sum())
                        if s < 3 or (n - s) < 3:
                            break
                    else:
                        pass

                    with np.errstate(invalid='ignore'):
                        r = float(shared['a'].corr(shared['b']))
                    if np.isnan(r):
                        continue

                    lo, hi = fisher_ci(r, n)
                    if np.isnan(lo) or lo <= 0:
                        continue  # only keep where correlation is distinguishable from 0

                    stab = _stability_check(shared, r)
                    edge = bivariate_bernoulli_edge(r, price_a, price_b)
                    joint_model = price_a * price_b * edge if not np.isnan(edge) else np.nan

                    rows.append({
                        'team':        team,
                        'name_a':      pa,
                        'stat_a':      stat_a,
                        'thresh_a':    thresh_a,
                        'ticker_a':    ticker_a,
                        'name_b':      pb,
                        'stat_b':      stat_b,
                        'thresh_b':    thresh_b,
                        'ticker_b':    ticker_b,
                        'n':           n,
                        'r':           round(r, 3),
                        'ci_lo':       lo,
                        'ci_hi':       hi,
                        'p_a':         round(price_a, 3),
                        'p_b':         round(price_b, 3),
                        'joint_model': round(joint_model, 4) if not np.isnan(joint_model) else np.nan,
                        'edge_ratio':  round(edge, 3) if not np.isnan(edge) else np.nan,
                        **stab,
                    })

    return rows


# ─── TASK 4: NET-OF-COST GATE ────────────────────────────────────────────────

def add_orderbook_costs(rows: list) -> list:
    """
    Fetch live bid/ask for each unique ticker. Add round-trip cost and net edge.

    For a 1-unit parlay at independent price p_a × p_b:
      gross_edge  = P_model(both) − p_a × p_b
                  = r × sqrt(p_a(1−p_a) × p_b(1−p_b))   [in $ per unit]
      rt_cost     = (ask_a − bid_a) + (ask_b − bid_b)    [one-way spread, $ per unit]
      net_edge    = gross_edge − rt_cost
    net_edge > 0 means the modeled correlation lift exceeds the cost of entering both legs.
    """
    tickers = sorted({r['ticker_a'] for r in rows} | {r['ticker_b'] for r in rows})
    bid_ask = {}
    print(f"  Fetching live orderbook for {len(tickers)} tickers...")
    for tkt in tickers:
        bid_ask[tkt] = fetch_bid_ask(tkt)
        time.sleep(0.08)

    for row in rows:
        bid_a, ask_a = bid_ask.get(row['ticker_a'], (None, None))
        bid_b, ask_b = bid_ask.get(row['ticker_b'], (None, None))
        row.update({'bid_a': bid_a, 'ask_a': ask_a, 'bid_b': bid_b, 'ask_b': ask_b})

        if None in (bid_a, ask_a, bid_b, ask_b):
            row.update({'rt_cost': np.nan, 'gross_edge': np.nan, 'net_edge': np.nan})
            continue

        # Use ask as the marginal price (conservative: you pay the ask)
        p_a = ask_a
        p_b = ask_b
        r   = row['r']
        gross  = r * math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
        rt     = (ask_a - bid_a) + (ask_b - bid_b)
        net    = gross - rt

        row['gross_edge'] = round(gross, 4)
        row['rt_cost']    = round(rt,    4)
        row['net_edge']   = round(net,   4)

    return rows


# ─── HTML ─────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; padding: 32px 24px; }
h1 { font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }
.sub { color: #94a3b8; font-size: 0.875rem; margin-bottom: 40px; }
section { margin-bottom: 52px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #f1f5f9;
     border-left: 3px solid #8b5cf6; padding-left: 12px; margin-bottom: 6px; }
.desc { font-size: 0.8rem; color: #64748b; margin-bottom: 14px; padding-left: 15px; }
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid #1e293b; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
thead th { background: #1e293b; color: #94a3b8; font-weight: 600;
           padding: 10px 12px; text-align: left; white-space: nowrap;
           border-bottom: 1px solid #334155; }
tbody tr { border-bottom: 1px solid #1e293b; }
tbody tr:hover { background: #1a2235; }
td { padding: 9px 12px; vertical-align: middle; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.tag { background: #2d1b69; color: #c4b5fd; border-radius: 4px;
       padding: 1px 6px; font-size: 0.74rem; font-weight: 600; white-space: nowrap; }
.badge { background: #1e293b; color: #cbd5e1; border-radius: 4px;
         padding: 3px 8px; font-size: 0.78rem; font-weight: 700; }
.pos  { color: #34d399; } .neg { color: #f87171; } .warn { color: #fbbf24; }
.stat-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 36px; }
.stat-card { background: #1e293b; border-radius: 8px; padding: 16px 20px; min-width: 140px; }
.stat-card .v { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
.stat-card .l { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
.notice { background: #1e293b; border-left: 3px solid #f59e0b; border-radius: 6px;
          padding: 14px 18px; font-size: 0.82rem; color: #94a3b8; margin-bottom: 32px; }
.notice strong { color: #fbbf24; }
.glossary { background: #1e293b; border-radius: 8px; padding: 18px 22px;
            font-size: 0.8rem; color: #94a3b8; margin-bottom: 40px; }
.glossary h3 { font-size: 0.85rem; color: #cbd5e1; margin-bottom: 10px; }
.glossary dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
.glossary dt { color: #c4b5fd; font-weight: 600; }
"""


def _f(v, d=3):
    try:
        f = float(v)
        return '--' if math.isnan(f) else f"{f:.{d}f}"
    except Exception:
        return '--'

def _pct(v):
    try:
        f = float(v)
        return '--' if math.isnan(f) else f"{f*100:.1f}%"
    except Exception:
        return '--'

def _cents(v):
    try:
        f = float(v)
        return '--' if math.isnan(f) else f"{f*100:+.1f}¢"
    except Exception:
        return '--'


def _row_html(row):
    label_a = f"<b>{row['name_a']}</b> <span class='tag'>{row['thresh_a']}+ {STAT_LABEL[row['stat_a']]}</span>"
    label_b = f"<b>{row['name_b']}</b> <span class='tag'>{row['thresh_b']}+ {STAT_LABEL[row['stat_b']]}</span>"

    stab_flag = row.get('stability_note', '')
    stab_cls  = '' if row.get('stable', False) else 'warn'
    stab_html = f"<span class='{stab_cls}'>{stab_flag}</span>"

    net = row.get('net_edge', np.nan)
    try:
        net_cls = 'pos' if float(net) > 0 else 'neg'
    except Exception:
        net_cls = ''

    splits = ""
    if row.get('r_h1') is not None:
        splits = f"<br><span style='font-size:0.72rem;color:#475569'>H1={row['r_h1']} H2={row['r_h2']} L20={row['r_l20']}</span>"

    return f"""<tr>
  <td><span class='badge'>{row['team']}</span></td>
  <td>{label_a}<br>{label_b}</td>
  <td class='num'>{row['n']}</td>
  <td class='num'>{_f(row['r'], 3)}</td>
  <td class='num'>{_f(row['ci_lo'], 3)}</td>
  <td class='num'>{_pct(row['p_a'])} / {_pct(row['p_b'])}</td>
  <td class='num'>{_pct(row.get('joint_model'))}</td>
  <td class='num'>{_f(row.get('edge_ratio'), 2)}x</td>
  <td class='num'>{_cents(row.get('gross_edge'))}</td>
  <td class='num'>{_cents(row.get('rt_cost'))}</td>
  <td class='num {net_cls}'>{_cents(net)}</td>
  <td>{stab_html}{splits}</td>
</tr>"""


TABLE_HEADER = """<tr>
  <th>Team</th><th>Pair</th><th>n</th><th>r (thresh-matched)</th><th>CI lo</th>
  <th>Kalshi p_a / p_b</th><th>Model joint</th><th>Edge ratio</th>
  <th>Gross edge</th><th>RT cost</th><th>Net edge</th><th>Stability</th>
</tr>"""


def make_html(shortlist: list, candidates: list, n_games: int) -> str:
    today = datetime.date.today().isoformat()
    n_cand = len(candidates)
    n_short = len(shortlist)

    stat_cards = f"""
<div class='stat-grid'>
  <div class='stat-card'><div class='v'>{n_games:,}</div><div class='l'>Regular season player-games</div></div>
  <div class='stat-card'><div class='v'>{n_cand}</div><div class='l'>Finals pairs with CI lo &gt; 0</div></div>
  <div class='stat-card'><div class='v'>{n_short}</div><div class='l'>Pass all gates (stable + net &gt; 0)</div></div>
</div>"""

    notice = """<div class='notice'>
  <strong>Conditional on product check (Task 1):</strong>
  Kalshi does not currently expose a multi-leg player prop parlay via the API.
  The edge modeled here is realizable only if a combo product exists (verify in the Kalshi UI).
  Round-trip cost is the sum of bid-ask spreads on both legs — one-way entry cost on a hold-to-expiry position.
</div>"""

    glossary = """<div class='glossary'>
  <h3>Column guide</h3>
  <dl>
    <dt>r (thresh-matched)</dt><dd>Pearson r built at the EXACT Kalshi line threshold, not the median. Both binary series use the same threshold from Kalshi pricing.</dd>
    <dt>CI lo</dt><dd>Fisher 95% lower bound on r. Positive = correlation distinguishable from 0 at 95% confidence.</dd>
    <dt>Kalshi p_a / p_b</dt><dd>Ask prices from live Kalshi orderbook — the price you actually pay for yes.</dd>
    <dt>Model joint</dt><dd>p_a·p_b + r·√(p_a(1−p_a)·p_b(1−p_b)) — bivariate Bernoulli. Higher than p_a·p_b because r &gt; 0.</dd>
    <dt>Edge ratio</dt><dd>Model joint / (p_a·p_b). Above 1.0 means correlation adds lift over independent price.</dd>
    <dt>Gross edge</dt><dd>Model joint − p_a·p_b in cents per $1 parlay unit = r·√(p_a(1−p_a)·p_b(1−p_b)).</dd>
    <dt>RT cost</dt><dd>(ask_a − bid_a) + (ask_b − bid_b): one-way spread cost to enter both legs.</dd>
    <dt>Net edge</dt><dd>Gross edge − RT cost. Positive = correlation lift exceeds entry spread. This is the gate.</dd>
    <dt>Stability</dt><dd>H1/H2/L20 r values. Flagged if r flips sign between halves or L20 contradicts full-season sign.</dd>
  </dl>
</div>"""

    if shortlist:
        short_rows = '\n'.join(_row_html(r) for r in shortlist)
        short_section = f"""
<section>
  <h2>Shortlist — Pass All Gates</h2>
  <p class='desc'>CI lo &gt; 0 · stable (no sign flip across splits) · net edge &gt; 0 · NYK + SAS only</p>
  <div class='table-wrap'>
    <table><thead>{TABLE_HEADER}</thead><tbody>{short_rows}</tbody></table>
  </div>
</section>"""
    else:
        short_section = """<section>
  <h2>Shortlist — Pass All Gates</h2>
  <p class='desc' style='color:#f87171;padding-left:0'>
    Zero pairs pass all three gates (CI lo &gt; 0, stable, net edge &gt; 0).
    This is a valid null result — no actionable correlation parlay identified for the Finals.
  </p>
</section>"""

    if candidates:
        cand_rows = '\n'.join(_row_html(r) for r in candidates)
        cand_section = f"""
<section>
  <h2>All Candidates — CI lo &gt; 0 (pre-stability/cost filter)</h2>
  <p class='desc'>All Finals pairs where correlation CI lower bound is positive · unstable and negative-net-edge pairs included for inspection</p>
  <div class='table-wrap'>
    <table><thead>{TABLE_HEADER}</thead><tbody>{cand_rows}</tbody></table>
  </div>
</section>"""
    else:
        cand_section = ""

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>NBA Finals Prop Correlations — Threshold-Matched</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>NBA Finals Prop Correlations</h1>
  <p class='sub'>2025-26 Regular Season (nba_api) × Kalshi Playoff Prices · Threshold-matched · Generated {today}</p>
  {stat_cards}
  {notice}
  {glossary}
  {short_section}
  {cand_section}
</body>
</html>"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=== NBA Finals Prop Correlation Analysis (threshold-matched) ===\n")

    print("[1/5] Loading regular season game logs...")
    df_logs = fetch_regular_season_logs()
    n_games = len(df_logs)
    print(f"  {n_games:,} player-game rows")

    print("[2/5] Building player game dictionaries...")
    player_dict = build_player_game_dict(df_logs)
    print(f"  {len(player_dict)} players")

    print("[3/5] Loading Kalshi playoff markets...")
    kalshi_markets = load_kalshi_markets()
    live_count = sum(
        1 for s in kalshi_markets.values()
        for t in s.values()
        for info in t.values()
        if info.get('live_price') is not None
    )
    print(f"  {len(kalshi_markets)} players, {live_count} live-price markets")

    print("[4/5] Computing threshold-matched correlations for Finals teams...")
    candidates = compute_finals_pairs(player_dict, kalshi_markets)
    print(f"  {len(candidates)} pairs with CI lo > 0")

    if not candidates:
        print("\nNo passing pairs found. Generating null-result report.")
        html = make_html([], [], n_games)
        html_path = os.path.join(OUT_DIR, "nba_api_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Saved: {os.path.basename(html_path)}")
        return

    print("[5/5] Fetching live orderbook (bid/ask) and computing net-of-cost edge...")
    candidates = add_orderbook_costs(candidates)

    # Shortlist: stable + net_edge > 0
    shortlist = [
        r for r in candidates
        if r.get('stable', False) and not np.isnan(r.get('net_edge', np.nan)) and r['net_edge'] > 0
    ]
    # Sort candidates by net_edge descending
    candidates.sort(key=lambda r: r.get('net_edge') or -9999, reverse=True)
    shortlist.sort(key=lambda r: r.get('net_edge') or -9999, reverse=True)

    print(f"\n{'='*60}")
    print(f"SHORTLIST: {len(shortlist)} pair(s) pass all gates (CI lo > 0, stable, net edge > 0)")
    if shortlist:
        print(f"{'Team':<5} {'Pair':<55} {'r':>6} {'n':>4} {'CI lo':>6} {'p_a':>5} {'p_b':>5} "
              f"{'EdgeR':>6} {'NetEdge':>9} {'Stab'}")
        print('-' * 120)
        for row in shortlist:
            pair = (f"{row['name_a']} {row['thresh_a']}+{STAT_LABEL[row['stat_a']]} / "
                    f"{row['name_b']} {row['thresh_b']}+{STAT_LABEL[row['stat_b']]}")
            print(f"{row['team']:<5} {pair:<55} {row['r']:>6.3f} {row['n']:>4} "
                  f"{row['ci_lo']:>6.3f} {row['p_a']:>5.2f} {row['p_b']:>5.2f} "
                  f"{row.get('edge_ratio', 0):>6.2f}x "
                  f"{row.get('net_edge',0)*100:>+7.1f}¢ "
                  f"{row.get('stability_note','')}")
    else:
        print("Zero pairs pass all gates. Clean null — no actionable parlay identified.")

    print(f"\nAll {len(candidates)} candidates (pre-stability/cost):")
    print(f"{'Team':<5} {'Pair':<55} {'r':>6} {'n':>4} {'CI lo':>6} {'Net¢':>7} {'Stab'}")
    print('-' * 100)
    for row in candidates:
        pair = (f"{row['name_a']} {row['thresh_a']}+{STAT_LABEL[row['stat_a']]} / "
                f"{row['name_b']} {row['thresh_b']}+{STAT_LABEL[row['stat_b']]}")
        net_s = f"{row.get('net_edge',0)*100:>+7.1f}¢" if not np.isnan(row.get('net_edge', np.nan)) else '     --'
        stab_s = '' if row.get('stable') else f"[{row.get('stability_note','')}]"
        print(f"{row['team']:<5} {pair:<55} {row['r']:>6.3f} {row['n']:>4} "
              f"{row['ci_lo']:>6.3f} {net_s} {stab_s}")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "nba_api_finals_pairs.csv")
    pd.DataFrame(candidates).drop(columns=['stable'], errors='ignore').to_csv(csv_path, index=False)
    print(f"\nSaved: {os.path.basename(csv_path)} ({len(candidates)} rows)")

    # Generate HTML
    html_path = os.path.join(OUT_DIR, "nba_api_report.html")
    html = make_html(shortlist, candidates, n_games)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {os.path.basename(html_path)}")
    print("\nDone.")


if __name__ == '__main__':
    main()
