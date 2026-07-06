import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import requests
import pandas as pd
import pybaseball as pb
from pybaseball import playerid_lookup, statcast_batter, statcast_pitcher, schedule_and_record
import datetime
import numpy as np
from scipy.stats import norm
import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL     = "https://api.elections.kalshi.com/trade-api/v2"
SEASON_START = "2026-01-01"
SEASON_END   = datetime.date.today().isoformat()
CACHE_DIR    = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

PLAYER_SERIES = ['KXMLBHIT', 'KXMLBHR', 'KXMLBHRR', 'KXMLBTB']
PITCHER_SERIES = ['KXMLBKS']
TEAM_SERIES   = ['KXMLBGAME', 'KXMLBSPREAD', 'KXMLBTOTAL', 'KXMLBTEAMTOTAL', 'KXMLBF5SPREAD']

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
          "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
TB_MAP     = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}

# ─── API HELPERS ──────────────────────────────────────────────────────────────

def pregame_price(ticker):
    if ticker in _price_cache:
        return _price_cache[ticker]

    parts    = ticker.split("-")
    date_str = parts[1]
    year  = int(date_str[0:2]) + 2000
    month = MONTHS[date_str[2:5]]
    day   = int(date_str[5:7])
    hour  = int(date_str[7:9])
    minute = int(date_str[9:11])
    end_ts   = int(datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())
    start_ts = end_ts - 24 * 3600
    params   = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60}

    response = requests.get(f"{BASE_URL}/historical/markets/{ticker}/candlesticks", params=params)
    if response.status_code == 404:
        series   = ticker.split("-")[0]
        response = requests.get(f"{BASE_URL}/series/{series}/markets/{ticker}/candlesticks", params=params)
    if response.status_code not in (200, 404):
        _price_cache[ticker] = None
        return None

    try:
        data = response.json()
    except Exception:
        _price_cache[ticker] = None
        return None
    candlesticks = data.get("candlesticks", [])
    candlestick  = candlesticks[-1] if candlesticks else None
    if not candlestick:
        _price_cache[ticker] = None
        return None

    yes_bid = candlestick.get("yes_bid") or {}
    yes_ask = candlestick.get("yes_ask") or {}
    bid_str = yes_bid.get("close_dollars") or yes_bid.get("close")
    ask_str = yes_ask.get("close_dollars") or yes_ask.get("close")
    if bid_str is None or ask_str is None:
        _price_cache[ticker] = None
        return None
    price = round((float(bid_str) + float(ask_str)) / 2, 4)
    _price_cache[ticker] = price
    return price


def get_markets(series, ticker_prefix=None, max_pages=20):
    """Fetch markets for a Kalshi series, optionally filtered by ticker prefix."""
    year_prefix = f"{series}-{str(datetime.date.today().year)[2:]}"
    params = {"limit": 1000, "series_ticker": series, "ticker_prefix": ticker_prefix or year_prefix}
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
    """get_markets with a daily disk cache — re-fetches once per calendar day."""
    import json
    today = datetime.date.today().isoformat()
    if series in _markets_cache:
        return _markets_cache[series]
    path = os.path.join(CACHE_DIR, f"markets_{series}_{today}.json")
    if os.path.exists(path):
        with open(path) as f:
            markets = json.load(f)
    else:
        markets = get_markets(series)
        with open(path, 'w') as f:
            json.dump(markets, f)
    _markets_cache[series] = markets
    return markets


def ticker_to_datetime(ticker):
    """Returns (date_str YYYY-MM-DD, time_utc_minutes) for game-order sorting."""
    date_str = ticker.split("-")[1]
    date = f"{int(date_str[0:2])+2000}-{MONTHS[date_str[2:5]]:02d}-{date_str[5:7]}"
    mins = int(date_str[7:9]) * 60 + int(date_str[9:11])
    return date, mins

def ticker_to_date(ticker):
    return ticker_to_datetime(ticker)[0]

# ─── ROSTER DISCOVERY ─────────────────────────────────────────────────────────

def get_team_roster(team_abbr, series_list=None):
    """
    Scrape Kalshi player-prop series to find all unique players with markets
    for the given team. Returns list of (firstname, lastname) tuples.
    """
    if series_list is None:
        series_list = PLAYER_SERIES
    seen = set()
    for series in series_list:
        for m in get_markets(series):
            parts = m['ticker'].split('-')
            if len(parts) < 4 or not parts[2].startswith(team_abbr):
                continue
            # extract name from title: first two words are "Firstname Lastname ..."
            words = m.get('title', '').split()
            if len(words) >= 2:
                seen.add((words[0], words[1]))
    return list(seen)

# ─── CACHING ──────────────────────────────────────────────────────────────────

def _statcast_cache_path(name, kind):
    return os.path.join(CACHE_DIR, f"{name}_{kind}.parquet")

def _load_cached_statcast(path):
    if not os.path.exists(path):
        return None, SEASON_START
    df = pd.read_parquet(path)
    last_date = df['game_date'].max()
    next_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return df, next_date

def _save_statcast(path, df):
    df.to_parquet(path, index=False)

_price_cache_path = os.path.join(CACHE_DIR, "prices.parquet")

def _load_price_cache():
    if not os.path.exists(_price_cache_path):
        return {}
    df = pd.read_parquet(_price_cache_path)
    # NaN in parquet = cached None (no trading data) — restore as None so the ticker is skipped
    return {t: (None if pd.isna(p) else p) for t, p in zip(df['ticker'], df['price'])}

def _save_price_cache(cache: dict):
    df = pd.DataFrame(list(cache.items()), columns=['ticker', 'price'])
    df.to_parquet(_price_cache_path, index=False)

_price_cache = _load_price_cache()

# ─── STATCAST PATH (unused by main build — kept for quick_correlation) ────────

def _aggregate_raw(raw, is_pitcher=False):
    if raw.empty:
        cols = ['game_pk', 'game_date', 'ks'] if is_pitcher else ['game_pk', 'game_date', 'hits', 'hr', 'tb', 'rbi', 'hrr']
        return pd.DataFrame(columns=cols)
    if is_pitcher:
        raw['is_k'] = raw['events'] == 'strikeout'
        return raw.groupby('game_pk').agg(
            game_date=('game_date', 'first'),
            ks=('is_k', 'sum'),
        ).reset_index()
    raw['is_hit'] = raw['events'].isin(HIT_EVENTS)
    raw['is_hr']  = raw['events'] == 'home_run'
    raw['tb']     = raw['events'].map(TB_MAP).fillna(0)
    raw['rbi']    = (raw['post_bat_score'] - raw['bat_score']).clip(lower=0)
    g = raw.groupby('game_pk').agg(
        game_date=('game_date', 'first'),
        hits=('is_hit', 'sum'),
        hr=('is_hr',   'sum'),
        tb=('tb',      'sum'),
        rbi=('rbi',    'sum'),
    ).reset_index()
    g['hrr'] = g['hits'] + g['rbi']  # H+RBI proxy for H+R+RBI (runs scored not in statcast per-PA)
    return g


def _fetch_and_cache(firstname, lastname, is_pitcher=False):
    slug      = f"{firstname}_{lastname}".lower()
    kind      = "pitcher" if is_pitcher else "batter"
    path      = _statcast_cache_path(slug, kind)
    cached, fetch_from = _load_cached_statcast(path)

    if fetch_from <= SEASON_END:
        print(f"    fetching statcast {firstname} {lastname} from {fetch_from}...")
        fetcher = statcast_pitcher if is_pitcher else statcast_batter
        lookup  = playerid_lookup(lastname, firstname)
        if lookup.empty:
            print(f"    no player ID found for {firstname} {lastname} — try checking the name spelling")
            return cached if cached is not None else pd.DataFrame()
        pid = lookup['key_mlbam'].iloc[0]
        new_raw = fetcher(fetch_from, SEASON_END, pid)
        if not new_raw.empty:
            new_g = _aggregate_raw(new_raw.copy(), is_pitcher)
            new_g['game_date'] = new_g['game_date'].astype(str)
            cached = pd.concat([cached, new_g], ignore_index=True).drop_duplicates('game_pk') \
                     if cached is not None else new_g
            _save_statcast(path, cached)
        elif cached is None:
            return pd.DataFrame()
    else:
        print(f"    statcast {firstname} {lastname} loaded from cache.")
    return cached


def compute_game_stats(firstname, lastname):
    return _fetch_and_cache(firstname, lastname, is_pitcher=False)

def compute_pitcher_game_stats(firstname, lastname):
    return _fetch_and_cache(firstname, lastname, is_pitcher=True)

# ─── TEAM GAME RESULTS ────────────────────────────────────────────────────────

def get_team_game_results(team_abbr):
    """Game results from pybaseball schedule_and_record."""
    df = schedule_and_record(2026, team_abbr)
    df = df[df['R'].notna() & df['RA'].notna()].copy()
    # pybaseball Date format: "Apr 1" — add year for parsing
    df['game_date'] = pd.to_datetime(
        df['Date'].str.replace(r'^[A-Za-z]+,\s*', '', regex=True)
                  .str.replace(r'\s*\(\d+\)', '', regex=True) + ' 2026',
        format='%b %d %Y'
    ).dt.strftime('%Y-%m-%d')
    df['game_order']   = df.groupby('game_date').cumcount()  # 0 = first game, 1 = second (DH)
    df['runs_scored']  = df['R'].astype(int)
    df['runs_allowed'] = df['RA'].astype(int)
    df['run_diff']     = df['runs_scored'] - df['runs_allowed']
    df['total_runs']   = df['runs_scored'] + df['runs_allowed']
    df['win']          = df['W/L'].str.startswith('W').astype(int)
    return df[['game_date','game_order','runs_scored','runs_allowed','run_diff','total_runs','win']]

# ─── END STATCAST PATH ────────────────────────────────────────────────────────

# ─── PRICE FETCH ──────────────────────────────────────────────────────────────

STAT_COL = {
    'KXMLBHIT': 'hits',
    'KXMLBHR':  'hr',
    'KXMLBHRR': 'hrr',
    'KXMLBTB':  'tb',
    'KXMLBKS':  'ks',
}

def fetch_prices_parallel(tickers, max_workers=10):
    """Fetch pregame_price for multiple tickers concurrently, skipping cached ones."""
    uncached = [t for t in tickers if t not in _price_cache]
    results  = {t: _price_cache[t] for t in tickers if t in _price_cache}
    if uncached:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(pregame_price, t): t for t in uncached}
            for f in as_completed(futures):
                results[futures[f]] = f.result()
        _save_price_cache(_price_cache)
    return results


def player_prop_table(firstname, lastname, series, markets=None, game_stats=None):
    """
    Returns dict: {prop_label -> Series(surprise, index=game_date)}
    Pass pre-fetched markets and game_stats to avoid redundant API/statcast calls.
    """
    full_name  = f"{firstname} {lastname}"
    is_pitcher = (series == 'KXMLBKS')
    if game_stats is None:
        game_stats = compute_pitcher_game_stats(firstname, lastname) if is_pitcher \
                     else compute_game_stats(firstname, lastname)
    stat_col = STAT_COL[series]

    threshold_markets = {}
    for m in (markets if markets is not None else get_markets(series)):
        title    = m.get('title', '')
        subtitle = m.get('subtitle', '')
        if full_name not in title and full_name not in subtitle:
            continue
        parts = m['ticker'].split('-')
        if len(parts) < 4:
            continue
        threshold = int(parts[-1])
        threshold_markets.setdefault(threshold, []).append(m['ticker'])

    # fetch all prices in parallel across all thresholds at once
    all_tickers = [t for tickers in threshold_markets.values() for t in tickers]
    price_map   = fetch_prices_parallel(all_tickers)
    none_count  = sum(v is None for v in price_map.values())
    print(f"  {full_name} {series}: {len(all_tickers)} tickers, {none_count} None")

    # build prices_df with game_order so doubleheader games don't cross-merge
    prices_rows = [{'ticker': t, 'game_date': ticker_to_datetime(t)[0],
                    'time_utc': ticker_to_datetime(t)[1], 'pregame_price': price_map[t]}
                   for t in all_tickers]
    prices_df_all = pd.DataFrame(prices_rows).dropna(subset=['pregame_price'])
    # tickers for the same game share the same time — dense rank within date gives game_order
    prices_df_all['game_order'] = (
        prices_df_all.groupby('game_date')['time_utc']
        .transform(lambda x: x.rank(method='dense').astype(int) - 1)
    )

    # add game_order to stats: lower game_pk = earlier game on same date
    stats = game_stats.copy()
    stats['game_order'] = (
        stats.groupby('game_date')['game_pk']
        .rank(method='dense').astype(int) - 1
    )

    result = {}
    for threshold, tickers in threshold_markets.items():
        t_prices = prices_df_all[prices_df_all['ticker'].isin(tickers)][
            ['game_date', 'game_order', 'pregame_price']
        ]
        merged = stats.merge(t_prices, on=['game_date', 'game_order'])
        if merged.empty:
            continue
        merged['outcome']  = (merged[stat_col] >= threshold).astype(int)
        merged['surprise'] = merged['outcome'] - merged['pregame_price']
        label = f"{full_name} {series} {threshold}+"
        result[label] = merged.set_index('game_date')['surprise'].dropna()

    return result


TEAM_OUTCOME_FN = {
    'KXMLBGAME':      lambda row, _: row['win'],
    'KXMLBSPREAD':    lambda row, t: int(row['run_diff'] >= t),
    'KXMLBTOTAL':     lambda row, t: int(row['total_runs'] >= t),
    'KXMLBTEAMTOTAL': lambda row, t: int(row['runs_scored'] >= t),
    'KXMLBF5SPREAD':  lambda row, t: int(row['run_diff'] >= t),  # full-game diff as proxy
}

def team_prop_table(team_abbr, series, markets=None):
    """
    Returns dict: {prop_label -> Series(surprise, index=game_date)}
    for team-level series.
    Pass pre-fetched markets to avoid redundant API calls in bulk builds.
    """
    game_results = get_team_game_results(team_abbr)
    outcome_fn   = TEAM_OUTCOME_FN.get(series)
    if not outcome_fn:
        return {}

    # build game_order map from all tickers: (date, time_utc) -> game_order
    all_entries = []
    for m in (markets if markets is not None else get_markets(series)):
        parts = m['ticker'].split('-')
        if len(parts) < 3 or team_abbr not in m['ticker']:
            continue
        last   = parts[-1]
        digits = ''.join(c for c in last if c.isdigit())
        threshold = int(digits) if digits else 0
        date, mins = ticker_to_datetime(m['ticker'])
        all_entries.append({'ticker': m['ticker'], 'game_date': date,
                            'time_utc': mins, 'threshold': threshold})

    # dense-rank times within each date to get game_order (0 = first game)
    if not all_entries:
        return {}
    entries_df = pd.DataFrame(all_entries)
    entries_df['game_order'] = (
        entries_df.groupby('game_date')['time_utc']
        .transform(lambda x: x.rank(method='dense').astype(int) - 1)
    )

    threshold_markets = {}
    for _, row in entries_df.iterrows():
        threshold_markets.setdefault(row['threshold'], []).append(row.to_dict())

    result = {}
    for threshold, entries in threshold_markets.items():
        rows = []
        for entry in entries:
            game = game_results[
                (game_results['game_date'] == entry['game_date']) &
                (game_results['game_order'] == entry['game_order'])
            ]
            if game.empty:
                continue
            price = pregame_price(entry['ticker'])
            if price is None:
                continue
            outcome  = outcome_fn(game.iloc[0], threshold)
            rows.append({'game_date': entry['game_date'], 'surprise': outcome - price})
        if not rows:
            continue
        df    = pd.DataFrame(rows).set_index('game_date')['surprise']
        label = f"{team_abbr} {series} {threshold}+" if threshold else f"{team_abbr} {series} WIN"
        result[label] = df
    return result

# ─── BUILD FULL PROP DICT ─────────────────────────────────────────────────────

def build_prop_dict(team_abbr, series_list=None):
    """
    Build prop dict from Kalshi market results — no statcast needed.
    outcome = market result field (yes=1 / no=0).
    Filters to markets whose ticker contains team_abbr.
    """
    if series_list is None:
        series_list = PLAYER_SERIES + PITCHER_SERIES + TEAM_SERIES

    # collect all resolved markets for this team
    all_markets = []
    for series in series_list:
        print(f"  fetching {series}...")
        for m in get_markets_cached(series):
            if m.get('result') not in ('yes', 'no'):
                continue
            # parts[2] is the player/team segment — must start with team_abbr
            # (avoids false matches when team_abbr appears in the game event string)
            parts = m['ticker'].split('-')
            if len(parts) < 3 or not parts[2].startswith(team_abbr):
                continue
            all_markets.append(m)
    print(f"  {len(all_markets)} resolved markets")

    # use last_price_dollars from market data — no extra API calls needed.
    # markets close at first pitch so last_price is the final pregame consensus.
    groups = {}
    skipped = 0
    for m in all_markets:
        ticker  = m['ticker']
        last_p  = m.get('last_price_dollars') or '0'
        price   = float(last_p)
        if price <= 0.01:   # no meaningful trading activity
            skipped += 1
            continue
        outcome  = 1 if m['result'] == 'yes' else 0
        surprise = outcome - price
        date, mins = ticker_to_datetime(ticker)
        game_key   = f"{date}-{mins:04d}"  # unique per game; handles doubleheaders

        parts     = ticker.split('-')
        series    = parts[0]
        threshold = parts[-1]
        title     = m.get('title', '')
        if len(parts) == 3:
            # team prop (no player segment): "KXMLBTOTAL 12+"
            label = f"{series} {threshold}+"
        else:
            # player prop: "Mookie Betts KXMLBHIT 1+"
            name  = title.split(':')[0].strip() if ':' in title else parts[2]
            label = f"{name} {series} {threshold}+"

        groups.setdefault(label, []).append({
            'game_key': game_key, 'surprise': surprise,
            'outcome': outcome, 'price': price
        })

    used = len(all_markets) - skipped
    if all_markets:
        print(f"  {used}/{len(all_markets)} markets had last_price > $0.01 ({used/len(all_markets):.0%})")
    else:
        print("  0 resolved markets")

    prop_dict = {}
    raw_dict  = {}  # label -> DataFrame[game_key, outcome, price]
    for label, rows in groups.items():
        df = pd.DataFrame(rows).drop_duplicates('game_key').set_index('game_key')
        if len(df) >= 3:
            prop_dict[label] = df['surprise']
            raw_dict[label]  = df[['outcome', 'price']]

    print(f"Built {len(prop_dict)} props total.")
    return prop_dict, raw_dict

# ─── CORRELATION & FISHER CI ──────────────────────────────────────────────────

def _label_entity(label):
    """
    Extract the 'thing being measured' from a prop label.
    Player props  → player name   ("Mookie Betts KXMLBHIT 1+" → "Mookie Betts")
    Team props    → series name   ("KXMLBTOTAL 12+"           → "KXMLBTOTAL")
    Used to prevent same-player or same-series pairs in a parlay.
    """
    for s in PLAYER_SERIES + PITCHER_SERIES + TEAM_SERIES:
        if f' {s} ' in label:          # player prop: name appears before series
            return label.split(f' {s} ')[0].strip()
        if label.startswith(f'{s} '):  # team prop: label starts with series
            return s
    return label


def _corr_pvalue(r, n, one_sided=False):
    """P-value for Pearson r via Fisher z.
    one_sided=True tests r > 0 (upper tail); use this for positive-correlation search."""
    if n <= 3 or np.isnan(r):
        return 1.0
    z = np.arctanh(r) * np.sqrt(n - 3)
    if one_sided:
        return float(norm.sf(z))   # P(Z > z), so r < 0 → p near 1 and fails BH
    return float(2 * (1 - norm.cdf(abs(z))))


def fisher_ci(r, n, alpha=0.05):
    if n <= 3 or np.isnan(r):
        return (np.nan, np.nan)
    z     = np.arctanh(r)
    se    = 1 / np.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    return (round(np.tanh(z - zcrit * se), 3), round(np.tanh(z + zcrit * se), 3))


def correlation_matrix(prop_dict):
    labels   = list(prop_dict.keys())
    corr_mat = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    n_mat    = pd.DataFrame(np.zeros((len(labels), len(labels)), dtype=int), index=labels, columns=labels)
    ci_mat   = {}

    for a, b in itertools.combinations(labels, 2):
        shared = pd.concat([prop_dict[a], prop_dict[b]], axis=1, join='inner').dropna()
        n      = len(shared)
        with np.errstate(invalid='ignore', divide='ignore'):
            r = shared.iloc[:, 0].corr(shared.iloc[:, 1]) if n > 2 else np.nan
        corr_mat.loc[a, b] = r
        corr_mat.loc[b, a] = r
        n_mat.loc[a, b]    = n
        n_mat.loc[b, a]    = n
        ci_mat[(a, b)]     = fisher_ci(r, n)

    print("Correlation matrix:\n", corr_mat.round(3))
    print("\nShared games (n):\n", n_mat)
    return corr_mat, n_mat, ci_mat

# ─── PARLAY FINDER ────────────────────────────────────────────────────────────

def parlay_finder(prop_dict, n_legs=2, top_n=10, min_n=20, fdr_q=0.10):
    """
    Find parlay combinations with statistically significant positive correlation.

    Two gates, applied in order:
      1. n >= min_n shared games (drops underpowered pairs before FDR)
      2. BH-FDR at fdr_q using one-sided p-values (r > 0) across ALL remaining pairs

    One-sided p-values mean negative correlations automatically fail BH —
    no redundant CI gate needed. CI is shown for interpretation only.
    Ranked by min CI lower bound across pairs (not avg_r).
    """
    labels = list(prop_dict.keys())

    _team_series = set(TEAM_SERIES)

    # Step 1: compute all pairwise stats
    pair_cache = {}
    for a, b in itertools.combinations(labels, 2):
        ea, eb = _label_entity(a), _label_entity(b)
        if ea == eb:
            continue  # same player (different threshold/stat) or nested team totals
        if ea in _team_series or eb in _team_series:
            continue  # player vs team metric, or team metric vs team metric — entangled by construction
        shared = pd.concat([prop_dict[a], prop_dict[b]], axis=1, join='inner').dropna()
        n = len(shared)
        if n < min_n:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            r = shared.iloc[:, 0].corr(shared.iloc[:, 1]) if n > 2 else np.nan
        lo, hi = fisher_ci(r, n)
        p = _corr_pvalue(r, n, one_sided=True)  # one-sided: testing r > 0
        pair_cache[(a, b)] = {'r': r, 'n': n, 'ci': (lo, hi), 'p': p}

    # Step 2: BH-FDR across all valid pairs (true BH: reject 1..k_max)
    if pair_cache:
        pair_keys = list(pair_cache.keys())
        pvals     = np.array([pair_cache[k]['p'] for k in pair_keys])
        m         = len(pvals)
        order     = np.argsort(pvals)
        sorted_p  = pvals[order]
        below     = sorted_p <= fdr_q * np.arange(1, m + 1) / m
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

    # Step 3: combos where every pair has n >= min_n and passes BH-FDR (one-sided)
    # CI is kept for display only — BH is the single significance gate
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

    print(f"\nTop {top_n} {n_legs}-leg parlays (CI lo > 0, BH-FDR q<={fdr_q:.0%}, min_n={min_n}):")
    if not results:
        print("  (no significant combinations — try reducing min_n or gathering more data)")
    for i, res in enumerate(results[:top_n], 1):
        print(f"\n#{i}  min_CI_lo={res['min_lo']:.3f}")
        for leg in res['combo']:
            print(f"  • {leg}")
        for p in res['pairs']:
            a, b = p['pair']
            print(f"    [{a}] x [{b}]  r={p['r']:.3f}  95%CI={p['ci']}  n={p['n']}  p={p['p']:.4f}")

    return results


def parlay_edge_rows(results, raw_dict, team):
    """
    For each parlay combo in results, compute observed joint probability,
    implied-independent joint probability, and edge ratio.
    Returns list of dicts suitable for pd.DataFrame / CSV.
    """
    rows = []
    for res in results:
        combo = res['combo']
        n_legs = len(combo)
        pair_stats = res['pairs']

        # join all legs on game_key (inner join)
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

        n_shared = len(shared)
        # average pregame implied probability for each leg
        mean_prices = [shared[f'p{i}'].mean() for i in range(n_legs)]
        # observed joint hit rate in shared games
        all_yes = (shared[[f'out{i}' for i in range(n_legs)]] == 1).all(axis=1)
        p_joint = all_yes.mean()
        p_indep = float(np.prod(mean_prices))
        # ratio: > 1 means positive correlation creates edge vs independence assumption
        ratio = p_joint / p_indep if p_indep > 0 else float('nan')

        row = {
            'team':   team,
            'n_legs': n_legs,
            'leg_1':  combo[0],
            'leg_2':  combo[1],
            'leg_3':  combo[2] if n_legs > 2 else '',
            'r_min':  round(min(p['r'] for p in pair_stats), 3),
            'r_max':  round(max(p['r'] for p in pair_stats), 3),
            'ci_lo':  round(res['min_lo'], 3),
            'n_shared': n_shared,
            'avg_p_leg1': round(mean_prices[0], 3),
            'avg_p_leg2': round(mean_prices[1], 3),
            'avg_p_leg3': round(mean_prices[2], 3) if n_legs > 2 else '',
            'p_joint_observed': round(p_joint, 4),
            'p_independent':    round(p_indep, 4),
            'edge_ratio':       round(ratio, 2),  # 1.5 = 50% more likely than indep
        }
        rows.append(row)
    return rows


# ─── VALIDATION HELPERS ───────────────────────────────────────────────────────

def find_season_midpoint(team_raw_dicts):
    """Median game_key across all teams/props — used as train/test cutoff."""
    all_keys = []
    for raw_dict in team_raw_dicts.values():
        for df in raw_dict.values():
            all_keys.extend(df.index.tolist())
    sorted_keys = sorted(set(all_keys))
    mid = sorted_keys[len(sorted_keys) // 2]
    print(f"  first game: {sorted_keys[0]}  midpoint: {mid}  last: {sorted_keys[-1]}")
    print(f"  ({len(sorted_keys)} unique game slots total)")
    return mid


def split_raw_dict(raw_dict, cutoff, min_n=3):
    """Split raw_dict into (train, test) at cutoff game_key."""
    train, test = {}, {}
    for label, df in raw_dict.items():
        before = df[df.index <= cutoff]
        after  = df[df.index >  cutoff]
        if len(before) >= min_n:
            train[label] = before
        if len(after) >= min_n:
            test[label]  = after
    return train, test


def raw_to_prop_dict(raw_dict):
    return {lbl: df['outcome'] - df['price'] for lbl, df in raw_dict.items()}


def pair_edge_on(label_a, label_b, raw_dict, min_n=3):
    """Edge ratio for an (a, b) pair using the given raw_dict slice."""
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


# ─── RUN ──────────────────────────────────────────────────────────────────────

# All 30 MLB teams — Kalshi abbreviations (verified from ticker data)
TEAMS_TO_ANALYZE = [
    'LAD', 'NYY', 'BOS', 'ATL', 'PHI', 'HOU', 'CHC', 'NYM', 'SD', 'SF',
    'MIL', 'CLE', 'TOR', 'MIN', 'SEA', 'TEX', 'AZ', 'STL', 'BAL', 'TB',
    'DET', 'KC', 'ATH', 'LAA', 'CIN', 'COL', 'MIA', 'PIT', 'WSH', 'CWS',
]

significant: dict[str, dict] = {}
all_csv_rows: list = []
all_team_raw: dict = {}   # saved for validation pass

for team in TEAMS_TO_ANALYZE:
    print(f"\n{'='*60}")
    print(f"  {team}")
    print('='*60)

    prop_dict, raw_dict = build_prop_dict(team)
    if len(prop_dict) < 5:
        print(f"  skipping — only {len(prop_dict)} props")
        continue

    all_team_raw[team] = raw_dict

    print(f"\nProp dict ({len(prop_dict)} props):")
    for label, s in prop_dict.items():
        print(f"  {label}: n={len(s)}")

    two_leg   = parlay_finder(prop_dict, n_legs=2, top_n=5, min_n=8)
    three_leg = parlay_finder(prop_dict, n_legs=3, top_n=3, min_n=8)

    if two_leg or three_leg:
        significant[team] = {'2-leg': two_leg, '3-leg': three_leg}
        all_csv_rows.extend(parlay_edge_rows(two_leg,   raw_dict, team))
        all_csv_rows.extend(parlay_edge_rows(three_leg, raw_dict, team))

print("\n\n" + "="*60)
print("SUMMARY - teams with significant parlay combinations")
print("="*60)
if significant:
    for team, res in significant.items():
        n2, n3 = len(res['2-leg']), len(res['3-leg'])
        print(f"  {team}: {n2} two-leg, {n3} three-leg")
else:
    print("  No significant combinations found across any team.")

# ─── CSV EXPORT ───────────────────────────────────────────────────────────────

if all_csv_rows:
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'parlays.csv')
    csv_path = os.path.normpath(csv_path)
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"\nSaved {len(all_csv_rows)} parlay combinations to {csv_path}")

# ─── TRAIN / TEST VALIDATION ──────────────────────────────────────────────────

print("\n\n" + "="*60)
print("TRAIN/TEST VALIDATION")
print("="*60)
print("Finding season midpoint...")
cutoff = find_season_midpoint(all_team_raw)
print(f"Cutoff: {cutoff}\n")

val_rows = []

for team, raw_dict in all_team_raw.items():
    train_raw, test_raw = split_raw_dict(raw_dict, cutoff, min_n=3)
    train_prop = raw_to_prop_dict(train_raw)

    if len(train_prop) < 5:
        continue

    # discover pairs on first-half data only (lower min_n since ~half the games)
    with np.errstate(invalid='ignore', divide='ignore'):
        train_results = parlay_finder(train_prop, n_legs=2, top_n=5, min_n=5,
                                      fdr_q=0.10)

    for res in train_results:
        for ps in res['pairs']:
            a, b = ps['pair']
            train_edge, train_n = pair_edge_on(a, b, train_raw)
            test_edge,  test_n  = pair_edge_on(a, b, test_raw)
            val_rows.append({
                'team':        team,
                'leg_1':       a,
                'leg_2':       b,
                'train_r':     round(ps['r'], 3),
                'train_n':     train_n,
                'train_edge':  round(train_edge, 2) if not np.isnan(train_edge) else np.nan,
                'test_n':      test_n,
                'test_edge':   round(test_edge, 2)  if not np.isnan(test_edge)  else np.nan,
                'edge_held':   (not np.isnan(test_edge)) and test_edge >= 1.2,
            })

val_df = pd.DataFrame(val_rows)

# deduplicate — same pair can appear for multiple combos
val_df = val_df.drop_duplicates(subset=['team', 'leg_1', 'leg_2'])

val_path = os.path.join(os.path.dirname(__file__), '..', 'parlays_validation.csv')
val_df.to_csv(os.path.normpath(val_path), index=False)

# summary stats
valid = val_df.dropna(subset=['test_edge'])
print(f"Pairs selected in training:        {len(val_df)}")
print(f"Pairs with test data (n>=3):       {len(valid)}")
print(f"Pairs where test_edge >= 1.2:      {valid['edge_held'].sum()}  "
      f"({valid['edge_held'].mean():.0%} of those with test data)")
print()
print(f"Median train edge:   {val_df['train_edge'].median():.2f}x")
print(f"Median test  edge:   {valid['test_edge'].median():.2f}x")
print()

# show all pairs sorted by test_edge
print("All validated pairs (sorted by test_edge):")
display_cols = ['team', 'leg_1', 'leg_2', 'train_r', 'train_n',
                'train_edge', 'test_n', 'test_edge']
print(valid.sort_values('test_edge', ascending=False)[display_cols].to_string(index=False))

# distribution of edge ratio change
print("\nEdge ratio: train vs test")
buckets = [
    ('collapsed  (<0.5x)',  valid['test_edge'] < 0.5),
    ('shrunk (0.5-1.0x)',   (valid['test_edge'] >= 0.5) & (valid['test_edge'] < 1.0)),
    ('flat   (1.0-1.2x)',   (valid['test_edge'] >= 1.0) & (valid['test_edge'] < 1.2)),
    ('held   (1.2-2.0x)',   (valid['test_edge'] >= 1.2) & (valid['test_edge'] < 2.0)),
    ('strong (2.0x+)',      valid['test_edge'] >= 2.0),
]
for label, mask in buckets:
    n = mask.sum()
    print(f"  {label}: {n} pairs ({n/len(valid):.0%})")
