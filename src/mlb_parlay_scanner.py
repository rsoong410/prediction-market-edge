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
SEASON_YEAR  = 2026
SEASON_START = f"{SEASON_YEAR}-01-01"
SEASON_END   = datetime.date.today().isoformat()
CACHE_DIR    = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Train/test split of the season. None = calendar midpoint between the first and
# last game actually played; set e.g. "2026-06-15" to pin the cutoff by hand.
# Games on the split date itself fall in the first half.
SPLIT_DATE  = None
TRAIN_MIN_N = 5   # min shared games for a pair to be discoverable in the first half
TEST_MIN_N  = 3   # min shared games in the second half for the pair to be gradeable
HELD_EDGE   = 1.2 # second-half edge ratio a pair must clear to count as "held"
MAX_PAIR_R  = 0.95 # above this two binary series are effectively the same event, not a relationship

# Running tally of pairs rejected by the pre-FDR gates, so a run that finds nothing
# can still report why. Snapshot and diff it to scope counts to one phase.
FILTER_STATS = {'degenerate': 0, 'zero_joint': 0, 'uninformative_joint': 0}

# Pairs clearing BH, split by the sign of r. Same snapshot-and-diff pattern.
BH_STATS = {'positive': 0, 'negative': 0}

DIRECTIONS = ('positive', 'negative', 'both')
SCAN_DIRECTION = 'both'  # direction used by the run block below; parlay_finder still defaults to 'positive'

# A negatively-correlated parlay is the mirror of a positive one: it HOLDS when the
# joint stays proportionally as far BELOW independence as a positive one sits above.
HELD_EDGE_NEG = 1 / HELD_EDGE

# For a negative pair the signal is a joint that fires less often than independence
# predicts. That is only informative when independence predicted a meaningful number
# of joint hits — with two longshots at n=5 you expect ~0 anyway, so observing 0
# says nothing. Require this many expected joint hits before trusting a negative pair.
MIN_EXPECTED_JOINT = 1.0

PLAYER_SERIES = ['KXMLBHIT', 'KXMLBHR', 'KXMLBHRR', 'KXMLBTB']
PITCHER_SERIES = ['KXMLBKS']
TEAM_SERIES   = ['KXMLBGAME', 'KXMLBSPREAD', 'KXMLBTOTAL', 'KXMLBTEAMTOTAL', 'KXMLBF5SPREAD']

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
          "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
TB_MAP     = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}

# api helpers
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

# roster discovery
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

# caching
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

# statcast path (unused by main build — kept for quick_correlation)
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

# team game results
def get_team_game_results(team_abbr):
    """Game results from pybaseball schedule_and_record."""
    df = schedule_and_record(SEASON_YEAR, team_abbr)
    df = df[df['R'].notna() & df['RA'].notna()].copy()
    # pybaseball Date format: "Apr 1" — add year for parsing
    df['game_date'] = pd.to_datetime(
        df['Date'].str.replace(r'^[A-Za-z]+,\s*', '', regex=True)
                  .str.replace(r'\s*\(\d+\)', '', regex=True) + f' {SEASON_YEAR}',
        format='%b %d %Y'
    ).dt.strftime('%Y-%m-%d')
    df['game_order']   = df.groupby('game_date').cumcount()  # 0 = first game, 1 = second (DH)
    df['runs_scored']  = df['R'].astype(int)
    df['runs_allowed'] = df['RA'].astype(int)
    df['run_diff']     = df['runs_scored'] - df['runs_allowed']
    df['total_runs']   = df['runs_scored'] + df['runs_allowed']
    df['win']          = df['W/L'].str.startswith('W').astype(int)
    return df[['game_date','game_order','runs_scored','runs_allowed','run_diff','total_runs','win']]

# end statcast path
# price fetch
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

# build full prop dict
def build_prop_dict(team_abbr, series_list=None):
    """
    Build prop dict from Kalshi market results — no statcast needed.
    outcome = market result field (yes=1 / no=0).
    Filters to markets whose ticker contains team_abbr.
    """
    if series_list is None:
        series_list = PLAYER_SERIES + PITCHER_SERIES + TEAM_SERIES

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

# correlation & fisher ci
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
    """
    P-value for Pearson r via Fisher z.

    one_sided=True  tests H1: r > 0 (upper tail only). Negative r yields p near 1,
                    so negative pairs are auto-failed by BH — the positive-only search.
    one_sided=False tests H1: r != 0 (both tails). Required whenever negative pairs
                    should be discoverable, since it judges |r| rather than r.

    Default is two-sided; the one-sided path is opt-in so switching direction is
    always an explicit choice at the call site.
    """
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

# parlay finder
def parlay_finder(prop_dict, n_legs=2, top_n=10, min_n=20, fdr_q=0.10,
                  raw_dict=None, max_r=MAX_PAIR_R, direction='positive'):
    """
    Find parlay combinations whose legs are significantly correlated.

    direction:
      'positive' (default) — legs move together; the joint is UNDERpriced under the
                             independence assumption, so the parlay is buyable.
                             Uses one-sided p-values, preserving historical behavior.
      'negative'           — legs move apart; the joint is OVERpriced, so the trade is
                             the other side. Uses two-sided p-values.
      'both'               — report either sign. Two-sided p-values.

    Gates, applied in order:
      1. n >= min_n shared games (drops underpowered pairs before FDR)
      2. |r| < max_r — two binary series that move together almost perfectly are the
         same event wearing two labels (a player's 1+ and 2+ thresholds, a prop listed
         only on games the other was also listed on). These produce r ~ 0.99 on tiny
         samples and would otherwise dominate the rankings.
      3. coherence between r and the observed joint (needs raw_dict), keyed on sign of r:
           r > 0 → the joint must have occurred at least once. A pair claiming positive
                   correlation that never once hit together is the artifact signature:
                   when both legs always lose, surprise collapses to -price and what
                   gets measured is the correlation of the two PRICES.
           r < 0 → a zero joint is coherent with the claim, so it is allowed, but
                   independence must have predicted at least MIN_EXPECTED_JOINT hits.
                   Otherwise "fewer joints than expected" is unmeasurable.
      4. BH-FDR at fdr_q across ALL surviving pairs

    Gates 2 and 3 run before FDR so artifacts don't inflate m and dilute the correction.
    BH runs ONCE over the combined p-value list — positive and negative candidates share
    a single family. Splitting them into two corrections would inflate the true FDR.
    Direction filtering is applied AFTER BH, to the survivors, so the family size m is
    identical no matter which direction is requested.

    Ranked by strength of the CI bound facing away from zero: max CI upper bound
    (most negative) for negative pairs, min CI lower bound for positive pairs.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    labels = list(prop_dict.keys())
    dropped_r = dropped_joint = dropped_uninformative = 0
    # one-sided only for the positive-only search; any search that must surface
    # negative r needs both tails or negative pairs are auto-failed by construction
    one_sided = (direction == 'positive')

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
        if np.isnan(r) or abs(r) >= max_r:
            dropped_r += 1
            continue
        if raw_dict is not None:
            st = pair_stats_on(a, b, raw_dict, min_n=min_n)
            joint, indep = st.get('p_joint', np.nan), st.get('p_indep', np.nan)
            if np.isnan(joint) or np.isnan(indep):
                dropped_joint += 1
                continue
            if r > 0 and joint <= 0:
                dropped_joint += 1
                continue
            if r < 0 and st['n'] * indep < MIN_EXPECTED_JOINT:
                dropped_uninformative += 1
                continue
        lo, hi = fisher_ci(r, n)
        p = _corr_pvalue(r, n, one_sided=one_sided)
        pair_cache[(a, b)] = {'r': r, 'n': n, 'ci': (lo, hi), 'p': p,
                              'direction': 'positive' if r > 0 else 'negative'}

    FILTER_STATS['degenerate'] += dropped_r
    FILTER_STATS['zero_joint'] += dropped_joint
    FILTER_STATS['uninformative_joint'] += dropped_uninformative
    if dropped_r or dropped_joint or dropped_uninformative:
        print(f"  filtered {dropped_r} degenerate (|r|>={max_r}), "
              f"{dropped_joint} incoherent-joint, "
              f"{dropped_uninformative} uninformative-negative pairs before FDR")

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

    # Step 2b: direction filter, applied to BH SURVIVORS only. Doing it here rather
    # than before Step 2 keeps the family size m identical across direction settings,
    # so a pair's significance never depends on which direction was requested.
    bh_by_dir = {'positive': 0, 'negative': 0}
    for k in fdr_pass:
        bh_by_dir[pair_cache[k]['direction']] += 1
    BH_STATS['positive'] += bh_by_dir['positive']
    BH_STATS['negative'] += bh_by_dir['negative']

    if direction != 'both':
        fdr_pass = {k for k in fdr_pass if pair_cache[k]['direction'] == direction}
    if bh_by_dir['positive'] or bh_by_dir['negative']:
        print(f"  BH passed: {bh_by_dir['positive']} positive, {bh_by_dir['negative']} negative "
              f"(of {len(pair_cache)} tested, direction={direction})")

    def _lookup(a, b):
        return pair_cache.get((a, b)) or pair_cache.get((b, a))

    def _key(a, b):
        return (a, b) if (a, b) in pair_cache else (b, a)

    # Step 3: combos where every pair has n >= min_n and passes BH-FDR
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

        dirs = {ps['direction'] for ps in pair_stats}
        combo_dir = dirs.pop() if len(dirs) == 1 else 'mixed'
        # Strength = the CI bound facing AWAY from zero, so positive and negative
        # combos rank on the same scale (higher = further from zero = stronger).
        strength = min((ps['ci'][0] if ps['r'] > 0 else -ps['ci'][1]) for ps in pair_stats)
        results.append({'combo': combo, 'min_lo': min(ps['ci'][0] for ps in pair_stats),
                        'strength': strength, 'direction': combo_dir, 'pairs': pair_stats})

    results.sort(key=lambda x: x['strength'], reverse=True)

    print(f"\nTop {top_n} {n_legs}-leg parlays "
          f"(direction={direction}, BH-FDR q<={fdr_q:.0%}, min_n={min_n}):")
    if not results:
        print("  (no significant combinations — try reducing min_n or gathering more data)")
    for i, res in enumerate(results[:top_n], 1):
        print(f"\n#{i}  [{res['direction']}]  strength={res['strength']:.3f}")
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
        mean_prices = [shared[f'p{i}'].mean() for i in range(n_legs)]
        all_yes = (shared[[f'out{i}' for i in range(n_legs)]] == 1).all(axis=1)
        p_joint = all_yes.mean()
        p_indep = float(np.prod(mean_prices))
        # Raw, un-inverted ratio. > 1: the joint fires more often than independent
        # pricing implies (underpriced, buy). < 1: less often (overpriced, sell/fade).
        # Deliberately NOT abs()'d or flipped — the sign carries the trade direction.
        ratio = p_joint / p_indep if p_indep > 0 else float('nan')

        if np.isnan(ratio) or ratio == 1:
            tradeable_side = ''
        else:
            tradeable_side = 'buy_yes' if ratio > 1 else 'buy_no'

        row = {
            'team':   team,
            'n_legs': n_legs,
            'leg_1':  combo[0],
            'leg_2':  combo[1],
            'leg_3':  combo[2] if n_legs > 2 else '',
            'direction':      res.get('direction', 'positive'),
            'tradeable_side': tradeable_side,
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


# validation helpers
def find_season_split(team_raw_dicts, split_date=None):
    """
    Cutoff game_key separating the first half of the season from the second.

    Default is the calendar midpoint between the first and last game with resolved
    markets — not the median game slot, which skews toward whichever stretch of the
    schedule happened to have the most props listed. Returned cutoff is inclusive:
    games on the split date itself land in the first half.
    """
    all_keys = sorted({k for raw_dict in team_raw_dicts.values()
                         for df in raw_dict.values() for k in df.index})
    if not all_keys:
        raise SystemExit("No resolved games found — nothing to split.")

    first_date = datetime.date.fromisoformat(all_keys[0][:10])
    last_date  = datetime.date.fromisoformat(all_keys[-1][:10])
    split = (datetime.date.fromisoformat(split_date) if split_date
             else first_date + datetime.timedelta(days=(last_date - first_date).days // 2))

    cutoff  = f"{split.isoformat()}-9999"
    n_train = sum(1 for k in all_keys if k <= cutoff)
    n_test  = len(all_keys) - n_train

    print(f"  season spans {first_date} → {last_date} ({len(all_keys)} game slots)")
    print(f"  split at {split} ({'manual' if split_date else 'calendar midpoint'})")
    print(f"  first half: {first_date} → {split}  ({n_train} slots)")
    print(f"  second half: {split + datetime.timedelta(days=1)} → {last_date}  ({n_test} slots)")
    if n_test == 0:
        raise SystemExit("Second half is empty — nothing to validate against.")
    return cutoff


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


_EMPTY_PAIR = {'edge': np.nan, 'r': np.nan, 'n': 0, 'p_joint': np.nan, 'p_indep': np.nan}

def pair_stats_on(label_a, label_b, raw_dict, min_n=3):
    """
    Edge ratio, surprise correlation, and shared-game count for an (a, b) pair
    on the given raw_dict slice. Reporting r alongside edge separates the two ways
    a pair can fail out-of-sample: the correlation vanished, or it survived but the
    joint rate still didn't beat the independent price.
    """
    if label_a not in raw_dict or label_b not in raw_dict:
        return dict(_EMPTY_PAIR)
    da = raw_dict[label_a].rename(columns={'outcome': 'out_a', 'price': 'p_a'})
    db = raw_dict[label_b].rename(columns={'outcome': 'out_b', 'price': 'p_b'})
    shared = da.join(db, how='inner').dropna()
    n = len(shared)
    if n < min_n:
        return dict(_EMPTY_PAIR, n=n)

    p_joint = ((shared['out_a'] == 1) & (shared['out_b'] == 1)).mean()
    p_indep = shared['p_a'].mean() * shared['p_b'].mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (shared['out_a'] - shared['p_a']).corr(shared['out_b'] - shared['p_b']) if n > 2 else np.nan
    return {
        'edge':    p_joint / p_indep if p_indep > 0 else np.nan,
        'r':       r,
        'n':       n,
        'p_joint': p_joint,
        'p_indep': p_indep,
    }


# run
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

    two_leg   = parlay_finder(prop_dict, n_legs=2, top_n=5, min_n=8, raw_dict=raw_dict,
                              direction=SCAN_DIRECTION)
    three_leg = parlay_finder(prop_dict, n_legs=3, top_n=3, min_n=8, raw_dict=raw_dict,
                              direction=SCAN_DIRECTION)

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

# csv export
if all_csv_rows:
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'parlays.csv')
    csv_path = os.path.normpath(csv_path)
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"\nSaved {len(all_csv_rows)} parlay combinations to {csv_path}")

# first half / second half validation
print("\n\n" + "="*60)
print(f"FIRST HALF vs SECOND HALF — {SEASON_YEAR} SEASON")
print("="*60)
cutoff = find_season_split(all_team_raw, SPLIT_DATE)
print()

def _r2(v, places=2):
    return round(v, places) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan

VAL_COLS = ['team', 'leg_1', 'leg_2', 'direction', 'tradeable_side',
            'train_r', 'train_ci_lo', 'train_n', 'train_edge',
            'test_r', 'test_n', 'test_edge', 'r_held', 'edge_held']

def _edge_held(direction, edge):
    """
    Did the out-of-sample edge survive, in the direction the pair was discovered?

    positive: joint must still fire >= HELD_EDGE x more often than independence.
    negative: mirror image — joint must still fire <= 1/HELD_EDGE as often. Using the
              reciprocal rather than a separate constant keeps the two bars equally
              strict in proportional terms.
    """
    if edge is None or np.isnan(edge):
        return False
    return bool(edge >= HELD_EDGE) if direction == 'positive' else bool(edge <= HELD_EDGE_NEG)

def _r_held(direction, r):
    """Correlation kept its sign out of sample."""
    if r is None or np.isnan(r):
        return False
    return bool(r > 0) if direction == 'positive' else bool(r < 0)

val_rows = []
_filters_before = dict(FILTER_STATS)
_bh_before = dict(BH_STATS)

for team, raw_dict in all_team_raw.items():
    train_raw, test_raw = split_raw_dict(raw_dict, cutoff, min_n=TEST_MIN_N)
    train_prop = raw_to_prop_dict(train_raw)

    if len(train_prop) < 5:
        continue

    # discover pairs on first-half data ONLY — the second half is never seen here
    with np.errstate(invalid='ignore', divide='ignore'):
        train_results = parlay_finder(train_prop, n_legs=2, top_n=5,
                                      min_n=TRAIN_MIN_N, fdr_q=0.10,
                                      raw_dict=train_raw, direction=SCAN_DIRECTION)

    for res in train_results:
        for ps in res['pairs']:
            a, b = ps['pair']
            pair_dir = ps['direction']
            tr = pair_stats_on(a, b, train_raw, min_n=TRAIN_MIN_N)
            te = pair_stats_on(a, b, test_raw,  min_n=TEST_MIN_N)
            val_rows.append({
                'team':       team,
                'leg_1':      a,
                'leg_2':      b,
                'direction':  pair_dir,
                'tradeable_side': 'buy_yes' if pair_dir == 'positive' else 'buy_no',
                'train_r':    _r2(ps['r'], 3),
                'train_ci_lo': ps['ci'][0],
                'train_n':    tr['n'],
                'train_edge': _r2(tr['edge']),
                'test_r':     _r2(te['r'], 3),
                'test_n':     te['n'],
                'test_edge':  _r2(te['edge']),
                'r_held':     _r_held(pair_dir, te['r']),
                'edge_held':  _edge_held(pair_dir, te['edge']),
            })

val_df = pd.DataFrame(val_rows, columns=VAL_COLS)

# deduplicate — same pair can appear for multiple combos
val_df = val_df.drop_duplicates(subset=['team', 'leg_1', 'leg_2'])

val_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'parlays_validation.csv'))
val_df.to_csv(val_path, index=False)
print(f"\nSaved {len(val_df)} first-half pairs to {val_path}")

val_filtered = {k: FILTER_STATS[k] - _filters_before[k] for k in FILTER_STATS}
val_bh       = {k: BH_STATS[k] - _bh_before[k] for k in BH_STATS}
valid = val_df.dropna(subset=['test_edge'])

pos_df, neg_df = val_df[val_df['direction'] == 'positive'], val_df[val_df['direction'] == 'negative']
pos_valid = valid[valid['direction'] == 'positive']
neg_valid = valid[valid['direction'] == 'negative']

# An empty result is a real finding, not an error — it still gets a report explaining
# which gate emptied the set, so a stale report from a previous run is never left behind.
print(f"\nPairs discovered in first half:        {len(val_df)}  "
      f"({len(pos_df)} positive, {len(neg_df)} negative)")
print(f"Pairs gradeable in second half (n>={TEST_MIN_N}): {len(valid)}  "
      f"({len(pos_valid)} positive, {len(neg_valid)} negative)")
print(f"BH passed in validation: {val_bh['positive']} positive, {val_bh['negative']} negative")
print(f"Rejected pre-FDR: {val_filtered['degenerate']} degenerate (|r|>={MAX_PAIR_R}), "
      f"{val_filtered['zero_joint']} incoherent joint, "
      f"{val_filtered['uninformative_joint']} uninformative negative")
for _dname, _dvalid in (('positive', pos_valid), ('negative', neg_valid)):
    if len(_dvalid):
        _bar = f">= {HELD_EDGE}" if _dname == 'positive' else f"<= {HELD_EDGE_NEG:.3f}"
        print(f"  {_dname}: {int(_dvalid['edge_held'].sum())}/{len(_dvalid)} held "
              f"(edge {_bar}), median test edge {_dvalid['test_edge'].median():.2f}x")

buckets = [
    ('collapsed  (<0.5x)',  valid['test_edge'] < 0.5),
    ('shrunk (0.5-1.0x)',   (valid['test_edge'] >= 0.5) & (valid['test_edge'] < 1.0)),
    ('flat   (1.0-1.2x)',   (valid['test_edge'] >= 1.0) & (valid['test_edge'] < 1.2)),
    ('held   (1.2-2.0x)',   (valid['test_edge'] >= 1.2) & (valid['test_edge'] < 2.0)),
    ('strong (2.0x+)',      valid['test_edge'] >= 2.0),
]

if valid.empty:
    print("\nNothing to grade — no first-half pair cleared the gates with enough "
          "second-half games.")
else:
    print(f"Correlation kept its sign:             {valid['r_held'].sum()}  ({valid['r_held'].mean():.0%})")
    print(f"Edge held in its own direction:        {valid['edge_held'].sum()}  ({valid['edge_held'].mean():.0%})")
    print()
    print(f"Median r:     first half {val_df['train_r'].median():.3f}  →  second half {valid['test_r'].median():.3f}")
    print(f"Median edge:  first half {val_df['train_edge'].median():.2f}x  →  second half {valid['test_edge'].median():.2f}x")
    print()

    print("All graded pairs (sorted by second-half edge):")
    display_cols = ['team', 'leg_1', 'leg_2', 'direction', 'tradeable_side',
                    'train_r', 'train_n', 'train_edge', 'test_r', 'test_n', 'test_edge']
    print(valid.sort_values('test_edge', ascending=False)[display_cols].to_string(index=False))

    print("\nSecond-half edge ratio distribution")
    for label, mask in buckets:
        n = int(mask.sum())
        print(f"  {label}: {n} pairs ({n/len(valid):.0%})")

# html report
def _fmt(v, places=2, suffix=''):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:.{places}f}{suffix}"

def _val_rows_html(df):
    out = []
    for _, r in df.iterrows():
        held = r['edge_held']
        verdict = ("<span class='held'>HELD</span>" if held else
                   "<span class='faded'>faded</span>")
        drift = r['test_edge'] - r['train_edge'] if not (
            np.isnan(r['test_edge']) or np.isnan(r['train_edge'])) else np.nan
        out.append(f"""
        <tr class="{'row-held' if held else ''}">
          <td><span class='team-badge'>{r['team']}</span></td>
          <td class='legs-cell'><span class='leg'>{r['leg_1']}</span><span class='leg'>{r['leg_2']}</span></td>
          <td class='num'>{_fmt(r['train_r'], 3)}</td>
          <td class='num'>{r['train_n']}</td>
          <td class='num'>{_fmt(r['train_edge'], 2, '×')}</td>
          <td class='num sep'>{_fmt(r['test_r'], 3)}</td>
          <td class='num'>{r['test_n']}</td>
          <td class='num'><strong>{_fmt(r['test_edge'], 2, '×')}</strong></td>
          <td class='num'>{_fmt(drift, 2, '×')}</td>
          <td>{verdict}</td>
        </tr>""")
    return '\n'.join(out)

VAL_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; padding: 32px 24px; }
h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; color: #f8fafc; }
.subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 28px; }
h2 { font-size: 1.15rem; font-weight: 600; color: #f1f5f9;
     border-left: 3px solid #3b82f6; padding-left: 12px; margin-bottom: 6px; }
.desc { font-size: 0.8rem; color: #64748b; margin-bottom: 14px; padding-left: 15px; }
section { margin-bottom: 48px; }
.cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 36px; }
.card { background: #1e293b; border-radius: 8px; padding: 14px 18px; min-width: 150px; }
.card .k { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;
           letter-spacing: 0.06em; margin-bottom: 4px; }
.card .v { font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
.card .sub { font-size: 0.72rem; color: #64748b; margin-top: 2px; }
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid #1e293b; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
thead th { background: #1e293b; color: #94a3b8; font-weight: 600;
           padding: 10px 12px; text-align: left; white-space: nowrap;
           border-bottom: 1px solid #334155; cursor: help; }
tbody tr { border-bottom: 1px solid #1e293b; }
tbody tr:hover { background: #1a2235; }
tr.row-held { background: rgba(52, 211, 153, 0.06); }
td { padding: 9px 12px; vertical-align: middle; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.sep { border-left: 1px solid #334155; }
th.sep { border-left: 1px solid #475569; }
.legs-cell { line-height: 1.6; }
.leg { display: block; }
.team-badge { background: #1e293b; color: #cbd5e1; border-radius: 4px;
              padding: 3px 8px; font-size: 0.78rem; font-weight: 700;
              letter-spacing: 0.05em; white-space: nowrap; }
.held { color: #34d399; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.05em; }
.faded { color: #64748b; font-size: 0.75rem; }
.note { background: #1e293b; border-radius: 8px; padding: 18px 22px;
        font-size: 0.8rem; color: #94a3b8; line-height: 1.6; margin-bottom: 36px; }
.note h3 { font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px; }
.note ul { margin: 8px 0 0 18px; }
"""

def _agg(series, fmt="{:.2f}", how='median'):
    """Format an aggregate, or an em dash when there is nothing to aggregate."""
    if len(series) == 0 or series.isna().all():
        return '—'
    return fmt.format(series.median() if how == 'median' else series.mean())

split_day  = cutoff[:10]
# Positive sections keep their original semantics: only positive-direction pairs,
# ranked by edge descending (furthest ABOVE 1.0 first).
held_df    = (pos_valid[pos_valid['edge_held']].sort_values('test_edge', ascending=False)
              if len(pos_valid) else pos_valid)
faded_df   = (pos_valid[~pos_valid['edge_held']].sort_values('test_edge', ascending=False)
              if len(pos_valid) else pos_valid)
# Negative pairs rank the other way: furthest BELOW 1.0 first, so ascending edge.
neg_sorted = neg_valid.sort_values('test_edge', ascending=True) if len(neg_valid) else neg_valid

empty_banner = ''
if valid.empty:
    empty_banner = f"""
<div class='note' style='border-left:3px solid #fbbf24'>
  <h3>No pairs survived to grading</h3>
  Discovery ran on games through {split_day} and produced
  <strong>{len(val_df)}</strong> gradeable pair(s). Before FDR, the gates rejected
  <strong>{val_filtered['degenerate']}</strong> pairs as degenerate (|r| &ge; {MAX_PAIR_R} —
  two labels for the same event) and <strong>{val_filtered['zero_joint']}</strong> whose legs
  never once hit together.
  <ul>
    <li>This is <em>not</em> evidence that correlation edge is absent — it means the
        first half alone did not contain enough co-listed games to support a single
        defensible pair at min_n={TRAIN_MIN_N}.</li>
    <li>Kalshi lists a given prop on only a subset of games, so halving an already
        sparse panel leaves very few overlapping observations per pair.</li>
    <li>Re-run once the full season has resolved, or pool pairs by prop type rather
        than by named player, to put real power behind this test.</li>
  </ul>
</div>"""
bucket_html = ''.join(
    f"<div class='card'><div class='k'>{lbl.split('(')[0].strip()}</div>"
    f"<div class='v'>{int(mask.sum()) if len(valid) else 0}</div>"
    f"<div class='sub'>{lbl[lbl.find('('):]}</div></div>"
    for lbl, mask in buckets
)

val_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB {SEASON_YEAR} — First Half vs Second Half</title>
<style>{VAL_CSS}</style>
</head>
<body>
<h1>MLB {SEASON_YEAR}: First Half vs Second Half</h1>
<p class='subtitle'>Pairs discovered on games through <strong>{split_day}</strong>, then graded on
everything after · Kalshi market data · BH-FDR q=10% · generated {SEASON_END}</p>

<div class='cards'>
  <div class='card'><div class='k'>Discovered</div><div class='v'>{len(val_df)}</div><div class='sub'>first-half pairs</div></div>
  <div class='card'><div class='k'>Positive</div><div class='v'>{len(pos_df)}</div><div class='sub'>{len(pos_valid)} gradeable</div></div>
  <div class='card'><div class='k'>Negative</div><div class='v'>{len(neg_df)}</div><div class='sub'>{len(neg_valid)} gradeable</div></div>
  <div class='card'><div class='k'>Gradeable</div><div class='v'>{len(valid)}</div><div class='sub'>n&nbsp;&ge;&nbsp;{TEST_MIN_N} after split</div></div>
  <div class='card'><div class='k'>Edge held</div><div class='v'>{int(valid['edge_held'].sum()) if len(valid) else 0}</div><div class='sub'>{_agg(valid['edge_held'], '{:.0%}', 'mean')} of graded</div></div>
  <div class='card'><div class='k'>r still &gt; 0</div><div class='v'>{int(valid['r_held'].sum()) if len(valid) else 0}</div><div class='sub'>{_agg(valid['r_held'], '{:.0%}', 'mean')} of graded</div></div>
  <div class='card'><div class='k'>Median edge</div><div class='v'>{_agg(valid['test_edge'], '{:.2f}×')}</div><div class='sub'>was {_agg(val_df['train_edge'], '{:.2f}×')} in H1</div></div>
</div>
{empty_banner}

<div class='note'>
  <h3>How to read this</h3>
  Every pair here was selected using <em>only</em> games on or before {split_day} — the second-half
  columns are genuinely out-of-sample. A pair counts as <span class='held'>HELD</span> when its
  second-half edge ratio stays on the side it was discovered on: ≥ {HELD_EDGE}× for positive
  pairs, ≤ {HELD_EDGE_NEG:.3f}× for negative ones.
  <ul>
    <li><strong>r</strong> — Pearson correlation of the surprise series (outcome − implied price).</li>
    <li><strong>Edge</strong> — observed joint hit rate ÷ the rate implied by pricing the legs independently.
        Shown raw and un-inverted, so &gt; 1 means the joint is underpriced and &lt; 1 means overpriced.</li>
    <li><strong>Δ Edge</strong> — second half minus first half. For positive pairs a large negative
        drift is the overfit tell; for negative pairs it is a large positive drift.</li>
    <li>Second-half samples are small and the season is still running, so treat individual rows as
        weak evidence; the aggregate hold rate is the meaningful number.</li>
  </ul>
</div>

<section>
  <h2>Held up — positive ({len(held_df)})</h2>
  <p class='desc'>Positively-correlated pairs whose second-half edge is still ≥ {HELD_EDGE}×. Buy Yes.</p>
  <div class='table-wrap'>
  <table>
    <thead><tr>
      <th>Team</th><th>Props</th>
      <th title="First-half correlation">H1 r</th>
      <th title="First-half shared games">H1 n</th>
      <th title="First-half edge ratio">H1 edge</th>
      <th class='sep' title="Second-half correlation">H2 r</th>
      <th title="Second-half shared games">H2 n</th>
      <th title="Second-half edge ratio">H2 edge</th>
      <th title="Change in edge ratio">&Delta; edge</th>
      <th>Verdict</th>
    </tr></thead>
    <tbody>{_val_rows_html(held_df) or "<tr><td colspan='10'>Nothing held.</td></tr>"}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>Negative — overpriced joints ({len(neg_sorted)})</h2>
  <p class='desc'>Negatively-correlated pairs: the legs fire together <em>less</em> often than
  independent pricing implies, so the joint is overpriced. Sorted by how far below 1.0 the
  second-half edge sits — furthest below first. Held at ≤ {HELD_EDGE_NEG:.3f}×. Buy No.</p>
  <div class='table-wrap'>
  <table>
    <thead><tr>
      <th>Team</th><th>Props</th>
      <th>H1 r</th><th>H1 n</th><th>H1 edge</th>
      <th class='sep'>H2 r</th><th>H2 n</th><th>H2 edge</th>
      <th>&Delta; edge</th><th>Verdict</th>
    </tr></thead>
    <tbody>{_val_rows_html(neg_sorted) or "<tr><td colspan='10'>No negative pairs survived to grading.</td></tr>"}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>Faded — positive ({len(faded_df)})</h2>
  <p class='desc'>Discovered as positive in the first half but the edge did not survive the second.</p>
  <div class='table-wrap'>
  <table>
    <thead><tr>
      <th>Team</th><th>Props</th>
      <th>H1 r</th><th>H1 n</th><th>H1 edge</th>
      <th class='sep'>H2 r</th><th>H2 n</th><th>H2 edge</th>
      <th>&Delta; edge</th><th>Verdict</th>
    </tr></thead>
    <tbody>{_val_rows_html(faded_df) or "<tr><td colspan='10'>Nothing faded.</td></tr>"}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>Second-half edge distribution</h2>
  <p class='desc'>Where the {len(valid)} graded pairs landed.</p>
  <div class='cards'>{bucket_html}</div>
</section>
</body>
</html>"""

report_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'mlb_validation_report.html'))
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(val_html)
print(f"\nSaved {report_path}")
