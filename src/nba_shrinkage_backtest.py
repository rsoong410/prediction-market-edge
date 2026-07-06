"""
Backtest: does regular-season same-game correlation systematically overstate
playoff same-game correlation?

Hard ceiling (stated in output): we cannot measure 2026 Finals correlation in
advance, and per-pair playoff samples are tiny.  This estimates the general
reg-season → playoff shrinkage from historical playoffs and compares it to the
Kalshi RFQ maker's implied haircut (~0.3 on our 0.35 = their 0.10).  A
favorable result would NOT prove edge on specific 2026 pairs.

Method:
  For each season 2015-16 … 2024-25:
    For each same-team player pair where both appear in reg-season AND playoffs:
      - Compute reg-season median threshold per stat from reg-season games
      - Build binary series at that threshold for BOTH reg-season and playoff games
      - Compute Pearson phi (= phi coefficient on binary data) in each period
  Pool all (reg_r, po_r) observations.
  Regression + mean-ratio + joint-prediction error.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, time, datetime, itertools
import pandas as pd
import numpy as np
from scipy.stats import linregress, t as t_dist
from nba_api.stats.endpoints import leaguegamelog

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SRC_DIR   = os.path.dirname(__file__)
CACHE_DIR = os.path.join(SRC_DIR, ".cache_shrink")
OUT_DIR   = os.path.join(SRC_DIR, "..")
os.makedirs(CACHE_DIR, exist_ok=True)

STAT_COLS = ['PTS', 'REB', 'AST', 'FG3M']

# Seasons to backtest — intentionally excludes the ongoing 2025-26 season
SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24', '2024-25',
]

MIN_REG_N = 30   # minimum shared reg-season games per pair
MIN_PO_N  = 5    # minimum shared playoff games per pair

MAKER_HAIRCUT = 0.30   # implied from Kalshi quotes: their ~0.10 / our ~0.35


# ─── DATA FETCHING ───────────────────────────────────────────────────────────

def fetch_logs(season: str, season_type: str, max_retries: int = 3) -> pd.DataFrame:
    key   = 'reg' if 'Regular' in season_type else 'po'
    fname = f"logs_{season}_{key}.csv"
    path  = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    for attempt in range(max_retries):
        try:
            print(f"    API: {season} {season_type} (attempt {attempt+1})")
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation='P',
                timeout=90,
            )
            df = log.get_data_frames()[0]
            df.to_csv(path, index=False)
            time.sleep(1.2)
            return df
        except Exception as e:
            print(f"    Retry ({e})")
            time.sleep(3.0)
    raise RuntimeError(f"Failed to fetch {season} {season_type}")


def build_player_dict(df: pd.DataFrame) -> dict:
    """
    {player_name: {'team': str, 'games': DataFrame(index=date, cols=STAT_COLS)}}
    Uses primary team (most games) — handles mid-season trades.
    """
    raw = {}
    for _, row in df.iterrows():
        name  = str(row['PLAYER_NAME'])
        date  = str(row['GAME_DATE'])[:10]
        team  = str(row['TEAM_ABBREVIATION'])
        stats = {s: int(row.get(s) or 0) for s in STAT_COLS}
        raw.setdefault(name, []).append({'date': date, 'team': team, **stats})

    result = {}
    for name, rows in raw.items():
        sub  = pd.DataFrame(rows)
        team = sub['team'].value_counts().index[0]
        games = (sub[sub['team'] == team]
                 .drop_duplicates('date')
                 .set_index('date')
                 [STAT_COLS])
        result[name] = {'team': team, 'games': games}
    return result


# ─── PER-SEASON ANALYSIS ─────────────────────────────────────────────────────

def _var_ok(series: pd.Series, min_each: int = 2) -> bool:
    """True if both 0 and 1 appear at least min_each times."""
    s = int(series.sum())
    return s >= min_each and (len(series) - s) >= min_each


def process_season(season: str) -> list:
    reg_df = fetch_logs(season, 'Regular Season')
    po_df  = fetch_logs(season, 'Playoffs')

    reg_dict = build_player_dict(reg_df)
    po_dict  = build_player_dict(po_df)

    po_teams = {d['team'] for d in po_dict.values()}

    # group reg-season players by team (restrict to playoff teams)
    team_players: dict = {}
    for pname, pdata in reg_dict.items():
        team = pdata['team']
        if team in po_teams:
            team_players.setdefault(team, []).append(pname)

    obs = []
    for team, players in team_players.items():
        po_eligible = {p for p, d in po_dict.items() if d['team'] == team}
        eligible    = [p for p in players if p in po_eligible]
        if len(eligible) < 2:
            continue

        for pa, pb in itertools.combinations(eligible, 2):
            rg_a = reg_dict[pa]['games']
            rg_b = reg_dict[pb]['games']
            po_a = po_dict[pa]['games']
            po_b = po_dict[pb]['games']

            for stat_a in STAT_COLS:
                if stat_a not in rg_a.columns or stat_a not in po_a.columns:
                    continue
                reg_col_a = rg_a[stat_a].fillna(0).astype(int)
                thresh_a  = max(1, int(np.floor(reg_col_a.median())))
                reg_bin_a = (reg_col_a >= thresh_a).astype(int)
                po_bin_a  = (po_a[stat_a].fillna(0).astype(int) >= thresh_a).astype(int)

                for stat_b in STAT_COLS:
                    if stat_b not in rg_b.columns or stat_b not in po_b.columns:
                        continue
                    reg_col_b = rg_b[stat_b].fillna(0).astype(int)
                    thresh_b  = max(1, int(np.floor(reg_col_b.median())))
                    reg_bin_b = (reg_col_b >= thresh_b).astype(int)
                    po_bin_b  = (po_b[stat_b].fillna(0).astype(int) >= thresh_b).astype(int)

                    # inner-join on shared game dates
                    reg_sh = pd.concat(
                        [reg_bin_a.rename('a'), reg_bin_b.rename('b')],
                        axis=1, join='inner'
                    ).dropna()
                    po_sh  = pd.concat(
                        [po_bin_a.rename('a'), po_bin_b.rename('b')],
                        axis=1, join='inner'
                    ).dropna()

                    if len(reg_sh) < MIN_REG_N or len(po_sh) < MIN_PO_N:
                        continue
                    # phi requires both outcomes to appear in each period
                    if not all(_var_ok(reg_sh[c]) for c in ('a','b')):
                        continue
                    if not all(_var_ok(po_sh[c]) for c in ('a','b')):
                        continue

                    with np.errstate(invalid='ignore'):
                        reg_r = float(reg_sh['a'].corr(reg_sh['b']))
                        po_r  = float(po_sh['a'].corr(po_sh['b']))
                    if np.isnan(reg_r) or np.isnan(po_r):
                        continue

                    # playoff marginals (for joint prediction)
                    p_a  = float(po_sh['a'].mean())
                    p_b  = float(po_sh['b'].mean())
                    # bivariate Bernoulli joint using reg_r but playoff marginals
                    pred = p_a*p_b + reg_r * np.sqrt(p_a*(1-p_a)*p_b*(1-p_b))
                    pred = float(np.clip(pred, 0, 1))
                    real = float(((po_sh['a']==1) & (po_sh['b']==1)).mean())

                    obs.append({
                        'season':     season,
                        'team':       team,
                        'name_a':     pa,
                        'stat_a':     stat_a,
                        'thresh_a':   thresh_a,
                        'name_b':     pb,
                        'stat_b':     stat_b,
                        'thresh_b':   thresh_b,
                        'n_reg':      len(reg_sh),
                        'n_po':       len(po_sh),
                        'reg_r':      round(reg_r, 4),
                        'po_r':       round(po_r,  4),
                        'p_po_a':     round(p_a,   4),
                        'p_po_b':     round(p_b,   4),
                        'pred_joint': round(pred,  4),
                        'real_joint': round(real,  4),
                    })
    return obs


# ─── POOLED ANALYSIS ─────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame):
    n = len(df)

    print()
    print("=" * 65)
    print("POOLED DATASET")
    print(f"  Seasons:                 {df['season'].nunique()}  "
          f"({df['season'].min()} – {df['season'].max()})")
    print(f"  Observations (pair×stat):  {n:,}")
    print(f"  Unique player pairs:       {df.groupby(['season','team','name_a','name_b']).ngroups:,}")
    print(f"  Unique team-seasons:       {df.groupby(['season','team']).ngroups:,}")
    print(f"  Total playoff pair-games:  {df['n_po'].sum():,}")
    print(f"  Median reg-season n:       {df['n_reg'].median():.0f}")
    print(f"  Median playoff n:          {df['n_po'].median():.0f}  "
          f"(range {df['n_po'].min()}–{df['n_po'].max()})")
    print(f"  Mean reg_r:                {df['reg_r'].mean():.3f}  "
          f"(std {df['reg_r'].std():.3f})")
    print(f"  Mean po_r:                 {df['po_r'].mean():.3f}  "
          f"(std {df['po_r'].std():.3f})")

    # ── 1. Regression: po_r ~ reg_r ──────────────────────────────────────────
    x = df['reg_r'].values
    y = df['po_r'].values
    slope, intercept, rval, pval, se = linregress(x, y)
    ci_half = t_dist.ppf(0.975, df=n-2) * se

    print()
    print("── Regression: po_r ~ reg_r ──────────────────────────────────────")
    print(f"  slope β:    {slope:+.3f}  [95% CI: {slope-ci_half:+.3f}, {slope+ci_half:+.3f}]")
    print(f"  intercept:  {intercept:+.3f}")
    print(f"  R² (of regression):  {rval**2:.3f}  (Pearson r = {rval:.3f})")
    print(f"  p-value:    {pval:.2e}")
    print(f"  NOTE: attenuation bias pushes β toward zero — treat as lower bound on true slope.")

    # ── 2. Mean ratio po_r / reg_r (only positive reg_r pairs) ──────────────
    pos = df[df['reg_r'] > 0.05].copy()
    ratios = pos['po_r'] / pos['reg_r']
    mu_r   = ratios.mean()
    std_r  = ratios.std()
    n_r    = len(ratios)
    ci_r   = t_dist.ppf(0.975, df=n_r-1) * std_r / np.sqrt(n_r)
    med_r  = ratios.median()

    print()
    print("── Mean ratio po_r / reg_r  (where reg_r > 0.05) ─────────────────")
    print(f"  N:           {n_r:,}")
    print(f"  Mean ratio:  {mu_r:.3f}  [95% CI: {mu_r-ci_r:.3f}, {mu_r+ci_r:.3f}]")
    print(f"  Median:      {med_r:.3f}")
    print(f"  Std dev:     {std_r:.3f}")

    # ── 3. Joint over-prediction ──────────────────────────────────────────────
    df['joint_err'] = df['pred_joint'] - df['real_joint']
    mu_e   = df['joint_err'].mean()
    std_e  = df['joint_err'].std()
    ci_e   = t_dist.ppf(0.975, df=n-1) * std_e / np.sqrt(n)

    # Split by sign of reg_r to isolate positive-correlation pairs
    pos_reg = df[df['reg_r'] > 0]
    mu_e_pos  = pos_reg['joint_err'].mean()
    std_e_pos = pos_reg['joint_err'].std()
    ci_e_pos  = t_dist.ppf(0.975, df=len(pos_reg)-1) * std_e_pos / np.sqrt(len(pos_reg))

    print()
    print("── Bivariate Bernoulli joint over-prediction (pred − realized) ────")
    print(f"  All pairs (N={n:,}):")
    print(f"    Mean error: {mu_e:+.4f}  [95% CI: {mu_e-ci_e:+.4f}, {mu_e+ci_e:+.4f}]")
    print(f"  Positive-reg_r pairs only (N={len(pos_reg):,}):")
    print(f"    Mean error: {mu_e_pos:+.4f}  [95% CI: {mu_e_pos-ci_e_pos:+.4f}, {mu_e_pos+ci_e_pos:+.4f}]")
    print(f"  Interpretation: positive = model over-predicts playoff joint probability.")

    # ── 4. Decile breakdown ───────────────────────────────────────────────────
    df['reg_r_decile'] = pd.cut(df['reg_r'], bins=5,
                                 labels=['<-0.2','-0.2–0','-0–0.2','0.2–0.4','>0.4'])
    agg = df.groupby('reg_r_decile', observed=True).agg(
        n=('po_r','count'),
        mean_po_r=('po_r','mean'),
        mean_reg_r=('reg_r','mean'),
    ).reset_index()
    agg['ratio'] = agg['mean_po_r'] / agg['mean_reg_r']

    print()
    print("── Mean po_r by reg_r quintile ───────────────────────────────────")
    print(f"  {'reg_r bin':<12} {'n':>6} {'mean_reg_r':>12} {'mean_po_r':>10} {'ratio':>8}")
    for _, row in agg.iterrows():
        ratio_s = f"{row['ratio']:.2f}" if not np.isnan(row['ratio']) and abs(row['ratio']) < 100 else "  —"
        print(f"  {str(row['reg_r_decile']):<12} {int(row['n']):>6} {row['mean_reg_r']:>12.3f} "
              f"{row['mean_po_r']:>10.3f} {ratio_s:>8}")

    # ── 5. VERDICT ────────────────────────────────────────────────────────────
    slope_lo = slope - ci_half
    slope_hi = slope + ci_half

    print()
    print("=" * 65)
    print("VERDICT")
    print()
    print(f"  Maker implied haircut:   {MAKER_HAIRCUT:.2f}  "
          f"(their ~0.10 / our ~0.35 on Finals pairs)")
    print(f"  Empirical β [CI]:        {slope:.3f}  [{slope_lo:.3f}, {slope_hi:.3f}]")
    print(f"  Empirical mean ratio:    {mu_r:.3f}  [{mu_r-ci_r:.3f}, {mu_r+ci_r:.3f}]")
    print()

    # Decision rule.
    # β is the empirical "fraction of reg_r that persists to playoffs."
    # MAKER_HAIRCUT is the fraction the maker already credits in their pricing.
    # If β > MAKER_HAIRCUT: maker under-credits, edge is plausible.
    # If β < MAKER_HAIRCUT: maker over-credits, parlay is expensive vs. history.
    if slope_lo > MAKER_HAIRCUT:
        verdict = (f"MAKER TOO AGGRESSIVE — β lower bound {slope_lo:.2f} > "
                   f"maker's {MAKER_HAIRCUT:.2f} haircut factor. "
                   "Historically more correlation persists than the maker credits. "
                   "Edge plausible — but see hard ceiling.")
    elif slope_hi > MAKER_HAIRCUT:
        verdict = (f"INDETERMINATE — β CI [{slope_lo:.2f}, {slope_hi:.2f}] straddles "
                   f"maker's {MAKER_HAIRCUT:.2f}. Cannot distinguish. Default: no edge.")
    else:
        # β upper bound is BELOW the maker's haircut factor.
        # The maker already credits MORE correlation than history warrants.
        verdict = (f"NO EDGE — β CI [{slope_lo:.2f}, {slope_hi:.2f}] is entirely BELOW "
                   f"maker's {MAKER_HAIRCUT:.2f} haircut factor. "
                   f"History: only ~{slope:.0%} of reg_r persists to playoffs; "
                   f"for our reg_r~0.35 that is ~{slope*0.35:.2f}, "
                   f"while the maker already credits ~0.10. "
                   "The maker over-credits correlation vs. history. "
                   "Buying this parlay pays above historically-justified fair value.")

    print(f"  {verdict}")
    print()
    print("  HARD CEILING:")
    print("  We cannot measure the 2026 Finals correlation in advance, and per-pair")
    print("  playoff samples are tiny.  This test estimates the GENERAL reg-season →")
    print("  playoff shrinkage from historical playoffs and compares it to the maker's")
    print("  haircut.  A favorable result would NOT prove edge on the specific 2026")
    print("  Finals pairs — only that the maker's haircut is, on average, too aggressive.")
    print()
    print("  Additional caveats:")
    print("  · Attenuation bias in reg_r biases β downward — true slope may be higher.")
    print("  · Observations are clustered by team×season×player-pair; raw CIs are tight.")
    print("  · The Finals context (intense scouting, small n) may differ from average.")

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=== NBA Reg-Season → Playoff Correlation Shrinkage Backtest ===")
    print(f"  Seasons: {SEASONS[0]} – {SEASONS[-1]}  ({len(SEASONS)} seasons)")
    print(f"  Min reg n: {MIN_REG_N}  |  Min playoff n: {MIN_PO_N}")
    print()

    all_obs = []
    for season in SEASONS:
        print(f"[{season}]")
        try:
            obs = process_season(season)
            print(f"  → {len(obs):,} observations")
            all_obs.extend(obs)
        except Exception as e:
            print(f"  ERROR: {e} — skipping")

    if not all_obs:
        print("No observations collected. Exiting.")
        return

    df = pd.DataFrame(all_obs)

    # Save raw data
    csv_path = os.path.join(OUT_DIR, "nba_shrinkage_obs.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw observations: {os.path.basename(csv_path)} ({len(df):,} rows)")

    df = analyze(df)

    # Quick season-by-season summary
    print()
    print("── Season-by-season summary ──────────────────────────────────────")
    print(f"  {'Season':<10} {'N obs':>7} {'po_games':>9} {'mean reg_r':>11} {'mean po_r':>10} {'β approx':>9}")
    for season in sorted(df['season'].unique()):
        sub = df[df['season']==season]
        x   = sub['reg_r'].values
        y   = sub['po_r'].values
        if len(x) < 10:
            b = float('nan')
        else:
            b = linregress(x, y).slope
        print(f"  {season:<10} {len(sub):>7,} {sub['n_po'].sum():>9,} "
              f"{sub['reg_r'].mean():>11.3f} {sub['po_r'].mean():>10.3f} "
              f"{b:>9.3f}")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
