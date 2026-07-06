# Sports Parlay Correlation Analysis

Quantitative research pipeline that hunts for **statistically genuine correlations between player/team prop outcomes** in MLB and NBA, using Kalshi prediction-market data as the source of pregame implied probabilities.

Sportsbooks price multi-leg parlays assuming each leg is *independent* — multiplying individual probabilities together. If two legs are actually correlated (e.g. a starting pitcher's strikeouts and his team winning), the true joint probability is higher than the "independent" price implies, and the parlay is underpriced. This project measures that gap directly from historical outcomes and validates it out-of-sample rather than reporting in-sample correlation as if it were edge.

## What it does

1. **Pull markets** — fetch resolved Kalshi prop markets (hits, HR, strikeouts, spreads, totals, etc.) and their pregame implied probabilities via the historical candlesticks API.
2. **Pull outcomes** — cross-reference against real play-by-play/box-score data (`pybaseball` for MLB, `nba_api` for NBA) to know what actually happened, independent of the market's own settlement.
3. **Correlate** — for every pair of props on the same team, compute the Pearson correlation of the "surprise" series (observed outcome − implied probability) across shared games, with a Fisher CI and Benjamini-Hochberg FDR correction across the many pairs tested.
4. **Score edge** — compare the observed joint hit rate to the hit rate implied by independence (`edge_ratio = P(both hit) / (P(a) × P(b))`).
5. **Validate out-of-sample** — split each season at its midpoint, discover correlated pairs on the first half only, and check whether the edge actually holds on the second half. This is the step that separates a real signal from overfitting noise.
6. **Report** — render the surviving combinations to a static HTML report, ranked by a credibility score (`log(edge) × √n × CI-lower`).

## Repo layout

```
src/
  mlb_parlay_scanner.py       MLB: full scanner + train/test validation (pybaseball + Kalshi)
  nba_parlay_scanner.py       NBA: same methodology applied league-wide (Kalshi + nba_api)
  nba_finals_correlation.py   NBA: targeted same-game correlation for a specific Finals matchup,
                               with live order-book cost estimates
  nba_shrinkage_backtest.py   Backtest across 10 historical NBA seasons: does regular-season
                               correlation systematically overstate playoff correlation?
generate_report.py            Builds the MLB HTML report from the scanner's CSV output
examples/                     Sample HTML reports from a prior run, so you can see the output
                               without needing API access
```

## Methodology notes / honesty about limits

- Correlation is measured on **shared resolved games only** — small-n pairs are penalized via the Fisher CI, not just the point estimate.
- The train/test split exists specifically to catch pairs whose correlation is a fluke of one half-season.
- `nba_shrinkage_backtest.py` exists because regular-season correlation is not a safe proxy for playoff correlation (different lineups, different opponent quality, series-length survivorship) — it quantifies the shrinkage instead of assuming it away.
- None of this is a betting recommendation; it's a statistical audit of one specific market-pricing assumption (leg independence).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python src/mlb_parlay_scanner.py      # → parlays.csv, parlays_validation.csv
python generate_report.py             # → parlays_report.html

python src/nba_parlay_scanner.py      # → nba_parlays*.csv, nba_report.html, nba_validation_report.html
python src/nba_finals_correlation.py  # → nba_api_finals_pairs.csv, nba_api_report.html
python src/nba_shrinkage_backtest.py  # → nba_shrinkage_obs.csv
```

All scripts cache upstream API/data pulls under `src/.cache*/` so re-runs are fast and don't hammer either API.

## Stack

Python · pandas / NumPy / SciPy (stats) · `pybaseball` · `nba_api` · Kalshi REST API
