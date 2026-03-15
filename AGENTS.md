# Repository Context

## Purpose
This repo contains a Python cryptocurrency trading bot focused on short-term BTC trading using five-minute candles from `bitcoin_price_history_five_minute_candlesticks.csv`.

The current implementation is a research/backtesting bot first, with a saved model state and a CCXT execution wrapper ready for future live integration.

## Current Strategy
- Feature set: rolling returns, EMA gaps, ATR, RSI, breakout/pullback, and distance from recent highs/lows.
- Learning loop: a custom incremental softmax classifier updates over time with newly labeled candles.
- Trade style: day/swing trades with a tuned hold horizon around `432` bars, which is `36` hours on five-minute candles.
- Position limit: up to `5` open trades.
- Spot fees: `0.5%` per side.
- Margin fees: `0.05%` per side.
- Margin leverage cap: `5x`.
- Best current behavior is mostly selective margin longs during strong bullish continuation regimes, with spot and short logic still available but triggered less often.

## Important Files
- `trading_bot/config.py`: strategy and research defaults plus candidate search space.
- `trading_bot/data.py`: CSV loading, feature engineering, and target creation.
- `trading_bot/model.py`: incremental online classifier and persistence helpers.
- `trading_bot/portfolio.py`: position accounting, fees, exits, and trade logs.
- `trading_bot/backtest.py`: walk-forward simulation, optimization, metrics, and model updates.
- `trading_bot/exchange.py`: CCXT wrapper for spot and margin execution.
- `trading_bot/cli.py`: CLI entrypoints for training and incremental updates.
- `artifacts/bot_state.joblib`: saved trained model state from the latest run.

## Verified Commands
- `python -m trading_bot.cli train --csv bitcoin_price_history_five_minute_candlesticks.csv --output-dir artifacts`
- `python -m trading_bot.cli update-model --state artifacts/bot_state.joblib --csv <new-candles.csv>`

## Latest Results
Run date: March 14, 2026

- Validation: `+12.48%` net return, `7.69%` max drawdown
- Held-out test: `+1.47%` net return, `6.13%` max drawdown
- Best tuned parameters:
  - `horizon_bars=432`
  - `max_hold_bars=432`
  - `cooldown_bars=72`
  - `spot_trades=4`
  - `margin_trades=70`

## Environment Notes
- Local `.deps/` and `.vendor/` folders were unreliable for compiled packages on this machine's Python 3.14 setup.
- Host-level user Python packages were installed so escalated host Python runs work reliably.
- If future work needs reproducible local environments, prefer creating a fresh venv once the Python packaging issue is resolved rather than depending on `.deps/` or `.vendor/`.

## Next Good Improvements
- Add a dedicated live runner that fetches fresh candles and routes orders through `CCXTExecutionClient`.
- Add richer regime detection and exchange-specific margin configuration.
- Add more explicit regression tests around fee handling, leverage, and `update-model`.
