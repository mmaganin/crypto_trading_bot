# crypto_trading_bot
An automated cryptocurrency trading bot research project built in Python around BTC five-minute candlestick history.

The current bot:

- trains on `bitcoin_price_history_five_minute_candlesticks.csv`
- builds technical features from BTC OHLCV data
- uses an incremental online classifier that updates as new candles arrive
- supports both spot and margin trade simulation
- enforces the requested fee model: `0.5%` per side for spot and `0.05%` per side for margin
- caps margin leverage at `5x`
- limits concurrency to `5` open trades
- focuses on short-term swing/day-style trades, with the tuned strategy currently holding positions for up to about `36` hours

## Commands

Train, optimize, and save the latest bot state:

```powershell
python -m trading_bot.cli train --csv bitcoin_price_history_five_minute_candlesticks.csv --output-dir artifacts
```

Incrementally update the saved model with newly arrived candles:

```powershell
python -m trading_bot.cli update-model --state artifacts/bot_state.joblib --csv path\to\new_candles.csv
```

## Current Backtest Snapshot

Latest run on March 14, 2026:

- Validation split: `+12.48%` net return, `7.69%` max drawdown
- Held-out test split: `+1.47%` net return, `6.13%` max drawdown
- Best tuned configuration: `432`-bar horizon, `432`-bar max hold, `72`-bar cooldown

Artifacts are written to `artifacts/`:

- `test_metrics.json`
- `test_equity_curve.csv`
- `test_trades.csv`
- `bot_state.joblib`

## Exchange Execution

`trading_bot/exchange.py` includes a `CCXTExecutionClient` wrapper for spot and margin order submission. Exchange-specific margin details still vary by venue, so treat live order execution as a supervised next step rather than blindly enabling it.
