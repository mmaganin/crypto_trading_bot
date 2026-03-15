# Repository Context

## Purpose
This repo contains a Python cryptocurrency trading bot focused on short-term BTC trading from historical candle CSVs.

The research/backtesting pipeline now supports any regular candle interval up to `4h`, with `5m` remaining the baseline interval the strategy was originally tuned and benchmarked on.

The current implementation is a research/backtesting bot first, with a saved model state and a CCXT execution wrapper ready for future live integration.

## Current Strategy
- Feature set: rolling returns, EMA gaps, ATR, RSI, breakout/pullback, and distance from recent highs/lows.
- Data contract: minimum required input is `timestamp + OHLC`; `volume`, `vwap`, and `transactions` are now optional.
- Interval handling: the loader auto-detects the candle interval, normalizes the BTC CSV's naive wall-clock spring-DST skips (for example `01:55 -> 03:00`) before interval inference, rejects intervals above `4h`, and rescales bar-based windows/horizons from the original `5m` baseline.
- Learning loop: a custom incremental softmax classifier updates over time with newly labeled candles.
- Research selection: candidate search now uses `3` rolling validation folds across the recent pre-test history and prefers the profile with the strongest worst-fold return before tie-breaking on average fold return and average score.
- Trade style: the current best `5m` behavior is a slower, more selective margin-long bias with a tuned hold horizon around `576` five-minute bars (`48` hours) and a `120` bar cooldown.
- Position limit: up to `5` open trades.
- Spot fees: `0.5%` per side.
- Margin fees: `0.05%` per side.
- Margin leverage cap: `5x`.
- Best current behavior is mostly selective margin longs during strong bullish continuation regimes, with spot and short logic still available but triggered less often.
- Trade-entry gating now enforces the tuned `margin_threshold` and `spot_threshold`, so the parameter search affects live entry decisions instead of only labels and exits.

## Data Requirements
- Required columns: one timestamp column plus `open`, `high`, `low`, and `close`.
- Supported timestamp aliases include names like `timestamp`, `timestamp_ms`, `open_time`, `date`, `datetime`, and `time`.
- Optional columns: `volume`, `vwap`, and `transactions`.
- Timestamp cadence must be regular and no larger than `4h`.
- Naive text timestamps are parsed as wall-clock times. If they only break cadence because of the U.S. spring DST skip, the loader now compresses that skipped hour back to the dominant interval before validating cadence.
- `update-model` now persists the trained candle interval in the saved state and rejects new candle files that do not match that interval.

## Important Files
- `trading_bot/config.py`: strategy and research defaults, multi-fold validation settings, candidate profiles, and bar-scaling helpers for non-5m candle intervals.
- `trading_bot/data.py`: CSV loading, timestamp normalization, interval validation, feature engineering, and target creation.
- `trading_bot/model.py`: incremental online classifier and persistence helpers.
- `trading_bot/portfolio.py`: position accounting, fees, exits, and trade logs.
- `trading_bot/backtest.py`: walk-forward simulation, multi-fold validation selection, interval-aware optimization/metrics, state persistence, and model updates.
- `trading_bot/exchange.py`: CCXT wrapper for spot and margin execution.
- `trading_bot/cli.py`: CLI entrypoints for training and incremental updates.
- `artifacts/bot_state.joblib`: saved trained model state from the latest run.

## Verified Commands
- `python -m trading_bot.cli train --csv bitcoin_price_history_five_minute_candlesticks.csv --output-dir artifacts`
- `python -m trading_bot.cli update-model --state artifacts/bot_state.joblib --csv <new-candles.csv>`

Both commands now accept OHLC-only CSVs as long as they include a timestamp column and use a regular interval of `4h` or smaller.

## Latest Results
Run date: March 15, 2026

- These benchmark results are from the original `5m` BTC dataset after fixing naive spring-DST timestamp handling, wiring the entry thresholds into signal gating, and switching candidate selection to `3` rolling validation folds.
- Validation fold average: `+4.11%` net return, `6.68%` max drawdown
- Validation stability: worst fold `+0.63%`, average fold `+4.11%`
- Held-out test: `+7.13%` net return, `5.20%` max drawdown, `1.613` Sharpe, `1.520` profit factor
- Best tuned parameters:
  - `horizon_bars=576`
  - `max_hold_bars=576`
  - `cooldown_bars=120`
  - `margin_threshold=0.001`
  - `spot_threshold=0.025`
  - `stop_loss_atr_multiple=2.0`
  - `take_profit_atr_multiple=6.0`
  - `spot_trades=0`
  - `margin_trades=51`

## Environment Notes
- Local `.deps/` and `.vendor/` folders were unreliable for compiled packages on this machine's Python 3.14 setup.
- Host-level user Python packages were installed so escalated host Python runs work reliably.
- If future work needs reproducible local environments, prefer creating a fresh venv once the Python packaging issue is resolved rather than depending on `.deps/` or `.vendor/`.

## Next Good Improvements
- Add a dedicated live runner that fetches fresh candles and routes orders through `CCXTExecutionClient`.
- Add richer regime detection and exchange-specific margin configuration.
- Add explicit regression tests around fee handling, leverage, and DST/fallback timestamp edge cases from real exchange exports.
