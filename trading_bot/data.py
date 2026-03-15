from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW_TO_STANDARD_COLUMNS = {
    "open_price": "open",
    "highest_price": "high",
    "lowest_price": "low",
    "close_price": "close",
    "volume": "volume",
    "vol_weight_avg_price": "vwap",
    "num_transactions": "transactions",
    "unix_tmstmp_at_open": "timestamp_ms",
    "readable_time_at_open": "opened_at_text",
}

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_72",
    "body_ratio",
    "range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "vwap_gap",
    "volume_change_1",
    "volume_ratio_24",
    "transactions_ratio_24",
    "volatility_12",
    "volatility_24",
    "volatility_72",
    "ema_gap_12",
    "ema_gap_48",
    "ema_gap_288",
    "trend_strength",
    "atr_ratio_14",
    "rsi_14",
    "rsi_42",
    "breakout_24",
    "pullback_24",
    "distance_from_high_48",
    "distance_from_low_48",
]


@dataclass(frozen=True)
class DatasetSplits:
    train_end: int
    validation_end: int
    total_rows: int

    @property
    def validation_start(self) -> int:
        return self.train_end

    @property
    def test_start(self) -> int:
        return self.validation_end


def load_candles(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame.rename(columns=RAW_TO_STANDARD_COLUMNS)
    frame["opened_at"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    numeric_columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame["transactions"] = frame["transactions"].astype(int)
    frame = (
        frame.sort_values("timestamp_ms")
        .drop_duplicates(subset="timestamp_ms", keep="last")
        .reset_index(drop=True)
    )
    return frame


def compute_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_price = frame["open"]
    volume = frame["volume"]
    transactions = frame["transactions"].astype(float)
    vwap = frame["vwap"]

    for bars in (1, 3, 6, 12, 24, 72):
        frame[f"return_{bars}"] = close.pct_change(bars)

    frame["body_ratio"] = (close - open_price) / open_price
    frame["range_ratio"] = (high - low) / open_price
    frame["upper_wick_ratio"] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    frame["lower_wick_ratio"] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price
    frame["vwap_gap"] = (vwap - close) / close
    frame["volume_change_1"] = volume.pct_change()
    frame["volume_ratio_24"] = volume / volume.rolling(24).mean() - 1.0
    frame["transactions_ratio_24"] = transactions / transactions.rolling(24).mean() - 1.0

    base_return = frame["return_1"]
    for window in (12, 24, 72):
        frame[f"volatility_{window}"] = base_return.rolling(window).std()

    for span in (12, 48, 288):
        ema = close.ewm(span=span, adjust=False).mean()
        frame[f"ema_gap_{span}"] = close / ema - 1.0
    frame["trend_strength"] = (frame["ema_gap_12"] - frame["ema_gap_48"]) + frame["ema_gap_288"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_ratio_14"] = true_range.rolling(14).mean() / close

    frame["rsi_14"] = _relative_strength_index(close, 14)
    frame["rsi_42"] = _relative_strength_index(close, 42)

    rolling_high_24 = high.rolling(24).max()
    rolling_low_24 = low.rolling(24).min()
    rolling_high_48 = high.rolling(48).max()
    rolling_low_48 = low.rolling(48).min()
    frame["breakout_24"] = close / rolling_high_24.shift(1) - 1.0
    frame["pullback_24"] = close / rolling_low_24.shift(1) - 1.0
    frame["distance_from_high_48"] = close / rolling_high_48 - 1.0
    frame["distance_from_low_48"] = close / rolling_low_48 - 1.0

    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return frame


def attach_targets(
    feature_frame: pd.DataFrame,
    horizon_bars: int,
    label_return_threshold: float,
) -> pd.DataFrame:
    dataset = feature_frame.copy()
    dataset["future_return"] = dataset["close"].shift(-horizon_bars) / dataset["close"] - 1.0
    dataset["label"] = 1
    dataset.loc[dataset["future_return"] > label_return_threshold, "label"] = 2
    dataset.loc[dataset["future_return"] < -label_return_threshold, "label"] = 0
    dataset["label"] = dataset["label"].astype(int)
    return dataset


def split_dataset(total_rows: int, train_ratio: float, validation_ratio: float) -> DatasetSplits:
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))
    return DatasetSplits(train_end=train_end, validation_end=validation_end, total_rows=total_rows)


def _relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return (rsi / 100.0).clip(0.0, 1.0)
