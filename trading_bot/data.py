from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import scale_bar_count


MAX_SUPPORTED_CANDLE_INTERVAL_MS = 4 * 60 * 60 * 1000

RAW_TO_STANDARD_COLUMNS = {
    "open": "open",
    "open_price": "open",
    "high": "high",
    "high_price": "high",
    "highest_price": "high",
    "low": "low",
    "low_price": "low",
    "lowest_price": "low",
    "close": "close",
    "close_price": "close",
    "volume": "volume",
    "vwap": "vwap",
    "vol_weight_avg_price": "vwap",
    "volume_weighted_average_price": "vwap",
    "transactions": "transactions",
    "num_transactions": "transactions",
    "trade_count": "transactions",
    "timestamp": "timestamp_ms",
    "timestamp_ms": "timestamp_ms",
    "unix_timestamp": "timestamp_ms",
    "unix_tmstmp_at_open": "timestamp_ms",
    "open_time": "timestamp_ms",
    "opentime": "timestamp_ms",
    "opened_at": "opened_at_text",
    "opened_at_text": "opened_at_text",
    "readable_time_at_open": "opened_at_text",
    "datetime": "opened_at_text",
    "date": "opened_at_text",
    "time": "opened_at_text",
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


@dataclass(frozen=True)
class CandleInterval:
    milliseconds: int

    @property
    def minutes(self) -> float:
        return self.milliseconds / 60_000.0

    @property
    def bars_per_year(self) -> float:
        milliseconds_per_year = 365 * 24 * 60 * 60 * 1000
        return milliseconds_per_year / self.milliseconds

    @property
    def label(self) -> str:
        return format_candle_interval(self.milliseconds)

    def scale_bars(self, base_bars: int, minimum: int = 1) -> int:
        return scale_bar_count(base_bars, self.milliseconds, minimum=minimum)


def load_candles(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = _standardize_columns(frame)

    if "timestamp_ms" in frame.columns:
        frame["timestamp_ms"] = _coerce_timestamp_ms(frame["timestamp_ms"])
    elif "opened_at_text" in frame.columns:
        frame["timestamp_ms"] = _coerce_timestamp_ms(frame["opened_at_text"])
    else:
        raise ValueError("Candle data must include a timestamp column alongside OHLC prices.")

    required_columns = ("open", "high", "low", "close")
    missing_required = [column for column in required_columns if column not in frame.columns]
    if missing_required:
        missing_list = ", ".join(sorted(missing_required))
        raise ValueError(f"Candle data is missing required OHLC columns: {missing_list}.")

    frame["opened_at"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    numeric_columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]
    for column in numeric_columns:
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["timestamp_ms", "open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Candle data contains missing values in the timestamp or OHLC columns.")

    if "transactions" in frame.columns:
        frame["transactions"] = frame["transactions"].round().fillna(0).astype(int)
    frame = (
        frame.sort_values("timestamp_ms")
        .drop_duplicates(subset="timestamp_ms", keep="last")
        .reset_index(drop=True)
    )
    return frame


def infer_candle_interval(
    candles: pd.DataFrame,
    *,
    expected_interval_ms: int | None = None,
    max_interval_ms: int = MAX_SUPPORTED_CANDLE_INTERVAL_MS,
) -> CandleInterval:
    if "timestamp_ms" not in candles.columns:
        raise ValueError("Candles must include timestamp_ms to infer the candle interval.")

    timestamps = (
        pd.Series(candles["timestamp_ms"])
        .dropna()
        .astype("int64")
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(timestamps) < 2:
        if expected_interval_ms is None:
            raise ValueError("At least two distinct candles are required to infer the candle interval.")
        return CandleInterval(milliseconds=int(expected_interval_ms))

    deltas = timestamps.diff().dropna().astype("int64")
    if (deltas <= 0).any():
        raise ValueError("Candles must be strictly ordered in increasing timestamp order.")

    unique_deltas = sorted(int(delta) for delta in deltas.unique())
    interval_ms = unique_deltas[0]
    if len(unique_deltas) != 1:
        raise ValueError("Candles must use a consistent interval with no missing gaps.")
    if expected_interval_ms is not None and interval_ms != expected_interval_ms:
        raise ValueError(
            f"Candle interval mismatch: expected {format_candle_interval(expected_interval_ms)} "
            f"but received {format_candle_interval(interval_ms)} candles."
        )
    if interval_ms > max_interval_ms:
        raise ValueError(
            f"Candle interval {format_candle_interval(interval_ms)} is unsupported. "
            "Use candles that are four hours or smaller."
        )
    return CandleInterval(milliseconds=interval_ms)


def compute_feature_frame(
    candles: pd.DataFrame,
    candle_interval: CandleInterval | None = None,
) -> pd.DataFrame:
    candle_interval = candle_interval or infer_candle_interval(candles)
    frame = candles.copy()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_price = frame["open"]
    vwap = (
        frame["vwap"]
        if "vwap" in frame.columns
        else (open_price + high + low + close) / 4.0
    )

    return_windows = {bars: candle_interval.scale_bars(bars, minimum=1) for bars in (1, 3, 6, 12, 24, 72)}
    volatility_windows = {bars: candle_interval.scale_bars(bars, minimum=2) for bars in (12, 24, 72)}
    ema_spans = {bars: candle_interval.scale_bars(bars, minimum=1) for bars in (12, 48, 288)}
    rsi_periods = {bars: candle_interval.scale_bars(bars, minimum=2) for bars in (14, 42)}
    range_windows = {bars: candle_interval.scale_bars(bars, minimum=1) for bars in (24, 48)}
    atr_window = candle_interval.scale_bars(14, minimum=1)

    for bars, scaled_bars in return_windows.items():
        frame[f"return_{bars}"] = close.pct_change(scaled_bars)

    frame["body_ratio"] = (close - open_price) / open_price
    frame["range_ratio"] = (high - low) / open_price
    frame["upper_wick_ratio"] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    frame["lower_wick_ratio"] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price
    frame["vwap_gap"] = (vwap - close) / close

    if "volume" in frame.columns:
        volume = frame["volume"]
        volume_ratio_window = return_windows[24]
        frame["volume_change_1"] = volume.pct_change()
        frame["volume_ratio_24"] = volume / volume.rolling(volume_ratio_window).mean() - 1.0
    else:
        frame["volume_change_1"] = 0.0
        frame["volume_ratio_24"] = 0.0

    if "transactions" in frame.columns:
        transactions = frame["transactions"].astype(float)
        transactions_ratio_window = return_windows[24]
        frame["transactions_ratio_24"] = transactions / transactions.rolling(transactions_ratio_window).mean() - 1.0
    else:
        frame["transactions_ratio_24"] = 0.0

    base_return = frame["return_1"]
    for window, scaled_window in volatility_windows.items():
        frame[f"volatility_{window}"] = base_return.rolling(scaled_window).std(ddof=0)

    for span, scaled_span in ema_spans.items():
        ema = close.ewm(span=scaled_span, adjust=False).mean()
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
    frame["atr_ratio_14"] = true_range.rolling(atr_window).mean() / close

    frame["rsi_14"] = _relative_strength_index(close, rsi_periods[14])
    frame["rsi_42"] = _relative_strength_index(close, rsi_periods[42])

    rolling_high_24 = high.rolling(range_windows[24]).max()
    rolling_low_24 = low.rolling(range_windows[24]).min()
    rolling_high_48 = high.rolling(range_windows[48]).max()
    rolling_low_48 = low.rolling(range_windows[48]).min()
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


def format_candle_interval(interval_ms: int) -> str:
    if interval_ms % (60 * 60 * 1000) == 0:
        hours = interval_ms // (60 * 60 * 1000)
        return f"{hours}h"
    if interval_ms % (60 * 1000) == 0:
        minutes = interval_ms // (60 * 1000)
        return f"{minutes}m"
    if interval_ms % 1000 == 0:
        seconds = interval_ms // 1000
        return f"{seconds}s"
    return f"{interval_ms}ms"


def _relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return (rsi / 100.0).clip(0.0, 1.0)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {
        column: RAW_TO_STANDARD_COLUMNS.get(_normalize_column_name(column), _normalize_column_name(column))
        for column in frame.columns
    }
    frame = frame.rename(columns=renamed_columns)
    return frame.loc[:, ~frame.columns.duplicated(keep="first")]


def _normalize_column_name(column: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())
    return normalized.strip("_")


def _coerce_timestamp_ms(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        absolute_max = float(numeric.abs().max())
        if absolute_max < 1e11:
            scaled = numeric * 1_000.0
        elif absolute_max < 1e14:
            scaled = numeric
        elif absolute_max < 1e17:
            scaled = numeric / 1_000.0
        else:
            scaled = numeric / 1_000_000.0
        return pd.Series(np.rint(scaled), index=values.index).astype("int64")

    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Unable to parse candle timestamps.")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((parsed - epoch) // pd.Timedelta(milliseconds=1)).astype("int64")
