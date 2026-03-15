from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest import build_trade_proposal, build_validation_windows, update_model_from_new_candles
from trading_bot.config import (
    FeeModel,
    ResearchConfig,
    StrategyParameters,
    scale_research_config,
    scale_strategy_parameters,
)
from trading_bot.data import FEATURE_COLUMNS, attach_targets, compute_feature_frame, infer_candle_interval, load_candles
from trading_bot.model import IncrementalDirectionalModel, ModelPrediction
from trading_bot.portfolio import Portfolio


def synthetic_candles(
    rows: int = 800,
    interval_ms: int = 300_000,
    include_optional: bool = True,
) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 20_000 + index * 0.8 + np.sin(index / 6) * 120 + np.cos(index / 17) * 55
    open_price = close - np.sin(index / 5) * 8
    high = close + 12
    low = close - 12
    timestamps = 1_700_000_000_000 + (index.astype(int) * interval_ms)
    opened_at = pd.to_datetime(timestamps, unit="ms", utc=True)

    frame = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "timestamp_ms": timestamps.astype(int),
            "opened_at": opened_at,
        }
    )
    if include_optional:
        frame["volume"] = 100 + (index % 20) * 2
        frame["transactions"] = (50 + (index % 15)).astype(int)
        frame["vwap"] = close + 1
    return frame


class TradingBotTests(unittest.TestCase):
    def test_feature_pipeline_produces_expected_columns(self) -> None:
        feature_frame = compute_feature_frame(synthetic_candles())
        for column in FEATURE_COLUMNS:
            self.assertIn(column, feature_frame.columns)
        self.assertGreater(feature_frame[FEATURE_COLUMNS].notna().sum().sum(), 0)

    def test_feature_pipeline_supports_ohlc_only_hourly_data(self) -> None:
        feature_frame = compute_feature_frame(synthetic_candles(interval_ms=60 * 60 * 1000, include_optional=False))
        self.assertGreater(feature_frame[FEATURE_COLUMNS].notna().sum().sum(), 0)
        self.assertTrue((feature_frame["volume_change_1"].fillna(0.0) == 0.0).all())
        self.assertTrue((feature_frame["transactions_ratio_24"].fillna(0.0) == 0.0).all())

    def test_load_candles_accepts_ohlc_only_csv(self) -> None:
        candles = synthetic_candles(rows=12, interval_ms=4 * 60 * 60 * 1000, include_optional=False)
        raw_frame = pd.DataFrame(
            {
                "Date": candles["opened_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z"),
                "Open": candles["open"],
                "High": candles["high"],
                "Low": candles["low"],
                "Close": candles["close"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "ohlc.csv"
            raw_frame.to_csv(csv_path, index=False)
            loaded = load_candles(csv_path)

        interval = infer_candle_interval(loaded)
        self.assertEqual(interval.milliseconds, 4 * 60 * 60 * 1000)
        self.assertTrue({"timestamp_ms", "open", "high", "low", "close"}.issubset(loaded.columns))

    def test_load_candles_normalizes_spring_dst_wall_clock_gap(self) -> None:
        opened_at = pd.Series(
            [
                "03/10/2024 01:45:00",
                "03/10/2024 01:50:00",
                "03/10/2024 01:55:00",
                "03/10/2024 03:00:00",
                "03/10/2024 03:05:00",
                "03/10/2024 03:10:00",
            ]
        )
        raw_frame = pd.DataFrame(
            {
                "readable_time_at_open": opened_at,
                "open_price": [1, 2, 3, 4, 5, 6],
                "highest_price": [2, 3, 4, 5, 6, 7],
                "lowest_price": [0, 1, 2, 3, 4, 5],
                "close_price": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "spring_dst.csv"
            raw_frame.to_csv(csv_path, index=False)
            loaded = load_candles(csv_path)

        interval = infer_candle_interval(loaded)
        self.assertEqual(interval.milliseconds, 5 * 60 * 1000)
        self.assertTrue((loaded["timestamp_ms"].diff().dropna() == 5 * 60 * 1000).all())

    def test_rejects_candles_larger_than_four_hours(self) -> None:
        candles = synthetic_candles(rows=12, interval_ms=6 * 60 * 60 * 1000, include_optional=False)
        with self.assertRaisesRegex(ValueError, "four hours or smaller"):
            infer_candle_interval(candles)

    def test_rescales_bar_based_settings_for_hourly_candles(self) -> None:
        interval_ms = 60 * 60 * 1000
        scaled_strategy = scale_strategy_parameters(StrategyParameters(), interval_ms)
        scaled_research = scale_research_config(ResearchConfig(), interval_ms)

        self.assertEqual(scaled_strategy.horizon_bars, 48)
        self.assertEqual(scaled_strategy.max_hold_bars, 48)
        self.assertEqual(scaled_strategy.cooldown_bars, 12)
        self.assertEqual(scaled_research.recent_candle_buffer, 42)
        self.assertEqual(scaled_research.minimum_training_rows, 208)

    def test_validation_windows_cover_three_recent_pretest_folds(self) -> None:
        research_config = ResearchConfig(minimum_training_rows=0)
        windows = build_validation_windows(total_rows=1000, research_config=research_config)
        self.assertEqual(windows, [(500, 600), (600, 700), (700, 800)])

    def test_incremental_model_round_trips_through_disk(self) -> None:
        candles = synthetic_candles()
        feature_frame = compute_feature_frame(candles)
        dataset = attach_targets(feature_frame, horizon_bars=24, label_return_threshold=0.001)
        valid = dataset[FEATURE_COLUMNS].notna().all(axis=1) & dataset["future_return"].notna()
        features = dataset.loc[valid, FEATURE_COLUMNS].to_numpy(dtype=float)
        labels = dataset.loc[valid, "label"].to_numpy(dtype=np.int64)
        future_returns = dataset.loc[valid, "future_return"].to_numpy(dtype=float)

        model = IncrementalDirectionalModel(StrategyParameters())
        model.fit_initial(features[:200], labels[:200], future_returns[:200])
        original_prediction = model.predict(features[220])

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.joblib"
            model.save(state_path, extra_state={"marker": "ok"})
            reloaded_model, extra_state = IncrementalDirectionalModel.load(state_path)

        reloaded_prediction = reloaded_model.predict(features[220])
        self.assertEqual(extra_state["marker"], "ok")
        self.assertAlmostEqual(original_prediction.expected_return, reloaded_prediction.expected_return, places=8)

    def test_update_model_rejects_interval_mismatch(self) -> None:
        candles = synthetic_candles(include_optional=False)
        feature_frame = compute_feature_frame(candles)
        dataset = attach_targets(feature_frame, horizon_bars=24, label_return_threshold=0.001)
        valid = dataset[FEATURE_COLUMNS].notna().all(axis=1) & dataset["future_return"].notna()
        features = dataset.loc[valid, FEATURE_COLUMNS].to_numpy(dtype=float)
        labels = dataset.loc[valid, "label"].to_numpy(dtype=np.int64)
        future_returns = dataset.loc[valid, "future_return"].to_numpy(dtype=float)

        model = IncrementalDirectionalModel(StrategyParameters())
        model.fit_initial(features[:200], labels[:200], future_returns[:200])

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.joblib"
            csv_path = Path(temp_dir) / "hourly.csv"
            model.save(
                state_path,
                extra_state={
                    "recent_candles": dataset.tail(120).to_dict(orient="records"),
                    "last_trained_timestamp_ms": int(dataset.loc[valid, "timestamp_ms"].max()),
                    "candle_interval_ms": 300_000,
                    "candle_interval": "5m",
                },
            )

            hourly_candles = synthetic_candles(rows=12, interval_ms=60 * 60 * 1000, include_optional=False)
            hourly_candles.drop(columns=["opened_at"]).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "mismatch"):
                update_model_from_new_candles(state_path, csv_path)

    def test_trade_proposal_respects_constraints(self) -> None:
        feature_frame = compute_feature_frame(synthetic_candles())
        row = feature_frame.iloc[500].copy()
        row["rsi_14"] = 0.82
        row["ema_gap_48"] = 0.02
        row["ema_gap_288"] = 0.01
        row["distance_from_low_48"] = 0.025
        row["distance_from_high_48"] = -0.01
        row["trend_strength"] = 0.015
        row["atr_ratio_14"] = 0.004
        row["breakout_24"] = 0.001

        portfolio = Portfolio(starting_cash=10_000, fee_model=FeeModel(), max_open_positions=5)
        prediction = ModelPrediction(
            probability_down=0.18,
            probability_flat=0.20,
            probability_up=0.62,
            expected_return=0.03,
        )
        proposal = build_trade_proposal(
            prediction=prediction,
            row=row,
            portfolio=portfolio,
            current_equity=10_000,
            strategy_parameters=StrategyParameters(),
            fee_model=FeeModel(),
            entry_index=501,
            created_index=500,
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertLessEqual(proposal.leverage, 5.0)
        self.assertIn(proposal.market, {"spot", "margin"})


if __name__ == "__main__":
    unittest.main()
