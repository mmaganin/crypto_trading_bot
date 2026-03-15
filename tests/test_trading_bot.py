from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest import build_trade_proposal
from trading_bot.config import FeeModel, StrategyParameters
from trading_bot.data import FEATURE_COLUMNS, attach_targets, compute_feature_frame
from trading_bot.model import IncrementalDirectionalModel, ModelPrediction
from trading_bot.portfolio import Portfolio


def synthetic_candles(rows: int = 800) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 20_000 + index * 0.8 + np.sin(index / 6) * 120 + np.cos(index / 17) * 55
    open_price = close - np.sin(index / 5) * 8
    high = close + 12
    low = close - 12
    volume = 100 + (index % 20) * 2
    transactions = 50 + (index % 15)
    vwap = close + 1
    timestamps = 1_700_000_000_000 + (index.astype(int) * 300_000)
    opened_at = pd.to_datetime(timestamps, unit="ms", utc=True)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": vwap,
            "transactions": transactions.astype(int),
            "timestamp_ms": timestamps.astype(int),
            "opened_at": opened_at,
        }
    )


class TradingBotTests(unittest.TestCase):
    def test_feature_pipeline_produces_expected_columns(self) -> None:
        feature_frame = compute_feature_frame(synthetic_candles())
        for column in FEATURE_COLUMNS:
            self.assertIn(column, feature_frame.columns)
        self.assertGreater(feature_frame[FEATURE_COLUMNS].notna().sum().sum(), 0)

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
