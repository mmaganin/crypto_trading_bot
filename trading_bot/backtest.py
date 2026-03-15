from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FeeModel, ResearchConfig, StrategyParameters, candidate_parameters, scale_research_config
from .data import (
    CandleInterval,
    FEATURE_COLUMNS,
    attach_targets,
    compute_feature_frame,
    format_candle_interval,
    infer_candle_interval,
    load_candles,
    split_dataset,
)
from .model import IncrementalDirectionalModel, ModelPrediction
from .portfolio import PendingOrder, Portfolio


@dataclass
class SimulationResult:
    metrics: dict[str, Any]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    strategy_parameters: StrategyParameters
    model: IncrementalDirectionalModel
    recent_candles: pd.DataFrame
    last_trained_timestamp_ms: int
    evaluation_name: str
    candle_interval_ms: int


def run_research_pipeline(
    candles: pd.DataFrame,
    fee_model: FeeModel,
    research_config: ResearchConfig,
    base_strategy_parameters: StrategyParameters,
) -> dict[str, Any]:
    candle_interval = infer_candle_interval(candles)
    scaled_research_config = scale_research_config(research_config, candle_interval.milliseconds)
    feature_frame = compute_feature_frame(candles, candle_interval)
    splits = split_dataset(
        len(feature_frame),
        scaled_research_config.train_ratio,
        scaled_research_config.validation_ratio,
    )

    validation_results: list[SimulationResult] = []
    for candidate in candidate_parameters(base_strategy_parameters, candle_interval.milliseconds):
        candidate_dataset = attach_targets(
            feature_frame,
            horizon_bars=candidate.horizon_bars,
            label_return_threshold=candidate.label_return_threshold,
        )
        validation_results.append(
            summarize_validation_results(
                _run_validation_folds(
                    dataset=candidate_dataset,
                    fee_model=fee_model,
                    research_config=scaled_research_config,
                    strategy_parameters=candidate,
                    candle_interval=candle_interval,
                    total_rows=len(feature_frame),
                )
            )
        )

    best_validation = max(
        validation_results,
        key=lambda result: (
            float(result.metrics["minimum_fold_return_pct"]),
            float(result.metrics["average_fold_return_pct"]),
            float(result.metrics["score"]),
        ),
    )

    test_dataset = attach_targets(
        feature_frame,
        horizon_bars=best_validation.strategy_parameters.horizon_bars,
        label_return_threshold=best_validation.strategy_parameters.label_return_threshold,
    )
    test_result = simulate_walk_forward(
        dataset=test_dataset,
        fee_model=fee_model,
        research_config=scaled_research_config,
        strategy_parameters=best_validation.strategy_parameters,
        evaluation_start=splits.test_start,
        evaluation_end=len(test_dataset),
        evaluation_name="test",
        candle_interval=candle_interval,
    )
    trained_model, training_state = fit_model_on_full_history(
        dataset=test_dataset,
        strategy_parameters=best_validation.strategy_parameters,
        recent_candle_buffer=scaled_research_config.recent_candle_buffer,
        candle_interval=candle_interval,
    )
    return {
        "validation": best_validation,
        "test": replace(
            test_result,
            model=trained_model,
            recent_candles=training_state["recent_candles"],
            last_trained_timestamp_ms=training_state["last_trained_timestamp_ms"],
        ),
        "best_strategy_parameters": best_validation.strategy_parameters,
        "research_config": scaled_research_config,
        "candle_interval": candle_interval,
    }


def _run_validation_folds(
    dataset: pd.DataFrame,
    fee_model: FeeModel,
    research_config: ResearchConfig,
    strategy_parameters: StrategyParameters,
    candle_interval: CandleInterval,
    total_rows: int,
) -> list[SimulationResult]:
    validation_windows = build_validation_windows(total_rows, research_config)
    return [
        simulate_walk_forward(
            dataset=dataset,
            fee_model=fee_model,
            research_config=research_config,
            strategy_parameters=strategy_parameters,
            evaluation_start=window_start,
            evaluation_end=window_end,
            evaluation_name="validation",
            candle_interval=candle_interval,
        )
        for window_start, window_end in validation_windows
    ]


def build_validation_windows(total_rows: int, research_config: ResearchConfig) -> list[tuple[int, int]]:
    splits = split_dataset(total_rows, research_config.train_ratio, research_config.validation_ratio)
    if research_config.validation_folds <= 1:
        return [(splits.validation_start, splits.validation_end)]

    fold_size = max(1, int(total_rows * research_config.validation_ratio / (research_config.validation_folds - 1)))
    first_start = max(research_config.minimum_training_rows, splits.validation_start - fold_size)
    windows: list[tuple[int, int]] = []
    for fold_index in range(research_config.validation_folds):
        window_start = first_start + fold_index * fold_size
        window_end = window_start + fold_size
        if window_end > splits.validation_end:
            break
        windows.append((window_start, window_end))

    if not windows:
        return [(splits.validation_start, splits.validation_end)]
    return windows


def summarize_validation_results(results: list[SimulationResult]) -> SimulationResult:
    if not results:
        raise ValueError("At least one validation result is required.")

    aggregated_metrics = _aggregate_validation_metrics(results)
    return replace(results[-1], metrics=aggregated_metrics)


def _aggregate_validation_metrics(results: list[SimulationResult]) -> dict[str, Any]:
    def average(metric_name: str) -> float:
        return float(np.mean([float(result.metrics[metric_name]) for result in results]))

    fold_returns = [float(result.metrics["total_return_pct"]) for result in results]

    return {
        "ending_equity": round(average("ending_equity"), 2),
        "net_profit": round(average("net_profit"), 2),
        "total_return_pct": round(average("total_return_pct"), 2),
        "max_drawdown_pct": round(average("max_drawdown_pct"), 2),
        "sharpe_ratio": round(average("sharpe_ratio"), 3),
        "profit_factor": round(average("profit_factor"), 3),
        "win_rate": round(average("win_rate"), 2),
        "num_trades": int(round(average("num_trades"))),
        "spot_trades": int(round(average("spot_trades"))),
        "margin_trades": int(round(average("margin_trades"))),
        "total_fees_paid": round(average("total_fees_paid"), 2),
        "score": round(average("score"), 3),
        "minimum_fold_return_pct": round(min(fold_returns), 2),
        "average_fold_return_pct": round(float(np.mean(fold_returns)), 2),
        "validation_folds": len(results),
        "evaluation_name": "validation",
        "strategy_parameters": results[-1].strategy_parameters.to_dict(),
        "candle_interval_ms": results[-1].candle_interval_ms,
        "candle_interval": format_candle_interval(results[-1].candle_interval_ms),
    }


def simulate_walk_forward(
    dataset: pd.DataFrame,
    fee_model: FeeModel,
    research_config: ResearchConfig,
    strategy_parameters: StrategyParameters,
    evaluation_start: int,
    evaluation_end: int,
    evaluation_name: str,
    candle_interval: CandleInterval,
) -> SimulationResult:
    features = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = dataset["label"].to_numpy(dtype=np.int64)
    future_returns = dataset["future_return"].to_numpy(dtype=float)
    signal_mask = np.isfinite(features).all(axis=1)
    label_mask = signal_mask & np.isfinite(future_returns)
    training_mask = label_mask & (np.arange(len(dataset)) < evaluation_start)

    if int(training_mask.sum()) < research_config.minimum_training_rows:
        raise ValueError("Not enough training rows were available before the evaluation window.")

    model = IncrementalDirectionalModel(strategy_parameters)
    model.fit_initial(features[training_mask], labels[training_mask], future_returns[training_mask])

    portfolio = Portfolio(
        starting_cash=research_config.starting_cash,
        fee_model=fee_model,
        max_open_positions=strategy_parameters.max_open_positions,
    )
    pending_order: PendingOrder | None = None
    last_entry_index = -10_000
    equity_records: list[dict[str, Any]] = []

    for bar_index in range(evaluation_start, evaluation_end):
        row = dataset.iloc[bar_index]

        if pending_order is not None and pending_order.entry_index == bar_index:
            position = portfolio.open_position(
                order=pending_order,
                entry_price=float(row["open"]),
                opened_at=row["opened_at"],
                max_hold_bars=strategy_parameters.max_hold_bars,
            )
            if position is not None:
                last_entry_index = bar_index
            pending_order = None

        for position in list(portfolio.positions):
            exit_decision = portfolio.maybe_exit_position(
                position=position,
                bar_open=float(row["open"]),
                bar_high=float(row["high"]),
                bar_low=float(row["low"]),
                bar_close=float(row["close"]),
                bar_index=bar_index,
            )
            if exit_decision is not None:
                portfolio.close_position(
                    position=position,
                    exit_price=exit_decision.price,
                    closed_at=row["opened_at"],
                    bar_index=bar_index,
                    reason=exit_decision.reason,
                )

        matured_index = bar_index - strategy_parameters.horizon_bars
        if matured_index >= 0 and label_mask[matured_index]:
            model.update(features[matured_index], int(labels[matured_index]), float(future_returns[matured_index]))

        close_price = float(row["close"])
        current_equity = portfolio.equity(close_price)
        equity_records.append(
            {
                "opened_at": row["opened_at"],
                "close_price": close_price,
                "equity": current_equity,
                "cash": portfolio.cash,
                "open_positions": len(portfolio.positions),
            }
        )

        if (
            pending_order is None
            and bar_index + 1 < evaluation_end
            and signal_mask[bar_index]
            and len(portfolio.positions) < strategy_parameters.max_open_positions
            and bar_index - last_entry_index >= strategy_parameters.cooldown_bars
        ):
            prediction = model.predict(features[bar_index])
            proposal = build_trade_proposal(
                prediction=prediction,
                row=row,
                portfolio=portfolio,
                current_equity=current_equity,
                strategy_parameters=strategy_parameters,
                fee_model=fee_model,
                entry_index=bar_index + 1,
                created_index=bar_index,
            )
            if proposal is not None:
                pending_order = proposal

    final_row = dataset.iloc[evaluation_end - 1]
    for position in list(portfolio.positions):
        portfolio.close_position(
            position=position,
            exit_price=float(final_row["close"]),
            closed_at=final_row["opened_at"],
            bar_index=evaluation_end - 1,
            reason="end_of_backtest",
        )

    equity_curve = pd.DataFrame(equity_records)
    trades = pd.DataFrame([trade.to_record() for trade in portfolio.closed_trades])
    metrics = calculate_metrics(
        equity_curve=equity_curve,
        trades=trades,
        starting_cash=research_config.starting_cash,
        total_fees_paid=portfolio.total_fees_paid,
        bars_per_year=candle_interval.bars_per_year,
    )
    metrics["evaluation_name"] = evaluation_name
    metrics["strategy_parameters"] = strategy_parameters.to_dict()
    metrics["candle_interval_ms"] = candle_interval.milliseconds
    metrics["candle_interval"] = candle_interval.label

    return SimulationResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        strategy_parameters=strategy_parameters,
        model=model,
        recent_candles=dataset.tail(research_config.recent_candle_buffer).copy(),
        last_trained_timestamp_ms=int(dataset.loc[label_mask, "timestamp_ms"].max()),
        evaluation_name=evaluation_name,
        candle_interval_ms=candle_interval.milliseconds,
    )


def fit_model_on_full_history(
    dataset: pd.DataFrame,
    strategy_parameters: StrategyParameters,
    recent_candle_buffer: int,
    candle_interval: CandleInterval,
) -> tuple[IncrementalDirectionalModel, dict[str, Any]]:
    features = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = dataset["label"].to_numpy(dtype=np.int64)
    future_returns = dataset["future_return"].to_numpy(dtype=float)
    label_mask = np.isfinite(features).all(axis=1) & np.isfinite(future_returns)
    model = IncrementalDirectionalModel(strategy_parameters)
    model.fit_initial(features[label_mask], labels[label_mask], future_returns[label_mask])
    training_state = {
        "recent_candles": dataset.tail(recent_candle_buffer).copy(),
        "last_trained_timestamp_ms": int(dataset.loc[label_mask, "timestamp_ms"].max()),
        "candle_interval_ms": candle_interval.milliseconds,
        "candle_interval": candle_interval.label,
    }
    return model, training_state


def save_simulation_outputs(
    output_dir: str | Path,
    simulation_result: SimulationResult,
    research_config: ResearchConfig,
    fee_model: FeeModel,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / f"{simulation_result.evaluation_name}_metrics.json"
    equity_path = output_path / f"{simulation_result.evaluation_name}_equity_curve.csv"
    trades_path = output_path / f"{simulation_result.evaluation_name}_trades.csv"
    state_path = output_path / "bot_state.joblib"

    metrics_payload = {
        **simulation_result.metrics,
        "research_config": research_config.to_dict(),
        "fee_model": asdict(fee_model),
    }
    metrics_path.write_text(json_dumps(metrics_payload), encoding="utf-8")
    simulation_result.equity_curve.to_csv(equity_path, index=False)
    simulation_result.trades.to_csv(trades_path, index=False)
    simulation_result.model.save(
        state_path,
        extra_state={
            "recent_candles": simulation_result.recent_candles.to_dict(orient="records"),
            "last_trained_timestamp_ms": simulation_result.last_trained_timestamp_ms,
            "research_config": research_config.to_dict(),
            "fee_model": asdict(fee_model),
            "candle_interval_ms": simulation_result.candle_interval_ms,
            "candle_interval": format_candle_interval(simulation_result.candle_interval_ms),
        },
    )
    return {
        "metrics": metrics_path,
        "equity_curve": equity_path,
        "trades": trades_path,
        "state": state_path,
    }


def update_model_from_new_candles(
    state_path: str | Path,
    candle_csv_path: str | Path,
    recent_candle_buffer: int = 500,
) -> dict[str, Any]:
    model, extra_state = IncrementalDirectionalModel.load(state_path)
    historic_tail = pd.DataFrame(extra_state["recent_candles"])
    expected_interval = _resolve_candle_interval(extra_state, historic_tail)
    new_candles = load_candles(candle_csv_path)

    if len(new_candles) >= 2:
        infer_candle_interval(new_candles, expected_interval_ms=expected_interval.milliseconds)

    combined = (
        pd.concat([historic_tail, new_candles], ignore_index=True)
        .sort_values("timestamp_ms")
        .drop_duplicates(subset="timestamp_ms", keep="last")
        .reset_index(drop=True)
    )
    combined_interval = infer_candle_interval(combined, expected_interval_ms=expected_interval.milliseconds)
    dataset = attach_targets(
        compute_feature_frame(combined, combined_interval),
        horizon_bars=model.strategy_parameters.horizon_bars,
        label_return_threshold=model.strategy_parameters.label_return_threshold,
    )
    features = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    label_mask = np.isfinite(features).all(axis=1) & np.isfinite(dataset["future_return"].to_numpy(dtype=float))
    last_trained_timestamp_ms = int(extra_state["last_trained_timestamp_ms"])
    update_mask = label_mask & (dataset["timestamp_ms"] > last_trained_timestamp_ms)

    update_count = int(update_mask.sum())
    for _, row in dataset.loc[update_mask].iterrows():
        feature_vector = row[FEATURE_COLUMNS].to_numpy(dtype=float)
        model.update(feature_vector, int(row["label"]), float(row["future_return"]))

    new_last_trained = int(dataset.loc[label_mask, "timestamp_ms"].max())
    scaled_recent_candle_buffer = max(8, recent_candle_buffer)
    model.save(
        state_path,
        extra_state={
            **extra_state,
            "recent_candles": dataset.tail(scaled_recent_candle_buffer).to_dict(orient="records"),
            "last_trained_timestamp_ms": new_last_trained,
            "candle_interval_ms": expected_interval.milliseconds,
            "candle_interval": expected_interval.label,
        },
    )
    return {
        "updated_rows": update_count,
        "last_trained_timestamp_ms": new_last_trained,
        "state_path": str(state_path),
        "candle_interval": expected_interval.label,
    }


def build_trade_proposal(
    prediction: ModelPrediction,
    row: pd.Series,
    portfolio: Portfolio,
    current_equity: float,
    strategy_parameters: StrategyParameters,
    fee_model: FeeModel,
    entry_index: int,
    created_index: int,
) -> PendingOrder | None:
    edge = prediction.expected_return
    direction_bias = prediction.probability_up - prediction.probability_down
    confidence = abs(direction_bias)
    atr_ratio = max(float(row["atr_ratio_14"]), 0.0015)
    trend_strength = float(row["trend_strength"])
    rsi_14 = float(row["rsi_14"])
    ema_gap_48 = float(row["ema_gap_48"])
    ema_gap_288 = float(row["ema_gap_288"])
    breakout_24 = float(row["breakout_24"])
    distance_from_low_48 = float(row["distance_from_low_48"])
    distance_from_high_48 = float(row["distance_from_high_48"])
    cash_available = portfolio.cash * (1.0 - strategy_parameters.min_cash_buffer)
    if cash_available <= 0.0:
        return None

    minimum_margin_edge = strategy_parameters.margin_threshold + fee_model.margin_round_trip
    minimum_spot_edge = strategy_parameters.spot_threshold + fee_model.spot_round_trip

    bullish_score = sum(
        [
            rsi_14 >= 0.75,
            0.008 <= ema_gap_48 <= 0.035,
            distance_from_low_48 >= 0.015,
            trend_strength >= 0.008,
            atr_ratio <= 0.010,
            breakout_24 >= -0.002,
            prediction.probability_down <= 0.40,
        ]
    )
    bearish_score = sum(
        [
            rsi_14 <= 0.18,
            -0.040 <= ema_gap_48 <= -0.010,
            distance_from_high_48 <= -0.020,
            trend_strength <= -0.010,
            atr_ratio <= 0.010,
            breakout_24 <= -0.010,
            prediction.probability_up <= 0.35,
        ]
    )

    market: str | None = None
    side: str | None = None
    leverage = 1.0
    fee_buffer = fee_model.margin_round_trip

    if (
        bullish_score >= 7
        and ema_gap_288 > 0.0
        and trend_strength >= 0.015
        and edge >= minimum_spot_edge
        and prediction.probability_up >= strategy_parameters.spot_probability
    ):
        market = "spot"
        side = "long"
        fee_buffer = fee_model.spot_round_trip
    elif (
        bullish_score >= 5
        and ema_gap_288 >= 0.0
        and edge >= minimum_margin_edge
        and prediction.probability_up >= strategy_parameters.margin_probability
        and prediction.probability_down <= 0.45
    ):
        market = "margin"
        side = "long"
        leverage = resolve_leverage(max(confidence, bullish_score / 6.0), atr_ratio, strategy_parameters, fee_model)
    elif (
        bearish_score >= 7
        and prediction.probability_down >= 0.60
        and edge <= -minimum_margin_edge
    ):
        market = "margin"
        side = "short"
        leverage = resolve_leverage(max(confidence, bearish_score / 6.0), atr_ratio, strategy_parameters, fee_model)
    else:
        return None

    signal_strength = max(confidence, bullish_score / 6.0 if side == "long" else bearish_score / 6.0)
    if market == "margin" and signal_strength < strategy_parameters.min_confidence:
        return None

    sizing_multiplier = min(1.4, max(0.55, signal_strength / max(strategy_parameters.min_confidence, 0.01)))
    if market == "spot":
        committed_capital = min(
            cash_available,
            current_equity * strategy_parameters.spot_position_fraction * sizing_multiplier,
        )
    else:
        committed_capital = min(
            cash_available,
            current_equity * strategy_parameters.margin_collateral_fraction * sizing_multiplier,
        )

    if committed_capital <= 0.0:
        return None

    if market == "margin":
        target_return = max(0.08, atr_ratio * strategy_parameters.take_profit_atr_multiple, abs(edge) * 1.5)
        stop_return = max(0.025, atr_ratio * strategy_parameters.stop_loss_atr_multiple, fee_buffer * 2.0)
        target_return = min(target_return, 0.20)
        stop_return = min(stop_return, 0.05)
    else:
        target_return = max(0.04, atr_ratio * strategy_parameters.take_profit_atr_multiple, abs(edge) * 1.3)
        stop_return = max(0.015, atr_ratio * strategy_parameters.stop_loss_atr_multiple, fee_buffer * 1.6)
        target_return = min(target_return, 0.12)
        stop_return = min(stop_return, 0.05)
    if target_return <= fee_buffer:
        return None

    return PendingOrder(
        entry_index=entry_index,
        created_index=created_index,
        market=market,
        side=side,
        leverage=leverage,
        committed_capital=float(committed_capital),
        stop_return=float(stop_return),
        target_return=float(target_return),
        expected_return=float(edge),
        confidence=float(signal_strength),
    )


def resolve_leverage(
    confidence: float,
    atr_ratio: float,
    strategy_parameters: StrategyParameters,
    fee_model: FeeModel,
) -> float:
    volatility_penalty = min(1.0, max(0.45, 0.008 / max(atr_ratio, 0.001)))
    raw_leverage = (1.25 + confidence * 3.0) * volatility_penalty
    return float(
        min(
            fee_model.max_leverage,
            max(strategy_parameters.leverage_floor, min(strategy_parameters.leverage_ceiling, raw_leverage)),
        )
    )


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    starting_cash: float,
    total_fees_paid: float,
    bars_per_year: float,
) -> dict[str, Any]:
    if equity_curve.empty:
        return {
            "ending_equity": starting_cash,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "spot_trades": 0,
            "margin_trades": 0,
            "total_fees_paid": total_fees_paid,
            "score": -999.0,
        }

    equity_curve = equity_curve.copy()
    equity_curve["return"] = equity_curve["equity"].pct_change().fillna(0.0)
    running_peak = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / running_peak - 1.0
    ending_equity = float(equity_curve["equity"].iloc[-1])
    total_return_pct = (ending_equity / starting_cash - 1.0) * 100.0
    sharpe_denominator = float(equity_curve["return"].std())
    annualization_factor = np.sqrt(bars_per_year)
    sharpe_ratio = (
        float(equity_curve["return"].mean()) / sharpe_denominator * annualization_factor
        if sharpe_denominator > 0.0
        else 0.0
    )

    if trades.empty:
        profit_factor = 0.0
        win_rate = 0.0
        spot_trades = 0
        margin_trades = 0
    else:
        gross_profit = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum())
        gross_loss = abs(float(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else float("inf")
        win_rate = float((trades["net_pnl"] > 0).mean())
        spot_trades = int((trades["market"] == "spot").sum())
        margin_trades = int((trades["market"] == "margin").sum())

    score = total_return_pct - abs(float(drawdown.min()) * 100.0) * 0.65 + sharpe_ratio * 0.20
    if trades.empty:
        score -= 25.0

    return {
        "ending_equity": round(ending_equity, 2),
        "net_profit": round(ending_equity - starting_cash, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(abs(float(drawdown.min()) * 100.0), 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else "inf",
        "win_rate": round(win_rate * 100.0, 2),
        "num_trades": int(len(trades)),
        "spot_trades": spot_trades,
        "margin_trades": margin_trades,
        "total_fees_paid": round(total_fees_paid, 2),
        "score": round(score, 3),
    }


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


def _resolve_candle_interval(extra_state: dict[str, Any], historic_tail: pd.DataFrame) -> CandleInterval:
    interval_ms = extra_state.get("candle_interval_ms")
    if interval_ms is None:
        interval_label = extra_state.get("candle_interval")
        if isinstance(interval_label, str) and interval_label.endswith("m"):
            interval_ms = int(interval_label[:-1]) * 60 * 1000
        elif isinstance(interval_label, str) and interval_label.endswith("h"):
            interval_ms = int(interval_label[:-1]) * 60 * 60 * 1000

    if interval_ms is None:
        return infer_candle_interval(historic_tail)

    return infer_candle_interval(historic_tail, expected_interval_ms=int(interval_ms))
