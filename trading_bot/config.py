from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


BASE_CANDLE_INTERVAL_MS = 5 * 60 * 1000


def scale_bar_count(base_bars: int, candle_interval_ms: int, minimum: int = 1) -> int:
    if candle_interval_ms <= 0:
        raise ValueError("Candle interval must be positive.")
    scaled = int(round(base_bars * BASE_CANDLE_INTERVAL_MS / candle_interval_ms))
    return max(minimum, scaled)


@dataclass(frozen=True)
class FeeModel:
    spot_fee_rate: float = 0.005
    margin_fee_rate: float = 0.0005
    max_leverage: float = 5.0

    @property
    def spot_round_trip(self) -> float:
        return self.spot_fee_rate * 2.0

    @property
    def margin_round_trip(self) -> float:
        return self.margin_fee_rate * 2.0


@dataclass(frozen=True)
class StrategyParameters:
    horizon_bars: int = 576
    label_return_threshold: float = 0.002
    margin_threshold: float = 0.001
    spot_threshold: float = 0.025
    min_confidence: float = 0.05
    margin_probability: float = 0.35
    spot_probability: float = 0.52
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 6.0
    max_hold_bars: int = 576
    cooldown_bars: int = 144
    max_open_positions: int = 5
    spot_position_fraction: float = 0.10
    margin_collateral_fraction: float = 0.04
    min_cash_buffer: float = 0.15
    leverage_floor: float = 1.5
    leverage_ceiling: float = 5.0
    initial_epochs: int = 4
    alpha: float = 0.0004
    l1_ratio: float = 0.18
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchConfig:
    starting_cash: float = 10_000.0
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    recent_candle_buffer: int = 500
    minimum_training_rows: int = 2_500

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.validation_ratio

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scale_strategy_parameters(parameters: StrategyParameters, candle_interval_ms: int) -> StrategyParameters:
    return replace(
        parameters,
        horizon_bars=scale_bar_count(parameters.horizon_bars, candle_interval_ms, minimum=1),
        max_hold_bars=scale_bar_count(parameters.max_hold_bars, candle_interval_ms, minimum=1),
        cooldown_bars=scale_bar_count(parameters.cooldown_bars, candle_interval_ms, minimum=0),
    )


def scale_research_config(config: ResearchConfig, candle_interval_ms: int) -> ResearchConfig:
    return replace(
        config,
        recent_candle_buffer=scale_bar_count(config.recent_candle_buffer, candle_interval_ms, minimum=8),
        minimum_training_rows=scale_bar_count(config.minimum_training_rows, candle_interval_ms, minimum=32),
    )


def candidate_parameters(
    base: StrategyParameters,
    candle_interval_ms: int = BASE_CANDLE_INTERVAL_MS,
) -> list[StrategyParameters]:
    profiles = (
        {},
        {
            "horizon_bars": 288,
            "label_return_threshold": 0.0010,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0220,
            "max_hold_bars": 288,
            "stop_loss_atr_multiple": 1.8,
            "take_profit_atr_multiple": 4.5,
            "cooldown_bars": 48,
        },
        {
            "horizon_bars": 288,
            "label_return_threshold": 0.0015,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0240,
            "max_hold_bars": 288,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 5.0,
            "cooldown_bars": 72,
        },
        {
            "horizon_bars": 576,
            "label_return_threshold": 0.0015,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0220,
            "max_hold_bars": 576,
            "stop_loss_atr_multiple": 1.8,
            "take_profit_atr_multiple": 5.0,
            "cooldown_bars": 72,
        },
        {
            "horizon_bars": 576,
            "label_return_threshold": 0.0020,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0250,
            "max_hold_bars": 576,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 6.0,
            "cooldown_bars": 72,
        },
        {
            "horizon_bars": 576,
            "label_return_threshold": 0.0020,
            "margin_threshold": 0.0012,
            "spot_threshold": 0.0280,
            "max_hold_bars": 576,
            "stop_loss_atr_multiple": 2.4,
            "take_profit_atr_multiple": 7.0,
            "cooldown_bars": 96,
        },
        {
            "horizon_bars": 432,
            "label_return_threshold": 0.0015,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0220,
            "max_hold_bars": 432,
            "stop_loss_atr_multiple": 1.8,
            "take_profit_atr_multiple": 4.5,
            "cooldown_bars": 48,
        },
        {
            "horizon_bars": 432,
            "label_return_threshold": 0.0020,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0240,
            "max_hold_bars": 432,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 5.5,
            "cooldown_bars": 72,
        },
        {
            "horizon_bars": 288,
            "label_return_threshold": 0.0020,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0260,
            "max_hold_bars": 288,
            "stop_loss_atr_multiple": 2.2,
            "take_profit_atr_multiple": 5.5,
            "cooldown_bars": 72,
        },
        {
            "horizon_bars": 576,
            "label_return_threshold": 0.0015,
            "margin_threshold": 0.0008,
            "spot_threshold": 0.0220,
            "max_hold_bars": 576,
            "stop_loss_atr_multiple": 1.8,
            "take_profit_atr_multiple": 5.5,
            "cooldown_bars": 96,
        },
        {
            "horizon_bars": 576,
            "label_return_threshold": 0.0015,
            "margin_threshold": 0.0010,
            "spot_threshold": 0.0240,
            "max_hold_bars": 432,
            "stop_loss_atr_multiple": 1.6,
            "take_profit_atr_multiple": 5.0,
            "cooldown_bars": 48,
        },
        {
            "horizon_bars": 288,
            "label_return_threshold": 0.0015,
            "margin_threshold": 0.0008,
            "spot_threshold": 0.0220,
            "max_hold_bars": 432,
            "stop_loss_atr_multiple": 1.6,
            "take_profit_atr_multiple": 4.5,
            "cooldown_bars": 96,
        },
    )
    seen: set[tuple[Any, ...]] = set()
    variants: list[StrategyParameters] = []
    for profile in profiles:
        candidate = scale_strategy_parameters(replace(base, **profile), candle_interval_ms)
        fingerprint = tuple(candidate.to_dict().items())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        variants.append(candidate)
    return variants
