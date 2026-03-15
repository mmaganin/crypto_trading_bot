from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import FeeModel


@dataclass
class PendingOrder:
    entry_index: int
    created_index: int
    market: str
    side: str
    leverage: float
    committed_capital: float
    stop_return: float
    target_return: float
    expected_return: float
    confidence: float


@dataclass
class OpenPosition:
    position_id: int
    opened_at_index: int
    opened_at: pd.Timestamp
    market: str
    side: str
    entry_price: float
    quantity: float
    committed_capital: float
    leverage: float
    stop_price: float
    target_price: float
    liquidation_price: float | None
    expiry_index: int
    open_fee: float
    expected_return: float
    confidence: float

    def fee_rate(self, fee_model: FeeModel) -> float:
        return fee_model.spot_fee_rate if self.market == "spot" else fee_model.margin_fee_rate


@dataclass
class ClosedTrade:
    position_id: int
    market: str
    side: str
    opened_at: str
    closed_at: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    bars_held: int
    gross_pnl: float
    net_pnl: float
    return_on_capital: float
    fees_paid: float
    exit_reason: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExitDecision:
    price: float
    reason: str


class Portfolio:
    def __init__(self, starting_cash: float, fee_model: FeeModel, max_open_positions: int) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.fee_model = fee_model
        self.max_open_positions = max_open_positions
        self.positions: list[OpenPosition] = []
        self.closed_trades: list[ClosedTrade] = []
        self.position_sequence = 0
        self.total_fees_paid = 0.0

    def equity(self, mark_price: float) -> float:
        total_equity = self.cash
        for position in self.positions:
            estimated_close_fee = position.quantity * mark_price * position.fee_rate(self.fee_model)
            if position.market == "spot":
                total_equity += position.quantity * mark_price - estimated_close_fee
                continue
            unrealized_pnl = self._gross_pnl(position, mark_price)
            total_equity += max(0.0, position.committed_capital + unrealized_pnl - estimated_close_fee)
        return total_equity

    def open_position(
        self,
        order: PendingOrder,
        entry_price: float,
        opened_at: pd.Timestamp,
        max_hold_bars: int,
    ) -> OpenPosition | None:
        if len(self.positions) >= self.max_open_positions:
            return None
        if order.market == "spot" and order.side != "long":
            return None

        fee_rate = self.fee_model.spot_fee_rate if order.market == "spot" else self.fee_model.margin_fee_rate
        leverage = 1.0 if order.market == "spot" else min(order.leverage, self.fee_model.max_leverage)
        notional = order.committed_capital * leverage
        quantity = notional / entry_price
        open_fee = notional * fee_rate
        cash_needed = order.committed_capital + open_fee
        if cash_needed > self.cash or quantity <= 0.0:
            return None

        self.cash -= cash_needed
        self.total_fees_paid += open_fee
        self.position_sequence += 1

        if order.side == "long":
            stop_price = entry_price * (1.0 - order.stop_return)
            target_price = entry_price * (1.0 + order.target_return)
            liquidation_price = (
                entry_price * (1.0 - 1.0 / leverage) if order.market == "margin" and leverage > 1.0 else None
            )
        else:
            stop_price = entry_price * (1.0 + order.stop_return)
            target_price = entry_price * (1.0 - order.target_return)
            liquidation_price = (
                entry_price * (1.0 + 1.0 / leverage) if order.market == "margin" and leverage > 1.0 else None
            )

        position = OpenPosition(
            position_id=self.position_sequence,
            opened_at_index=order.entry_index,
            opened_at=opened_at,
            market=order.market,
            side=order.side,
            entry_price=entry_price,
            quantity=quantity,
            committed_capital=order.committed_capital,
            leverage=leverage,
            stop_price=stop_price,
            target_price=target_price,
            liquidation_price=liquidation_price,
            expiry_index=order.entry_index + max_hold_bars,
            open_fee=open_fee,
            expected_return=order.expected_return,
            confidence=order.confidence,
        )
        self.positions.append(position)
        return position

    def maybe_exit_position(
        self,
        position: OpenPosition,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_index: int,
    ) -> ExitDecision | None:
        if position.side == "long":
            if position.liquidation_price is not None and bar_open <= position.liquidation_price:
                return ExitDecision(bar_open, "liquidation_gap")
            if bar_open <= position.stop_price:
                return ExitDecision(bar_open, "stop_gap")
            if bar_open >= position.target_price:
                return ExitDecision(bar_open, "target_gap")
            if position.liquidation_price is not None and bar_low <= position.liquidation_price:
                return ExitDecision(position.liquidation_price, "liquidated")
            if bar_low <= position.stop_price:
                return ExitDecision(position.stop_price, "stop_loss")
            if bar_high >= position.target_price:
                return ExitDecision(position.target_price, "take_profit")
        else:
            if position.liquidation_price is not None and bar_open >= position.liquidation_price:
                return ExitDecision(bar_open, "liquidation_gap")
            if bar_open >= position.stop_price:
                return ExitDecision(bar_open, "stop_gap")
            if bar_open <= position.target_price:
                return ExitDecision(bar_open, "target_gap")
            if position.liquidation_price is not None and bar_high >= position.liquidation_price:
                return ExitDecision(position.liquidation_price, "liquidated")
            if bar_high >= position.stop_price:
                return ExitDecision(position.stop_price, "stop_loss")
            if bar_low <= position.target_price:
                return ExitDecision(position.target_price, "take_profit")

        if bar_index >= position.expiry_index:
            return ExitDecision(bar_close, "time_exit")
        return None

    def close_position(
        self,
        position: OpenPosition,
        exit_price: float,
        closed_at: pd.Timestamp,
        bar_index: int,
        reason: str,
    ) -> ClosedTrade:
        fee_rate = position.fee_rate(self.fee_model)
        close_fee = position.quantity * exit_price * fee_rate
        gross_pnl = self._gross_pnl(position, exit_price)
        if position.market == "spot":
            cash_delta = position.quantity * exit_price - close_fee
        else:
            cash_delta = max(0.0, position.committed_capital + gross_pnl - close_fee)
        self.cash += cash_delta
        self.total_fees_paid += close_fee

        closed_trade = ClosedTrade(
            position_id=position.position_id,
            market=position.market,
            side=position.side,
            opened_at=position.opened_at.isoformat(),
            closed_at=closed_at.isoformat(),
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            bars_held=bar_index - position.opened_at_index,
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl - position.open_fee - close_fee,
            return_on_capital=(gross_pnl - position.open_fee - close_fee) / position.committed_capital,
            fees_paid=position.open_fee + close_fee,
            exit_reason=reason,
        )
        self.positions = [current for current in self.positions if current.position_id != position.position_id]
        self.closed_trades.append(closed_trade)
        return closed_trade

    @staticmethod
    def _gross_pnl(position: OpenPosition, exit_price: float) -> float:
        if position.side == "long":
            return position.quantity * (exit_price - position.entry_price)
        return position.quantity * (position.entry_price - exit_price)
