from __future__ import annotations

from typing import Any

import ccxt


class CCXTExecutionClient:
    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        secret: str,
        password: str | None = None,
        sandbox: bool = False,
    ) -> None:
        exchange_class = getattr(ccxt, exchange_id)
        credentials: dict[str, Any] = {"apiKey": api_key, "secret": secret, "enableRateLimit": True}
        if password:
            credentials["password"] = password
        self.exchange = exchange_class(credentials)
        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

    def load_markets(self, reload: bool = False) -> dict[str, Any]:
        return self.exchange.load_markets(reload)

    def ensure_symbol(self, symbol: str) -> None:
        if symbol not in self.exchange.markets:
            raise ValueError(f"{symbol} is not available on {self.exchange.id}.")

    def create_spot_market_order(self, symbol: str, side: str, amount: float) -> dict[str, Any]:
        self.exchange.options["defaultType"] = "spot"
        self.ensure_symbol(symbol)
        return self.exchange.create_order(symbol, "market", side, amount, None)

    def create_margin_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: float,
        margin_mode: str = "isolated",
    ) -> dict[str, Any]:
        self.ensure_symbol(symbol)
        leverage = float(leverage)
        params: dict[str, Any] = {"marginMode": margin_mode}
        if self.exchange.has.get("setLeverage"):
            try:
                self.exchange.set_leverage(leverage, symbol, {"marginMode": margin_mode})
            except Exception:
                params["leverage"] = leverage
        else:
            params["leverage"] = leverage
        return self.exchange.create_order(symbol, "market", side, amount, None, params)

    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def close(self) -> None:
        close_method = getattr(self.exchange, "close", None)
        if close_method is not None:
            close_method()
