import uuid
import time
import logging
from typing import Dict, List, Optional, Any
import ccxt

logger = logging.getLogger(__name__)

from src.exchanges.fees import get_fee_rate


class PaperSimulator:
    """Simulates a trading account with fake balances and order execution."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        base_currency: str = "USDT",
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,  # 0.1% fee
        redis_client=None,
        ws_manager=None,
    ):
        self.exchange = exchange
        self.base_currency = base_currency
        self.fee_rate = fee_rate
        self.redis_client = redis_client
        self.ws_manager = ws_manager
        self.balances: Dict[str, float] = {base_currency: initial_balance}
        self.orders: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    def _get_price(self, symbol: str) -> float:
        """Get current mid price for a symbol, preferring live WebSocket data."""
        # Try WebSocket ticker first
        if self.ws_manager is not None:
            ws_ticker = self.ws_manager.get_ticker(symbol)
            if ws_ticker is not None:
                bid = ws_ticker.get('bid')
                ask = ws_ticker.get('ask')
                last = ws_ticker.get('last')
                if bid is not None and ask is not None:
                    return (bid + ask) / 2
                if last is not None:
                    return last
        # Fallback to REST
        ticker = self.exchange.fetch_ticker(symbol)
        return (ticker['bid'] + ticker['ask']) / 2 if ticker['bid'] and ticker['ask'] else ticker['last']

    def _deduct_fee(self, currency: str, amount: float) -> float:
        """Return amount after fee deduction."""
        return amount * (1 - self.fee_rate)

    def _get_fee_rate(self, symbol: str) -> float:
        """Return the taker fee rate for the symbol, using Redis cache if available."""
        return get_fee_rate(
            self.exchange,
            symbol,
            redis_client=self.redis_client,
            default=self.fee_rate,
        )

    def get_balance(self, currency: str) -> float:
        return self.balances.get(currency, 0.0)

    def fetch_balance(self) -> Dict[str, float]:
        return dict(self.balances)

    def create_market_buy_order(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        """Buy base currency using quote currency (e.g., spend USDT to buy BTC)."""
        base, quote = symbol.split('/')
        price = self._get_price(symbol)
        base_amount = quote_amount / price
        fee_rate = self._get_fee_rate(symbol)
        fee = base_amount * fee_rate
        net_base = base_amount - fee

        if self.balances.get(quote, 0) < quote_amount:
            raise ValueError(f"Insufficient {quote} balance")

        self.balances[quote] -= quote_amount
        self.balances[base] = self.balances.get(base, 0) + net_base

        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'type': 'market',
            'side': 'buy',
            'amount': base_amount,
            'price': price,
            'cost': quote_amount,
            'fee': {'cost': fee, 'currency': base},
            'status': 'closed',
            'timestamp': int(time.time() * 1000),
        }
        self.orders.append(order)
        self.trades.append(order)
        logger.info("Paper %s %s: %s %s @ %s", order['side'].upper(), order['symbol'], order['amount'], order['cost'], order['price'])
        return order

    def create_market_sell_order(self, symbol: str, base_amount: float) -> Dict[str, Any]:
        """Sell base currency to receive quote currency."""
        base, quote = symbol.split('/')
        price = self._get_price(symbol)
        quote_amount = base_amount * price
        fee_rate = self._get_fee_rate(symbol)
        fee = quote_amount * fee_rate
        net_quote = quote_amount - fee

        if self.balances.get(base, 0) < base_amount:
            raise ValueError(f"Insufficient {base} balance")

        self.balances[base] -= base_amount
        self.balances[quote] = self.balances.get(quote, 0) + net_quote

        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'type': 'market',
            'side': 'sell',
            'amount': base_amount,
            'price': price,
            'cost': quote_amount,
            'fee': {'cost': fee, 'currency': quote},
            'status': 'closed',
            'timestamp': int(time.time() * 1000),
        }
        self.orders.append(order)
        self.trades.append(order)
        logger.info("Paper %s %s: %s %s @ %s", order['side'].upper(), order['symbol'], order['amount'], order['cost'], order['price'])
        return order

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        # In this simple simulator, market orders fill immediately, so no open orders.
        return []

    def cancel_order(self, order_id: str) -> bool:
        # No open orders to cancel.
        return False

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return self.trades
