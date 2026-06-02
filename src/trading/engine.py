import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any

from src.config.settings import settings
from src.exchanges.factory import get_exchange
from src.exchanges.market_data import get_available_pairs, get_tickers, get_order_book
from src.trading.paper_simulator import PaperSimulator
from src.trading.live_trader import LiveTrader
from src.llm.cache import get_cached_ollama_response
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_coin_selection_prompt,
    build_strategy_prompt,
)
from src.strategies.llm_parser import create_strategy_from_llm
from src.strategies.validator import validate_signal
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

COIN_REVALUATION_INTERVAL = 300  # seconds (5 minutes)
STRATEGY_INTERVAL = 60           # seconds (1 minute)
POSITION_SIZE_FRACTION = 0.1     # fraction of base currency balance per trade


class TradingEngine:
    def __init__(self):
        self.exchange = get_exchange()
        self.base_currency = settings.BASE_CURRENCY
        self.max_coins = settings.MAX_COINS
        self.redis = get_redis_client()

        if settings.TRADING_MODE == "paper":
            self.trader = PaperSimulator(
                self.exchange,
                base_currency=self.base_currency,
                initial_balance=10000.0,
            )
        else:
            self.trader = LiveTrader(self.exchange)

        self.current_coins: List[str] = []
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self._load_state()

    def _load_state(self):
        """Load current coins and positions from Redis."""
        coins_json = self.redis.get("trading:current_coins")
        if coins_json:
            self.current_coins = json.loads(coins_json)
        positions_json = self.redis.get("trading:positions")
        if positions_json:
            self.positions = json.loads(positions_json)

    def _save_state(self):
        """Persist current coins and positions to Redis."""
        self.redis.set("trading:current_coins", json.dumps(self.current_coins))
        self.redis.set("trading:positions", json.dumps(self.positions))

    async def run(self):
        """Main loop that runs forever."""
        logger.info("Trading engine started.")
        while True:
            try:
                await self._reevaluate_coins()
                for symbol in self.current_coins:
                    await self._process_coin(symbol)
                self._save_state()
            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
            await asyncio.sleep(STRATEGY_INTERVAL)

    async def _reevaluate_coins(self):
        """Use LLM to select which coins to trade."""
        # Only re-evaluate every COIN_REVALUATION_INTERVAL
        last_key = "trading:last_coin_eval"
        last_eval = self.redis.get(last_key)
        now = time.time()
        if last_eval and (now - float(last_eval)) < COIN_REVALUATION_INTERVAL:
            return

        available_pairs = get_available_pairs(self.exchange, self.base_currency)
        if not available_pairs:
            logger.warning("No available pairs found.")
            return

        # Fetch tickers for a subset to keep prompt size manageable
        sample_pairs = available_pairs[:50]
        tickers = get_tickers(self.exchange, sample_pairs)

        prompt = build_coin_selection_prompt(
            available_pairs=sample_pairs,
            current_coins=self.current_coins,
            max_coins=self.max_coins,
            base_currency=self.base_currency,
            tickers=tickers,
        )
        response = get_cached_ollama_response(prompt, SYSTEM_PROMPT, ttl=300)
        try:
            new_coins = json.loads(response)
            if isinstance(new_coins, list):
                # Validate that coins are in available pairs
                valid_coins = [c for c in new_coins if c in available_pairs]
                self.current_coins = valid_coins[: self.max_coins]
                logger.info(f"Selected coins: {self.current_coins}")
        except json.JSONDecodeError:
            logger.error("Failed to parse coin selection response.")

        self.redis.set(last_key, now)

    async def _process_coin(self, symbol: str):
        """Fetch market data, get LLM strategy, validate, and execute."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            order_book = get_order_book(self.exchange, symbol, limit=5)
            balance = self.trader.fetch_balance()
            open_positions = [
                pos for pos in self.positions.values() if pos.get("symbol") == symbol
            ]

            prompt = build_strategy_prompt(
                symbol=symbol,
                ticker=ticker,
                order_book=order_book,
                balance=balance,
                open_positions=open_positions,
            )
            response = get_cached_ollama_response(prompt, SYSTEM_PROMPT, ttl=60)
            strategy = create_strategy_from_llm(response)
            signal = strategy.generate_signal({})
            validated = validate_signal(signal)

            if validated.action != "HOLD":
                await self._execute_signal(symbol, validated)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    async def _execute_signal(self, symbol: str, signal):
        """Execute a BUY or SELL signal."""
        base, quote = symbol.split("/")
        balance = self.trader.fetch_balance()

        if signal.action == "BUY":
            # Use a fraction of available quote balance
            quote_balance = balance.get(quote, 0.0)
            amount = quote_balance * POSITION_SIZE_FRACTION
            if amount <= 0:
                logger.info(f"Insufficient {quote} to buy {symbol}")
                return
            try:
                order = self.trader.create_market_buy_order(symbol, amount)
                logger.info(f"BUY {symbol}: {order}")
                # Record position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "buy",
                    "amount": order["amount"],
                    "price": order["price"],
                    "timestamp": order["timestamp"],
                }
            except Exception as e:
                logger.error(f"Buy order failed for {symbol}: {e}")

        elif signal.action == "SELL":
            base_balance = balance.get(base, 0.0)
            if base_balance <= 0:
                logger.info(f"No {base} balance to sell {symbol}")
                return
            try:
                order = self.trader.create_market_sell_order(symbol, base_balance)
                logger.info(f"SELL {symbol}: {order}")
                # Remove position
                self.positions.pop(symbol, None)
            except Exception as e:
                logger.error(f"Sell order failed for {symbol}: {e}")
