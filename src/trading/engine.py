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
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm
from src.strategies.validator import validate_signal
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

COIN_REVALUATION_INTERVAL = 300  # seconds (5 minutes)
STRATEGY_INTERVAL = 60           # seconds (1 minute)
POSITION_SIZE_FRACTION = 0.1     # fraction of base currency balance per trade
STOP_LOSS_PCT = 0.05            # 5% below entry price
TAKE_PROFIT_PCT = 0.10          # 10% above entry price


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
        self.trade_history: List[Dict[str, Any]] = []
        self.initial_balance: float = 0.0
        self.notifier = None
        self._load_state()
        # Restore paper simulator state from trade history
        if settings.TRADING_MODE == "paper":
            self._restore_paper_state()
        # Ensure trading is not paused on startup
        self.redis.delete("trading:paused")

    def set_notifier(self, notifier):
        """Attach a notification service (e.g., TelegramBot)."""
        self.notifier = notifier

    def _restore_paper_state(self):
        """Replay trade history to restore paper simulator balances."""
        # Reset simulator to initial state
        self.trader.balances = {self.base_currency: self.initial_balance}
        self.trader.trades = []
        for trade in self.trade_history:
            symbol = trade['symbol']
            side = trade['side']
            amount = trade['amount']
            price = trade['price']
            cost = trade['cost']
            fee = trade.get('fee', {})
            fee_cost = fee.get('cost', 0)
            fee_currency = fee.get('currency', '')
            base, quote = symbol.split('/')
            if side == 'buy':
                # Deduct quote, add base minus fee
                self.trader.balances[quote] = self.trader.balances.get(quote, 0) - cost
                net_base = amount - fee_cost if fee_currency == base else amount
                self.trader.balances[base] = self.trader.balances.get(base, 0) + net_base
            elif side == 'sell':
                # Deduct base, add quote minus fee
                self.trader.balances[base] = self.trader.balances.get(base, 0) - amount
                net_quote = cost - fee_cost if fee_currency == quote else cost
                self.trader.balances[quote] = self.trader.balances.get(quote, 0) + net_quote
            self.trader.trades.append(trade)

    def _load_state(self):
        """Load current coins and positions from Redis."""
        coins_json = self.redis.get("trading:current_coins")
        if coins_json:
            self.current_coins = json.loads(coins_json)
        positions_json = self.redis.get("trading:positions")
        if positions_json:
            self.positions = json.loads(positions_json)
            # Ensure risk fields exist (for backward compatibility)
            for pos in self.positions.values():
                if "stop_loss" not in pos:
                    pos["stop_loss"] = pos["price"] * (1 - STOP_LOSS_PCT)
                if "take_profit" not in pos:
                    pos["take_profit"] = pos["price"] * (1 + TAKE_PROFIT_PCT)
        trades_json = self.redis.get("trading:trade_history")
        if trades_json:
            self.trade_history = json.loads(trades_json)
        init_bal = self.redis.get("trading:initial_balance")
        if init_bal:
            self.initial_balance = float(init_bal)
        else:
            balance = self.trader.fetch_balance()
            self.initial_balance = balance.get(self.base_currency, 0.0)
            self.redis.set("trading:initial_balance", self.initial_balance)

    async def _save_state(self):
        """Persist current coins and positions to Redis."""
        await asyncio.to_thread(self.redis.set, "trading:current_coins", json.dumps(self.current_coins))
        await asyncio.to_thread(self.redis.set, "trading:positions", json.dumps(self.positions))
        # Keep only the last 1000 trades to avoid unbounded growth
        self.trade_history = self.trade_history[-1000:]
        await asyncio.to_thread(self.redis.set, "trading:trade_history", json.dumps(self.trade_history))

    async def run(self):
        """Main loop that runs forever."""
        logger.info("Trading engine started.")
        startup_notified = False
        while True:
            try:
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    await asyncio.sleep(STRATEGY_INTERVAL)
                    continue

                if not startup_notified and self.notifier:
                    await self.notifier.send_notification("🚀 Trading engine started.")
                    startup_notified = True

                await self._reevaluate_coins()
                for symbol in self.current_coins:
                    await self._process_coin(symbol)
                await self._check_risk_management()
                await self._save_state()
            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
            await asyncio.sleep(STRATEGY_INTERVAL)

    async def _reevaluate_coins(self):
        """Use LLM to select which coins to trade."""
        # Only re-evaluate every COIN_REVALUATION_INTERVAL
        last_key = "trading:last_coin_eval"
        last_eval = await asyncio.to_thread(self.redis.get, last_key)
        now = time.time()
        if last_eval and (now - float(last_eval)) < COIN_REVALUATION_INTERVAL:
            return

        available_pairs = await asyncio.to_thread(get_available_pairs, self.exchange, self.base_currency)
        if not available_pairs:
            logger.warning("No available pairs found.")
            return

        # Fetch balance and compute per-coin budget
        balance = await asyncio.to_thread(self.trader.fetch_balance)
        base_balance = balance.get(self.base_currency, 0.0)
        per_coin_budget = base_balance / self.max_coins if self.max_coins > 0 else 0.0

        # Fetch tickers for a subset to keep prompt size manageable
        sample_pairs = available_pairs[:50]
        tickers = await asyncio.to_thread(get_tickers, self.exchange, sample_pairs)

        # Extract minimum trade limits from exchange markets
        market_limits = {}
        for symbol in sample_pairs:
            market = self.exchange.markets.get(symbol, {})
            limits = market.get('limits', {})
            min_cost = limits.get('cost', {}).get('min')
            min_amount = limits.get('amount', {}).get('min')
            market_limits[symbol] = {
                'min_cost': min_cost,
                'min_amount': min_amount,
            }

        prompt = build_coin_selection_prompt(
            available_pairs=sample_pairs,
            current_coins=self.current_coins,
            max_coins=self.max_coins,
            base_currency=self.base_currency,
            tickers=tickers,
            base_balance=base_balance,
            per_coin_budget=per_coin_budget,
            market_limits=market_limits,
        )
        response = await asyncio.to_thread(get_cached_ollama_response, prompt, SYSTEM_PROMPT, 300)
        try:
            new_coins = json.loads(response)
            if isinstance(new_coins, list):
                # Validate that coins are in available pairs
                valid_coins = [c for c in new_coins if c in available_pairs]
                self.current_coins = valid_coins[: self.max_coins]
                logger.info(f"Selected coins: {self.current_coins}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🔄 Coins updated: {', '.join(self.current_coins) if self.current_coins else 'None'}"
                    )
        except json.JSONDecodeError:
            logger.error("Failed to parse coin selection response.")
            if self.notifier:
                await self.notifier.send_notification("❌ Failed to parse coin selection response.")

        await asyncio.to_thread(self.redis.set, last_key, now)

    async def _process_coin(self, symbol: str):
        """Fetch market data, get LLM strategy, validate, and execute."""
        try:
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            order_book = await asyncio.to_thread(get_order_book, self.exchange, symbol, 20)
            balance = await asyncio.to_thread(self.trader.fetch_balance)
            open_positions = [
                pos for pos in self.positions.values() if pos.get("symbol") == symbol
            ]

            # Compute per-coin budget for this coin
            base_balance = balance.get(self.base_currency, 0.0)
            per_coin_budget = base_balance / self.max_coins if self.max_coins > 0 else 0.0

            prompt = build_strategy_prompt(
                symbol=symbol,
                ticker=ticker,
                order_book=order_book,
                balance=balance,
                open_positions=open_positions,
                per_coin_budget=per_coin_budget,
                max_coins=self.max_coins,
            )
            response = await asyncio.to_thread(get_cached_ollama_response, prompt, SYSTEM_PROMPT, 60)
            strategy = create_strategy_from_llm(response)
            signal = strategy.generate_signal({})
            validated = validate_signal(signal)

            # Log and notify the decision
            logger.info(f"Decision for {symbol}: {validated.action} (confidence: {validated.confidence:.2f})")
            if self.notifier:
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(validated.action, "❓")
                await self.notifier.send_notification(
                    f"{emoji} {symbol}: {validated.action} "
                    f"(confidence: {validated.confidence:.2f}) – {validated.reasoning}"
                )

            if validated.action != "HOLD":
                await self._execute_signal(symbol, validated)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            if self.notifier:
                await self.notifier.send_notification(f"❌ Error processing {symbol}: {e}")

    def get_profit_summary(self) -> Dict[str, float]:
        """Return profit/loss summary."""
        balance = self.trader.fetch_balance()
        current_balance = balance.get(self.base_currency, 0.0)
        open_value = 0.0
        for sym, pos in self.positions.items():
            try:
                ticker = self.exchange.fetch_ticker(sym)
                price = ticker['last']
                open_value += pos['amount'] * price
            except Exception:
                pass
        total_value = current_balance + open_value
        pnl = total_value - self.initial_balance
        pnl_percent = (pnl / self.initial_balance * 100) if self.initial_balance else 0.0
        return {
            "initial_balance": self.initial_balance,
            "current_balance": current_balance,
            "open_value": open_value,
            "total_pnl": pnl,
            "pnl_percent": pnl_percent,
        }

    async def _check_risk_management(self):
        """Check open positions and close if stop-loss or take-profit is hit."""
        for symbol, pos in list(self.positions.items()):
            try:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                current_price = ticker['last']
                if current_price <= pos["stop_loss"]:
                    logger.info(f"Stop-loss triggered for {symbol} at {current_price}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⛔ Stop‑loss triggered for {symbol} at {current_price:.4f}"
                        )
                    await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss"))
                elif current_price >= pos["take_profit"]:
                    logger.info(f"Take-profit triggered for {symbol} at {current_price}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"✅ Take‑profit triggered for {symbol} at {current_price:.4f}"
                        )
                    await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit"))
            except Exception as e:
                logger.error(f"Risk check failed for {symbol}: {e}")

    async def _execute_signal(self, symbol: str, signal):
        """Execute a BUY or SELL signal."""
        base, quote = symbol.split("/")
        balance = await asyncio.to_thread(self.trader.fetch_balance)

        if signal.action == "BUY":
            # Use per-coin budget as the buy amount, capped at available balance
            quote_balance = balance.get(quote, 0.0)
            per_coin_budget = quote_balance / self.max_coins if self.max_coins > 0 else 0.0
            amount = min(per_coin_budget, quote_balance)
            if amount <= 0:
                logger.info(f"Insufficient {quote} to buy {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(f"⚠️ Insufficient {quote} to buy {symbol}")
                return
            try:
                order = await asyncio.to_thread(self.trader.create_market_buy_order, symbol, amount)
                logger.info(f"BUY {symbol}: {order}")
                # Update or create position
                if symbol in self.positions:
                    # Accumulate: weighted average price
                    old_amount = self.positions[symbol]["amount"]
                    old_price = self.positions[symbol]["price"]
                    new_amount = old_amount + order["amount"]
                    new_price = ((old_amount * old_price) + (order["amount"] * order["price"])) / new_amount
                    self.positions[symbol]["amount"] = new_amount
                    self.positions[symbol]["price"] = new_price
                    self.positions[symbol]["stop_loss"] = new_price * (1 - STOP_LOSS_PCT)
                    self.positions[symbol]["take_profit"] = new_price * (1 + TAKE_PROFIT_PCT)
                else:
                    entry_price = order["price"]
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "side": "buy",
                        "amount": order["amount"],
                        "price": entry_price,
                        "timestamp": order["timestamp"],
                        "stop_loss": entry_price * (1 - STOP_LOSS_PCT),
                        "take_profit": entry_price * (1 + TAKE_PROFIT_PCT),
                    }
                self.trade_history.append(order)
                await self._save_state()
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🟢 BUY {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    )
            except Exception as e:
                logger.error(f"Buy order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(f"❌ Buy order failed for {symbol}: {e}")

        elif signal.action == "SELL":
            # Sell the exact position amount if we have one, otherwise sell all base balance
            pos = self.positions.get(symbol)
            if pos:
                sell_amount = pos["amount"]
            else:
                sell_amount = balance.get(base, 0.0)
            if sell_amount <= 0:
                logger.info(f"No {base} to sell for {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(f"⚠️ No {base} to sell for {symbol}")
                return
            try:
                order = await asyncio.to_thread(self.trader.create_market_sell_order, symbol, sell_amount)
                logger.info(f"SELL {symbol}: {order}")
                # Remove position
                self.positions.pop(symbol, None)
                self.trade_history.append(order)
                await self._save_state()
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🔴 SELL {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    )
            except Exception as e:
                logger.error(f"Sell order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(f"❌ Sell order failed for {symbol}: {e}")
