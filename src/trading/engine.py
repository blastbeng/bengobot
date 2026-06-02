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
from src.database import load_trading_state, save_trading_state, delete_trading_state

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
                initial_balance=settings.PAPER_INITIAL_BALANCE,
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
        self._ensure_cost_basis()
        # Ensure trading is not paused on startup
        self.redis.delete("trading:paused")

    def set_notifier(self, notifier):
        """Attach a notification service (e.g., TelegramBot)."""
        self.notifier = notifier

    def _restore_paper_state(self):
        """Replay trade history to restore paper simulator balances and positions with cost basis."""
        # Reset simulator to initial state
        self.trader.balances = {self.base_currency: self.initial_balance}
        self.trader.trades = []
        positions = {}
        for trade in self.trade_history:
            symbol = trade['symbol']
            side = trade['side']
            amount = trade['amount']
            price = trade['price']
            cost = trade['cost']
            fee = trade.get('fee', {})
            fee_cost = float(fee.get('cost', 0) or 0)
            fee_currency = fee.get('currency', '')
            base, quote = symbol.split('/')

            if side == 'buy':
                # Update balances
                self.trader.balances[quote] = self.trader.balances.get(quote, 0) - cost
                net_base = amount - (fee_cost if fee_currency == base else 0.0)
                self.trader.balances[base] = self.trader.balances.get(base, 0) + net_base

                # Update position
                if symbol in positions:
                    pos = positions[symbol]
                    pos['cost_basis'] += cost
                    pos['net_base'] += net_base
                    pos['amount'] += net_base
                    pos['price'] = pos['cost_basis'] / pos['net_base'] if pos['net_base'] else price
                else:
                    entry_price = cost / net_base if net_base else price
                    positions[symbol] = {
                        'symbol': symbol,
                        'side': 'buy',
                        'amount': net_base,
                        'price': entry_price,
                        'cost_basis': cost,
                        'net_base': net_base,
                        'timestamp': trade['timestamp'],
                        'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
                        'take_profit': entry_price * (1 + TAKE_PROFIT_PCT),
                    }
            elif side == 'sell':
                # Update balances
                self.trader.balances[base] = self.trader.balances.get(base, 0) - amount
                net_quote = cost - (fee_cost if fee_currency == quote else 0.0)
                self.trader.balances[quote] = self.trader.balances.get(quote, 0) + net_quote
                # Remove position
                positions.pop(symbol, None)

            self.trader.trades.append(trade)

        self.positions = positions

    def _ensure_cost_basis(self):
        """If positions lack cost_basis, compute it from amount and price (backward compat)."""
        for sym, pos in self.positions.items():
            if 'cost_basis' not in pos or 'net_base' not in pos:
                # Assume no fees for old positions; cost_basis = amount * price
                pos['cost_basis'] = pos['amount'] * pos['price']
                pos['net_base'] = pos['amount']

    async def _reconcile_positions(self):
        """Detect and handle external changes: delisted coins, externally sold positions."""
        # --- Delisted coins ---
        available_pairs = await asyncio.to_thread(get_available_pairs, self.exchange, self.base_currency)
        for coin in list(self.current_coins):
            if coin not in available_pairs:
                logger.warning(f"Coin {coin} no longer available. Removing from tracking.")
                self.current_coins.remove(coin)
                if coin in self.positions:
                    pos = self.positions.pop(coin)
                    trade = {
                        "symbol": coin,
                        "side": "sell",
                        "amount": pos["amount"],
                        "price": 0.0,
                        "cost": 0.0,
                        "fee": {"cost": 0.0, "currency": self.base_currency},
                        "timestamp": time.time(),
                        "note": "delisted"
                    }
                    self.trade_history.append(trade)
                    logger.warning(f"Delisted coin {coin}: recorded forced sell of {pos['amount']} at 0.")

        # --- Externally modified balances ---
        for symbol, pos in list(self.positions.items()):
            base = symbol.split('/')[0]
            try:
                actual_balance = await asyncio.to_thread(self.trader.get_balance, base)
            except Exception as e:
                logger.error(f"Failed to fetch balance for {base}: {e}")
                continue

            recorded_amount = pos.get("amount", 0.0)
            if actual_balance < recorded_amount - 1e-8:
                # External sell detected
                sold_amount = recorded_amount - actual_balance
                try:
                    ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                    current_price = ticker['last']
                except Exception:
                    current_price = pos.get("price", 0.0)  # fallback to entry price
                cost = sold_amount * current_price
                fee_rate = 0.001  # assume 0.1% fee
                fee_cost = cost * fee_rate
                trade = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": sold_amount,
                    "price": current_price,
                    "cost": cost,
                    "fee": {"cost": fee_cost, "currency": self.base_currency},
                    "timestamp": time.time(),
                    "note": "external_sell"
                }
                self.trade_history.append(trade)
                logger.warning(
                    f"External sell detected for {symbol}: {sold_amount} sold at ~{current_price}. "
                    f"Updating position from {recorded_amount} to {actual_balance}."
                )
                if actual_balance == 0.0:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]["amount"] = actual_balance
            elif actual_balance > recorded_amount + 1e-8:
                # External deposit – sync to actual balance
                logger.warning(
                    f"Balance of {base} increased externally from {recorded_amount} to {actual_balance}. "
                    f"Updating position."
                )
                self.positions[symbol]["amount"] = actual_balance

    def _load_state(self):
        """Load current coins, positions, trade history, and initial balance from SQLite."""
        state = load_trading_state()

        self.current_coins = state.get("current_coins", [])
        self.positions = state.get("positions", {})
        # Ensure risk fields exist (for backward compatibility)
        for pos in self.positions.values():
            if "stop_loss" not in pos:
                pos["stop_loss"] = pos["price"] * (1 - STOP_LOSS_PCT)
            if "take_profit" not in pos:
                pos["take_profit"] = pos["price"] * (1 + TAKE_PROFIT_PCT)

        self.trade_history = state.get("trade_history", [])

        if "initial_balance" in state:
            self.initial_balance = float(state["initial_balance"])
        else:
            # Compute and persist initial balance
            if settings.TRADING_MODE == "paper":
                self.initial_balance = settings.PAPER_INITIAL_BALANCE
            else:
                balance = self.trader.fetch_balance()
                self.initial_balance = balance.get(self.base_currency, 0.0)
            save_trading_state("initial_balance", self.initial_balance)

    async def _save_state(self):
        """Persist current coins, positions, and trade history to SQLite."""
        await asyncio.to_thread(save_trading_state, "current_coins", self.current_coins)
        await asyncio.to_thread(save_trading_state, "positions", self.positions)
        # Keep only the last 1000 trades to avoid unbounded growth
        self.trade_history = self.trade_history[-1000:]
        await asyncio.to_thread(save_trading_state, "trade_history", self.trade_history)

    async def run(self):
        """Main loop that runs forever."""
        logger.info("Trading engine started.")
        while True:
            try:
                await self._reconcile_positions()
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    await asyncio.sleep(STRATEGY_INTERVAL)
                    continue

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

        # Build market_limits with a concrete min_cost for each symbol
        market_limits = {}
        for symbol in sample_pairs:
            market = self.exchange.markets.get(symbol, {})
            limits = market.get('limits', {})
            min_cost = limits.get('cost', {}).get('min')
            min_amount = limits.get('amount', {}).get('min')
            ticker = tickers.get(symbol, {})
            last_price = ticker.get('last', 0)

            # Compute a numeric min_cost
            if min_cost is not None:
                numeric_min_cost = float(min_cost)
            elif min_amount is not None and last_price:
                numeric_min_cost = float(min_amount) * last_price
            else:
                numeric_min_cost = 0.0  # no known limit

            market_limits[symbol] = {
                'min_cost': numeric_min_cost,
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
        logger.info(f"LLM coin selection raw response: {response}")

        try:
            new_coins = json.loads(response)
            if isinstance(new_coins, list):
                # Validate that coins are in available pairs
                valid_coins = [c for c in new_coins if c in available_pairs]
                self.current_coins = valid_coins[: self.max_coins]
        except json.JSONDecodeError:
            logger.error("Failed to parse coin selection response.")
            new_coins = []

        # Fallback: if LLM returned no coins, pick top-volume affordable coins
        if not self.current_coins:
            logger.warning("LLM returned no coins – using volume-based fallback.")
            # Sort sample_pairs by 24h volume descending
            def volume(sym):
                t = tickers.get(sym, {})
                return t.get('quoteVolume', 0) or 0
            sorted_pairs = sorted(sample_pairs, key=volume, reverse=True)
            fallback_coins = []
            for sym in sorted_pairs:
                min_cost = market_limits.get(sym, {}).get('min_cost', 0)
                if per_coin_budget >= min_cost:
                    fallback_coins.append(sym)
                if len(fallback_coins) >= self.max_coins:
                    break
            self.current_coins = fallback_coins

        logger.info(f"Selected coins: {self.current_coins}")
        if self.notifier:
            await self.notifier.send_notification(
                f"🔄 Coins updated: {', '.join(self.current_coins) if self.current_coins else 'None'}"
            )

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
        total_fees = sum(
            float(t.get('fee', {}).get('cost', 0) or 0)
            for t in self.trade_history
        )
        total_value = current_balance + open_value
        pnl = total_value - self.initial_balance
        pnl_percent = (pnl / self.initial_balance * 100) if self.initial_balance else 0.0
        return {
            "initial_balance": self.initial_balance,
            "current_balance": current_balance,
            "open_value": open_value,
            "total_pnl": pnl,
            "pnl_percent": pnl_percent,
            "total_fees": total_fees,
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
                # Extract fee info for cost basis tracking
                fee = order.get('fee', {})
                fee_cost = float(fee.get('cost', 0.0) or 0.0)
                fee_currency = fee.get('currency', '')
                cost_basis = order['cost']
                net_base = order['amount'] - (fee_cost if fee_currency == base else 0.0)

                if symbol in self.positions:
                    # Accumulate: weighted average price with cost basis
                    old_cost_basis = self.positions[symbol].get("cost_basis", self.positions[symbol]["amount"] * self.positions[symbol]["price"])
                    old_net_base = self.positions[symbol].get("net_base", self.positions[symbol]["amount"])
                    new_cost_basis = old_cost_basis + cost_basis
                    new_net_base = old_net_base + net_base
                    new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
                    self.positions[symbol]["amount"] = new_net_base
                    self.positions[symbol]["price"] = new_price
                    self.positions[symbol]["cost_basis"] = new_cost_basis
                    self.positions[symbol]["net_base"] = new_net_base
                    self.positions[symbol]["stop_loss"] = new_price * (1 - STOP_LOSS_PCT)
                    self.positions[symbol]["take_profit"] = new_price * (1 + TAKE_PROFIT_PCT)
                else:
                    entry_price = cost_basis / net_base if net_base > 0 else order["price"]
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "side": "buy",
                        "amount": net_base,
                        "price": entry_price,
                        "timestamp": order["timestamp"],
                        "stop_loss": entry_price * (1 - STOP_LOSS_PCT),
                        "take_profit": entry_price * (1 + TAKE_PROFIT_PCT),
                        "cost_basis": cost_basis,
                        "net_base": net_base,
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
            # Determine the amount of base currency to sell
            pos = self.positions.get(symbol)
            if pos:
                gross_amount = pos["amount"]
            else:
                gross_amount = balance.get(base, 0.0)
            if gross_amount <= 0:
                logger.info(f"No {base} to sell for {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(f"⚠️ No {base} to sell for {symbol}")
                return
            try:
                order = await asyncio.to_thread(self.trader.create_market_sell_order, symbol, gross_amount)
                logger.info(f"SELL {symbol}: {order}")
                # Compute realized P&L
                fee = order.get('fee', {})
                fee_cost = float(fee.get('cost', 0.0) or 0.0)
                fee_currency = fee.get('currency', '')
                net_quote = order['cost'] - (fee_cost if fee_currency == quote else 0.0)
                if pos:
                    cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                    realized_pnl = net_quote - cost_basis
                else:
                    realized_pnl = 0.0
                order["realized_pnl"] = realized_pnl
                order["cost_basis"] = pos.get("cost_basis", 0.0) if pos else 0.0
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
