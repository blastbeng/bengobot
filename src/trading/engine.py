import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any

from src.config.settings import settings
from src.exchanges.fees import get_fee_rate
from src.exchanges.factory import get_exchange
from src.exchanges.market_data import get_available_pairs, get_tickers, get_order_book, get_multi_timeframe_ohlcv
from src.trading.paper_simulator import PaperSimulator
from src.trading.live_trader import LiveTrader
from src.llm.cache import get_cached_llm_response
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_coin_selection_prompt,
    build_strategy_prompt,
    compute_atr,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_ema,
    compute_stochastic,
    compute_adx,
    compute_obv,
    compute_mfi,
    compute_cci,
    compute_williams_r,
)
try:
    from src.news.fetcher import discover_trending_coins
except ImportError:
    discover_trending_coins = None
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm
from src.strategies.validator import validate_signal
from src.utils.redis_client import get_redis_client
from src.database import load_trading_state, save_trading_state, delete_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db

logger = logging.getLogger(__name__)

COIN_REVALUATION_INTERVAL = 900  # seconds (15 minutes)
STRATEGY_INTERVAL = 600           # seconds (10 minutes)


class TradingEngine:
    def __init__(self):
        self.exchange = get_exchange()
        self.base_currency = settings.BASE_CURRENCY
        self.max_coins = settings.MAX_COINS
        self.effective_max_coins = self.max_coins
        self.redis = get_redis_client()

        if settings.TRADING_MODE == "paper":
            self.trader = PaperSimulator(
                self.exchange,
                base_currency=self.base_currency,
                initial_balance=settings.PAPER_INITIAL_BALANCE,
                redis_client=self.redis,
            )
        else:
            self.trader = LiveTrader(self.exchange)

        self.current_coins: List[Dict[str, str]] = []   # each dict: {"symbol": ..., "timeframe": ...}
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

    def _get_sentiment_str(self, symbol: str) -> str:
        """Get a short news sentiment string for notifications."""
        if not settings.NEWS_ENABLED:
            return ""
        try:
            base_coin = symbol.split("/")[0] if "/" in symbol else symbol
            agg_sent = get_aggregate_sentiment_from_db(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if agg_sent:
                compound = agg_sent["avg_compound"]
                sentiment_label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
                return f"📰 {sentiment_label} ({compound:+.2f}, {agg_sent['total_articles']} articles)"
        except Exception:
            pass
        return ""

    async def _refresh_news_cache(self):
        """Periodically fetch news for tracked coins and top-volume coins to keep cache warm."""
        if not settings.NEWS_ENABLED:
            return
        try:
            from src.news.fetcher import fetch_news_for_symbol
        except ImportError:
            logger.warning("News module not available; skipping background news refresh.")
            return

        # Perform an initial fetch immediately so news is available on startup.
        # Run all fetches in parallel with a timeout to avoid blocking the bot.
        try:
            symbols_to_refresh = set(entry["symbol"] for entry in self.current_coins)
            if symbols_to_refresh:
                logger.info(
                    f"Starting background initial news fetch for {len(symbols_to_refresh)} coins "
                    f"(timeout={settings.NEWS_INITIAL_FETCH_TIMEOUT_SECONDS}s). Trading continues immediately."
                )

                async def _fetch_and_store(sym: str):
                    try:
                        articles = await asyncio.to_thread(fetch_news_for_symbol, sym)
                        if articles:
                            base_coin = sym.split("/")[0] if "/" in sym else sym
                            await asyncio.to_thread(store_news_articles, base_coin, articles)
                    except Exception as e:
                        logger.debug(f"Initial news fetch failed for {sym}: {e}")

                await asyncio.wait_for(
                    asyncio.gather(*[_fetch_and_store(sym) for sym in symbols_to_refresh]),
                    timeout=settings.NEWS_INITIAL_FETCH_TIMEOUT_SECONDS,
                )
                logger.info("Initial news fetch complete.")
        except asyncio.TimeoutError:
            logger.warning(
                "Initial news fetch timed out. Some news will be missing until the next refresh cycle."
            )
        except Exception as e:
            logger.warning(f"Initial news fetch error: {e}")

        # Log current news table status
        try:
            from src.database import get_news_for_symbol
            for sym in symbols_to_refresh:
                base_coin = sym.split("/")[0] if "/" in sym else sym
                articles = get_news_for_symbol(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                logger.info(f"News DB check: {base_coin} has {len(articles)} articles.")
        except Exception as e:
            logger.warning(f"News DB status check failed: {e}")

        while True:
            try:
                cycle_start = time.time()
                # Determine symbols to refresh: current coins + top 10 by volume
                symbols_to_refresh = set(entry["symbol"] for entry in self.current_coins)
                try:
                    available_pairs = await asyncio.to_thread(
                        get_available_pairs, self.exchange, self.base_currency
                    )
                    tickers = await asyncio.to_thread(get_tickers, self.exchange, available_pairs[:50])
                    sorted_by_vol = sorted(
                        available_pairs[:50],
                        key=lambda s: tickers.get(s, {}).get("quoteVolume", 0) or 0,
                        reverse=True,
                    )[:50]
                    symbols_to_refresh.update(sorted_by_vol)
                except Exception as e:
                    logger.warning(f"Could not determine top-volume coins for news refresh: {e}")

                for sym in symbols_to_refresh:
                    try:
                        articles = await asyncio.to_thread(fetch_news_for_symbol, sym)
                        if articles:
                            base_coin = sym.split("/")[0] if "/" in sym else sym
                            await asyncio.to_thread(store_news_articles, base_coin, articles)
                    except Exception as e:
                        logger.debug(f"News refresh failed for {sym}: {e}")

                logger.debug(f"News cache refreshed for {len(symbols_to_refresh)} symbols in {time.time() - cycle_start:.2f}s")
            except Exception as e:
                logger.error(f"Background news refresh error: {e}")

            # Clean up old news articles
            try:
                from src.database import cleanup_old_news
                await asyncio.to_thread(cleanup_old_news, settings.NEWS_RETENTION_SECONDS)
            except Exception as e:
                logger.warning(f"News cleanup failed: {e}")

            await asyncio.sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)

    def _restore_paper_state(self):
        """Replay trade history to restore paper simulator balances and positions with cost basis."""
        old_positions = self.positions.copy()  # preserve LLM-set risk params
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
                cost_basis = cost + (fee_cost if fee_currency == quote else 0.0)
                self.trader.balances[base] = self.trader.balances.get(base, 0) + net_base

                # Update position
                if symbol in positions:
                    pos = positions[symbol]
                    pos['cost_basis'] += cost_basis
                    pos['net_base'] += net_base
                    pos['amount'] += net_base
                    pos['price'] = pos['cost_basis'] / pos['net_base'] if pos['net_base'] else price
                else:
                    entry_price = cost_basis / net_base if net_base else price
                    positions[symbol] = {
                        'symbol': symbol,
                        'side': 'buy',
                        'amount': net_base,
                        'price': entry_price,
                        'cost_basis': cost_basis,
                        'net_base': net_base,
                        'timestamp': trade['timestamp'],
                        # stop_loss and take_profit will be set by the LLM later
                        'stop_loss': None,
                        'take_profit': None,
                    }
            elif side == 'sell':
                # Update balances
                self.trader.balances[base] = self.trader.balances.get(base, 0) - amount
                net_quote = cost - (fee_cost if fee_currency == quote else 0.0)
                self.trader.balances[quote] = self.trader.balances.get(quote, 0) + net_quote
                # Remove position
                positions.pop(symbol, None)

            self.trader.trades.append(trade)

        # Merge saved risk parameters from old positions (LLM-defined)
        for sym, pos in positions.items():
            if sym in old_positions:
                old = old_positions[sym]
                # Only override if the old position had explicit risk params (from LLM)
                if "stop_loss" in old:
                    pos["stop_loss"] = old["stop_loss"]
                if "take_profit" in old:
                    pos["take_profit"] = old["take_profit"]
                if "trailing_stop" in old:
                    pos["trailing_stop"] = old["trailing_stop"]
                if "trailing_stop_distance_pct" in old:
                    pos["trailing_stop_distance_pct"] = old["trailing_stop_distance_pct"]
                if "max_hold_time_seconds" in old:
                    pos["max_hold_time_seconds"] = old["max_hold_time_seconds"]
                if "trailing_stop_activation_pct" in old:
                    pos["trailing_stop_activation_pct"] = old["trailing_stop_activation_pct"]
                if "timeframe" in old:
                    pos["timeframe"] = old["timeframe"]
        # Re-apply force_close for positions still missing risk parameters
        for sym, pos in positions.items():
            if "stop_loss" not in pos or "take_profit" not in pos:
                pos["_force_close"] = True
        self.positions = positions

    def _ensure_cost_basis(self):
        """If positions lack cost_basis, compute it from amount and price (backward compat)."""
        for sym, pos in self.positions.items():
            if 'cost_basis' not in pos or 'net_base' not in pos:
                # Assume no fees for old positions; cost_basis = amount * price
                pos['cost_basis'] = pos['amount'] * pos['price']
                pos['net_base'] = pos['amount']

    def _compute_performance_metrics(self) -> Dict[str, Any]:
        """Analyze trade history to produce per-coin and per-strategy performance summaries."""
        from collections import defaultdict

        now = time.time()
        coin_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0, "last_trade_ts": 0})
        strategy_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0})
        coin_stop_losses = defaultdict(int)
        coin_hold_times = defaultdict(list)

        for trade in self.trade_history:
            if trade.get("side") != "sell":
                continue
            symbol = trade["symbol"]
            pnl = trade.get("realized_pnl", 0.0)
            strategy = trade.get("strategy_type", "unknown")
            exit_reason = trade.get("exit_reason", "")
            if exit_reason == "stop_loss":
                coin_stop_losses[symbol] += 1
            hold_time = trade.get("hold_time_seconds")
            if hold_time is not None:
                coin_hold_times[symbol].append(hold_time)

            coin_stats[symbol]["trades"] += 1
            coin_stats[symbol]["total_pnl"] += pnl
            if pnl > 0:
                coin_stats[symbol]["wins"] += 1
            coin_stats[symbol]["last_trade_ts"] = max(coin_stats[symbol]["last_trade_ts"], trade.get("timestamp", 0) / 1000.0)

            strategy_stats[strategy]["trades"] += 1
            strategy_stats[strategy]["total_pnl"] += pnl
            if pnl > 0:
                strategy_stats[strategy]["wins"] += 1

        # Convert to dicts with win rates
        coin_perf = {}
        for sym, s in coin_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0.0
            coin_perf[sym] = {
                "trades": s["trades"],
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl": round(s["total_pnl"], 4),
                "last_trade_seconds_ago": round(now - s["last_trade_ts"]) if s["last_trade_ts"] else None,
                "stop_loss_hits": coin_stop_losses.get(sym, 0),
                "avg_hold_time_seconds": round(sum(coin_hold_times[sym]) / len(coin_hold_times[sym]), 1) if coin_hold_times.get(sym) else None,
            }

        strategy_perf = {}
        for st, s in strategy_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0.0
            strategy_perf[st] = {
                "trades": s["trades"],
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl": round(s["total_pnl"], 4),
            }

        # Overall equity curve summary: last 10 trades P&L trend
        recent_sells = [t for t in self.trade_history if t.get("side") == "sell"][-10:]
        recent_pnl = [t.get("realized_pnl", 0.0) for t in recent_sells]
        total_recent_pnl = sum(recent_pnl)
        trend = "up" if total_recent_pnl > 0 else "down" if total_recent_pnl < 0 else "flat"

        # Compute drawdown
        equity_series = []
        running_equity = 0.0
        for trade in self.trade_history:
            if trade.get("side") == "sell":
                running_equity += trade.get("realized_pnl", 0.0)
            equity_series.append(running_equity)
        peak = max(equity_series) if equity_series else 0.0
        current_equity = equity_series[-1] if equity_series else 0.0
        drawdown_pct = ((peak - current_equity) / peak * 100) if peak > 0 else 0.0

        return {
            "coin_performance": coin_perf,
            "strategy_performance": strategy_perf,
            "equity_curve": {
                "total_pnl": round(sum(t.get("realized_pnl", 0.0) for t in self.trade_history if t.get("side") == "sell"), 4),
                "recent_10_trades_pnl": round(total_recent_pnl, 4),
                "trend": trend,
                "drawdown_pct": round(drawdown_pct, 2),
            },
        }

    async def _reconcile_positions(self):
        """Detect and handle external changes: delisted coins, externally sold positions."""
        # --- Delisted coins ---
        available_pairs = await asyncio.to_thread(get_available_pairs, self.exchange, self.base_currency)
        for entry in list(self.current_coins):
            coin = entry["symbol"]
            if coin not in available_pairs:
                logger.warning(f"Coin {coin} no longer available. Removing from tracking.")
                self.current_coins.remove(entry)
                if coin in self.positions:
                    pos = self.positions.pop(coin)
                    cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                    trade = {
                        "symbol": coin,
                        "side": "sell",
                        "amount": pos["amount"],
                        "price": 0.0,
                        "cost": 0.0,
                        "fee": {"cost": 0.0, "currency": self.base_currency},
                        "timestamp": time.time() * 1000,
                        "note": "delisted",
                        "exit_reason": "delisted",
                        "realized_pnl": -cost_basis,
                        "cost_basis": cost_basis,
                    }
                    self.trade_history.append(trade)
                    await asyncio.to_thread(insert_trade, trade)
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
                fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
                fee_cost = cost * fee_rate
                trade = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": sold_amount,
                    "price": current_price,
                    "cost": cost,
                    "fee": {"cost": fee_cost, "currency": self.base_currency},
                    "timestamp": time.time() * 1000,
                    "note": "external_sell",
                    "exit_reason": "external_sell"
                }
                # Compute realized P&L for the externally sold portion
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_base = pos.get("net_base", pos["amount"])
                prorated_cost_basis = cost_basis * (sold_amount / net_base) if net_base > 0 else 0.0
                net_quote = cost - fee_cost
                trade["realized_pnl"] = net_quote - prorated_cost_basis
                trade["cost_basis"] = prorated_cost_basis
                self.trade_history.append(trade)
                await asyncio.to_thread(insert_trade, trade)
                logger.warning(
                    f"External sell detected for {symbol}: {sold_amount} sold at ~{current_price}. "
                    f"Updating position from {recorded_amount} to {actual_balance}."
                )
                if actual_balance == 0.0:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]["amount"] = actual_balance
                    self.positions[symbol]["cost_basis"] = cost_basis - prorated_cost_basis
                    self.positions[symbol]["net_base"] = net_base - sold_amount
                    new_net_base = self.positions[symbol]["net_base"]
                    new_cost_basis = self.positions[symbol]["cost_basis"]
                    self.positions[symbol]["price"] = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
            elif actual_balance > recorded_amount + 1e-8:
                # External deposit – sync to actual balance
                logger.warning(
                    f"Balance of {base} increased externally from {recorded_amount} to {actual_balance}. "
                    f"Updating position."
                )
                self.positions[symbol]["amount"] = actual_balance
                self.positions[symbol]["net_base"] = actual_balance
                cost_basis = self.positions[symbol].get("cost_basis", 0.0)
                self.positions[symbol]["price"] = cost_basis / actual_balance if actual_balance > 0 else 0.0

        # --- Close positions that were loaded without LLM risk parameters ---
        for symbol, pos in list(self.positions.items()):
            if pos.get("_force_close"):
                logger.info(f"Forcing close of {symbol} because it lacks LLM risk parameters.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🔻 Closing {symbol} – missing LLM risk parameters."
                    )
                signal = Signal(action="SELL", confidence=1.0, reasoning="Missing LLM risk parameters")
                await self._execute_signal(symbol, signal, exit_reason="force_close")

        # Persist any changes made during reconciliation
        await self._save_state()

    def _load_state(self):
        """Load current coins, positions, trade history, and initial balance from SQLite."""
        state = load_trading_state()

        raw_coins = state.get("current_coins", [])
        # Convert old format (list of strings) to new format if needed
        if raw_coins and isinstance(raw_coins[0], str):
            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
            self.current_coins = [{"symbol": s, "timeframe": default_tf} for s in raw_coins]
        else:
            self.current_coins = raw_coins
        self.positions = state.get("positions", {})
        # Remove any position that lacks LLM-defined risk parameters.
        # Such positions cannot be managed safely.
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if "stop_loss" not in pos or "take_profit" not in pos:
                logger.warning(
                    f"Position for {symbol} is missing stop_loss/take_profit. "
                    f"It will be closed because all risk parameters must come from the LLM."
                )
                pos["_force_close"] = True

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
        # Start background news refresh task
        asyncio.create_task(self._refresh_news_cache())
        while True:
            try:
                await self._reconcile_positions()
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    await asyncio.sleep(STRATEGY_INTERVAL)
                    continue

                await self._reevaluate_coins()
                await self._close_removed_positions()
                for coin_entry in self.current_coins:
                    await self._process_coin(coin_entry)
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
        if last_eval and (now - float(last_eval)) < COIN_REVALUATION_INTERVAL and self.current_coins:
            return

        available_pairs = await asyncio.to_thread(get_available_pairs, self.exchange, self.base_currency)
        if not available_pairs:
            logger.warning("No available pairs found.")
            return

        # --- News-driven coin discovery: add trending coins not in the top 50 ---
        if settings.NEWS_ENABLED and settings.NEWS_COIN_DISCOVERY_ENABLED and discover_trending_coins is not None:
            try:
                discovered = await asyncio.to_thread(
                    discover_trending_coins,
                    self.base_currency,
                    available_pairs,
                    max_coins=settings.NEWS_COIN_DISCOVERY_MAX_COINS,
                    min_sentiment=settings.NEWS_COIN_DISCOVERY_MIN_SENTIMENT,
                    min_articles=settings.NEWS_COIN_DISCOVERY_MIN_ARTICLES,
                )
                # Add discovered coins to the front of the list so they are included in the sample
                for pair in discovered:
                    if pair not in available_pairs:
                        available_pairs.insert(0, pair)
                if discovered:
                    logger.info(f"Added {len(discovered)} news-discovered coins to candidate pool.")
            except Exception as e:
                logger.warning(f"News coin discovery failed: {e}")

        # Fetch balance and compute per-coin budget
        balance = await asyncio.to_thread(self.trader.fetch_balance)
        base_balance = balance.get(self.base_currency, 0.0)
        per_coin_budget = base_balance / self.max_coins if self.max_coins > 0 else 0.0

        # Fetch tickers for a subset to keep prompt size manageable
        sample_pairs = available_pairs[:50]
        tickers = await asyncio.to_thread(get_tickers, self.exchange, sample_pairs)

        # Fetch news sentiment for all candidate coins
        news_sentiment = {}
        if settings.NEWS_ENABLED:
            for sym in sample_pairs:
                try:
                    base_coin = sym.split("/")[0] if "/" in sym else sym
                    agg = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                    if agg:
                        news_sentiment[base_coin] = agg
                except Exception as e:
                    logger.debug(f"Could not fetch news sentiment for {sym}: {e}")

        # Overall market trend (use BTC/USDT as benchmark)
        market_trend = None
        btc_symbol = "BTC/USDT"
        if btc_symbol in tickers:
            btc_ticker = tickers[btc_symbol]
            market_trend = {
                "symbol": btc_symbol,
                "change_24h": btc_ticker.get("percentage"),
                "last": btc_ticker.get("last"),
            }
        elif sample_pairs:
            # fallback to first available pair
            first = sample_pairs[0]
            if first in tickers:
                t = tickers[first]
                market_trend = {
                    "symbol": first,
                    "change_24h": t.get("percentage"),
                    "last": t.get("last"),
                }

        # Determine top coins by volume for OHLCV fetch (limit to 20 to avoid rate limits)
        def _volume(sym):
            t = tickers.get(sym, {})
            return t.get('quoteVolume', 0) or 0
        sorted_by_vol = sorted(sample_pairs, key=_volume, reverse=True)[:20]

        # Fetch multi-timeframe OHLCV for these coins
        ohlcv_data = {}
        if settings.OHLCV_TIMEFRAMES:
            async def fetch_ohlcv_for_symbol(sym):
                try:
                    data = await asyncio.to_thread(
                        get_multi_timeframe_ohlcv, self.exchange, sym, settings.OHLCV_TIMEFRAMES, limit=50
                    )
                    return sym, data
                except Exception as e:
                    logger.warning(f"OHLCV fetch failed for {sym}: {e}")
                    return sym, {}
            tasks = [fetch_ohlcv_for_symbol(sym) for sym in sorted_by_vol]
            results = await asyncio.gather(*tasks)
            ohlcv_data = dict(results)

        # Compute indicators for each coin with OHLCV data
        coin_indicators = {}
        for sym, tf_data in ohlcv_data.items():
            for tf in settings.OHLCV_TIMEFRAMES:
                if tf in tf_data and tf_data[tf]:
                    candles = tf_data[tf]
                    if len(candles) >= 26:
                        closes = [c[4] for c in candles]
                        highs = [c[2] for c in candles]
                        lows = [c[3] for c in candles]
                        volumes = [c[5] for c in candles]
                        ind = {}
                        ind['rsi'] = compute_rsi(closes)
                        macd_val, macd_sig, macd_hist = compute_macd(closes)
                        ind['macd'] = macd_val
                        ind['macd_signal'] = macd_sig
                        ind['macd_hist'] = macd_hist
                        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes)
                        ind['bb_upper'] = bb_upper
                        ind['bb_middle'] = bb_middle
                        ind['bb_lower'] = bb_lower
                        ema_9_list = compute_ema(closes, 9)
                        ema_21_list = compute_ema(closes, 21)
                        ind['ema_9'] = ema_9_list[-1] if ema_9_list else None
                        ind['ema_21'] = ema_21_list[-1] if ema_21_list else None
                        stochastic_k, stochastic_d = compute_stochastic(highs, lows, closes)
                        ind['stochastic_k'] = stochastic_k
                        ind['stochastic_d'] = stochastic_d
                        adx_val, plus_di, minus_di = compute_adx(highs, lows, closes)
                        ind['adx'] = adx_val
                        ind['plus_di'] = plus_di
                        ind['minus_di'] = minus_di
                        ind['obv'] = compute_obv(closes, volumes)
                        ind['mfi'] = compute_mfi(highs, lows, closes, volumes)
                        ind['cci'] = compute_cci(highs, lows, closes)
                        ind['williams_r'] = compute_williams_r(highs, lows, closes)
                        coin_indicators[sym] = ind
                    break

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

        # Compute effective max coins based on budget and minimum trade costs
        min_costs = [lim['min_cost'] for lim in market_limits.values() if lim['min_cost'] > 0]
        if min_costs:
            min_min_cost = min(min_costs)
            max_affordable = int(base_balance // min_min_cost) if min_min_cost > 0 else self.max_coins
        else:
            max_affordable = self.max_coins
        self.effective_max_coins = max(1, min(self.max_coins, max_affordable)) if max_affordable > 0 else 0

        if self.effective_max_coins == 0:
            logger.warning("Insufficient balance to trade any coin. Clearing coin list.")
            self.current_coins = []
            await asyncio.to_thread(self.redis.set, last_key, now)
            return

        # Recompute per-coin budget with the effective max
        per_coin_budget = base_balance / self.effective_max_coins

        perf = self._compute_performance_metrics()
        prompt = build_coin_selection_prompt(
            available_pairs=sample_pairs,
            current_coins=self.current_coins,
            max_coins=self.effective_max_coins,
            base_currency=self.base_currency,
            tickers=tickers,
            base_balance=base_balance,
            per_coin_budget=per_coin_budget,
            market_limits=market_limits,
            performance=perf,
            ohlcv_data=ohlcv_data,
            market_trend=market_trend,
            news_sentiment=news_sentiment,
            coin_indicators=coin_indicators,
        )
        response = await asyncio.to_thread(get_cached_llm_response, prompt, SYSTEM_PROMPT, 300)
        logger.info(f"LLM coin selection raw response: {response}")

        try:
            parsed = json.loads(response)
            new_coins: List[Dict[str, str]] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "symbol" in item:
                        sym = item["symbol"]
                        if sym in available_pairs:
                            tf = item.get("timeframe")
                            if tf not in settings.OHLCV_TIMEFRAMES:
                                tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                            new_coins.append({"symbol": sym, "timeframe": tf})
                    elif isinstance(item, str):
                        # backward compatibility: plain string
                        if item in available_pairs:
                            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                            new_coins.append({"symbol": item, "timeframe": default_tf})
                # Deduplicate by symbol, keeping first occurrence
                seen = set()
                deduped = []
                for entry in new_coins:
                    sym = entry["symbol"]
                    if sym not in seen:
                        seen.add(sym)
                        deduped.append(entry)
                self.current_coins = deduped[: self.effective_max_coins]
            else:
                logger.error("LLM coin selection response is not a list.")
        except json.JSONDecodeError:
            logger.error("Failed to parse coin selection response.")

        # Fallback: if LLM returned no coins, pick top-volume affordable coins
        if not self.current_coins:
            logger.warning("LLM returned no coins – using volume-based fallback.")
            # Sort sample_pairs by 24h volume descending
            sorted_pairs = sorted(sample_pairs, key=_volume, reverse=True)
            fallback_coins: List[Dict[str, str]] = []
            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
            for sym in sorted_pairs:
                min_cost = market_limits.get(sym, {}).get('min_cost', 0)
                if per_coin_budget >= min_cost:
                    fallback_coins.append({"symbol": sym, "timeframe": default_tf})
                if len(fallback_coins) >= self.effective_max_coins:
                    break
            self.current_coins = fallback_coins

        coin_labels = [f"{c['symbol']}({c['timeframe']})" for c in self.current_coins]
        logger.info(f"Selected coins: {coin_labels}")
        if self.notifier:
            await self.notifier.send_notification(
                f"🔄 Coins updated: {', '.join(coin_labels) if coin_labels else 'None'}"
            )

        await asyncio.to_thread(self.redis.set, last_key, now)

    async def _close_removed_positions(self):
        """Handle positions for coins that are no longer in the current selection.
        Instead of force-selling, we keep the position and let risk management close it naturally.
        """
        current_symbols = {entry["symbol"] for entry in self.current_coins}
        removed = [sym for sym in self.positions if sym not in current_symbols]
        for sym in removed:
            logger.info(f"Coin {sym} removed from selection. Position will be managed by risk parameters until closed.")
            if self.notifier:
                await self.notifier.send_notification(
                    f"📤 {sym} removed from active coins. Existing position will be held and managed by stop-loss/take-profit."
                )

    async def _process_coin(self, coin_entry: Dict[str, str]):
        """Fetch market data, get LLM strategy, validate, and execute."""
        symbol = coin_entry["symbol"]
        assigned_tf = coin_entry["timeframe"]
        try:
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            order_book = await asyncio.to_thread(get_order_book, self.exchange, symbol, 20)
            balance = await asyncio.to_thread(self.trader.fetch_balance)
            base_balance = balance.get(self.base_currency, 0.0)
            if base_balance <= 0 or self.effective_max_coins == 0:
                logger.debug(f"Skipping {symbol}: insufficient balance or effective_max_coins=0")
                return

            # Fetch OHLCV for the coin's assigned timeframe
            ohlcv_data = {}
            if settings.OHLCV_TIMEFRAMES:
                try:
                    ohlcv_data = await asyncio.to_thread(
                        get_multi_timeframe_ohlcv, self.exchange, symbol, [assigned_tf], limit=50
                    )
                except Exception as e:
                    logger.warning(f"OHLCV fetch failed for {symbol}: {e}")
            open_positions = [
                pos for pos in self.positions.values() if pos.get("symbol") == symbol
            ]

            # Compute per-coin budget for this coin
            per_coin_budget = base_balance / self.effective_max_coins if self.effective_max_coins > 0 else 0.0

            perf = self._compute_performance_metrics()

            # --- Compute additional metrics for the LLM ---
            # Get indicator config for this coin (LLM-defined)
            ind_cfg = self.positions.get(symbol, {}).get('indicator_config') if symbol in self.positions else None
            rsi_period = ind_cfg.get('rsi_period', 14) if ind_cfg else 14
            macd_fast = ind_cfg.get('macd_fast', 12) if ind_cfg else 12
            macd_slow = ind_cfg.get('macd_slow', 26) if ind_cfg else 26
            macd_signal_period = ind_cfg.get('macd_signal', 9) if ind_cfg else 9
            bb_period = ind_cfg.get('bb_period', 20) if ind_cfg else 20
            bb_std = ind_cfg.get('bb_std', 2.0) if ind_cfg else 2.0
            ema_fast = ind_cfg.get('ema_fast', 9) if ind_cfg else 9
            ema_slow = ind_cfg.get('ema_slow', 21) if ind_cfg else 21
            stoch_k_period = ind_cfg.get('stoch_k_period', 14) if ind_cfg else 14
            stoch_d_period = ind_cfg.get('stoch_d_period', 3) if ind_cfg else 3
            adx_period = ind_cfg.get('adx_period', 14) if ind_cfg else 14
            mfi_period = ind_cfg.get('mfi_period', 14) if ind_cfg else 14
            cci_period = ind_cfg.get('cci_period', 20) if ind_cfg else 20
            willr_period = ind_cfg.get('willr_period', 14) if ind_cfg else 14

            atr = None
            rsi = None
            macd = None
            macd_signal = None
            macd_hist = None
            bb_upper = None
            bb_middle = None
            bb_lower = None
            ema_9 = None
            ema_21 = None
            stochastic_k = None
            stochastic_d = None
            adx = None
            plus_di = None
            minus_di = None
            obv = None
            mfi = None
            cci = None
            williams_r = None
            if ohlcv_data and assigned_tf in ohlcv_data:
                candles = ohlcv_data[assigned_tf]
                if candles:
                    atr = compute_atr(candles)
                    if len(candles) >= 26:
                        closes = [c[4] for c in candles]
                        highs = [c[2] for c in candles]
                        lows = [c[3] for c in candles]
                        volumes = [c[5] for c in candles]
                        rsi = compute_rsi(closes, period=rsi_period)
                        macd, macd_signal, macd_hist = compute_macd(closes, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)
                        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes, period=bb_period, std_dev=bb_std)
                        ema_9_list = compute_ema(closes, ema_fast)
                        ema_21_list = compute_ema(closes, ema_slow)
                        if ema_9_list:
                            ema_9 = ema_9_list[-1]
                        if ema_21_list:
                            ema_21 = ema_21_list[-1]
                        stochastic_k, stochastic_d = compute_stochastic(highs, lows, closes, period=stoch_k_period, smooth_k=stoch_d_period)
                        adx, plus_di, minus_di = compute_adx(highs, lows, closes, period=adx_period)
                        obv = compute_obv(closes, volumes)
                        mfi = compute_mfi(highs, lows, closes, volumes, period=mfi_period)
                        cci = compute_cci(highs, lows, closes, period=cci_period)
                        williams_r = compute_williams_r(highs, lows, closes, period=willr_period)

            # Extract raw candles for the assigned timeframe
            raw_candles = None
            if ohlcv_data and assigned_tf in ohlcv_data:
                raw_candles = ohlcv_data[assigned_tf]

            # Order book imbalance (bid volume / ask volume, top 5 levels)
            bids_vol = sum(bid[1] for bid in order_book.get('bids', [])[:5])
            asks_vol = sum(ask[1] for ask in order_book.get('asks', [])[:5])
            order_book_imbalance = bids_vol / asks_vol if asks_vol > 0 else None

            # --- Enhanced order book metrics ---
            spread_pct = None
            bid_wall_volume = None
            ask_wall_volume = None
            order_book_pressure = None
            depth_imbalances = None
            order_book_slope = None
            mid_price_bias = None

            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid = (best_bid + best_ask) / 2
                if mid > 0:
                    spread_pct = ((best_ask - best_bid) / mid) * 100

                # Wall volumes: cumulative volume within 1% of best price
                bid_threshold = best_bid * 0.99
                ask_threshold = best_ask * 1.01
                bid_wall_volume = sum(bid[1] for bid in bids if bid[0] >= bid_threshold)
                ask_wall_volume = sum(ask[1] for ask in asks if ask[0] <= ask_threshold)

                # Order book pressure: bid_wall / (bid_wall + ask_wall)
                total_wall = bid_wall_volume + ask_wall_volume
                if total_wall > 0:
                    order_book_pressure = bid_wall_volume / total_wall

                # --- Deeper order‑book metrics ---
                depth_imbalances = {}
                order_book_slope = None
                mid_price_bias = None

                # Depth imbalance at 0.5%, 1%, 2% from mid
                for pct in [0.005, 0.01, 0.02]:
                    bid_cutoff = mid * (1 - pct)
                    ask_cutoff = mid * (1 + pct)
                    bid_vol = sum(b[1] for b in bids if b[0] >= bid_cutoff)
                    ask_vol = sum(a[1] for a in asks if a[0] <= ask_cutoff)
                    total = bid_vol + ask_vol
                    imbalance = bid_vol / total if total > 0 else 0.5
                    depth_imbalances[f"{pct*100:.1f}%"] = round(imbalance, 3)

                # Order‑book slope: change in cumulative volume between 0.5% and 1% levels
                vol_05 = sum(b[1] for b in bids if b[0] >= mid * 0.995) + sum(a[1] for a in asks if a[0] <= mid * 1.005)
                vol_1 = sum(b[1] for b in bids if b[0] >= mid * 0.99) + sum(a[1] for a in asks if a[0] <= mid * 1.01)
                order_book_slope = (vol_1 - vol_05) / 0.005  # volume per 0.5% price move

                # Mid‑price bias: -1 (near bid) to +1 (near ask)
                if best_ask != best_bid:
                    mid_price_bias = (mid - best_bid) / (best_ask - best_bid) - 0.5  # range -0.5 to +0.5
                    mid_price_bias *= 2  # scale to -1..1

            # Fee rate for this symbol
            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)

            # Unrealized P&L for current position (if any)
            unrealized_pnl = None
            position_info = None
            if symbol in self.positions:
                pos = self.positions[symbol]
                position_info = pos
                current_price = ticker['last']
                entry_price = pos['price']
                amount = pos['amount']
                unrealized_pnl = (current_price - entry_price) * amount

            # Recent trade outcomes (last 5 closed trades)
            recent_trades = [
                t for t in self.trade_history if t.get("side") == "sell"
            ][-5:]
            recent_trades_summary = [
                {
                    "symbol": t["symbol"],
                    "realized_pnl": t.get("realized_pnl", 0.0),
                    "strategy": t.get("strategy_type", "unknown"),
                }
                for t in recent_trades
            ]

            prompt = build_strategy_prompt(
                symbol=symbol,
                ticker=ticker,
                order_book=order_book,
                balance=balance,
                open_positions=open_positions,
                per_coin_budget=per_coin_budget,
                max_coins=self.effective_max_coins,
                performance=perf,
                ohlcv_data=ohlcv_data,
                assigned_timeframe=assigned_tf,
                atr=atr,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_hist=macd_hist,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                ema_9=ema_9,
                ema_21=ema_21,
                stochastic_k=stochastic_k,
                stochastic_d=stochastic_d,
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                obv=obv,
                mfi=mfi,
                cci=cci,
                williams_r=williams_r,
                order_book_imbalance=order_book_imbalance,
                unrealized_pnl=unrealized_pnl,
                position_info=position_info,
                spread_pct=spread_pct,
                bid_wall_volume=bid_wall_volume,
                ask_wall_volume=ask_wall_volume,
                order_book_pressure=order_book_pressure,
                depth_imbalances=depth_imbalances,
                order_book_slope=order_book_slope,
                mid_price_bias=mid_price_bias,
                fee_rate=fee_rate,
                drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
                raw_candles=raw_candles,
                recent_trades=recent_trades_summary,
            )
            response = await asyncio.to_thread(get_cached_llm_response, prompt, SYSTEM_PROMPT, 60)
            strategy = create_strategy_from_llm(response)
            signal = strategy.generate_signal({})
            # Extract per-trade confidence threshold if present
            entry_conf_threshold = signal.strategy_params.get("entry_confidence_threshold") if signal.strategy_params else None
            validated = validate_signal(signal, fee_rate=fee_rate, entry_confidence_threshold=entry_conf_threshold)

            # Log raw response if validation turned a non-HOLD into HOLD
            if signal.action != "HOLD" and validated.action == "HOLD":
                logger.warning(
                    f"LLM signal for {symbol} rejected by validator. "
                    f"Original action={signal.action}, confidence={signal.confidence}, "
                    f"reasoning={validated.reasoning}. Raw LLM response: {response}"
                )

            # Log and notify the decision
            logger.info(f"Decision for {symbol}: {validated.action} (confidence: {validated.confidence:.2f})")
            if self.notifier:
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(validated.action, "❓")
                # Build a short indicator summary
                ind_parts = []
                if rsi is not None:
                    ind_parts.append(f"RSI={rsi:.1f}")
                if macd is not None and macd_signal is not None:
                    ind_parts.append(f"MACD={macd:.4f}/{macd_signal:.4f}")
                if bb_upper is not None:
                    ind_parts.append(f"BB={bb_lower:.2f}-{bb_upper:.2f}")
                if ema_9 is not None and ema_21 is not None:
                    ind_parts.append(f"EMA9/21={ema_9:.2f}/{ema_21:.2f}")
                if stochastic_k is not None:
                    ind_parts.append(f"StochK={stochastic_k:.1f}")
                if adx is not None:
                    ind_parts.append(f"ADX={adx:.1f}")
                if atr is not None:
                    ind_parts.append(f"ATR={atr:.4f}")
                if obv is not None:
                    ind_parts.append(f"OBV={obv:.2f}")
                if mfi is not None:
                    ind_parts.append(f"MFI={mfi:.2f}")
                if cci is not None:
                    ind_parts.append(f"CCI={cci:.2f}")
                if williams_r is not None:
                    ind_parts.append(f"WR={williams_r:.2f}")
                indicator_str = " | ".join(ind_parts) if ind_parts else ""
                sentiment_str = self._get_sentiment_str(symbol)
                msg = f"{emoji} {symbol}: {validated.action} (confidence: {validated.confidence:.2f}) – {validated.reasoning}"
                if indicator_str:
                    msg += f"\n📊 {indicator_str}"
                if sentiment_str:
                    msg += f"\n{sentiment_str}"
                await self.notifier.send_notification(msg)

            # Prevent SELL without an open position (no shorting)
            if validated.action == "SELL" and symbol not in self.positions:
                logger.info(f"Skipping SELL for {symbol}: no open position.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping SELL for {symbol}: no open position."
                    )
                return

            if validated.action != "HOLD":
                await self._execute_signal(symbol, validated, timeframe=assigned_tf, atr=atr)
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
        total_fees = 0.0
        for t in self.trade_history:
            fee = t.get('fee', {})
            fee_cost = float(fee.get('cost', 0) or 0)
            fee_currency = fee.get('currency', '')
            if fee_cost == 0.0:
                continue
            if fee_currency == self.base_currency:
                total_fees += fee_cost
            else:
                # fee is in the base coin (e.g., BTC) → convert using trade price
                price = t.get('price', 0.0)
                total_fees += fee_cost * price
        total_value = current_balance + open_value
        pnl = total_value - self.initial_balance
        pnl_percent = (pnl / self.initial_balance * 100) if self.initial_balance else 0.0

        # Win/Loss stats
        wins = 0
        losses = 0
        for t in self.trade_history:
            if t.get('side') == 'sell' and 'realized_pnl' in t:
                pnl_val = t['realized_pnl']
                if pnl_val > 0:
                    wins += 1
                elif pnl_val < 0:
                    losses += 1
        total_closed = wins + losses
        win_rate = (wins / total_closed) if total_closed > 0 else 0.0

        return {
            "initial_balance": self.initial_balance,
            "current_balance": current_balance,
            "open_value": open_value,
            "total_pnl": pnl,
            "pnl_percent": pnl_percent,
            "total_fees": total_fees,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
        }

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return current open positions as trade-like dicts with unrealized P&L."""
        open_trades = []
        for symbol, pos in self.positions.items():
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
            except Exception:
                current_price = pos['price']  # fallback to entry price

            entry_price = pos['price']
            amount = pos['amount']
            cost_basis = pos.get('cost_basis', amount * entry_price)
            unrealized_pnl = (current_price - entry_price) * amount
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            # Try to get fee from the most recent buy trade for this symbol
            fee = {}
            for t in reversed(self.trade_history):
                if t['symbol'] == symbol and t['side'] == 'buy':
                    fee = t.get('fee', {})
                    break

            open_trades.append({
                'symbol': symbol,
                'side': 'buy',
                'amount': amount,
                'price': entry_price,
                'timestamp': pos.get('timestamp', 0),
                'fee': fee,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'cost_basis': cost_basis,
            })
        return open_trades

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return performance summary grouped by coin and timeframe from trade_history table."""
        return get_performance()

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Return current risk/exposure metrics."""
        balance = self.trader.fetch_balance()
        total_balance = balance.get(self.base_currency, 0.0)

        pnl = total_balance - self.initial_balance
        pnl_pct = (pnl / self.initial_balance * 100) if self.initial_balance else 0.0

        # Open positions exposure and stop‑loss risk
        exposure = 0.0
        position_exposures = []
        total_stop_risk = 0.0
        for pos in self.positions.values():
            try:
                ticker = self.exchange.fetch_ticker(pos['symbol'])
                price = ticker['last'] if ticker and ticker.get('last') else 0.0
                pos_value = pos['amount'] * price
                exposure += pos_value
                position_exposures.append(pos_value)
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    total_stop_risk += loss_if_stop
            except Exception:
                pass

        total_portfolio_value = total_balance + exposure
        largest_position_exposure_pct = (
            (max(position_exposures) / total_portfolio_value * 100)
            if position_exposures and total_portfolio_value > 0
            else 0.0
        )

        # Drawdown from performance metrics
        perf = self._compute_performance_metrics()
        max_drawdown_pct = perf.get('equity_curve', {}).get('drawdown_pct', 0.0)

        # Trade statistics
        wins = []
        losses = []
        for t in self.trade_history:
            if t.get('side') == 'sell' and 'realized_pnl' in t:
                pnl_val = t['realized_pnl']
                if pnl_val > 0:
                    wins.append(pnl_val)
                elif pnl_val < 0:
                    losses.append(abs(pnl_val))
        total_trades = len(wins) + len(losses)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0

        return {
            'current_balance': total_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': pnl,
            'total_pnl_pct': pnl_pct,
            'open_positions_count': len(self.positions),
            'total_exposure': exposure,
            'base_currency': self.base_currency,
            'max_drawdown_pct': max_drawdown_pct,
            'largest_position_exposure_pct': largest_position_exposure_pct,
            'total_stop_loss_risk': total_stop_risk,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades,
        }

    async def _check_risk_management(self):
        """Check open positions and close if stop-loss, take-profit, or trailing stop is hit."""
        for symbol, pos in list(self.positions.items()):
            try:
                # Skip positions that don't have LLM-defined risk parameters yet
                if pos.get("stop_loss") is None or pos.get("take_profit") is None:
                    continue

                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                current_price = ticker['last']

                # Trailing stop update (only if enabled)
                if pos.get("trailing_stop") and pos.get("trailing_stop_distance_pct"):
                    # Check activation threshold
                    activation_pct = pos.get("trailing_stop_activation_pct")
                    if activation_pct is not None:
                        entry_price = pos["price"]
                        profit_pct = (current_price - entry_price) / entry_price
                        if profit_pct < activation_pct:
                            # Not yet activated; skip trailing stop update
                            pass
                        else:
                            distance = pos["trailing_stop_distance_pct"]
                            new_stop = current_price * (1 - distance)
                            if new_stop > pos["stop_loss"]:
                                pos["stop_loss"] = new_stop
                                logger.debug(f"Trailing stop updated for {symbol}: new stop {new_stop:.4f}")
                    else:
                        # No activation threshold – update immediately
                        distance = pos["trailing_stop_distance_pct"]
                        new_stop = current_price * (1 - distance)
                        if new_stop > pos["stop_loss"]:
                            pos["stop_loss"] = new_stop
                            logger.debug(f"Trailing stop updated for {symbol}: new stop {new_stop:.4f}")

                # --- News sentiment risk adjustment for open positions ---
                if settings.NEWS_ENABLED:
                    try:
                        base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                        agg_sent = get_aggregate_sentiment_from_db(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                        if agg_sent:
                            compound = agg_sent["avg_compound"]
                            # Force close if sentiment is extremely negative
                            if settings.NEWS_SENTIMENT_EXIT_ON_VERY_NEGATIVE and compound < settings.NEWS_SENTIMENT_EXIT_THRESHOLD:
                                logger.info(f"News sentiment extremely negative for {symbol} ({compound}). Forcing close.")
                                if self.notifier:
                                    await self.notifier.send_notification(
                                        f"🚨 Force closing {symbol}: news sentiment extremely negative ({compound})"
                                    )
                                await self._execute_signal(
                                    symbol,
                                    Signal(action="SELL", confidence=1.0, reasoning="News sentiment extremely negative"),
                                    exit_reason="news_sentiment_exit"
                                )
                                continue  # skip further checks for this symbol

                            # Tighten stop-loss if sentiment turns negative
                            if settings.NEWS_SENTIMENT_TIGHTEN_STOP and compound < settings.NEWS_SENTIMENT_TIGHTEN_STOP_THRESHOLD:
                                original_stop = pos["stop_loss"]
                                # Compute distance from current price to stop
                                distance = current_price - original_stop
                                if distance > 0:
                                    new_distance = distance * settings.NEWS_SENTIMENT_TIGHTEN_STOP_MULTIPLIER
                                    new_stop = current_price - new_distance
                                    # Only tighten, never loosen
                                    if new_stop > original_stop:
                                        pos["stop_loss"] = new_stop
                                        logger.info(
                                            f"Tightened stop for {symbol} due to negative news sentiment "
                                            f"({compound}): {original_stop:.4f} -> {new_stop:.4f}"
                                        )
                    except Exception as e:
                        logger.warning(f"Failed to apply news sentiment risk management for {symbol}: {e}")

                # Time‑based exit (LLM‑defined max hold time)
                max_hold = pos.get("max_hold_time_seconds")
                if max_hold is not None and max_hold > 0:
                    entry_ts = pos.get("timestamp", 0) / 1000.0  # convert ms to seconds
                    if time.time() - entry_ts > max_hold:
                        logger.info(f"Max hold time reached for {symbol} ({max_hold}s). Closing position.")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏰ Max hold time reached for {symbol} – closing position."
                            )
                        await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Max hold time"), exit_reason="max_hold_time")
                        continue   # skip further checks for this symbol

                if current_price <= pos["stop_loss"]:
                    logger.info(f"Stop-loss triggered for {symbol} at {current_price}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⛔ Stop‑loss triggered for {symbol} at {current_price:.4f}"
                        )
                    await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss"), exit_reason="stop_loss")
                elif current_price >= pos["take_profit"]:
                    logger.info(f"Take-profit triggered for {symbol} at {current_price}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"✅ Take‑profit triggered for {symbol} at {current_price:.4f}"
                        )
                    await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit"), exit_reason="take_profit")
            except Exception as e:
                logger.error(f"Risk check failed for {symbol}: {e}")

    async def _execute_signal(self, symbol: str, signal, timeframe: str = None, exit_reason: str = None, atr: Optional[float] = None):
        """Execute a BUY or SELL signal."""
        base, quote = symbol.split("/")
        balance = await asyncio.to_thread(self.trader.fetch_balance)

        if signal.action == "BUY":
            # Extract known parameters from the LLM's strategy_params (if any)
            params = signal.strategy_params or {}

            # Use LLM-provided risk parameters directly (no hardcoded minimums)
            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
            tp_pct = params["take_profit_pct"]
            min_tp = 2 * fee_rate + 0.001   # 0.1% margin
            if tp_pct < min_tp:
                logger.info(f"Take-profit {tp_pct:.4%} too low to cover fees (min {min_tp:.4%}). Skipping BUY for {symbol}.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping BUY for {symbol}: take-profit too low ({tp_pct:.4%})"
                    )
                return
            trailing_stop = params["trailing_stop"]
            trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")

            # Determine stop-loss percentage based on method
            stop_method = params.get("stop_loss_method", "fixed")
            if stop_method == "atr_multiple" and atr is not None and atr > 0:
                atr_mult = params["stop_loss_atr_multiple"]
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                current_price = ticker['last']
                sl_pct = (atr_mult * atr) / current_price
                logger.info(f"ATR-based stop: ATR={atr}, multiplier={atr_mult}, stop_loss_pct={sl_pct:.4%}")
            else:
                sl_pct = params["stop_loss_pct"]

            # Use per-coin budget as the buy amount, capped at available balance
            quote_balance = balance.get(quote, 0.0)
            position_fraction = params["position_size_fraction"]
            risk_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(signal.risk_level, 1.0)
            position_fraction *= risk_multiplier
            position_fraction = max(0.1, min(1.0, position_fraction))

            # --- News sentiment risk adjustment ---
            if settings.NEWS_ENABLED and settings.NEWS_SENTIMENT_RISK_ADJUSTMENT:
                try:
                    base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                    agg_sent = get_aggregate_sentiment_from_db(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                    if agg_sent:
                        compound = agg_sent["avg_compound"]
                        # Skip BUY if sentiment is extremely negative
                        if settings.NEWS_SENTIMENT_SKIP_BUY_ON_VERY_NEGATIVE and compound < settings.NEWS_SENTIMENT_NEGATIVE_THRESHOLD:
                            logger.info(f"Skipping BUY for {symbol}: news sentiment very negative ({compound})")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⚠️ Skipping BUY for {symbol}: very negative news sentiment ({compound})"
                                )
                            return
                        # Adjust position size multiplier based on sentiment
                        if compound < settings.NEWS_SENTIMENT_NEGATIVE_THRESHOLD:
                            multiplier = settings.NEWS_SENTIMENT_POSITION_SIZE_MULTIPLIER_NEGATIVE
                        elif compound > settings.NEWS_SENTIMENT_POSITIVE_THRESHOLD:
                            multiplier = settings.NEWS_SENTIMENT_POSITION_SIZE_MULTIPLIER_POSITIVE
                        else:
                            multiplier = 1.0
                        position_fraction *= multiplier
                        position_fraction = max(0.1, min(1.0, position_fraction))
                        logger.info(
                            f"News sentiment adjustment for {symbol}: compound={compound}, "
                            f"multiplier={multiplier}, new position_fraction={position_fraction}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to apply news sentiment adjustment for {symbol}: {e}")

            per_coin_budget = (quote_balance / self.effective_max_coins) * position_fraction if self.effective_max_coins > 0 else 0.0
            amount = min(per_coin_budget, quote_balance)

            # Apply max risk per trade if provided
            max_risk_pct = params.get("max_risk_per_trade_pct")
            if max_risk_pct is not None and sl_pct > 0:
                # Compute total portfolio value (quote balance + open positions value)
                total_value = quote_balance
                for sym, pos in self.positions.items():
                    try:
                        t = await asyncio.to_thread(self.exchange.fetch_ticker, sym)
                        total_value += pos['amount'] * t['last']
                    except Exception:
                        pass
                max_risk_amount = total_value * max_risk_pct
                # The loss if stop is hit: amount * sl_pct
                max_allowed_amount = max_risk_amount / sl_pct
                amount = min(amount, max_allowed_amount)
                logger.info(f"Max risk per trade: {max_risk_pct:.2%} of {total_value:.2f} = {max_risk_amount:.2f}, max allowed amount = {max_allowed_amount:.2f}")

            if amount <= 0:
                logger.info(f"Insufficient {quote} to buy {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(f"⚠️ Insufficient {quote} to buy {symbol}")
                return

            # Check minimum order size
            try:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                price = ticker['last']
                base_amount = amount / price
                market = self.exchange.markets.get(symbol, {})
                limits = market.get('limits', {})
                min_amount_limit = limits.get('amount', {}).get('min')
                min_cost_limit = limits.get('cost', {}).get('min')

                if min_amount_limit is not None and base_amount < float(min_amount_limit):
                    logger.info(f"BUY amount {base_amount:.6f} {base} below min amount {min_amount_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(f"⚠️ BUY skipped for {symbol}: amount too small")
                    return
                if min_cost_limit is not None and amount < float(min_cost_limit):
                    logger.info(f"BUY cost {amount:.2f} {quote} below min cost {min_cost_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(f"⚠️ BUY skipped for {symbol}: cost too small")
                    return
            except Exception as e:
                logger.warning(f"Could not verify min order size for {symbol}: {e}")

            try:
                order = await asyncio.to_thread(self.trader.create_market_buy_order, symbol, amount)
                logger.info(f"BUY {symbol}: {order}")
                # Update or create position
                # Extract fee info for cost basis tracking
                fee = order.get('fee', {})
                fee_cost = float(fee.get('cost', 0.0) or 0.0)
                fee_currency = fee.get('currency', '')

                # Fallback: compute fee from fee_rate if missing or zero
                if fee_cost == 0.0:
                    fee_cost = order['cost'] * fee_rate
                    fee_currency = quote  # assume fee is charged in quote currency
                    order['fee'] = {'cost': fee_cost, 'currency': fee_currency}

                cost_basis = order['cost'] + (fee_cost if fee_currency == quote else 0.0)
                net_base = order['amount'] - (fee_cost if fee_currency == base else 0.0)

                # Risk parameters are guaranteed by the validator
                # sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct are set above

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
                    self.positions[symbol]["stop_loss"] = new_price * (1 - sl_pct)
                    self.positions[symbol]["take_profit"] = new_price * (1 + tp_pct)
                    self.positions[symbol]["trailing_stop"] = trailing_stop
                    self.positions[symbol]["trailing_stop_distance_pct"] = trailing_stop_distance_pct
                    self.positions[symbol]["max_hold_time_seconds"] = params.get("max_hold_time_seconds")
                    self.positions[symbol]["trailing_stop_activation_pct"] = params.get("trailing_stop_activation_pct")
                    self.positions[symbol]["timeframe"] = timeframe
                    self.positions[symbol]["indicator_config"] = signal.indicator_config
                else:
                    entry_price = cost_basis / net_base if net_base > 0 else order["price"]
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "side": "buy",
                        "amount": net_base,
                        "price": entry_price,
                        "timestamp": order["timestamp"],
                        "stop_loss": entry_price * (1 - sl_pct),
                        "take_profit": entry_price * (1 + tp_pct),
                        "cost_basis": cost_basis,
                        "net_base": net_base,
                        "trailing_stop": trailing_stop,
                        "trailing_stop_distance_pct": trailing_stop_distance_pct,
                        "max_hold_time_seconds": params.get("max_hold_time_seconds"),
                        "trailing_stop_activation_pct": params.get("trailing_stop_activation_pct"),
                        "timeframe": timeframe,
                        "indicator_config": signal.indicator_config,
                    }
                order["strategy_type"] = signal.strategy_type
                order["timeframe"] = timeframe
                self.trade_history.append(order)
                await asyncio.to_thread(insert_trade, order)
                await self._save_state()
                if self.notifier:
                    sentiment_str = self._get_sentiment_str(symbol)
                    buy_msg = f"🟢 BUY {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    if sentiment_str:
                        buy_msg += f" | {sentiment_str}"
                    await self.notifier.send_notification(buy_msg)
            except Exception as e:
                logger.error(f"Buy order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(f"❌ Buy order failed for {symbol}: {e}")

        elif signal.action == "SELL":
            # Fetch fee rate for this symbol
            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
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

            # Check minimum sell size
            try:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                price = ticker['last']
                market = self.exchange.markets.get(symbol, {})
                limits = market.get('limits', {})
                min_amount_limit = limits.get('amount', {}).get('min')
                min_cost_limit = limits.get('cost', {}).get('min')
                if min_amount_limit is not None and gross_amount < float(min_amount_limit):
                    logger.info(f"SELL amount {gross_amount:.6f} {base} below min amount {min_amount_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(f"⚠️ SELL skipped for {symbol}: amount too small")
                    return
                if min_cost_limit is not None and gross_amount * price < float(min_cost_limit):
                    logger.info(f"SELL cost {gross_amount * price:.2f} {quote} below min cost {min_cost_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(f"⚠️ SELL skipped for {symbol}: cost too small")
                    return
            except Exception as e:
                logger.warning(f"Could not verify min sell size for {symbol}: {e}")

            try:
                order = await asyncio.to_thread(self.trader.create_market_sell_order, symbol, gross_amount)
                logger.info(f"SELL {symbol}: {order}")
                # Compute realized P&L
                fee = order.get('fee', {})
                fee_cost = float(fee.get('cost', 0.0) or 0.0)
                fee_currency = fee.get('currency', '')

                # Fallback: compute fee from fee_rate if missing or zero
                if fee_cost == 0.0:
                    fee_cost = order['cost'] * fee_rate
                    fee_currency = quote  # assume fee is charged in quote currency
                    order['fee'] = {'cost': fee_cost, 'currency': fee_currency}

                net_quote = order['cost'] - (fee_cost if fee_currency == quote else 0.0)
                if pos:
                    cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                    realized_pnl = net_quote - cost_basis
                else:
                    realized_pnl = 0.0
                order["realized_pnl"] = realized_pnl
                order["cost_basis"] = pos.get("cost_basis", 0.0) if pos else 0.0
                tf = timeframe or (pos.get("timeframe") if pos else None)
                order["timeframe"] = tf
                order["strategy_type"] = signal.strategy_type
                order["exit_reason"] = exit_reason
                if pos and "timestamp" in pos:
                    hold_time = (order["timestamp"] - pos["timestamp"]) / 1000.0
                    order["hold_time_seconds"] = hold_time
                else:
                    order["hold_time_seconds"] = None
                # Remove position
                self.positions.pop(symbol, None)
                self.trade_history.append(order)
                await asyncio.to_thread(insert_trade, order)
                await self._save_state()
                if self.notifier:
                    sentiment_str = self._get_sentiment_str(symbol)
                    sell_msg = f"🔴 SELL {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    if sentiment_str:
                        sell_msg += f" | {sentiment_str}"
                    await self.notifier.send_notification(sell_msg)
            except Exception as e:
                logger.error(f"Sell order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(f"❌ Sell order failed for {symbol}: {e}")
