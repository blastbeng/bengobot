import asyncio
import httpx
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
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
    compute_vwap,
    compute_ichimoku,
    compute_parabolic_sar,
    compute_keltner_channels,
    compute_pivot_points,
    compute_donchian_channels,
    _format_news_for_prompt,
)
try:
    from src.news.fetcher import discover_trending_coins
except ImportError:
    discover_trending_coins = None
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm
from src.strategies.validator import validate_signal
from src.utils.redis_client import get_redis_client
from src.database import load_trading_state, save_trading_state, delete_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db, get_news_for_symbol, get_ohlcv, get_latest_ohlcv_timestamp, insert_ohlcv_batch

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
        self.last_loss_time: Dict[str, float] = {}  # symbol -> timestamp of last losing trade
        self.cooldown_durations: Dict[str, float] = {}  # symbol -> cooldown seconds set by LLM
        self._last_strategy_eval: Dict[str, float] = {}   # symbol -> timestamp of last strategy evaluation
        self._strategy_intervals: Dict[str, float] = {}    # symbol -> custom interval in seconds
        self._coin_revaluation_interval = COIN_REVALUATION_INTERVAL
        self._daily_profit_target_pct: Optional[float] = None
        self.notifier = None
        self._load_state()
        # Restore paper simulator state from trade history
        if settings.TRADING_MODE == "paper":
            self._restore_paper_state()
        self._ensure_cost_basis()
        # Ensure trading is not paused on startup
        self.redis.delete("trading:paused")

        # Track quote currency spent in the current cycle to avoid over-allocating
        self._cycle_spent = 0.0
        self._market_breadth: Optional[Dict[str, Any]] = None

    def set_notifier(self, notifier):
        """Attach a notification service (e.g., TelegramBot)."""
        self.notifier = notifier

    def _get_sentiment_str(self, symbol: str) -> str:
        """Get a short news sentiment string for notifications, including an LLM summary."""
        if not settings.NEWS_ENABLED:
            return ""
        try:
            base_coin = symbol.split("/")[0] if "/" in symbol else symbol
            agg_sent = get_aggregate_sentiment_from_db(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if not agg_sent:
                return ""

            compound = agg_sent["avg_compound"]
            sentiment_label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
            total = agg_sent["total_articles"]

            # Try to get an LLM-generated summary of the news
            summary = ""
            try:
                articles = get_news_for_symbol(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                if articles:
                    formatted = _format_news_for_prompt(articles)
                    prompt = (
                        f"Here are recent news headlines and summaries for {base_coin}:\n\n"
                        f"{formatted}\n\n"
                        "Based on these articles, write a single very short sentence (max 15 words) "
                        "that explains the overall sentiment and the main reason for it. "
                        "Do not include any other text."
                    )
                    summary = get_cached_llm_response(prompt, "", ttl=300).strip()
                    # Limit length to avoid overly long notifications
                    if len(summary) > 120:
                        summary = summary[:117] + "..."
            except Exception:
                pass  # fallback to no summary

            if summary:
                return f"📰 {sentiment_label} ({compound:+.2f}, {total} articles) – {summary}"
            else:
                return f"📰 {sentiment_label} ({compound:+.2f}, {total} articles)"
        except Exception:
            pass
        return ""

    async def _get_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """Fetch Crypto Fear & Greed Index from alternative.me, cached in Redis."""
        if not getattr(settings, 'FEAR_GREED_ENABLED', True):
            return None
        cache_key = "fear_greed:index"
        try:
            cached = await asyncio.to_thread(self.redis.get, cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    entry = data.get("data", [{}])[0]
                    result = {
                        "value": int(entry.get("value", 50)),
                        "classification": entry.get("value_classification", "neutral"),
                    }
                    ttl = getattr(settings, 'FEAR_GREED_CACHE_TTL_SECONDS', 3600)
                    await asyncio.to_thread(self.redis.setex, cache_key, ttl, json.dumps(result))
                    return result
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}")
        return None

    async def _fetch_global_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetch global crypto market data (BTC dominance, total market cap) from CoinGecko, cached in Redis."""
        cache_key = "global_market:data"
        try:
            cached = await asyncio.to_thread(self.redis.get, cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/global",
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    gd = data.get("data", {})
                    result = {
                        "btc_dominance": round(gd.get("market_cap_percentage", {}).get("btc", 0.0), 2),
                        "eth_dominance": round(gd.get("market_cap_percentage", {}).get("eth", 0.0), 2),
                        "total_market_cap_usd": gd.get("total_market_cap", {}).get("usd", 0),
                        "total_volume_usd": gd.get("total_volume", {}).get("usd", 0),
                        "market_cap_change_24h_usd": round(gd.get("market_cap_change_percentage_24h_usd", 0.0), 2),
                    }
                    ttl = getattr(settings, 'GLOBAL_MARKET_DATA_CACHE_TTL_SECONDS', 1800)
                    await asyncio.to_thread(self.redis.setex, cache_key, ttl, json.dumps(result))
                    return result
        except Exception as e:
            logger.warning(f"Failed to fetch global market data: {e}")
        return None

    async def _fetch_altcoin_season_index(self) -> Optional[Dict[str, Any]]:
        """Fetch Altcoin Season Index from blockchaincenter.net, cached in Redis."""
        if not getattr(settings, 'ALTCOIN_SEASON_ENABLED', True):
            return None
        cache_key = "altcoin_season:index"
        try:
            cached = await asyncio.to_thread(self.redis.get, cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.blockchaincenter.net/altcoin-season-index",
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result = {
                        "value": int(data.get("value", 50)),
                        "description": data.get("description", ""),
                    }
                    ttl = getattr(settings, 'ALTCOIN_SEASON_CACHE_TTL_SECONDS', 3600)
                    await asyncio.to_thread(self.redis.setex, cache_key, ttl, json.dumps(result))
                    return result
        except Exception as e:
            logger.warning(f"Failed to fetch Altcoin Season Index: {e}")
        return None

    def _get_news_summary(self, symbol: str) -> str:
        """Return a very short summary of the latest news article for the symbol."""
        if not settings.NEWS_ENABLED:
            return ""
        try:
            base_coin = symbol.split("/")[0] if "/" in symbol else symbol
            articles = get_news_for_symbol(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if articles:
                # Use the most recent article's title, truncated
                title = articles[0].get("title", "")
                if title:
                    return title[:80] + ("..." if len(title) > 80 else "")
        except Exception:
            pass
        return ""

    async def _compute_volume_trend(self, symbol: str, current_volume: float) -> Optional[float]:
        """Compute volume trend as ratio of current 24h volume to EMA of past volumes.

        Returns the ratio (e.g., 2.0 means current volume is 2× the average).
        Returns None if volume data is unavailable.
        """
        if current_volume <= 0:
            return None

        redis_key = f"volume_trend:ema:{symbol}"
        alpha = 0.3  # EMA smoothing factor

        try:
            stored = await asyncio.to_thread(self.redis.get, redis_key)
            if stored is not None:
                old_avg = float(stored)
                new_avg = alpha * current_volume + (1 - alpha) * old_avg
                ratio = current_volume / old_avg if old_avg > 0 else 1.0
                # Store the updated average with 7-day TTL
                await asyncio.to_thread(self.redis.setex, redis_key, 7 * 24 * 3600, str(new_avg))
                return round(ratio, 3)
            else:
                # First observation: initialize with current volume, ratio = 1.0
                await asyncio.to_thread(self.redis.setex, redis_key, 7 * 24 * 3600, str(current_volume))
                return 1.0
        except Exception as e:
            logger.debug(f"Volume trend computation failed for {symbol}: {e}")
            return None

    async def _fetch_and_store_news_for_symbol(self, symbol: str):
        """Fetch news for a single symbol and store it in the database."""
        if not settings.NEWS_ENABLED:
            return
        try:
            from src.news.fetcher import fetch_news_for_symbol
            base_coin = symbol.split("/")[0] if "/" in symbol else symbol
            articles = await asyncio.to_thread(fetch_news_for_symbol, symbol)
            if articles:
                await asyncio.to_thread(store_news_articles, base_coin, articles)
        except Exception as e:
            logger.debug(f"News fetch/store failed for {symbol}: {e}")

    async def _refresh_current_coins_news_fast(self):
        """Fast news refresh loop – only for the coins currently tracked by the engine."""
        if not settings.NEWS_ENABLED:
            return
        # Fetch immediately on startup, then periodically
        while True:
            try:
                symbols = [entry["symbol"] for entry in self.current_coins]
                if symbols:
                    logger.debug(f"Fast news refresh for {len(symbols)} current coins")
                    await asyncio.gather(
                        *[self._fetch_and_store_news_for_symbol(sym) for sym in symbols]
                    )
            except Exception as e:
                logger.error(f"Fast news refresh error: {e}")
            await asyncio.sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)

    async def _refresh_news_cache(self):
        """Periodically fetch news for tracked coins and top-volume coins to keep cache warm."""
        if not settings.NEWS_ENABLED:
            return
        try:
            from src.news.fetcher import fetch_news_for_symbol
        except ImportError:
            logger.warning("News module not available; skipping background news refresh.")
            return

        while True:
            try:
                cycle_start = time.time()
                # Slow refresh: all available pairs EXCEPT the coins already handled by the fast loop
                current_symbols = {entry["symbol"] for entry in self.current_coins}
                symbols_to_refresh = set()
                try:
                    available_pairs = await asyncio.to_thread(
                        get_available_pairs, self.exchange, self.base_currency
                    )
                    symbols_to_refresh = set(available_pairs) - current_symbols
                except Exception as e:
                    logger.warning(f"Could not get available pairs for news refresh: {e}")

                for sym in symbols_to_refresh:
                    try:
                        articles = await asyncio.to_thread(fetch_news_for_symbol, sym)
                        if articles:
                            base_coin = sym.split("/")[0] if "/" in sym else sym
                            await asyncio.to_thread(store_news_articles, base_coin, articles)
                    except Exception as e:
                        logger.debug(f"News refresh failed for {sym}: {e}")
                    await asyncio.sleep(0.2)

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

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        """Convert a timeframe string (e.g., '1m', '5m', '1h') to milliseconds."""
        units = {
            'm': 60_000,
            'h': 3_600_000,
            'd': 86_400_000,
            'w': 604_800_000,
            'M': 2_592_000_000,  # approximate (30 days)
        }
        match = re.match(r'^(\d+)([mhdwM])$', timeframe)
        if not match:
            return 3_600_000  # default to 1h
        amount = int(match.group(1))
        unit = match.group(2)
        return amount * units.get(unit, 3_600_000)

    async def _backfill_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int):
        """Fetch and store all missing OHLCV candles between start_ms and end_ms."""
        logger.info(f"Backfill started for {symbol} {timeframe}: {start_ms} → {end_ms}")
        latest_ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, symbol, timeframe)
        if latest_ts is None:
            since = start_ms
        else:
            since = max(start_ms, latest_ts + 1)

        total_inserted = 0
        while since < end_ms:
            try:
                candles = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv, symbol, timeframe, since=since, limit=500
                )
            except Exception as e:
                logger.warning(f"fetch_ohlcv failed for {symbol} {timeframe} at {since}: {e}")
                break

            if not candles:
                break

            await asyncio.to_thread(insert_ohlcv_batch, symbol, timeframe, candles)
            batch_count = len(candles)
            total_inserted += batch_count
            logger.debug(f"Backfill batch: {symbol} {timeframe} fetched {batch_count} candles from {since}")

            last_ts = candles[-1][0]
            if last_ts <= since:
                # Avoid infinite loop if exchange returns same candle
                break
            since = last_ts + 1
            # Small delay to avoid rate limits
            await asyncio.sleep(0.2)

        logger.info(f"Backfill complete for {symbol} {timeframe}: {total_inserted} candles inserted")

    async def _fill_gaps(self, symbol: str, timeframe: str):
        """Detect and fill gaps in stored OHLCV data for a symbol/timeframe."""
        interval_ms = self._timeframe_to_ms(timeframe)
        if interval_ms <= 0:
            return

        # Get all stored timestamps
        candles = await asyncio.to_thread(get_ohlcv, symbol, timeframe, limit=50000)
        if len(candles) < 2:
            logger.debug(f"Not enough data to check gaps for {symbol} {timeframe}")
            return

        timestamps = sorted(c["timestamp"] for c in candles)

        # Find and fill gaps larger than 1.5x the expected interval
        gaps_found = 0
        gaps_filled = 0
        max_gaps_per_cycle = 5  # Limit gap fills per cycle to avoid rate limits
        for i in range(len(timestamps) - 1):
            if gaps_filled >= max_gaps_per_cycle:
                break
            gap = timestamps[i + 1] - timestamps[i]
            if gap > interval_ms * 1.5:
                gaps_found += 1
                gap_start = timestamps[i] + interval_ms
                gap_end = timestamps[i + 1] - interval_ms
                if gap_end > gap_start:
                    logger.info(f"Gap detected for {symbol} {timeframe}: {gap_start} → {gap_end} (size {gap}ms)")
                    await self._backfill_ohlcv(symbol, timeframe, gap_start, gap_end)
                    gaps_filled += 1

        if gaps_found == 0:
            logger.debug(f"No gaps found for {symbol} {timeframe}")
        else:
            logger.info(f"Gap check for {symbol} {timeframe}: {gaps_found} gaps found, {gaps_filled} filled")

    async def _backfill_new_coin(self, symbol: str):
        """Immediately backfill 30 days of OHLCV data for a newly selected coin."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 30 * 24 * 60 * 60 * 1000
        logger.info(f"Starting immediate backfill for newly selected coin {symbol}")
        for tf in settings.OHLCV_TIMEFRAMES:
            try:
                await self._backfill_ohlcv(symbol, tf, start_ms, now_ms)
                await self._fill_gaps(symbol, tf)
            except Exception as e:
                logger.error(f"Initial backfill failed for {symbol} {tf}: {e}")
        logger.info(f"Immediate backfill complete for {symbol}")

    async def _download_market_data_loop(self):
        """Periodically download and store OHLCV data for tracked coins, with gap detection."""
        # Initial delay to let the engine settle
        await asyncio.sleep(30)
        while True:
            try:
                if not self.current_coins:
                    logger.debug("No coins tracked; skipping market data download.")
                else:
                    logger.info("Starting market data download cycle...")
                    now_ms = int(time.time() * 1000)
                    start_ms = now_ms - 30 * 24 * 60 * 60 * 1000  # 30 days ago
                    for coin_entry in self.current_coins:
                        symbol = coin_entry["symbol"]
                        logger.debug(f"Downloading market data for {symbol}")
                        for tf in settings.OHLCV_TIMEFRAMES:
                            try:
                                await self._backfill_ohlcv(symbol, tf, start_ms, now_ms)
                                await self._fill_gaps(symbol, tf)
                            except Exception as e:
                                logger.warning(f"Market data download failed for {symbol} {tf}: {e}")
                        # Small delay between coins to avoid rate limits
                        await asyncio.sleep(0.5)
                    logger.info("Market data download cycle complete.")
            except Exception as e:
                logger.error(f"Market data download loop error: {e}", exc_info=True)

            await asyncio.sleep(settings.MARKET_DATA_REFRESH_SECONDS)

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
                if "trailing_take_profit" in old:
                    pos["trailing_take_profit"] = old["trailing_take_profit"]
                if "trailing_take_profit_distance_pct" in old:
                    pos["trailing_take_profit_distance_pct"] = old["trailing_take_profit_distance_pct"]
                if "breakeven_activation_pct" in old:
                    pos["breakeven_activation_pct"] = old["breakeven_activation_pct"]
                if "lock_profit_activation_pct" in old:
                    pos["lock_profit_activation_pct"] = old["lock_profit_activation_pct"]
                if "lock_profit_level_pct" in old:
                    pos["lock_profit_level_pct"] = old["lock_profit_level_pct"]
                if "partial_take_profit_pct" in old:
                    pos["partial_take_profit_pct"] = old["partial_take_profit_pct"]
                if "partial_take_profit_fraction" in old:
                    pos["partial_take_profit_fraction"] = old["partial_take_profit_fraction"]
                if "partial_tp_triggered" in old:
                    pos["partial_tp_triggered"] = old["partial_tp_triggered"]
                if "partial_take_profit_levels" in old:
                    pos["partial_take_profit_levels"] = old["partial_take_profit_levels"]
                if "partial_tp_levels_triggered" in old:
                    pos["partial_tp_levels_triggered"] = old["partial_tp_levels_triggered"]
                if "original_amount" in old:
                    pos["original_amount"] = old["original_amount"]
                if "cooldown_after_loss_seconds" in old:
                    pos["cooldown_after_loss_seconds"] = old["cooldown_after_loss_seconds"]
                if "news_sentiment_exit_threshold" in old:
                    pos["news_sentiment_exit_threshold"] = old["news_sentiment_exit_threshold"]
                if "max_unrealized_loss_pct" in old:
                    pos["max_unrealized_loss_pct"] = old["max_unrealized_loss_pct"]
                if "timeframe" in old:
                    pos["timeframe"] = old["timeframe"]
        # Re-apply force_close for positions still missing risk parameters
        for sym, pos in positions.items():
            if "stop_loss" not in pos or "take_profit" not in pos:
                pos["_force_close"] = True
        self.positions = positions
        logger.info("Restored paper trading state from %d historical trades", len(self.trade_history))

    def _ensure_cost_basis(self):
        """If positions lack cost_basis, compute it from amount and price (backward compat)."""
        for sym, pos in self.positions.items():
            if 'cost_basis' not in pos or 'net_base' not in pos:
                # Assume no fees for old positions; cost_basis = amount * price
                pos['cost_basis'] = pos['amount'] * pos['price']
                pos['net_base'] = pos['amount']

    def _daily_realized_pnl(self) -> float:
        """Return the sum of realized P&L for trades closed today (UTC)."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        total = 0.0
        for trade in self.trade_history:
            if trade.get("side") != "sell":
                continue
            ts = trade.get("timestamp", 0)
            if ts:
                trade_date = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).date()
                if trade_date == today:
                    total += trade.get("realized_pnl", 0.0)
        return total

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

        # Compute drawdown based on total equity (initial balance + cumulative realized P&L)
        equity_series = []
        running_equity = self.initial_balance
        for trade in self.trade_history:
            if trade.get("side") == "sell":
                running_equity += trade.get("realized_pnl", 0.0)
            equity_series.append(running_equity)
        peak = max(equity_series) if equity_series else self.initial_balance
        current_equity = equity_series[-1] if equity_series else self.initial_balance
        drawdown_pct = ((peak - current_equity) / peak * 100) if peak > 0 else 0.0

        daily_pnl = self._daily_realized_pnl()
        return {
            "coin_performance": coin_perf,
            "strategy_performance": strategy_perf,
            "equity_curve": {
                "total_pnl": round(sum(t.get("realized_pnl", 0.0) for t in self.trade_history if t.get("side") == "sell"), 4),
                "recent_10_trades_pnl": round(total_recent_pnl, 4),
                "trend": trend,
                "drawdown_pct": round(drawdown_pct, 2),
                "daily_pnl": round(daily_pnl, 4),
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
                        f"🔻 Closing {symbol} – missing LLM risk parameters.",
                        summary={
                            "symbol": symbol,
                            "action": "SELL",
                            "reason": "Missing LLM risk parameters",
                            "exit_reason": "force_close",
                        }
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

        logger.info(
            "Loaded trading state: %d coins, %d positions, %d trades",
            len(self.current_coins),
            len(self.positions),
            len(self.trade_history),
        )

    async def _save_state(self):
        """Persist current coins, positions, and trade history to SQLite."""
        await asyncio.to_thread(save_trading_state, "current_coins", self.current_coins)
        await asyncio.to_thread(save_trading_state, "positions", self.positions)
        # Keep only the last 1000 trades to avoid unbounded growth
        self.trade_history = self.trade_history[-1000:]
        await asyncio.to_thread(save_trading_state, "trade_history", self.trade_history)
        logger.debug("Saved trading state: %d coins, %d positions, %d trades",
                     len(self.current_coins), len(self.positions), len(self.trade_history))

    async def run(self):
        """Main loop that runs forever."""
        logger.info("Trading engine started.")
        # Start background news refresh task
        asyncio.create_task(self._refresh_news_cache())
        asyncio.create_task(self._refresh_current_coins_news_fast())
        # Start background market data download task
        asyncio.create_task(self._download_market_data_loop())
        while True:
            try:
                await self._reconcile_positions()

                # --- Daily profit target ---
                if self._daily_profit_target_pct is not None and self._daily_profit_target_pct > 0:
                    daily_pnl = self._daily_realized_pnl()
                    target_amount = self._daily_profit_target_pct * self.initial_balance
                    if daily_pnl >= target_amount:
                        logger.info(
                            f"Daily profit target reached: {daily_pnl:.2f} >= {target_amount:.2f}. Pausing trading."
                        )
                        await asyncio.to_thread(self.redis.set, "trading:paused", "1")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🏆 Daily profit target hit ({daily_pnl:.2f} / {target_amount:.2f}). Trading paused until tomorrow.",
                                summary={
                                    "action": "INFO",
                                    "reason": "Daily profit target hit",
                                    "daily_pnl": daily_pnl,
                                    "target_amount": target_amount,
                                }
                            )

                # Reset daily profit pause at midnight UTC
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == 0 and now_utc.minute < 5:
                    was_paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                    if was_paused:
                        logger.info("New day – resetting daily profit pause.")
                        await asyncio.to_thread(self.redis.delete, "trading:paused")
                        if self.notifier:
                            await self.notifier.send_notification(
                                "🌅 New day – trading resumed.",
                                summary={
                                    "action": "INFO",
                                    "reason": "New day - trading resumed",
                                }
                            )

                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    logger.info("Trading is paused. Skipping cycle.")
                    await asyncio.sleep(STRATEGY_INTERVAL)
                    continue

                # Periodic state debug log
                try:
                    bal = await asyncio.to_thread(self.trader.fetch_balance)
                    base_bal = bal.get(self.base_currency, 0.0)
                    logger.info(
                        f"Engine state: coins={len(self.current_coins)}, "
                        f"positions={len(self.positions)}, "
                        f"{self.base_currency}_balance={base_bal:.2f}, "
                        f"effective_max_coins={self.effective_max_coins}"
                    )
                except Exception:
                    logger.debug("Could not fetch balance for debug log")

                await self._reevaluate_coins()
                self._cycle_spent = 0.0
                now = time.time()
                for coin_entry in self.current_coins:
                    symbol = coin_entry["symbol"]
                    interval = self._strategy_intervals.get(symbol, STRATEGY_INTERVAL)
                    last_eval = self._last_strategy_eval.get(symbol, 0)
                    if now - last_eval >= interval:
                        await self._process_coin(coin_entry)
                        self._last_strategy_eval[symbol] = now
                await self._check_risk_management()
                await self._save_state()
            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
            # --- Dynamic sleep: wait until the next coin needs evaluation ---
            now = time.time()
            next_times = []
            for coin_entry in self.current_coins:
                symbol = coin_entry["symbol"]
                interval = self._strategy_intervals.get(symbol, STRATEGY_INTERVAL)
                last_eval = self._last_strategy_eval.get(symbol, 0)
                next_times.append(last_eval + interval)
            if next_times:
                earliest = min(next_times)
                sleep_seconds = max(1.0, earliest - now)
            else:
                sleep_seconds = STRATEGY_INTERVAL
            logger.debug(f"Sleeping for {sleep_seconds:.1f}s until next evaluation.")
            await asyncio.sleep(sleep_seconds)

    async def _reevaluate_coins(self):
        """Use LLM to select which coins to trade."""
        # Only re-evaluate every COIN_REVALUATION_INTERVAL
        last_key = "trading:last_coin_eval"
        last_eval = await asyncio.to_thread(self.redis.get, last_key)
        now = time.time()
        if last_eval and (now - float(last_eval)) < self._coin_revaluation_interval and self.current_coins:
            return

        old_coins = list(self.current_coins)
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
        # Apply sentiment filter if configured
        if settings.COIN_SELECTION_MIN_SENTIMENT > -1.0 and settings.NEWS_ENABLED:
            # Pre-fetch sentiment for all available pairs (or a larger batch) to filter
            # To avoid too many DB calls, we can fetch sentiment for the first N pairs
            candidate_pairs = available_pairs[:settings.COIN_SELECTION_MAX_PAIRS * 2]  # look at a larger pool
            filtered_pairs = []
            for sym in candidate_pairs:
                try:
                    base_coin = sym.split("/")[0] if "/" in sym else sym
                    agg = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                    if agg and agg["avg_compound"] >= settings.COIN_SELECTION_MIN_SENTIMENT:
                        filtered_pairs.append(sym)
                    elif not agg:
                        # No sentiment data – include by default (or you can skip)
                        filtered_pairs.append(sym)
                except Exception:
                    filtered_pairs.append(sym)  # include if error
            sample_pairs = filtered_pairs[:settings.COIN_SELECTION_MAX_PAIRS]
        else:
            sample_pairs = available_pairs[:settings.COIN_SELECTION_MAX_PAIRS]

        tickers = await asyncio.to_thread(get_tickers, self.exchange, sample_pairs)

        # --- Fetch order books for top coins to compute real spread/depth for scalping score ---
        top_n_for_ob = min(20, len(sample_pairs))
        top_by_vol = sorted(sample_pairs, key=lambda s: tickers.get(s, {}).get('quoteVolume', 0) or 0, reverse=True)[:top_n_for_ob]
        coin_spreads: Dict[str, float] = {}
        coin_depths: Dict[str, float] = {}
        for sym in top_by_vol:
            try:
                ob = await asyncio.to_thread(get_order_book, self.exchange, sym, 5)
                bids = ob.get('bids', [])
                asks = ob.get('asks', [])
                if bids and asks:
                    best_bid = bids[0][0]
                    best_ask = asks[0][0]
                    mid = (best_bid + best_ask) / 2
                    if mid > 0:
                        spread_pct = ((best_ask - best_bid) / mid) * 100
                        coin_spreads[sym] = round(spread_pct, 4)
                    # Depth: total volume within 1% of mid price
                    bid_vol = sum(b[1] for b in bids if b[0] >= mid * 0.99)
                    ask_vol = sum(a[1] for a in asks if a[0] <= mid * 1.01)
                    total_depth = bid_vol + ask_vol
                    coin_depths[sym] = round(total_depth, 2)
            except Exception as e:
                logger.debug(f"Order book fetch failed for {sym} during coin selection: {e}")

        # --- Compute scalping suitability scores for candidate coins ---
        coin_scores: Dict[str, float] = {}
        for sym in sample_pairs:
            try:
                t = tickers.get(sym, {})
                last = t.get('last', 0) or 0
                volume = t.get('quoteVolume', 0) or 0
                change_24h = abs(t.get('percentage', 0) or 0)  # absolute % change

                # Normalize volume (log scale to avoid one giant dominating)
                vol_score = min(1.0, math.log10(volume + 1) / 10.0) if volume > 0 else 0.0

                # Volatility proxy: use 24h change % (capped at 20%)
                vola_score = min(1.0, change_24h / 20.0)

                # Spread score: lower spread is better (1.0 for 0% spread, 0.0 for >=1% spread)
                sp = coin_spreads.get(sym)
                if sp is not None:
                    spread_score = max(0.0, 1.0 - sp / 1.0)  # 1% spread -> score 0
                else:
                    spread_score = 0.5  # unknown

                # Depth score: log scale, cap at 1.0
                depth = coin_depths.get(sym, 0)
                depth_score = min(1.0, math.log10(depth + 1) / 6.0) if depth > 0 else 0.0

                momentum_score = 1.0 if (t.get('percentage', 0) or 0) > 0 else 0.5

                # Composite score (weights adjusted to include depth)
                score = (0.25 * vol_score + 0.25 * vola_score + 0.25 * spread_score + 0.15 * depth_score + 0.10 * momentum_score)
                coin_scores[sym] = round(score, 3)
            except Exception:
                coin_scores[sym] = 0.0

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

        # Sentiment trend (delta from previous cycle)
        sentiment_trend: Dict[str, Optional[float]] = {}
        for sym in sample_pairs:
            base_coin = sym.split("/")[0] if "/" in sym else sym
            current_compound = None
            if base_coin in news_sentiment:
                current_compound = news_sentiment[base_coin].get("avg_compound")
            prev_key = f"sentiment:prev:{base_coin}"
            prev_raw = await asyncio.to_thread(self.redis.get, prev_key)
            prev_compound = float(prev_raw) if prev_raw else None
            if current_compound is not None:
                await asyncio.to_thread(self.redis.setex, prev_key, settings.NEWS_CACHE_TTL_SECONDS, str(current_compound))
            if current_compound is not None and prev_compound is not None:
                sentiment_trend[base_coin] = round(current_compound - prev_compound, 4)
            else:
                sentiment_trend[base_coin] = None

        # Volume trend (24h volume spike detection)
        volume_trends: Dict[str, Optional[float]] = {}
        for sym in sample_pairs:
            t = tickers.get(sym, {})
            vol = t.get('quoteVolume', 0) or 0
            if vol > 0:
                volume_trends[sym] = await self._compute_volume_trend(sym, vol)

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

        # Relative strength vs BTC
        relative_strength_btc: Dict[str, Dict[str, Any]] = {}
        btc_price = tickers.get("BTC/USDT", {}).get("last")
        btc_change_24h = tickers.get("BTC/USDT", {}).get("percentage")
        if btc_price and btc_price > 0:
            for sym in sample_pairs:
                t = tickers.get(sym, {})
                coin_price = t.get("last")
                if coin_price and coin_price > 0:
                    ratio = coin_price / btc_price
                    coin_change = t.get("percentage")
                    if coin_change is not None and btc_change_24h is not None:
                        # Relative performance: (1+coin_change)/(1+btc_change) - 1
                        rel_perf = ((1 + coin_change / 100) / (1 + btc_change_24h / 100) - 1) * 100
                    else:
                        rel_perf = None
                    relative_strength_btc[sym] = {
                        "ratio": round(ratio, 8),
                        "relative_24h_pct": round(rel_perf, 2) if rel_perf is not None else None,
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

        # Compute indicators for each coin with OHLCV data, for ALL timeframes
        coin_indicators = {}
        for sym, tf_data in ohlcv_data.items():
            coin_indicators[sym] = {}
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
                        ind['ichimoku'] = compute_ichimoku(highs, lows, closes)
                        ind['donchian_channels'] = compute_donchian_channels(highs, lows)
                        coin_indicators[sym][tf] = ind

        # Fetch historical OHLCV from database for longer-term trend analysis (up to 30 days)
        historical_ohlcv_summary = {}
        if settings.OHLCV_TIMEFRAMES:
            since_ms = int(time.time() * 1000) - 30 * 24 * 60 * 60 * 1000

            async def _fetch_historical_summary(sym):
                sym_summary = {}
                for tf in settings.OHLCV_TIMEFRAMES:
                    try:
                        db_candles = await asyncio.to_thread(
                            get_ohlcv, sym, tf, since_ms=since_ms, limit=500
                        )
                        if db_candles and len(db_candles) >= 2:
                            open_price = db_candles[0]["open"]
                            close_price = db_candles[-1]["close"]
                            high = max(c["high"] for c in db_candles)
                            low = min(c["low"] for c in db_candles)
                            volume = sum(c["volume"] for c in db_candles)
                            change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                            sym_summary[tf] = {
                                "candles": len(db_candles),
                                "change_pct": round(change_pct, 2),
                                "high": high,
                                "low": low,
                                "volume": volume,
                            }
                    except Exception as e:
                        logger.debug(f"Failed to fetch historical OHLCV for {sym} {tf}: {e}")
                return sym, sym_summary

            tasks = [_fetch_historical_summary(sym) for sym in sorted_by_vol]
            results = await asyncio.gather(*tasks)
            historical_ohlcv_summary = {sym: summary for sym, summary in results if summary}

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
            if self.notifier:
                await self.notifier.send_notification(
                    f"⚠️ Insufficient {self.base_currency} balance ({base_balance:.2f}) to trade any coin. "
                    f"Min cost required: {min_min_cost:.2f}. Depositing funds or resetting paper balance will fix this.",
                    summary={
                        "action": "HOLD",
                        "reason": "Insufficient balance",
                        "base_balance": base_balance,
                        "min_cost": min_min_cost,
                    }
                )
            return

        # Recompute per-coin budget with the effective max
        per_coin_budget = base_balance / self.effective_max_coins

        # Compute pairwise correlation matrix from OHLCV close prices
        correlation_matrix: Dict[str, Dict[str, float]] = {}
        if ohlcv_data and settings.OHLCV_TIMEFRAMES:
            primary_tf = settings.OHLCV_TIMEFRAMES[0]
            close_series: Dict[str, List[float]] = {}
            for sym in sorted_by_vol:
                if sym in ohlcv_data and primary_tf in ohlcv_data[sym]:
                    candles = ohlcv_data[sym][primary_tf]
                    if len(candles) >= 12:
                        close_series[sym] = [c[4] for c in candles]
            # Compute percentage returns
            returns_series: Dict[str, List[float]] = {}
            for sym, closes in close_series.items():
                returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                           for i in range(1, len(closes)) if closes[i - 1] != 0]
                if len(returns) >= 10:
                    returns_series[sym] = returns
            # Pairwise Pearson correlation
            corr_symbols = list(returns_series.keys())
            for sym_a in corr_symbols:
                correlation_matrix[sym_a] = {}
                for sym_b in corr_symbols:
                    if sym_a == sym_b:
                        correlation_matrix[sym_a][sym_b] = 1.0
                    elif sym_b in correlation_matrix and sym_a in correlation_matrix[sym_b]:
                        correlation_matrix[sym_a][sym_b] = correlation_matrix[sym_b][sym_a]
                    else:
                        ret_a = returns_series[sym_a]
                        ret_b = returns_series[sym_b]
                        min_len = min(len(ret_a), len(ret_b))
                        if min_len < 2:
                            continue
                        a = ret_a[-min_len:]
                        b = ret_b[-min_len:]
                        mean_a = sum(a) / min_len
                        mean_b = sum(b) / min_len
                        cov = sum((a[k] - mean_a) * (b[k] - mean_b) for k in range(min_len)) / min_len
                        std_a = (sum((x - mean_a) ** 2 for x in a) / min_len) ** 0.5
                        std_b = (sum((x - mean_b) ** 2 for x in b) / min_len) ** 0.5
                        if std_a > 0 and std_b > 0:
                            correlation_matrix[sym_a][sym_b] = round(cov / (std_a * std_b), 3)

        perf = self._compute_performance_metrics()
        fear_greed = await self._get_fear_greed_index()
        global_market = await self._fetch_global_market_data()
        # Current trading session
        now_utc = datetime.now(timezone.utc)
        utc_hour = now_utc.hour
        if 0 <= utc_hour < 7:
            session_label = "Asian"
        elif 7 <= utc_hour < 12:
            session_label = "European"
        elif 12 <= utc_hour < 20:
            session_label = "US"
        else:
            session_label = "Low activity"
        session_info = {"utc_hour": utc_hour, "session": session_label}

        # Market breadth: percentage of candidate coins with positive 24h change
        positive_count = sum(1 for sym in sample_pairs if (tickers.get(sym, {}).get('percentage') or 0) > 0)
        total_count = len(sample_pairs)
        market_breadth = {
            "positive_pct": round(positive_count / total_count * 100, 1) if total_count > 0 else 0.0,
            "positive_count": positive_count,
            "total_count": total_count,
        }
        self._market_breadth = market_breadth

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
            daily_pnl=perf["equity_curve"].get("daily_pnl"),
            coin_scores=coin_scores,
            coin_spreads=coin_spreads,
            coin_depths=coin_depths,
            historical_ohlcv_summary=historical_ohlcv_summary,
            correlation_matrix=correlation_matrix,
            fear_greed_index=fear_greed,
            relative_strength_btc=relative_strength_btc,
            session_info=session_info,
            sentiment_trend=sentiment_trend,
            volume_trends=volume_trends,
            market_breadth=market_breadth,
            btc_dominance=global_market.get("btc_dominance") if global_market else None,
            total_market_cap=global_market if global_market else None,
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(get_cached_llm_response, prompt, SYSTEM_PROMPT, 300),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.warning("LLM coin selection timed out. Falling back to volume-based selection.")
            response = None  # will trigger the fallback path below
        logger.info(f"LLM coin selection raw response: {response}")

        if response is not None:
            try:
                parsed = json.loads(response)
                new_coins: List[Dict[str, str]] = []
                llm_max_coins = None

                if isinstance(parsed, dict):
                    # New format: {"coins": [...], "max_coins": N}
                    coins_list = parsed.get("coins", [])
                    llm_max_coins = parsed.get("max_coins")
                    if not isinstance(coins_list, list):
                        logger.error("LLM coin selection 'coins' field is not a list.")
                        coins_list = []
                    for item in coins_list:
                        if isinstance(item, dict) and "symbol" in item:
                            sym = item["symbol"]
                            if sym in available_pairs:
                                tf = item.get("timeframe")
                                if tf not in settings.OHLCV_TIMEFRAMES:
                                    tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                new_coins.append({"symbol": sym, "timeframe": tf})
                        elif isinstance(item, str):
                            if item in available_pairs:
                                default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                new_coins.append({"symbol": item, "timeframe": default_tf})
                elif isinstance(parsed, list):
                    # Old format: plain list of objects or strings
                    for item in parsed:
                        if isinstance(item, dict) and "symbol" in item:
                            sym = item["symbol"]
                            if sym in available_pairs:
                                tf = item.get("timeframe")
                                if tf not in settings.OHLCV_TIMEFRAMES:
                                    tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                new_coins.append({"symbol": sym, "timeframe": tf})
                        elif isinstance(item, str):
                            if item in available_pairs:
                                default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                new_coins.append({"symbol": item, "timeframe": default_tf})
                else:
                    logger.error("LLM coin selection response is neither a list nor a dict.")

                # Deduplicate by symbol, keeping first occurrence
                seen = set()
                deduped = []
                for entry in new_coins:
                    sym = entry["symbol"]
                    if sym not in seen:
                        seen.add(sym)
                        deduped.append(entry)

                # Use the LLM's chosen number of coins to update effective_max_coins
                if llm_max_coins is not None and isinstance(llm_max_coins, int) and 0 <= llm_max_coins <= self.max_coins:
                    self.effective_max_coins = llm_max_coins
                else:
                    # Fallback: use the length of the deduped list, capped at the engine's max
                    self.effective_max_coins = min(len(deduped), self.effective_max_coins)

                # Optional: LLM can set the global coin re-evaluation interval
                new_interval = parsed.get("coin_revaluation_interval_seconds")
                if new_interval is not None:
                    if isinstance(new_interval, (int, float)) and new_interval >= 60:
                        self._coin_revaluation_interval = new_interval
                        logger.info(f"LLM set coin re-evaluation interval to {new_interval}s")
                    else:
                        logger.warning(f"Invalid coin_revaluation_interval_seconds: {new_interval}")

                # Optional: LLM can set a daily profit target
                profit_target = parsed.get("daily_profit_target_pct")
                if profit_target is not None:
                    if isinstance(profit_target, (int, float)) and 0.0 <= profit_target <= 1.0:
                        self._daily_profit_target_pct = profit_target
                        logger.info(f"LLM set daily profit target to {profit_target:.2%}")
                    else:
                        logger.warning(f"Invalid daily_profit_target_pct: {profit_target}")

                self.current_coins = deduped[: self.effective_max_coins]

                # If LLM explicitly chose zero coins, respect that and don't fall back to volume-based selection
                if not deduped or self.effective_max_coins == 0:
                    self.current_coins = []
                    self.effective_max_coins = 0
                    logger.info("LLM selected 0 coins – pausing trading until next evaluation.")
                    await asyncio.to_thread(self.redis.set, last_key, now)
                    return

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

        # Ensure all open positions remain in current_coins so they continue to be managed by the LLM strategy
        for symbol, pos in self.positions.items():
            if not any(entry["symbol"] == symbol for entry in self.current_coins):
                tf = pos.get("timeframe") or (settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h")
                self.current_coins.append({"symbol": symbol, "timeframe": tf})
                logger.info(f"Keeping {symbol} in current_coins due to open position (timeframe={tf})")

        # Trigger immediate backfill for newly selected coins
        old_symbols = {entry["symbol"] for entry in old_coins}
        new_symbols = {entry["symbol"] for entry in self.current_coins} - old_symbols
        for sym in new_symbols:
            logger.info(f"Triggering immediate backfill for newly selected coin {sym}")
            asyncio.create_task(self._backfill_new_coin(sym))

        # Also trigger immediate news fetch for newly selected coins
        if settings.NEWS_ENABLED:
            for sym in new_symbols:
                logger.info(f"Triggering immediate news fetch for newly selected coin {sym}")
                asyncio.create_task(self._fetch_and_store_news_for_symbol(sym))

        coin_labels = [f"{c['symbol']}({c['timeframe']})" for c in self.current_coins]
        logger.info(f"Selected coins: {coin_labels}")
        if not self.current_coins:
            logger.warning("No coins selected after evaluation. Bot will idle until next cycle.")
            if self.notifier:
                await self.notifier.send_notification(
                    f"⚠️ No coins selected. Bot will idle. "
                    f"Balance: {base_balance:.2f} {self.base_currency}, "
                    f"Per-coin budget: {per_coin_budget:.2f}",
                    summary={
                        "action": "HOLD",
                        "reason": "No coins selected",
                        "base_balance": base_balance,
                        "per_coin_budget": per_coin_budget,
                    }
                )
        elif self.notifier:
            await self.notifier.send_notification(
                f"🔄 Coins updated: {', '.join(coin_labels)}",
                summary={
                    "action": "INFO",
                    "reason": "Coins updated",
                    "coins": [c["symbol"] for c in self.current_coins],
                }
            )

        await asyncio.to_thread(self.redis.set, last_key, now)

    async def _process_coin(self, coin_entry: Dict[str, str]):
        """Fetch market data, get LLM strategy, validate, and execute."""
        symbol = coin_entry["symbol"]
        assigned_tf = coin_entry["timeframe"]

        # --- Cooldown after a losing trade (LLM-defined) ---
        last_loss = self.last_loss_time.get(symbol)
        if last_loss is not None:
            cooldown = self.cooldown_durations.get(symbol, 0)
            if cooldown > 0:
                elapsed = time.time() - last_loss
                if elapsed < cooldown:
                    remaining = cooldown - elapsed
                    logger.info(
                        f"Skipping {symbol}: cooldown active ({remaining:.0f}s remaining after loss)"
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏳ Skipping {symbol}: cooldown {remaining:.0f}s",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Cooldown active",
                                "cooldown_remaining_seconds": remaining,
                            }
                        )
                    return

        try:
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            # Relative strength vs BTC for this coin
            rel_strength_btc = None
            try:
                btc_ticker = await asyncio.to_thread(self.exchange.fetch_ticker, "BTC/USDT")
                btc_price = btc_ticker.get("last")
                if btc_price and btc_price > 0 and ticker.get("last"):
                    ratio = ticker["last"] / btc_price
                    coin_change = ticker.get("percentage")
                    btc_change = btc_ticker.get("percentage")
                    if coin_change is not None and btc_change is not None:
                        rel_perf = ((1 + coin_change / 100) / (1 + btc_change / 100) - 1) * 100
                    else:
                        rel_perf = None
                    rel_strength_btc = {
                        "ratio": round(ratio, 8),
                        "relative_24h_pct": round(rel_perf, 2) if rel_perf is not None else None,
                    }
            except Exception:
                pass
            order_book = await asyncio.to_thread(get_order_book, self.exchange, symbol, 20)
            # Fetch recent trades for micro-momentum and liquidity assessment
            recent_trades_raw = []
            try:
                recent_trades_raw = await asyncio.to_thread(
                    self.exchange.fetch_trades, symbol, limit=20
                )
            except Exception as e:
                logger.debug(f"Could not fetch recent trades for {symbol}: {e}")
            balance = await asyncio.to_thread(self.trader.fetch_balance)
            base_balance = balance.get(self.base_currency, 0.0)
            if base_balance <= 0 or self.effective_max_coins == 0:
                logger.warning(
                    f"Skipping {symbol}: {self.base_currency} balance={base_balance:.2f}, "
                    f"effective_max_coins={self.effective_max_coins}"
                )
                return

            # Fetch OHLCV for the coin's assigned timeframe
            ohlcv_data = {}
            if settings.OHLCV_TIMEFRAMES:
                try:
                    ohlcv_data = await asyncio.to_thread(
                        get_multi_timeframe_ohlcv, self.exchange, symbol, settings.OHLCV_TIMEFRAMES, limit=50
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
            ichimoku_tenkan = ind_cfg.get('ichimoku_tenkan', 9) if ind_cfg else 9
            ichimoku_kijun = ind_cfg.get('ichimoku_kijun', 26) if ind_cfg else 26
            ichimoku_senkou_b = ind_cfg.get('ichimoku_senkou_b', 52) if ind_cfg else 52
            donchian_period = ind_cfg.get('donchian_period', 20) if ind_cfg else 20

            # Compute indicators for all timeframes
            multi_tf_indicators: Dict[str, Dict[str, Any]] = {}
            multi_tf_raw_candles: Dict[str, List[List]] = {}
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
            ichimoku = None
            donchian_channels = None

            for tf in settings.OHLCV_TIMEFRAMES:
                if tf in ohlcv_data and ohlcv_data[tf]:
                    candles = ohlcv_data[tf]
                    multi_tf_raw_candles[tf] = candles
                    ind = {}
                    if len(candles) >= 2:
                        ind['atr'] = compute_atr(candles)
                    if len(candles) >= 26:
                        closes = [c[4] for c in candles]
                        highs = [c[2] for c in candles]
                        lows = [c[3] for c in candles]
                        volumes = [c[5] for c in candles]
                        ind['rsi'] = compute_rsi(closes, period=rsi_period)
                        macd_val, macd_sig, macd_hist_val = compute_macd(closes, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)
                        ind['macd'] = macd_val
                        ind['macd_signal'] = macd_sig
                        ind['macd_hist'] = macd_hist_val
                        bb_upper_val, bb_middle_val, bb_lower_val = compute_bollinger_bands(closes, period=bb_period, std_dev=bb_std)
                        ind['bb_upper'] = bb_upper_val
                        ind['bb_middle'] = bb_middle_val
                        ind['bb_lower'] = bb_lower_val
                        ema_9_list = compute_ema(closes, ema_fast)
                        ema_21_list = compute_ema(closes, ema_slow)
                        ind['ema_9'] = ema_9_list[-1] if ema_9_list else None
                        ind['ema_21'] = ema_21_list[-1] if ema_21_list else None
                        stoch_k, stoch_d = compute_stochastic(highs, lows, closes, period=stoch_k_period, smooth_k=stoch_d_period)
                        ind['stochastic_k'] = stoch_k
                        ind['stochastic_d'] = stoch_d
                        adx_val, plus_di_val, minus_di_val = compute_adx(highs, lows, closes, period=adx_period)
                        ind['adx'] = adx_val
                        ind['plus_di'] = plus_di_val
                        ind['minus_di'] = minus_di_val
                        ind['obv'] = compute_obv(closes, volumes)
                        ind['mfi'] = compute_mfi(highs, lows, closes, volumes, period=mfi_period)
                        ind['cci'] = compute_cci(highs, lows, closes, period=cci_period)
                        ind['williams_r'] = compute_williams_r(highs, lows, closes, period=willr_period)
                        ind['ichimoku'] = compute_ichimoku(highs, lows, closes, tenkan_period=ichimoku_tenkan, kijun_period=ichimoku_kijun, senkou_b_period=ichimoku_senkou_b)
                        ind['donchian_channels'] = compute_donchian_channels(highs, lows, period=donchian_period)
                    multi_tf_indicators[tf] = ind
                    # Keep the assigned timeframe's indicators for backward compatibility
                    if tf == assigned_tf:
                        atr = ind.get('atr')
                        rsi = ind.get('rsi')
                        macd = ind.get('macd')
                        macd_signal = ind.get('macd_signal')
                        macd_hist = ind.get('macd_hist')
                        bb_upper = ind.get('bb_upper')
                        bb_middle = ind.get('bb_middle')
                        bb_lower = ind.get('bb_lower')
                        ema_9 = ind.get('ema_9')
                        ema_21 = ind.get('ema_21')
                        stochastic_k = ind.get('stochastic_k')
                        stochastic_d = ind.get('stochastic_d')
                        adx = ind.get('adx')
                        plus_di = ind.get('plus_di')
                        minus_di = ind.get('minus_di')
                        obv = ind.get('obv')
                        mfi = ind.get('mfi')
                        cci = ind.get('cci')
                        williams_r = ind.get('williams_r')
                        ichimoku = ind.get('ichimoku')
                        donchian_channels = ind.get('donchian_channels')

            # Compute Parabolic SAR for the assigned timeframe
            parabolic_sar = None
            if assigned_tf in multi_tf_raw_candles:
                candles = multi_tf_raw_candles[assigned_tf]
                if len(candles) >= 2:
                    sar_highs = [c[2] for c in candles]
                    sar_lows = [c[3] for c in candles]
                    parabolic_sar = compute_parabolic_sar(sar_highs, sar_lows)

            # Compute Keltner Channels for the assigned timeframe
            keltner_channels = None
            if assigned_tf in multi_tf_raw_candles:
                candles = multi_tf_raw_candles[assigned_tf]
                if len(candles) >= 21:
                    kc_closes = [c[4] for c in candles]
                    kc_highs = [c[2] for c in candles]
                    kc_lows = [c[3] for c in candles]
                    keltner_channels = compute_keltner_channels(kc_closes, kc_highs, kc_lows)

            # Compute Pivot Points from the previous completed candle of the assigned timeframe
            pivot_points = None
            if assigned_tf in multi_tf_raw_candles:
                candles = multi_tf_raw_candles[assigned_tf]
                if len(candles) >= 2:
                    prev_candle = candles[-2]  # second-to-last candle is the last completed one
                    prev_high = prev_candle[2]
                    prev_low = prev_candle[3]
                    prev_close = prev_candle[4]
                    pivot_points = compute_pivot_points(prev_high, prev_low, prev_close)

            # Compute VWAP for each timeframe
            vwap_multi_tf: Dict[str, float] = {}
            for tf in settings.OHLCV_TIMEFRAMES:
                if tf in multi_tf_raw_candles:
                    tf_vwap = compute_vwap(multi_tf_raw_candles[tf])
                    if tf_vwap is not None:
                        vwap_multi_tf[tf] = tf_vwap
            vwap = vwap_multi_tf.get(assigned_tf) if assigned_tf else None

            # Compute ATR for each timeframe (volatility term structure)
            atr_multi_tf: Dict[str, float] = {}
            for tf in settings.OHLCV_TIMEFRAMES:
                if tf in multi_tf_raw_candles:
                    tf_atr = compute_atr(multi_tf_raw_candles[tf])
                    if tf_atr > 0:
                        atr_multi_tf[tf] = tf_atr

            # --- Market regime classification ---
            market_regime = "unknown"
            if adx is not None and atr is not None and atr > 0:
                current_price = ticker['last']
                if current_price > 0:
                    if adx > 25:
                        market_regime = "trending"
                    else:
                        market_regime = "ranging"
                    # Volatility: ATR as % of price
                    atr_pct = (atr / current_price) * 100
                    if atr_pct > 5.0:
                        market_regime += " (high volatility)"
                    elif atr_pct < 1.0:
                        market_regime += " (low volatility)"
                    else:
                        market_regime += " (normal volatility)"

            # Extract raw candles for the assigned timeframe (from multi-timeframe data)
            raw_candles = multi_tf_raw_candles.get(assigned_tf)

            # Fetch historical OHLCV from DB for backtest analysis (last 30 days)
            historical_ohlcv = None
            try:
                since_ms = int(time.time() * 1000) - 30 * 24 * 60 * 60 * 1000
                db_candles = await asyncio.to_thread(
                    get_ohlcv, symbol, assigned_tf, since_ms=since_ms, limit=500
                )
                if db_candles:
                    # Convert list of dicts to list of lists [ts, o, h, l, c, v] as expected by the prompt
                    historical_ohlcv = [
                        [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                        for c in db_candles
                    ]
            except Exception as e:
                logger.warning(f"Failed to fetch historical OHLCV for {symbol} {assigned_tf}: {e}")

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

                # Order book depth trend
                depth_trend = None
                if bid_wall_volume is not None and ask_wall_volume is not None:
                    current_depth = bid_wall_volume + ask_wall_volume
                    depth_key = f"depth:prev:{symbol}"
                    prev_raw = await asyncio.to_thread(self.redis.get, depth_key)
                    if prev_raw is not None:
                        prev_depth = float(prev_raw)
                        depth_trend = round(current_depth - prev_depth, 4)
                    await asyncio.to_thread(self.redis.setex, depth_key, 3600, str(current_depth))

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

            # --- Order book depth profile for scalping ---
            depth_profile = {}
            if bids and asks and mid > 0:
                for pct in [0.001, 0.002, 0.005, 0.01, 0.02]:
                    bid_cutoff = mid * (1 - pct)
                    ask_cutoff = mid * (1 + pct)
                    bid_vol = sum(b[1] for b in bids if b[0] >= bid_cutoff)
                    ask_vol = sum(a[1] for a in asks if a[0] <= ask_cutoff)
                    depth_profile[f"{pct*100:.1f}%"] = {
                        "bid_volume": round(bid_vol, 4),
                        "ask_volume": round(ask_vol, 4),
                    }

            # --- Scalping feasibility score ---
            scalping_score = None
            if spread_pct is not None and depth_profile and recent_trades_raw:
                # 1. Spread score: 1.0 if spread <= 0.05%, 0.0 if spread >= 0.5%
                spread_score = max(0.0, min(1.0, 1.0 - (spread_pct - 0.05) / 0.45)) if spread_pct > 0.05 else 1.0

                # 2. Depth score: use ask volume at 0.1% distance (if available)
                depth_01 = depth_profile.get("0.1%", {}).get("ask_volume", 0)
                depth_score = min(1.0, depth_01 / 1.0) if depth_01 else 0.0

                # 3. Trade frequency: trades per minute from recent_trades_raw
                if len(recent_trades_raw) >= 2:
                    timestamps = [t['timestamp'] for t in recent_trades_raw if 'timestamp' in t]
                    if len(timestamps) >= 2:
                        time_span_seconds = (max(timestamps) - min(timestamps)) / 1000.0
                        if time_span_seconds > 0:
                            trades_per_minute = len(timestamps) / (time_span_seconds / 60.0)
                        else:
                            trades_per_minute = 0
                    else:
                        trades_per_minute = 0
                else:
                    trades_per_minute = 0
                freq_score = min(1.0, trades_per_minute / 10.0)

                # 4. Volatility score: ATR% – moderate is best (0.5%–2%)
                if atr is not None and current_price > 0:
                    atr_pct = (atr / current_price) * 100
                    if atr_pct < 0.3:
                        vol_score = 0.2
                    elif atr_pct > 5.0:
                        vol_score = 0.3
                    else:
                        vol_score = max(0.0, 1.0 - abs(atr_pct - 1.5) / 3.5)
                else:
                    vol_score = 0.5

                # Composite score (equal weights)
                scalping_score = round(0.25 * spread_score + 0.25 * depth_score + 0.25 * freq_score + 0.25 * vol_score, 3)

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

            # Fetch minimum order size for this symbol
            market = self.exchange.markets.get(symbol, {})
            limits = market.get('limits', {})
            min_amount_raw = limits.get('amount', {}).get('min')
            min_cost_raw = limits.get('cost', {}).get('min')
            min_order_amount = float(min_amount_raw) if min_amount_raw is not None else None
            min_order_cost = float(min_cost_raw) if min_cost_raw is not None else None

            # Past trades for this specific coin (last 10 closed sells)
            past_trades = [
                t for t in self.trade_history
                if t.get("symbol") == symbol and t.get("side") == "sell"
            ][-10:]

            # Fetch aggregate sentiment for the symbol
            aggregate_sentiment = None
            if settings.NEWS_ENABLED:
                try:
                    base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                    aggregate_sentiment = await asyncio.to_thread(
                        get_aggregate_sentiment_from_db, base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS
                    )
                except Exception as e:
                    logger.debug(f"Could not fetch aggregate sentiment for {symbol}: {e}")

            # Sentiment trend for this coin
            sentiment_trend_val = None
            if aggregate_sentiment:
                base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                current_compound = aggregate_sentiment.get("avg_compound")
                prev_key = f"sentiment:prev:{base_coin}"
                prev_raw = await asyncio.to_thread(self.redis.get, prev_key)
                prev_compound = float(prev_raw) if prev_raw else None
                if current_compound is not None:
                    await asyncio.to_thread(self.redis.setex, prev_key, settings.NEWS_CACHE_TTL_SECONDS, str(current_compound))
                if current_compound is not None and prev_compound is not None:
                    sentiment_trend_val = round(current_compound - prev_compound, 4)

            # Volume trend (24h volume spike detection)
            volume_trend_val = None
            current_volume = ticker.get('quoteVolume', 0) or 0
            if current_volume > 0:
                volume_trend_val = await self._compute_volume_trend(symbol, current_volume)

            remaining = max(0.0, base_balance - self._cycle_spent)
            fear_greed = await self._get_fear_greed_index()
            global_market = await self._fetch_global_market_data()
            # Current trading session
            now_utc = datetime.now(timezone.utc)
            utc_hour = now_utc.hour
            if 0 <= utc_hour < 7:
                session_label = "Asian"
            elif 7 <= utc_hour < 12:
                session_label = "European"
            elif 12 <= utc_hour < 20:
                session_label = "US"
            else:
                session_label = "Low activity"
            session_info = {"utc_hour": utc_hour, "session": session_label}
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
                atr_multi_tf=atr_multi_tf,
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
                depth_profile=depth_profile,
                fee_rate=fee_rate,
                drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
                raw_candles=raw_candles,
                recent_trades=recent_trades_summary,
                historical_ohlcv=historical_ohlcv,
                min_order_amount=min_order_amount,
                min_order_cost=min_order_cost,
                all_coins=self.current_coins,
                past_trades=past_trades,
                aggregate_sentiment=aggregate_sentiment,
                cycle_spent=self._cycle_spent,
                remaining_balance=remaining,
                market_regime=market_regime,
                recent_trades_data=recent_trades_raw,
                multi_tf_raw_candles=multi_tf_raw_candles,
                multi_tf_indicators=multi_tf_indicators,
                scalping_feasibility_score=scalping_score,
                fear_greed_index=fear_greed,
                relative_strength_btc=rel_strength_btc,
                vwap=vwap,
                vwap_multi_tf=vwap_multi_tf,
                session_info=session_info,
                sentiment_trend=sentiment_trend_val,
                volume_trend=volume_trend_val,
                ichimoku=ichimoku,
                market_breadth=getattr(self, '_market_breadth', None),
                depth_trend=depth_trend,
                parabolic_sar=parabolic_sar,
                keltner_channels=keltner_channels,
                pivot_points=pivot_points,
                donchian_channels=donchian_channels,
                btc_dominance=global_market.get("btc_dominance") if global_market else None,
                total_market_cap=global_market if global_market else None,
            )
            logger.debug(f"LLM prompt for {symbol}: {len(prompt)} chars")
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(get_cached_llm_response, prompt, SYSTEM_PROMPT, 60),
                    timeout=120.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM strategy call timed out for {symbol}. Skipping this cycle.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⏱️ LLM timeout for {symbol}, skipping.",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "LLM timeout",
                        }
                    )
                return
            strategy = create_strategy_from_llm(response)
            signal = strategy.generate_signal({})
            current_price = ticker['last']
            validated = validate_signal(
                signal,
                fee_rate=fee_rate,
                atr=atr,
                price=current_price,
                spread_pct=spread_pct,
            )

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
                if ichimoku is not None:
                    ind_parts.append(f"Ichi T={ichimoku['tenkan_sen']:.2f}/K={ichimoku['kijun_sen']:.2f}")
                if donchian_channels is not None:
                    ind_parts.append(f"Donch={donchian_channels['lower']:.2f}-{donchian_channels['upper']:.2f}")
                indicator_str = " | ".join(ind_parts) if ind_parts else ""
                sentiment_str = self._get_sentiment_str(symbol)
                msg = f"{emoji} {symbol}: {validated.action} (confidence: {validated.confidence:.2f}) – {validated.reasoning}"
                if sentiment_str:
                    msg += f"\n{sentiment_str}"
                if getattr(validated, 'backtest_summary', None):
                    msg += f"\n📈 Backtest: {validated.backtest_summary}"
                if indicator_str:
                    msg += f"\n📊 {indicator_str}"
                # Build summary dict for logging
                decision_summary = {
                    "symbol": symbol,
                    "action": validated.action,
                    "confidence": validated.confidence,
                    "reason": validated.reasoning[:200],
                    "sentiment": aggregate_sentiment,
                    "indicators": {
                        "rsi": rsi,
                        "macd": macd,
                        "macd_signal": macd_signal,
                        "atr": atr,
                        "adx": adx,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                        "ema_9": ema_9,
                        "ema_21": ema_21,
                        "stochastic_k": stochastic_k,
                        "mfi": mfi,
                        "cci": cci,
                        "williams_r": williams_r,
                        "ichimoku": ichimoku,
                        "donchian_channels": donchian_channels,
                    },
                    "backtest": getattr(validated, 'backtest_summary', None),
                    "strategy_type": signal.strategy_type,
                    "market_regime": market_regime,
                    "scalping_score": scalping_score,
                }
                await self.notifier.send_notification(msg, summary=decision_summary)

            # --- LLM‑controlled trade filters ---
            params = signal.strategy_params or {}

            # Compute stop-loss percentage for max risk cap (needed for slippage check)
            sl_pct = None
            if validated.action == "BUY":
                stop_method = params.get("stop_loss_method", "fixed")
                if stop_method == "atr_multiple" and atr is not None and atr > 0:
                    atr_mult = params["stop_loss_atr_multiple"]
                    sl_pct = (atr_mult * atr) / current_price
                else:
                    sl_pct = params.get("stop_loss_pct")

            max_spread = params.get("max_spread_pct")
            if max_spread is not None and spread_pct is not None and spread_pct > max_spread:
                logger.info(f"Skipping {symbol}: spread {spread_pct:.4f}% exceeds LLM max {max_spread:.4f}%")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping {symbol}: spread too high ({spread_pct:.4f}% > {max_spread:.4f}%)",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Spread too high",
                            "spread_pct": spread_pct,
                            "max_spread_pct": max_spread,
                        }
                    )
                return

            min_conf = params.get("min_confidence")
            if min_conf is not None and validated.confidence < min_conf:
                logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below LLM min {min_conf:.2f}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping {symbol}: confidence too low ({validated.confidence:.2f})",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Confidence too low",
                            "confidence": validated.confidence,
                            "min_confidence": min_conf,
                        }
                    )
                return

            # --- LLM‑controlled minimum depth at take-profit ---
            min_depth_tp = params.get("min_depth_at_take_profit")
            if min_depth_tp is not None and min_depth_tp > 0:
                # Compute cumulative ask volume from mid to take-profit price
                tp_pct = params["take_profit_pct"]
                tp_price = current_price * (1 + tp_pct)
                asks = order_book.get('asks', [])
                cum_vol = 0.0
                for ask in asks:
                    if ask[0] <= tp_price:
                        cum_vol += ask[1]
                    else:
                        break
                if cum_vol < min_depth_tp:
                    logger.info(
                        f"Skipping {symbol}: ask depth up to take-profit ({tp_price:.4f}) is {cum_vol:.4f}, "
                        f"below LLM minimum {min_depth_tp:.4f}"
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ Skipping {symbol}: insufficient depth at take-profit ({cum_vol:.4f} < {min_depth_tp:.4f})",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Insufficient depth at take-profit",
                                "depth": cum_vol,
                                "min_depth": min_depth_tp,
                            }
                        )
                    return

            # --- LLM‑controlled max slippage for market buy ---
            max_slippage = params.get("max_slippage_pct")
            if max_slippage is not None and max_slippage > 0 and validated.action == "BUY":
                # Compute expected average fill price for the intended buy amount
                quote_balance = balance.get(self.base_currency, 0.0)
                position_fraction = params["position_size_fraction"]
                desired_quote = quote_balance * position_fraction
                # Apply max risk cap if present
                max_risk_pct = params.get("max_risk_per_trade_pct")
                if max_risk_pct is not None and sl_pct is not None and sl_pct > 0:
                    total_value = quote_balance
                    for sym, pos in self.positions.items():
                        try:
                            t = await asyncio.to_thread(self.exchange.fetch_ticker, sym)
                            total_value += pos['amount'] * t['last']
                        except Exception:
                            pass
                    max_risk_amount = total_value * max_risk_pct
                    max_allowed_amount = max_risk_amount / sl_pct
                    desired_quote = min(desired_quote, max_allowed_amount)
                # Cap at remaining cycle budget
                available = max(0.0, quote_balance - self._cycle_spent)
                desired_quote = min(desired_quote, available)
                if desired_quote > 0:
                    # Walk the order book asks to compute volume-weighted average price
                    asks = order_book.get('asks', [])
                    remaining = desired_quote
                    total_cost = 0.0
                    total_base = 0.0
                    for ask in asks:
                        price_level = ask[0]
                        volume = ask[1]
                        cost_at_level = price_level * volume
                        if cost_at_level >= remaining:
                            base_filled = remaining / price_level
                            total_cost += remaining
                            total_base += base_filled
                            remaining = 0
                            break
                        else:
                            total_cost += cost_at_level
                            total_base += volume
                            remaining -= cost_at_level
                    if remaining > 0:
                        logger.info(
                            f"Skipping BUY {symbol}: insufficient order book depth to fill "
                            f"{desired_quote:.2f} {self.base_currency} (remaining {remaining:.2f})"
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ Skipping BUY {symbol}: insufficient depth for order size",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Insufficient order book depth for buy size",
                                }
                            )
                        return
                    avg_price = total_cost / total_base if total_base > 0 else 0
                    best_ask = asks[0][0] if asks else 0
                    if best_ask > 0:
                        slippage_pct = (avg_price - best_ask) / best_ask * 100
                        if slippage_pct > max_slippage:
                            logger.info(
                                f"Skipping BUY {symbol}: expected slippage {slippage_pct:.4f}% "
                                f"exceeds LLM max {max_slippage:.4f}%"
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⚠️ Skipping BUY {symbol}: slippage too high ({slippage_pct:.4f}%)",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SKIP",
                                        "reason": "Slippage too high",
                                        "slippage_pct": slippage_pct,
                                        "max_slippage_pct": max_slippage,
                                    }
                                )
                            return

            # Respect the LLM's execute flag – skip trade if the LLM decided not to execute
            if not getattr(validated, 'execute', True):
                logger.info(f"LLM decided not to execute trade for {symbol}. Reason: {validated.reasoning}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⏭️ {symbol}: LLM skipped trade",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": "LLM execute flag false",
                        }
                    )
                return

            # Prevent SELL without an open position (no shorting)
            if validated.action == "SELL" and symbol not in self.positions:
                logger.info(f"Skipping SELL for {symbol}: no open position.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping SELL for {symbol}: no open position.",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "No open position",
                        }
                    )
                return

            if validated.action != "HOLD":
                await self._execute_signal(symbol, validated, timeframe=assigned_tf, atr=atr, spread_pct=spread_pct)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            if self.notifier:
                await self.notifier.send_notification(
                    f"❌ Error processing {symbol}: {e}",
                    summary={
                        "symbol": symbol,
                        "action": "ERROR",
                        "reason": str(e)[:200],
                    }
                )

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
            "total_fees": round(total_fees, 6),
            "total_fees_display": f"{total_fees:.6f}",
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

    async def sell_all_positions(self):
        """Sell all open positions at market price."""
        for symbol in list(self.positions.keys()):
            await self._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell all"),
                exit_reason="manual_sell_all"
            )

    async def sell_position(self, symbol: str):
        """Sell a specific open position at market price."""
        if symbol in self.positions:
            await self._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell"),
                exit_reason="manual_sell"
            )
        else:
            logger.warning(f"No open position for {symbol}")

    async def _check_risk_management(self):
        """Check open positions and close if stop-loss, take-profit, or trailing stop is hit."""
        for symbol, pos in list(self.positions.items()):
            try:
                # Skip positions that don't have LLM-defined risk parameters yet
                if pos.get("stop_loss") is None or pos.get("take_profit") is None:
                    continue

                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                current_price = ticker['last']

                # Fetch order book if needed for partial TP depth checks
                partial_levels_for_depth = pos.get("partial_take_profit_levels")
                need_order_book = partial_levels_for_depth and any(level.get("min_depth") is not None for level in partial_levels_for_depth)
                if need_order_book:
                    order_book = await asyncio.to_thread(get_order_book, self.exchange, symbol, 5)
                else:
                    order_book = None

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

                # --- Trailing take-profit ---
                if pos.get("trailing_take_profit") and pos.get("trailing_take_profit_distance_pct"):
                    ttp_dist = pos["trailing_take_profit_distance_pct"]
                    new_tp = current_price * (1 + ttp_dist)
                    if new_tp > pos["take_profit"]:
                        pos["take_profit"] = new_tp
                        logger.debug(f"Trailing take-profit updated for {symbol}: new TP {new_tp:.4f}")

                # --- Breakeven stop ---
                breakeven_activation = pos.get("breakeven_activation_pct")
                if breakeven_activation is not None and breakeven_activation > 0:
                    entry_price = pos["price"]
                    if current_price >= entry_price * (1 + breakeven_activation):
                        # Compute exact break-even price that covers exit fee
                        fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
                        if fee_rate < 1.0:
                            breakeven_price = entry_price / (1.0 - fee_rate)
                        else:
                            breakeven_price = entry_price  # fallback
                        if breakeven_price > pos["stop_loss"]:
                            pos["stop_loss"] = breakeven_price
                            logger.debug(f"Breakeven stop activated for {symbol}: new stop {breakeven_price:.4f}")

                # --- Lock profit (scalping) ---
                lock_activation = pos.get("lock_profit_activation_pct")
                lock_level = pos.get("lock_profit_level_pct")
                if lock_activation is not None and lock_level is not None and lock_activation > 0:
                    entry_price = pos["price"]
                    if current_price >= entry_price * (1 + lock_activation):
                        new_stop = entry_price * (1 + lock_level)
                        if new_stop > pos["stop_loss"]:
                            pos["stop_loss"] = new_stop
                            logger.debug(f"Lock-profit activated for {symbol}: new stop {new_stop:.4f} (guaranteed +{lock_level:.2%})")

                # --- Partial take-profit (scalping) ---
                partial_levels = pos.get("partial_take_profit_levels")
                if partial_levels:
                    # Multiple levels
                    triggered = pos.get("partial_tp_levels_triggered", [])
                    original_amount = pos.get("original_amount", pos["amount"])
                    for i, level in enumerate(partial_levels):
                        if i in triggered:
                            continue
                        lvl_pct = level["take_profit_pct"]
                        lvl_frac = level["fraction"]
                        entry_price = pos["price"]
                        # Time‑based cancellation
                        max_time = level.get("max_time_seconds")
                        if max_time is not None:
                            entry_ts = pos.get("timestamp", 0) / 1000.0
                            if time.time() - entry_ts > max_time:
                                logger.info(f"Partial TP level {i} for {symbol} expired (max {max_time}s). Cancelling.")
                                triggered.append(i)
                                pos["partial_tp_levels_triggered"] = triggered
                                continue
                        if current_price >= entry_price * (1 + lvl_pct):
                            # Check minimum depth if specified
                            min_depth = level.get("min_depth")
                            if min_depth is not None and order_book is not None:
                                tp_price = entry_price * (1 + lvl_pct)
                                asks = order_book.get('asks', [])
                                cum_vol = 0.0
                                for ask in asks:
                                    if ask[0] <= tp_price:
                                        cum_vol += ask[1]
                                    else:
                                        break
                                if cum_vol < min_depth:
                                    logger.info(
                                        f"Partial TP level {i} for {symbol}: insufficient depth "
                                        f"({cum_vol:.4f} < {min_depth:.4f}), skipping this cycle."
                                    )
                                    # Do not mark as triggered; will re-check next cycle
                                    continue
                            sell_amount = original_amount * lvl_frac
                            # Ensure we don't sell more than current position
                            sell_amount = min(sell_amount, pos["amount"])
                            if sell_amount <= 0:
                                triggered.append(i)
                                pos["partial_tp_levels_triggered"] = triggered
                                continue
                            logger.info(
                                f"Partial TP level {i} for {symbol}: selling {sell_amount:.6f} "
                                f"({lvl_frac:.0%} of original) at {current_price:.4f}"
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🔸 Partial TP [{i}] {symbol}: selling {sell_amount:.6f} @ {current_price:.4f}",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SELL",
                                        "reason": f"Partial take-profit level {i}",
                                        "price": current_price,
                                        "amount": sell_amount,
                                        "exit_reason": f"partial_take_profit_level_{i}",
                                    }
                                )
                            try:
                                order = await asyncio.to_thread(
                                    self.trader.create_market_sell_order, symbol, sell_amount
                                )
                            except Exception as e:
                                logger.error(f"Partial sell failed for {symbol}: {e}")
                                triggered.append(i)
                                pos["partial_tp_levels_triggered"] = triggered
                                continue

                            # Compute realized P&L for the sold portion
                            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
                            fee = order.get('fee', {})
                            fee_cost = float(fee.get('cost', 0.0) or 0.0)
                            fee_currency = fee.get('currency', '')
                            if fee_cost == 0.0:
                                fee_cost = order['cost'] * fee_rate
                                fee_currency = symbol.split('/')[1]
                                order['fee'] = {'cost': fee_cost, 'currency': fee_currency}
                            net_quote = order['cost'] - (fee_cost if fee_currency == symbol.split('/')[1] else 0.0)
                            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                            net_base = pos.get("net_base", pos["amount"])
                            prorated_cost_basis = cost_basis * (sell_amount / net_base) if net_base > 0 else 0.0
                            realized_pnl = net_quote - prorated_cost_basis

                            trade = {
                                "symbol": symbol,
                                "side": "sell",
                                "amount": sell_amount,
                                "price": order["price"],
                                "cost": order["cost"],
                                "fee": order["fee"],
                                "timestamp": order["timestamp"],
                                "exit_reason": f"partial_take_profit_level_{i}",
                                "realized_pnl": realized_pnl,
                                "cost_basis": prorated_cost_basis,
                                "strategy_type": pos.get("strategy_type", "unknown"),
                                "timeframe": pos.get("timeframe"),
                                "hold_time_seconds": (order["timestamp"] - pos["timestamp"]) / 1000.0 if "timestamp" in pos else None,
                            }
                            self.trade_history.append(trade)
                            await asyncio.to_thread(insert_trade, trade)

                            # Update position
                            pos["amount"] -= sell_amount
                            pos["cost_basis"] -= prorated_cost_basis
                            pos["net_base"] -= sell_amount
                            if pos["net_base"] > 0:
                                pos["price"] = pos["cost_basis"] / pos["net_base"]
                            else:
                                self.positions.pop(symbol, None)
                                logger.info(f"Position fully closed by partial TP levels for {symbol}")
                                break

                            triggered.append(i)
                            pos["partial_tp_levels_triggered"] = triggered

                            if self.notifier:
                                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                                await self.notifier.send_notification(
                                    f"🔸 Partial TP [{i}] {symbol}: sold {sell_amount:.6f} @ {order['price']:.4f} | "
                                    f"P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%) | Remaining: {pos['amount']:.6f}",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SELL",
                                        "reason": f"Partial take-profit level {i} executed",
                                        "price": order["price"],
                                        "amount": sell_amount,
                                        "realized_pnl": realized_pnl,
                                        "exit_reason": f"partial_take_profit_level_{i}",
                                    }
                                )
                else:
                    # Single partial TP (existing logic, unchanged)
                    partial_tp_pct = pos.get("partial_take_profit_pct")
                    partial_tp_fraction = pos.get("partial_take_profit_fraction")
                    if (
                        partial_tp_pct is not None
                        and partial_tp_fraction is not None
                        and not pos.get("partial_tp_triggered", False)
                    ):
                        entry_price = pos["price"]
                        if current_price >= entry_price * (1 + partial_tp_pct):
                            sell_amount = pos["amount"] * partial_tp_fraction
                            logger.info(
                                f"Partial take-profit triggered for {symbol}: selling {sell_amount:.6f} "
                                f"({partial_tp_fraction:.0%}) at {current_price:.4f}"
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🔸 Partial TP {symbol}: selling {sell_amount:.6f} @ {current_price:.4f}"
                                )
                            try:
                                order = await asyncio.to_thread(
                                    self.trader.create_market_sell_order, symbol, sell_amount
                                )
                            except Exception as e:
                                logger.error(f"Partial sell failed for {symbol}: {e}")
                                pos["partial_tp_triggered"] = True
                                continue

                            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
                            fee = order.get('fee', {})
                            fee_cost = float(fee.get('cost', 0.0) or 0.0)
                            fee_currency = fee.get('currency', '')
                            if fee_cost == 0.0:
                                fee_cost = order['cost'] * fee_rate
                                fee_currency = symbol.split('/')[1]
                                order['fee'] = {'cost': fee_cost, 'currency': fee_currency}
                            net_quote = order['cost'] - (fee_cost if fee_currency == symbol.split('/')[1] else 0.0)
                            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                            net_base = pos.get("net_base", pos["amount"])
                            prorated_cost_basis = cost_basis * (sell_amount / net_base) if net_base > 0 else 0.0
                            realized_pnl = net_quote - prorated_cost_basis

                            trade = {
                                "symbol": symbol,
                                "side": "sell",
                                "amount": sell_amount,
                                "price": order["price"],
                                "cost": order["cost"],
                                "fee": order["fee"],
                                "timestamp": order["timestamp"],
                                "exit_reason": "partial_take_profit",
                                "realized_pnl": realized_pnl,
                                "cost_basis": prorated_cost_basis,
                                "strategy_type": pos.get("strategy_type", "unknown"),
                                "timeframe": pos.get("timeframe"),
                                "hold_time_seconds": (order["timestamp"] - pos["timestamp"]) / 1000.0 if "timestamp" in pos else None,
                            }
                            self.trade_history.append(trade)
                            await asyncio.to_thread(insert_trade, trade)

                            pos["amount"] -= sell_amount
                            pos["cost_basis"] -= prorated_cost_basis
                            pos["net_base"] -= sell_amount
                            if pos["net_base"] > 0:
                                pos["price"] = pos["cost_basis"] / pos["net_base"]
                            else:
                                self.positions.pop(symbol, None)
                                logger.warning(f"Partial TP closed entire position for {symbol} unexpectedly.")
                                continue

                            pos["partial_tp_triggered"] = True

                            if pos.get("stop_loss") is not None and pos["stop_loss"] < entry_price:
                                pos["stop_loss"] = entry_price
                                logger.debug(f"Stop moved to breakeven after partial TP for {symbol}")

                            if self.notifier:
                                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                                await self.notifier.send_notification(
                                    f"🔸 Partial TP {symbol}: sold {sell_amount:.6f} @ {order['price']:.4f} | "
                                    f"P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%) | Remaining: {pos['amount']:.6f}"
                                )

                # --- News sentiment exit ---
                news_threshold = pos.get("news_sentiment_exit_threshold")
                if news_threshold is not None and settings.NEWS_ENABLED:
                    try:
                        base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                        agg = await asyncio.to_thread(
                            get_aggregate_sentiment_from_db, base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS
                        )
                        if agg and agg["avg_compound"] < news_threshold:
                            logger.info(
                                f"News sentiment exit for {symbol}: compound {agg['avg_compound']:.2f} < threshold {news_threshold}"
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"📰 Negative news exit for {symbol} (sentiment {agg['avg_compound']:.2f})",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SELL",
                                        "reason": "News sentiment exit",
                                        "sentiment": agg,
                                        "exit_reason": "news_sentiment_exit",
                                    }
                                )
                            await self._execute_signal(
                                symbol,
                                Signal(action="SELL", confidence=1.0, reasoning="News sentiment exit"),
                                exit_reason="news_sentiment_exit"
                            )
                            continue  # skip further checks for this symbol
                    except Exception as e:
                        logger.debug(f"News sentiment check failed for {symbol}: {e}")

                # --- Soft stop: max unrealized loss ---
                max_ul_pct = pos.get("max_unrealized_loss_pct")
                if max_ul_pct is not None and max_ul_pct > 0:
                    entry_price = pos["price"]
                    if current_price <= entry_price * (1 - max_ul_pct):
                        logger.info(f"Max unrealized loss reached for {symbol} ({max_ul_pct:.2%}). Closing position.")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"📉 Soft stop triggered for {symbol} at {current_price:.4f} (max loss {max_ul_pct:.2%})",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Max unrealized loss",
                                    "price": current_price,
                                    "exit_reason": "max_unrealized_loss",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Max unrealized loss"),
                            exit_reason="max_unrealized_loss"
                        )
                        continue

                # Time‑based exit (LLM‑defined max hold time)
                max_hold = pos.get("max_hold_time_seconds")
                if max_hold is not None and max_hold > 0:
                    entry_ts = pos.get("timestamp", 0) / 1000.0  # convert ms to seconds
                    if time.time() - entry_ts > max_hold:
                        logger.info(f"Max hold time reached for {symbol} ({max_hold}s). Closing position.")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏰ Max hold time reached for {symbol} – closing position.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Max hold time",
                                    "exit_reason": "max_hold_time",
                                }
                            )
                        await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Max hold time"), exit_reason="max_hold_time")
                        continue   # skip further checks for this symbol

                if current_price <= pos["stop_loss"]:
                    logger.info(f"Stop-loss triggered for {symbol} at {current_price}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⛔ Stop‑loss triggered for {symbol} at {current_price:.4f}",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Stop-loss",
                                "price": current_price,
                                "exit_reason": "stop_loss",
                            }
                        )
                    await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss"), exit_reason="stop_loss")
                elif current_price >= pos["take_profit"]:
                    logger.info(f"Take-profit triggered for {symbol} at {current_price}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"✅ Take‑profit triggered for {symbol} at {current_price:.4f}",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Take-profit",
                                "price": current_price,
                                "exit_reason": "take_profit",
                            }
                        )
                    await self._execute_signal(symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit"), exit_reason="take_profit")
            except Exception as e:
                logger.error(f"Risk check failed for {symbol}: {e}")

    async def _execute_signal(self, symbol: str, signal, timeframe: str = None, exit_reason: str = None, atr: Optional[float] = None, spread_pct: Optional[float] = None):
        """Execute a BUY or SELL signal."""
        base, quote = symbol.split("/")
        balance = await asyncio.to_thread(self.trader.fetch_balance)

        if signal.action == "BUY":
            # Extract known parameters from the LLM's strategy_params (if any)
            params = signal.strategy_params or {}

            # Use LLM-provided risk parameters directly (no hardcoded minimums)
            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
            tp_pct = params["take_profit_pct"]
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

            quote_balance = balance.get(quote, 0.0)
            position_fraction = params["position_size_fraction"]

            # Desired amount based on fraction of total available quote balance
            desired_amount = quote_balance * position_fraction

            # Apply max risk per trade cap if provided
            max_risk_pct = params.get("max_risk_per_trade_pct")
            if max_risk_pct is not None and sl_pct > 0:
                total_value = quote_balance
                for sym, pos in self.positions.items():
                    try:
                        t = await asyncio.to_thread(self.exchange.fetch_ticker, sym)
                        total_value += pos['amount'] * t['last']
                    except Exception:
                        pass
                max_risk_amount = total_value * max_risk_pct
                max_allowed_amount = max_risk_amount / sl_pct
                desired_amount = min(desired_amount, max_allowed_amount)
                logger.info(f"Max risk per trade: {max_risk_pct:.2%} of {total_value:.2f} = {max_risk_amount:.2f}, max allowed amount = {max_allowed_amount:.2f}")


            # --- Minimum absolute profit check (LLM‑defined) ---
            min_profit = params.get("min_profit_per_trade")
            if min_profit is not None and min_profit > 0:
                expected_gross_profit = desired_amount * tp_pct
                if expected_gross_profit < min_profit:
                    logger.info(
                        f"Skipping BUY {symbol}: expected gross profit {expected_gross_profit:.4f} {quote} "
                        f"below LLM minimum {min_profit:.4f}"
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ Skipping BUY {symbol}: profit too small ({expected_gross_profit:.4f} {quote})",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Expected profit below minimum",
                                "expected_profit": expected_gross_profit,
                                "min_profit": min_profit,
                            }
                        )
                    return

            # Cap at remaining available balance in this cycle
            available = max(0.0, quote_balance - self._cycle_spent)
            amount = min(desired_amount, available)

            if amount <= 0:
                logger.info(f"Insufficient {quote} to buy {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Insufficient {quote} to buy {symbol}",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Insufficient balance",
                        }
                    )
                return

            # Check minimum order size and adjust upward if needed
            try:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                price = ticker['last']
                base_amount = amount / price
                market = self.exchange.markets.get(symbol, {})
                limits = market.get('limits', {})
                min_amount_limit = limits.get('amount', {}).get('min')
                min_cost_limit = limits.get('cost', {}).get('min')

                # Determine the required minimum quote amount
                required_quote = amount
                if min_amount_limit is not None:
                    min_base = float(min_amount_limit)
                    required_quote = max(required_quote, min_base * price)
                if min_cost_limit is not None:
                    required_quote = max(required_quote, float(min_cost_limit))

                if required_quote > amount:
                    # Adjust amount upward to meet the minimum
                    old_amount = amount
                    amount = required_quote
                    # Check if the adjusted amount exceeds remaining cycle budget
                    if amount > available:
                        logger.info(
                            f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                            f"to meet minimum, but exceeds remaining cycle budget ({available:.2f}). Skipping."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ BUY skipped for {symbol}: amount adjusted to {amount:.2f} but insufficient remaining budget",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Adjusted amount exceeds remaining budget",
                                    "adjusted_amount": amount,
                                }
                            )
                        return
                    logger.info(
                        f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                        f"to meet exchange minimum"
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"ℹ️ {symbol}: buy amount adjusted to {amount:.2f} {quote} to meet minimum",
                            summary={
                                "symbol": symbol,
                                "action": "INFO",
                                "reason": "Buy amount adjusted to meet minimum",
                                "adjusted_amount": amount,
                            }
                        )
                    # Recalculate base_amount for the order
                    base_amount = amount / price
            except Exception as e:
                logger.warning(f"Could not verify/adjust min order size for {symbol}: {e}")

            try:
                order = await asyncio.to_thread(self.trader.create_market_buy_order, symbol, amount)
                logger.info(f"BUY {symbol}: {order}")
                self._cycle_spent += order['cost']
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
                    self.positions[symbol]["trailing_take_profit"] = params.get("trailing_take_profit", False)
                    self.positions[symbol]["trailing_take_profit_distance_pct"] = params.get("trailing_take_profit_distance_pct")
                    self.positions[symbol]["breakeven_activation_pct"] = params.get("breakeven_activation_pct")
                    self.positions[symbol]["lock_profit_activation_pct"] = params.get("lock_profit_activation_pct")
                    self.positions[symbol]["lock_profit_level_pct"] = params.get("lock_profit_level_pct")
                    # Multiple partial take-profit levels
                    partial_levels = params.get("partial_take_profit_levels")
                    if partial_levels:
                        self.positions[symbol]["partial_take_profit_levels"] = partial_levels
                        self.positions[symbol]["partial_tp_levels_triggered"] = []
                        # Clear single-level fields to avoid confusion
                        self.positions[symbol]["partial_take_profit_pct"] = None
                        self.positions[symbol]["partial_take_profit_fraction"] = None
                        self.positions[symbol]["partial_tp_triggered"] = None
                    else:
                        self.positions[symbol]["partial_take_profit_pct"] = params.get("partial_take_profit_pct")
                        self.positions[symbol]["partial_take_profit_fraction"] = params.get("partial_take_profit_fraction")
                        self.positions[symbol]["partial_tp_triggered"] = False
                    self.positions[symbol]["cooldown_after_loss_seconds"] = params["cooldown_after_loss_seconds"]
                    self.positions[symbol]["news_sentiment_exit_threshold"] = params.get("news_sentiment_exit_threshold")
                    self.positions[symbol]["max_unrealized_loss_pct"] = params.get("max_unrealized_loss_pct")
                    custom_interval = params.get("strategy_interval_seconds")
                    if custom_interval is not None:
                        self._strategy_intervals[symbol] = custom_interval
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
                        "trailing_take_profit": params.get("trailing_take_profit", False),
                        "trailing_take_profit_distance_pct": params.get("trailing_take_profit_distance_pct"),
                        "breakeven_activation_pct": params.get("breakeven_activation_pct"),
                        "lock_profit_activation_pct": params.get("lock_profit_activation_pct"),
                        "lock_profit_level_pct": params.get("lock_profit_level_pct"),
                        "partial_take_profit_levels": params.get("partial_take_profit_levels"),
                        "partial_tp_levels_triggered": [],
                        "original_amount": net_base,
                        "partial_take_profit_pct": params.get("partial_take_profit_pct") if not params.get("partial_take_profit_levels") else None,
                        "partial_take_profit_fraction": params.get("partial_take_profit_fraction") if not params.get("partial_take_profit_levels") else None,
                        "partial_tp_triggered": False if not params.get("partial_take_profit_levels") else None,
                        "cooldown_after_loss_seconds": params["cooldown_after_loss_seconds"],
                        "news_sentiment_exit_threshold": params.get("news_sentiment_exit_threshold"),
                        "max_unrealized_loss_pct": params.get("max_unrealized_loss_pct"),
                        "timeframe": timeframe,
                        "indicator_config": signal.indicator_config,
                    }
                    custom_interval = params.get("strategy_interval_seconds")
                    if custom_interval is not None:
                        self._strategy_intervals[symbol] = custom_interval
                order["strategy_type"] = signal.strategy_type
                order["timeframe"] = timeframe
                self.trade_history.append(order)
                await asyncio.to_thread(insert_trade, order)
                await self._save_state()
                if self.notifier:
                    buy_msg = f"🟢 BUY {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    await self.notifier.send_notification(
                        buy_msg,
                        summary={
                            "symbol": symbol,
                            "action": "BUY",
                            "price": order["price"],
                            "amount": order["amount"],
                            "confidence": signal.confidence,
                            "reason": signal.reasoning[:200],
                            "strategy_type": signal.strategy_type,
                            "indicators": {
                                "atr": atr,
                            },
                        }
                    )
            except Exception as e:
                logger.error(f"Buy order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Buy order failed for {symbol}: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Buy order failed: {e}"[:200],
                        }
                    )

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
                    await self.notifier.send_notification(
                        f"⚠️ No {base} to sell for {symbol}",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "No base balance to sell",
                        }
                    )
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
                        await self.notifier.send_notification(
                            f"⚠️ SELL skipped for {symbol}: amount too small",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Sell amount below minimum",
                            }
                        )
                    return
                if min_cost_limit is not None and gross_amount * price < float(min_cost_limit):
                    logger.info(f"SELL cost {gross_amount * price:.2f} {quote} below min cost {min_cost_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ SELL skipped for {symbol}: cost too small",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Sell cost below minimum",
                            }
                        )
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
                # Track loss timestamps for cooldown
                if realized_pnl < 0:
                    self.last_loss_time[symbol] = time.time()
                    # Store the cooldown duration from the position (set by LLM at BUY time)
                    cd = pos.get("cooldown_after_loss_seconds", 0) if pos else 0
                    self.cooldown_durations[symbol] = cd
                tf = timeframe or (pos.get("timeframe") if pos else None)
                order["timeframe"] = tf
                order["strategy_type"] = signal.strategy_type
                order["exit_reason"] = exit_reason
                order["exit_price"] = order["price"]
                if pos and "timestamp" in pos:
                    hold_time = (order["timestamp"] - pos["timestamp"]) / 1000.0
                    order["hold_time_seconds"] = hold_time
                else:
                    order["hold_time_seconds"] = None
                # Remove position
                self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self.trade_history.append(order)
                await asyncio.to_thread(insert_trade, order)
                await self._save_state()
                if self.notifier:
                    # Human-readable labels for common exit reasons
                    reason_labels = {
                        "manual_sell": "🖐️ Manual",
                        "manual_sell_all": "🖐️ Manual (Sell All)",
                        "stop_loss": "⛔ Stop-Loss",
                        "take_profit": "✅ Take-Profit",
                        "max_hold_time": "⏰ Max Hold Time",
                        "news_sentiment_exit": "📰 News Sentiment",
                        "force_close": "🔻 Force Close",
                        "external_sell": "🔄 External Sell",
                        "delisted": "🗑️ Delisted",
                    }
                    reason_label = reason_labels.get(exit_reason, exit_reason) if exit_reason else None
                    reason_str = f" [{reason_label}]" if reason_label else ""
                    sell_msg = f"🔴 SELL{reason_str} {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    # Add profit/loss info
                    if pos:
                        pnl_pct = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
                        sell_msg += f" | P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)"
                    await self.notifier.send_notification(
                        sell_msg,
                        summary={
                            "symbol": symbol,
                            "action": "SELL",
                            "price": order["price"],
                            "amount": order["amount"],
                            "confidence": signal.confidence,
                            "reason": signal.reasoning[:200],
                            "exit_reason": exit_reason,
                            "realized_pnl": realized_pnl,
                            "strategy_type": signal.strategy_type,
                            "indicators": {
                                "atr": atr,
                            },
                        }
                    )
            except Exception as e:
                logger.error(f"Sell order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Sell order failed for {symbol}: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Sell order failed: {e}"[:200],
                        }
                    )
