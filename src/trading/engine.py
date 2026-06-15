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
from src.exchanges.factory import get_exchange, get_pro_exchange
from src.exchanges.ws_manager import WebSocketManager
from src.exchanges.market_data import get_available_pairs, get_tickers, get_order_book, get_multi_timeframe_ohlcv
from src.trading.paper_simulator import PaperSimulator
from src.trading.live_trader import LiveTrader
from src.llm.cache import get_cached_llm_response, compute_market_hash
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_coin_selection_prompt,
    build_strategy_prompt,
    _format_news_for_prompt,
    compact_prompt,
    get_cached_news_summary,
)

COMPACTED_SYSTEM_PROMPT = compact_prompt(SYSTEM_PROMPT)
from src.indicators import (
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
    compute_all_indicators,
)
try:
    from src.news.fetcher import discover_trending_coins
except ImportError:
    discover_trending_coins = None
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm, LLMStrategy
from src.strategies.validator import validate_signal
from src.utils.redis_client import get_redis_client
from src.database import load_trading_state, save_trading_state, delete_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db, get_news_for_symbol, get_ohlcv, get_latest_ohlcv_timestamp, insert_ohlcv_batch, save_paper_balances, load_paper_balances, cleanup_old_ohlcv

logger = logging.getLogger(__name__)

COIN_REVALUATION_INTERVAL = 3600  # seconds (60 minutes)
DEFAULT_STRATEGY_INTERVAL = 600   # fallback when no timeframe or no coins (10 minutes)
MIN_COIN_REVALUATION_INTERVAL = 300  # seconds (5 minutes) – prevents rapid toggling
MIN_LLM_PAUSE_DURATION = 3600  # seconds (60 min) – LLM cannot resume before this
MAX_STOP_LOSS_REVIEWS = 10   # force-sell after this many consecutive stop-loss reviews
MAX_TAKE_PROFIT_REVIEWS = 10   # force-sell after this many consecutive take-profit reviews


class TradingEngine:
    def __init__(self):
        self.exchange = get_exchange()
        self.pro_exchange = get_pro_exchange()
        self.ws_manager = WebSocketManager(self.pro_exchange, [])
        self.base_currency = settings.BASE_CURRENCY
        self.max_coins = settings.MAX_COINS
        self.effective_max_coins = self.max_coins
        self.redis = get_redis_client()
        self._exchange_semaphore = asyncio.Semaphore(3)  # max 3 concurrent API calls

        if settings.TRADING_MODE == "paper":
            self.trader = PaperSimulator(
                self.exchange,
                base_currency=self.base_currency,
                initial_balance=settings.PAPER_INITIAL_BALANCE,
                redis_client=self.redis,
                ws_manager=self.ws_manager,
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
        self.notifier = None
        self._load_state()
        # Restore paper simulator state from trade history
        if settings.TRADING_MODE == "paper":
            self._restore_paper_state()
        self._ensure_cost_basis()
        # Ensure trading is not paused on startup and no stale pause keys remain
        pause_keys = [
            "trading:paused",
            "trading:pause_source",
            "trading:pause_start",
            "trading:pause_duration",
            "trading:pause_reason",
            "trading:llm_pause_time",
        ]
        for key in pause_keys:
            self.redis.delete(key)

        # Track quote currency spent in the current cycle to avoid over-allocating
        self._cycle_spent = 0.0
        self._coin_first_seen: Dict[str, float] = {}  # symbol -> timestamp when first added
        self._market_breadth: Optional[Dict[str, Any]] = None
        self._risk_lock = asyncio.Lock()
        self._coin_reeval_lock = asyncio.Lock()
        self._reeval_trigger = asyncio.Event()
        self._running = True
        self._last_state_save = 0
        self._last_eval_snapshot: Dict[str, Dict[str, float]] = {}  # symbol -> indicator snapshot
        self._pending_entries: Dict[str, Dict[str, Any]] = {}  # symbol -> pending entry condition info

        # Re-entrancy guards for periodic tasks
        self._reconcile_running = False
        self._reevaluate_running = False
        self._pause_check_running = False
        self._news_cache_running = False
        self._news_fast_running = False
        self._market_data_running = False
        self._full_breadth_running = False

    def set_notifier(self, notifier):
        """Attach a notification service (e.g., TelegramBot)."""
        self.notifier = notifier

    def trigger_coin_reevaluation(self):
        """Signal the periodic reevaluate loop to run immediately."""
        self._reeval_trigger.set()

    async def stop(self):
        """Gracefully stop the engine and all background tasks."""
        logger.info("Stopping trading engine...")
        self._running = False
        await self.ws_manager.stop()
        logger.info("Trading engine stopped.")

    async def _periodic_reconcile(self):
        """Run position reconciliation every 60 seconds."""
        while self._running:
            if self._reconcile_running:
                logger.warning("Reconcile still running; skipping this cycle.")
                await asyncio.sleep(60)
                continue
            self._reconcile_running = True
            try:
                await self._reconcile_positions()
            except Exception as e:
                logger.error(f"Reconcile error: {e}", exc_info=True)
            finally:
                self._reconcile_running = False
            await asyncio.sleep(60)

    async def _periodic_reevaluate(self):
        """Re-evaluate coin selection periodically."""
        while self._running:
            if self._reevaluate_running:
                logger.warning("Coin re-evaluation still running; skipping this cycle.")
                await asyncio.sleep(self._coin_revaluation_interval)
                continue
            self._reevaluate_running = True
            try:
                # Check if trading is paused
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    # Only run the pause/resume decision, skip coin selection
                    await self._check_pause_resume_decision()
                else:
                    await self._reevaluate_coins()
                    # Update WebSocket subscriptions to match current coins
                    current_symbols = [entry["symbol"] for entry in self.current_coins]
                    await self.ws_manager.update_subscriptions(current_symbols)
            except Exception as e:
                logger.error(f"Coin re-evaluation error: {e}", exc_info=True)
            finally:
                self._reevaluate_running = False
            try:
                await asyncio.wait_for(self._reeval_trigger.wait(), timeout=self._coin_revaluation_interval)
            except asyncio.TimeoutError:
                pass
            self._reeval_trigger.clear()

    async def _periodic_pause_check(self):
        """Check and handle auto-resume from pause (only for LLM-initiated pauses)."""
        while self._running:
            if self._pause_check_running:
                logger.warning("Pause check still running; skipping this cycle.")
                await asyncio.sleep(30)
                continue
            self._pause_check_running = True
            try:
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    # Only auto-resume if the pause was initiated by the LLM
                    source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    if source and (source.decode() if isinstance(source, bytes) else source) == "llm":
                        pause_start_raw = await asyncio.to_thread(self.redis.get, "trading:pause_start")
                        pause_duration_raw = await asyncio.to_thread(self.redis.get, "trading:pause_duration")
                        # --- Fallback if no pause_duration was set ---
                        if not pause_duration_raw:
                            # No LLM-set duration → resume after a safe default
                            default_max_pause = 7200  # 2 hours
                            try:
                                elapsed = time.time() - float(pause_start_raw)
                                if elapsed >= default_max_pause:
                                    logger.warning(
                                        "Pause has no duration; forcing auto‑resume after default fallback (2 hours)."
                                    )
                                    stored_reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
                                    stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                                    pause_keys = [
                                        "trading:paused",
                                        "trading:pause_source",
                                        "trading:pause_start",
                                        "trading:pause_duration",
                                        "trading:pause_reason",
                                        "trading:llm_pause_time",
                                    ]
                                    for key in pause_keys:
                                        await asyncio.to_thread(self.redis.delete, key)
                                    self._reeval_trigger.set()
                                    await asyncio.to_thread(self.redis.set, "trading:last_auto_resume", str(time.time()))
                                    await asyncio.to_thread(self.redis.setex, "trading:auto_resume_cooldown", 600, "1")
                                    if self.notifier:
                                        await self.notifier.send_notification(
                                            "⏰ Trading auto‑resumed after maximum pause duration (no LLM‑set duration).",
                                            summary={"action": "RESUME", "reason": "Fallback pause timeout"}
                                        )
                                else:
                                    # still waiting, but we already know there is no duration, don't spam log
                                    pass
                            except (ValueError, TypeError):
                                pass
                            return   # skip the original duration logic
                        if pause_start_raw and pause_duration_raw:
                            try:
                                pause_start = float(pause_start_raw)
                                pause_duration = int(pause_duration_raw)
                                if time.time() - pause_start >= pause_duration:
                                    logger.info("Pause duration elapsed – auto-resuming trading.")
                                    stored_reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
                                    stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                                    # Delete all pause keys
                                    pause_keys = [
                                        "trading:paused",
                                        "trading:pause_source",
                                        "trading:pause_start",
                                        "trading:pause_duration",
                                        "trading:pause_reason",
                                        "trading:llm_pause_time",
                                    ]
                                    for key in pause_keys:
                                        await asyncio.to_thread(self.redis.delete, key)
                                    self._reeval_trigger.set()
                                    await asyncio.to_thread(self.redis.set, "trading:last_auto_resume", str(time.time()))
                                    await asyncio.to_thread(self.redis.setex, "trading:auto_resume_cooldown", 600, "1")
                                    if self.notifier:
                                        reason_text = f" (was paused: {stored_reason})" if stored_reason else ""
                                        await self.notifier.send_notification(
                                            f"▶️ Trading auto-resumed after pause duration elapsed.{reason_text}",
                                            summary={"action": "RESUME", "reason": f"Pause duration elapsed{reason_text}"}
                                        )
                            except (ValueError, TypeError):
                                pass  # ignore malformed values
            except Exception as e:
                logger.error(f"Pause check error: {e}", exc_info=True)
            finally:
                self._pause_check_running = False
            await asyncio.sleep(30)

    async def _periodic_full_market_breadth(self):
        """Periodically compute market breadth over all available pairs."""
        await asyncio.sleep(60)  # initial delay
        while self._running:
            if self._full_breadth_running:
                logger.warning("Full market breadth computation still running; skipping this cycle.")
                await asyncio.sleep(300)
                continue
            self._full_breadth_running = True
            try:
                available_pairs = await asyncio.to_thread(
                    get_available_pairs, self.exchange, self.base_currency
                )
                if available_pairs:
                    # Limit to 500 pairs to avoid excessive API calls
                    breadth_pairs = available_pairs[:500]
                    breadth_tickers = await asyncio.to_thread(
                        get_tickers, self.exchange, breadth_pairs
                    )
                    positive_count = sum(
                        1 for sym in breadth_pairs
                        if (breadth_tickers.get(sym, {}).get('percentage') or 0) > 0
                    )
                    total_count = len(breadth_pairs)
                    full_market_breadth = {
                        "positive_pct": round(positive_count / total_count * 100, 1) if total_count > 0 else 0.0,
                        "positive_count": positive_count,
                        "total_count": total_count,
                    }
                    await asyncio.to_thread(
                        self.redis.setex, "market:breadth:full", 600, json.dumps(full_market_breadth)
                    )
                    logger.debug(f"Full market breadth updated: {full_market_breadth}")
            except Exception as e:
                logger.error(f"Full market breadth computation error: {e}", exc_info=True)
            finally:
                self._full_breadth_running = False
            await asyncio.sleep(300)  # every 5 minutes

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
                summary_raw = get_cached_news_summary(symbol)
                if isinstance(summary_raw, dict):
                    summary = summary_raw.get("summary", "")
                else:
                    summary = summary_raw
                if summary in ("No recent news.", "Could not generate summary."):
                    summary = ""
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
        """Fetch Altcoin Season Index from CoinMarketCap global metrics."""
        if not getattr(settings, 'ALTCOIN_SEASON_ENABLED', True):
            return None
        api_key = settings.CMC_API_KEY
        if not api_key:
            logger.warning("CMC_API_KEY not set; cannot fetch altcoin season index.")
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
                    "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                    headers={"X-CMC_PRO_API_KEY": api_key},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    btc_dominance = data["data"]["quote"]["USD"]["btc_dominance"]
                    altcoin_index = round(100.0 - btc_dominance, 2)
                    if altcoin_index > 75:
                        description = "Altcoin Season"
                    elif altcoin_index < 25:
                        description = "Bitcoin Season"
                    else:
                        description = "Neutral"
                    result = {
                        "value": altcoin_index,
                        "description": description,
                    }
                    ttl = getattr(settings, 'ALTCOIN_SEASON_CACHE_TTL_SECONDS', 3600)
                    await asyncio.to_thread(self.redis.setex, cache_key, ttl, json.dumps(result))
                    return result
                else:
                    logger.warning(
                        f"Altcoin Season Index API returned status {resp.status_code}: {resp.text[:200]}"
                    )
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

    async def _risk_management_loop(self):
        """Check stop-loss, take-profit, and other risk rules on every ticker update."""
        await asyncio.sleep(5)  # initial delay
        while self._running:
            try:
                if self.ws_manager.healthy:
                    update = await self.ws_manager.wait_for_update(timeout=5.0)
                else:
                    # WebSocket down – fall back to polling
                    await asyncio.sleep(5)
                await self._check_risk_management()
                await self._save_state()
            except Exception as e:
                logger.error(f"Risk management loop error: {e}", exc_info=True)

    async def _refresh_current_coins_news_fast(self):
        """Fast news refresh loop – only for the coins currently tracked by the engine."""
        if not settings.NEWS_ENABLED:
            return
        # Fetch immediately on startup, then periodically
        while self._running:
            if self._news_fast_running:
                logger.warning("Fast news refresh still running; skipping this cycle.")
                await asyncio.sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self._news_fast_running = True
            try:
                symbols = [entry["symbol"] for entry in self.current_coins]
                if symbols:
                    logger.debug(f"Fast news refresh for {len(symbols)} current coins")
                    await asyncio.gather(
                        *[self._fetch_and_store_news_for_symbol(sym) for sym in symbols]
                    )
            except Exception as e:
                logger.error(f"Fast news refresh error: {e}")
            finally:
                self._news_fast_running = False
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

        while self._running:
            if self._news_cache_running:
                logger.warning("News cache refresh still running; skipping this cycle.")
                await asyncio.sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self._news_cache_running = True
            try:
                cycle_start = time.time()
                # Slow refresh: all available pairs EXCEPT the coins already handled by the fast loop
                current_symbols = {entry["symbol"] for entry in self.current_coins}
                symbols_to_refresh = set()
                try:
                    available_pairs = await asyncio.to_thread(
                        get_available_pairs, self.exchange, self.base_currency
                    )
                    # Fetch tickers for a subset to determine top volume coins
                    # (limit to 200 to avoid excessive API calls)
                    sample_for_vol = available_pairs[:200]
                    tickers = await asyncio.to_thread(get_tickers, self.exchange, sample_for_vol)
                    def _vol(sym):
                        t = tickers.get(sym, {})
                        return t.get('quoteVolume', 0) or 0
                    top_volume_pairs = sorted(sample_for_vol, key=_vol, reverse=True)[
                        :settings.COIN_SELECTION_TOP_VOLUME_LIMIT
                    ]
                    symbols_to_refresh = set(top_volume_pairs) - current_symbols
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
            finally:
                self._news_cache_running = False

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

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
        return self._timeframe_to_ms(timeframe) // 1000

    async def _backfill_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int, max_candles: int = None, ignore_existing: bool = False):
        """Fetch and store all missing OHLCV candles between start_ms and end_ms."""
        logger.info(f"Backfill started for {symbol} {timeframe}: {start_ms} → {end_ms}")
        if ignore_existing:
            since = start_ms
        else:
            latest_ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, symbol, timeframe)
            if latest_ts is None:
                since = start_ms
            else:
                since = max(start_ms, latest_ts + 1)

        total_inserted = 0
        if max_candles is None:
            max_candles = settings.BACKFILL_MAX_CANDLES_PER_CALL
        while since < end_ms:
            try:
                async with self._exchange_semaphore:
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

            if total_inserted >= max_candles:
                logger.info(
                    f"Backfill limit reached for {symbol} {timeframe}: {total_inserted} candles inserted "
                    f"(max {max_candles}). Remaining range will be filled in next cycle."
                )
                break

            last_ts = candles[-1][0]
            if last_ts <= since:
                # Avoid infinite loop if exchange returns same candle
                break
            since = last_ts + 1
            # Small delay to avoid rate limits
            await asyncio.sleep(0.2)

        if total_inserted >= max_candles:
            logger.info(f"Backfill partial for {symbol} {timeframe}: {total_inserted} candles inserted (limit reached)")
        else:
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
                    await self._backfill_ohlcv(symbol, timeframe, gap_start, gap_end, ignore_existing=True)
                    gaps_filled += 1

        if gaps_found == 0:
            logger.debug(f"No gaps found for {symbol} {timeframe}")
        else:
            logger.info(f"Gap check for {symbol} {timeframe}: {gaps_found} gaps found, {gaps_filled} filled")

    async def _backfill_new_coin(self, symbol: str, timeframe: str):
        """Immediately backfill 30 days of OHLCV data for a newly selected coin (assigned timeframe only)."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 30 * 24 * 60 * 60 * 1000
        logger.info(f"Starting immediate backfill for newly selected coin {symbol} ({timeframe})")
        try:
            await self._backfill_ohlcv(symbol, timeframe, start_ms, now_ms)
            await self._fill_gaps(symbol, timeframe)
        except Exception as e:
            logger.error(f"Initial backfill failed for {symbol} {timeframe}: {e}")
        logger.info(f"Immediate backfill complete for {symbol} ({timeframe})")

    async def _download_market_data_loop(self):
        """Periodically download and store OHLCV data for tracked coins, with gap detection."""
        # Initial delay to let the engine settle
        await asyncio.sleep(30)
        while self._running:
            if self._market_data_running:
                logger.warning("Market data download still running; skipping this cycle.")
                await asyncio.sleep(settings.MARKET_DATA_REFRESH_SECONDS)
                continue
            self._market_data_running = True
            try:
                if not self.current_coins:
                    logger.debug("No coins tracked; skipping market data download.")
                else:
                    logger.info("Starting market data download cycle...")
                    now_ms = int(time.time() * 1000)
                    start_ms = now_ms - 30 * 24 * 60 * 60 * 1000  # 30 days ago
                    for coin_entry in self.current_coins:
                        symbol = coin_entry["symbol"]
                        tf = coin_entry["timeframe"]
                        logger.debug(f"Downloading market data for {symbol} ({tf})")
                        try:
                            await self._backfill_ohlcv(symbol, tf, start_ms, now_ms)
                            await self._fill_gaps(symbol, tf)
                        except Exception as e:
                            logger.warning(f"Market data download failed for {symbol} {tf}: {e}")
                        # Configurable delay between coins to avoid rate limits
                        await asyncio.sleep(settings.OHLCV_DOWNLOAD_COIN_DELAY_SECONDS)
                    logger.info("Market data download cycle complete.")
                    # Clean up old OHLCV data (older than 30 days)
                    await asyncio.to_thread(cleanup_old_ohlcv, 30)
            except Exception as e:
                logger.error(f"Market data download loop error: {e}", exc_info=True)
            finally:
                self._market_data_running = False

            await asyncio.sleep(settings.MARKET_DATA_REFRESH_SECONDS)

    def _restore_paper_state(self):
        """Load paper balances and reconcile with trade history.

        Instead of trusting the saved balances (which may be stale if a
        crash occurred before they were persisted), we always replay the
        full trade history from the initial balance.  This guarantees that
        the in-memory balances are consistent with the recorded trades.
        """
        saved_balances = load_paper_balances()

        # Always reconcile: replay trade history to compute expected balances
        expected_balances: Dict[str, float] = {self.base_currency: self.initial_balance}
        for trade in self.trade_history:
            symbol = trade["symbol"]
            base, quote = symbol.split("/")
            side = trade["side"]
            amount = trade["amount"]
            cost = trade.get("cost", amount * trade["price"])
            fee = trade.get("fee", {})
            fee_cost = float(fee.get("cost", 0) or 0)
            fee_currency = fee.get("currency", "")

            if side == "buy":
                expected_balances[quote] = expected_balances.get(quote, 0) - cost
                net_base = amount - (fee_cost if fee_currency == base else 0)
                expected_balances[base] = expected_balances.get(base, 0) + net_base
            elif side == "sell":
                net_quote = cost - (fee_cost if fee_currency == quote else 0)
                expected_balances[quote] = expected_balances.get(quote, 0) + net_quote
                expected_balances[base] = expected_balances.get(base, 0) - amount

        # Compare with saved balances and log warnings for mismatches
        if saved_balances:
            for currency in set(list(expected_balances.keys()) + list(saved_balances.keys())):
                expected_val = round(expected_balances.get(currency, 0), 8)
                saved_val = round(saved_balances.get(currency, 0), 8)
                if abs(expected_val - saved_val) > 0.01:
                    logger.warning(
                        "Paper balance mismatch for %s: expected=%s, saved=%s. Using reconciled balance.",
                        currency, expected_val, saved_val,
                    )

        self.trader.balances = expected_balances
        logger.info("Reconciled paper balances from trade history: %s", expected_balances)

        # Persist the reconciled balances so they are up-to-date on disk
        save_paper_balances(self.trader.balances)

        # Populate the simulator's trade list so the web dashboard can show history
        self.trader.trades = list(self.trade_history)

        # Positions are already loaded by _load_state() – nothing more to do.
        logger.info("Paper state restored: %d positions, %d trades",
                     len(self.positions), len(self.trade_history))

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

        # Count consecutive losing trades (most recent first)
        consecutive_losses = 0
        for trade in reversed(self.trade_history):
            if trade.get("side") == "sell":
                pnl = trade.get("realized_pnl", 0.0)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    break

        return {
            "coin_performance": coin_perf,
            "strategy_performance": strategy_perf,
            "equity_curve": {
                "total_pnl": round(sum(t.get("realized_pnl", 0.0) for t in self.trade_history if t.get("side") == "sell"), 4),
                "recent_10_trades_pnl": round(total_recent_pnl, 4),
                "trend": trend,
                "drawdown_pct": round(drawdown_pct, 2),
                "daily_pnl": round(daily_pnl, 4),
                "consecutive_losses": consecutive_losses,
            },
        }

    @staticmethod
    def _classify_market_regime(
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        bb_middle: Optional[float],
        atr: Optional[float],
        atr_percentile: Optional[float],
        current_price: float,
    ) -> str:
        """Classify market regime using multiple indicators."""
        if current_price <= 0:
            return "unknown"

        # --- Trend direction and strength ---
        trend_dir = "neutral"
        trend_strength = "weak"
        if adx is not None and plus_di is not None and minus_di is not None:
            if adx > 40:
                trend_strength = "strong"
            elif adx > 25:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"

            if plus_di > minus_di:
                trend_dir = "uptrend"
            elif minus_di > plus_di:
                trend_dir = "downtrend"
            else:
                trend_dir = "neutral"

        # --- Moving average alignment ---
        ma_alignment = "neutral"
        if ema_9 is not None and ema_21 is not None:
            if ema_9 > ema_21:
                ma_alignment = "bullish"
            else:
                ma_alignment = "bearish"

        # --- Volatility state ---
        volatility = "normal"
        if atr is not None and current_price > 0:
            atr_pct = (atr / current_price) * 100
            if atr_percentile is not None:
                if atr_percentile > 80:
                    volatility = "high"
                elif atr_percentile < 20:
                    volatility = "low"
                else:
                    volatility = "normal"
            else:
                # Fallback to simple thresholds
                if atr_pct > 5.0:
                    volatility = "high"
                elif atr_pct < 1.0:
                    volatility = "low"

        # --- Bollinger Band squeeze/expansion ---
        bb_state = ""
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < 0.02:   # very narrow bands
                bb_state = " squeeze"
            elif bb_width > 0.08: # wide bands
                bb_state = " expansion"

        # --- Compose final regime string ---
        if trend_strength in ("strong", "moderate") and trend_dir != "neutral":
            regime = f"{trend_strength} {trend_dir}"
        else:
            regime = "ranging"

        # Add volatility
        regime += f", {volatility} volatility"

        # Add Bollinger state if meaningful
        if bb_state:
            regime += bb_state

        # Add MA alignment if it conflicts with ADX trend (e.g., weak trend but bullish MA)
        if trend_strength == "weak" and ma_alignment != "neutral":
            regime += f" ({ma_alignment} MA bias)"

        return regime

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
                    await self._remove_coin_if_paused(coin)

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
                    ticker = self.ws_manager.get_ticker(symbol)
                    if ticker is None:
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
                    await self._remove_coin_if_paused(symbol)
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
        # Also persist paper balances if in paper mode
        if settings.TRADING_MODE == "paper":
            await asyncio.to_thread(save_paper_balances, self.trader.balances)
        logger.debug("Saved trading state: %d coins, %d positions, %d trades",
                     len(self.current_coins), len(self.positions), len(self.trade_history))

    async def run(self):
        """Main event‑driven loop using WebSocket ticker updates."""
        logger.info("Trading engine started.")
        await self.ws_manager.start()
        logger.info("WebSocket manager started.")

        # Start background tasks
        asyncio.create_task(self._refresh_news_cache())
        asyncio.create_task(self._refresh_current_coins_news_fast())
        asyncio.create_task(self._download_market_data_loop())
        asyncio.create_task(self._risk_management_loop())
        asyncio.create_task(self._periodic_reconcile())
        asyncio.create_task(self._periodic_reevaluate())
        asyncio.create_task(self._periodic_pause_check())
        asyncio.create_task(self._periodic_full_market_breadth())
        asyncio.create_task(self._check_pending_entries())

        # Initial coin selection and subscription update
        await self._reevaluate_coins()
        current_symbols = [entry["symbol"] for entry in self.current_coins]
        await self.ws_manager.update_subscriptions(current_symbols)

        while self._running:
            try:
                # Wait for a ticker update (or timeout after 1s to allow checking health)
                if self.ws_manager.healthy:
                    update = await self.ws_manager.wait_for_update(timeout=1.0)
                else:
                    logger.warning("WebSocket manager unhealthy – falling back to REST polling.")
                    await asyncio.sleep(1.0)

                # Process any coin whose evaluation interval has elapsed
                now = time.time()
                for coin_entry in self.current_coins:
                    symbol = coin_entry["symbol"]
                    default_interval = self._timeframe_to_seconds(coin_entry["timeframe"])
                    interval = self._strategy_intervals.get(symbol, default_interval)
                    last_eval = self._last_strategy_eval.get(symbol, 0)
                    if now - last_eval >= interval:
                        # Check if trading is paused (skip BUY signals)
                        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                        trading_paused = paused is not None and paused == b"1"
                        await self._process_coin(coin_entry, trading_paused=trading_paused)
                        self._last_strategy_eval[symbol] = now

                # Save state periodically (every 30 seconds)
                if now - self._last_state_save > 30:
                    await self._save_state()
                    self._last_state_save = now

            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _reevaluate_coins(self):
        """Use LLM to select which coins to trade."""
        async with self._coin_reeval_lock:
            return await self._reevaluate_coins_impl()

    async def _reevaluate_coins_impl(self):
        # Reset per-cycle spending tracker so new buys are not blocked by prior cycle spending
        self._cycle_spent = 0.0

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

        # Remove fully excluded symbols from the candidate pool
        sample_pairs = [
            sym for sym in sample_pairs
            if not any(
                entry.split("/")[0] == sym.split("/")[0] and
                entry.split("/")[1] == sym.split("/")[1] and
                len(entry.split("/")) == 2
                for entry in settings.EXCLUDED_PAIRS
            )
        ]

        tickers = await asyncio.to_thread(get_tickers, self.exchange, sample_pairs)

        # --- Limit candidate pool to top N by 24h volume ---
        def _volume(sym):
            t = tickers.get(sym, {})
            return t.get('quoteVolume', 0) or 0
        sample_pairs = sorted(sample_pairs, key=_volume, reverse=True)[:settings.COIN_SELECTION_TOP_VOLUME_LIMIT]

        # --- Fetch order books for these top coins to compute real spread/depth for scalping score ---
        top_n_for_ob = min(50, len(sample_pairs))
        top_by_vol = sample_pairs[:top_n_for_ob]  # already sorted by volume
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

        # Use the already volume‑sorted sample_pairs for OHLCV fetch (limit to 20 to avoid rate limits)
        sorted_by_vol = sample_pairs[:50]

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
                    ind = compute_all_indicators(candles)
                    if ind:
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
        altcoin_season = await self._fetch_altcoin_season_index()
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

        # Read full market breadth from Redis (computed by background task)
        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except Exception:
            pass

        # Store market status in Redis for the web dashboard
        market_status = {
            "fear_greed": fear_greed,
            "global_market": global_market,
            "altcoin_season": altcoin_season,
            "market_breadth": market_breadth,
            "full_market_breadth": full_market_breadth,
            "btc_dominance": global_market.get("btc_dominance") if global_market else None,
            "total_market_cap": global_market,
            "timestamp": time.time(),
        }
        await asyncio.to_thread(self.redis.setex, "market:status", 3600, json.dumps(market_status))

        # Check if trading is currently paused
        trading_paused_raw = await asyncio.to_thread(self.redis.get, "trading:paused")
        trading_paused_bool = trading_paused_raw is not None and trading_paused_raw == b"1"

        # Compute coin tenure for the prompt
        coin_tenure = {}
        for sym, first_seen in self._coin_first_seen.items():
            coin_tenure[sym] = round(now - first_seen)

        # Compute current max tenure per coin for the prompt
        coin_max_tenure = {}
        for entry in self.current_coins:
            if 'max_tenure_hours' in entry:
                coin_max_tenure[entry['symbol']] = entry['max_tenure_hours']

        # --- Warn if trading was recently auto-resumed ---
        auto_resume_note = ""
        last_auto_resume_raw = await asyncio.to_thread(self.redis.get, "trading:last_auto_resume")
        if last_auto_resume_raw:
            try:
                last_auto_resume_ts = float(last_auto_resume_raw)
                seconds_since = now - last_auto_resume_ts
                if seconds_since < self._coin_revaluation_interval * 2:
                    minutes_since = seconds_since / 60
                    auto_resume_note = (
                        f"\n**NOTE:** Trading was auto‑resumed {minutes_since:.1f} minutes ago after a pause. "
                        "Market conditions may not have changed significantly. "
                        "Consider whether conditions have actually improved enough to justify trading. "
                        "If you decide to pause again, set a longer `pause_duration_seconds` (e.g., 1800–7200) "
                        "to allow conditions to evolve; a very short pause will likely lead to the same outcome.\n"
                    )
            except (ValueError, TypeError):
                pass

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
            altcoin_season=altcoin_season,
            trading_paused=trading_paused_bool,
            open_positions=self.positions,
            coin_tenure=coin_tenure,
            coin_max_tenure=coin_max_tenure,
            full_market_breadth=full_market_breadth,
        )
        if auto_resume_note:
            prompt += "\n" + auto_resume_note
        # Build a market snapshot dict for caching
        market_snapshot = {
            "available_pairs": sample_pairs,
            "tickers": tickers,
            "market_limits": market_limits,
            "ohlcv_data": ohlcv_data,
            "news_sentiment": news_sentiment,
            "coin_indicators": coin_indicators,
            "performance": perf,
            "fear_greed": fear_greed,
            "global_market": global_market,
            "altcoin_season": altcoin_season,
            "session_info": session_info,
            "market_breadth": market_breadth,
            "btc_dominance": global_market.get("btc_dominance") if global_market else None,
            "total_market_cap": global_market if global_market else None,
            "trading_paused": trading_paused_bool,
            "open_positions": self.positions,
            "coin_tenure": coin_tenure,
            "coin_max_tenure": coin_max_tenure,
            "full_market_breadth": full_market_breadth,
            "base_balance": base_balance,
            "per_coin_budget": per_coin_budget,
            "max_coins": self.effective_max_coins,
            "current_coins": self.current_coins,
            "coin_scores": coin_scores,
            "coin_spreads": coin_spreads,
            "coin_depths": coin_depths,
            "historical_ohlcv_summary": historical_ohlcv_summary,
            "correlation_matrix": correlation_matrix,
            "relative_strength_btc": relative_strength_btc,
            "sentiment_trend": sentiment_trend,
            "volume_trends": volume_trends,
        }
        market_hash = compute_market_hash(market_snapshot)
        # Compute prompt complexity for temperature selection
        _st_values = [abs(v) for v in sentiment_trend.values() if v is not None]
        _st_mag = max(_st_values) if _st_values else None
        coin_selection_complexity = self._compute_prompt_complexity(
            num_candidates=len(sample_pairs),
            market_breadth=market_breadth,
            fear_greed=fear_greed,
            volatility_percentile=None,
            sentiment_trend_magnitude=_st_mag,
            conflicting_signals=False,
            is_critical=False,
        )
        effective_temp = self._get_effective_temperature("mind", coin_selection_complexity)

        parsed = {}
        max_retries = 2
        response = None
        llm_provider = None
        llm_model = None
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(prompt),
                    COMPACTED_SYSTEM_PROMPT,
                    300,
                    market_hash=market_hash,
                    model_type="mind",
                    temperature=effective_temp,
                )
                response = result["response"]
                llm_provider = result["provider"]
                llm_model = result["model"]
                break  # success
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(
                        f"LLM coin selection timed out (attempt {attempt+1}/{max_retries+1}). Retrying..."
                    )
                    await asyncio.sleep(1)
                else:
                    logger.warning(
                        "LLM coin selection timed out after all retries. Falling back to volume-based selection."
                    )
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"LLM coin selection failed with error: {e}. Retrying..."
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(
                        f"LLM coin selection failed after all retries: {e}. Falling back to volume-based selection."
                    )
        logger.info(f"LLM coin selection raw response: {response}")

        # Initialize variables that may be used later even if LLM fails
        parsed = {}
        pause_trading = None
        pause_reason = ""
        pause_duration = None

        # Retry JSON parsing if the first attempt fails
        if response is not None:
            try:
                json.loads(response)  # validate
            except json.JSONDecodeError:
                logger.warning("LLM coin selection response was not valid JSON. Retrying with correction prompt.")
                correction_prompt = (
                    "Your previous response was not valid JSON. "
                    "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                    "Here is the original request:\n\n" + prompt
                )
                try:
                    correction_result = await asyncio.to_thread(
                        get_cached_llm_response, compact_prompt(correction_prompt), COMPACTED_SYSTEM_PROMPT, 120,
                        model_type="actuator",
                        temperature=effective_temp,
                    )
                    response = correction_result["response"]
                    llm_provider = correction_result["provider"]
                    llm_model = correction_result["model"]
                    json.loads(response)  # validate the retry response
                except Exception as e:
                    logger.error(f"LLM coin selection still invalid after retry: {e}")
                    response = None

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
                                entry = {"symbol": sym, "timeframe": tf}
                                mth = item.get("max_tenure_hours")
                                if mth is not None:
                                    entry["max_tenure_hours"] = mth
                                new_coins.append(entry)
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
                                entry = {"symbol": sym, "timeframe": tf}
                                mth = item.get("max_tenure_hours")
                                if mth is not None:
                                    entry["max_tenure_hours"] = mth
                                new_coins.append(entry)
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

                # Remove excluded pairs
                deduped = [
                    e for e in deduped
                    if not self._is_excluded(e["symbol"], e["timeframe"])
                ]

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
                        clamped = max(new_interval, MIN_COIN_REVALUATION_INTERVAL)
                        self._coin_revaluation_interval = clamped
                        logger.info(f"LLM set coin re-evaluation interval to {clamped}s (requested {new_interval}s)")
                    else:
                        logger.warning(f"Invalid coin_revaluation_interval_seconds: {new_interval}")

                was_paused = trading_paused_bool   # captured earlier in the method
                # Optional: LLM can request to pause/resume trading
                pause_trading = parsed.get("pause_trading")
                pause_reason = parsed.get("pause_reason", "")

                # Normalise string booleans from the LLM (e.g. "false" → False)
                if isinstance(pause_trading, str):
                    low = pause_trading.strip().lower()
                    if low in ("true", "1"):
                        pause_trading = True
                    elif low in ("false", "0"):
                        pause_trading = False
                    else:
                        logger.warning(f"Unrecognised pause_trading string: {pause_trading}")
                        pause_trading = None

                # --- Auto-resume cooldown: ignore pause requests shortly after an auto-resume ---
                cooldown_active = await asyncio.to_thread(self.redis.get, "trading:auto_resume_cooldown")
                if cooldown_active and pause_trading is True:
                    logger.info(
                        "Ignoring LLM pause request because auto‑resume cooldown is active "
                        "(trading was recently auto‑resumed)."
                    )
                    pause_trading = None  # treat as "no decision"

                skip_resume = False
                if pause_trading is not None:
                    if isinstance(pause_trading, bool):
                        if pause_trading:
                            # Only pause if not already manually paused
                            current_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                            if current_source and (current_source.decode() if isinstance(current_source, bytes) else current_source) == "manual":
                                logger.info("LLM pause request ignored because trading is manually paused.")
                            else:
                                await asyncio.to_thread(self.redis.set, "trading:paused", "1")
                                await asyncio.to_thread(self.redis.set, "trading:pause_source", "llm")
                                await asyncio.to_thread(self.redis.set, "trading:pause_start", str(time.time()))
                                await asyncio.to_thread(self.redis.set, "trading:llm_pause_time", str(time.time()))
                                # Fallback if LLM did not provide pause_duration_seconds
                                if pause_duration is None:
                                    pause_duration = MIN_LLM_PAUSE_DURATION
                                    await asyncio.to_thread(
                                        self.redis.setex, "trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration))
                                    )
                                if pause_reason:
                                    await asyncio.to_thread(self.redis.set, "trading:pause_reason", pause_reason)
                                logger.info("LLM requested to pause trading.")
                        else:
                            # LLM requests resume – only allowed if the pause was LLM-initiated
                            current_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                            if current_source and current_source.decode() != "llm":
                                logger.info("LLM resume request ignored because pause was not initiated by LLM.")
                            else:
                                if trading_paused_bool:
                                    # Determine the required pause duration:
                                    # - the LLM-set pause_duration_seconds (if any) stored in Redis
                                    # - but never less than MIN_LLM_PAUSE_DURATION
                                    pause_start_raw = await asyncio.to_thread(self.redis.get, "trading:pause_start")
                                    pause_duration_raw = await asyncio.to_thread(self.redis.get, "trading:pause_duration")
                                    required_pause = MIN_LLM_PAUSE_DURATION
                                    if pause_duration_raw:
                                        try:
                                            llm_set_duration = int(pause_duration_raw)
                                            required_pause = max(MIN_LLM_PAUSE_DURATION, llm_set_duration)
                                        except (ValueError, TypeError):
                                            pass

                                    if pause_start_raw:
                                        try:
                                            pause_start = float(pause_start_raw)
                                            elapsed = time.time() - pause_start
                                            if elapsed < required_pause:
                                                remaining = required_pause - elapsed
                                                logger.info(
                                                    f"Ignoring LLM resume request: required pause duration "
                                                    f"({required_pause}s) not yet elapsed ({remaining:.0f}s remaining)."
                                                )
                                                skip_resume = True
                                        except (ValueError, TypeError):
                                            pass
                                    if not skip_resume:
                                        # Delete all pause keys
                                        pause_keys = [
                                            "trading:paused",
                                            "trading:pause_source",
                                            "trading:pause_start",
                                            "trading:pause_duration",
                                            "trading:pause_reason",
                                            "trading:llm_pause_time",
                                        ]
                                        for key in pause_keys:
                                            await asyncio.to_thread(self.redis.delete, key)
                                        logger.info("LLM requested to resume trading.")
                                        self._reeval_trigger.set()
                                else:
                                    # Trading is already active – LLM confirms to keep it active
                                    logger.info("LLM decided to keep trading active (already active).")
                    else:
                        logger.warning(f"Invalid pause_trading value: {pause_trading}")

                # Optional: LLM can set a pause duration after which trading auto-resumes
                pause_duration = parsed.get("pause_duration_seconds")
                if pause_duration is not None:
                    if isinstance(pause_duration, (int, float)) and pause_duration > 0:
                        await asyncio.to_thread(
                            self.redis.setex, "trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration))
                        )
                        logger.info(f"LLM set pause duration: {pause_duration}s")
                    else:
                        logger.warning(f"Invalid pause_duration_seconds: {pause_duration}")

                # Optional: LLM can set a global risk multiplier to scale all position sizes
                global_risk_mult = parsed.get("global_risk_multiplier")
                if global_risk_mult is not None:
                    if isinstance(global_risk_mult, (int, float)) and 0.0 <= global_risk_mult <= 1.0:
                        await asyncio.to_thread(
                            self.redis.setex, "trading:global_risk_multiplier", 3600, str(global_risk_mult)
                        )
                        logger.info(f"LLM set global risk multiplier: {global_risk_mult}")
                    else:
                        logger.warning(f"Invalid global_risk_multiplier: {global_risk_mult}")

                existing_coins = {c['symbol']: c for c in self.current_coins}
                for coin in deduped[: self.effective_max_coins]:
                    if coin['symbol'] in existing_coins and 'entry_time' in existing_coins[coin['symbol']]:
                        coin['entry_time'] = existing_coins[coin['symbol']]['entry_time']
                    else:
                        coin['entry_time'] = time.time()
                    # Preserve max_tenure_hours from existing coin if LLM didn't specify it
                    if 'max_tenure_hours' not in coin and coin['symbol'] in existing_coins and 'max_tenure_hours' in existing_coins[coin['symbol']]:
                        coin['max_tenure_hours'] = existing_coins[coin['symbol']]['max_tenure_hours']
                self.current_coins = deduped[: self.effective_max_coins]

                # If LLM explicitly chose zero coins, respect that and don't fall back to volume-based selection
                if not deduped or self.effective_max_coins == 0:
                    self.current_coins = []
                    self.effective_max_coins = 0
                    logger.info("LLM selected 0 coins – pausing trading until next evaluation.")

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
                # Apply minimum 24h volume filter if configured
                if settings.FALLBACK_MIN_24H_VOLUME > 0:
                    vol = _volume(sym)
                    if vol < settings.FALLBACK_MIN_24H_VOLUME:
                        continue
                min_cost = market_limits.get(sym, {}).get('min_cost', 0)
                if per_coin_budget >= min_cost:
                    if self._is_excluded(sym, default_tf):
                        continue
                    fallback_coins.append({"symbol": sym, "timeframe": default_tf})
                if len(fallback_coins) >= self.effective_max_coins:
                    break
            existing_coins = {c['symbol']: c for c in self.current_coins}
            for coin in fallback_coins:
                if coin['symbol'] in existing_coins and 'entry_time' in existing_coins[coin['symbol']]:
                    coin['entry_time'] = existing_coins[coin['symbol']]['entry_time']
                else:
                    coin['entry_time'] = time.time()
            self.current_coins = fallback_coins

        # Ensure all open positions remain in current_coins so they continue to be managed by the LLM strategy
        for symbol, pos in self.positions.items():
            if not any(entry["symbol"] == symbol for entry in self.current_coins):
                tf = pos.get("timeframe") or (settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h")
                self.current_coins.append({"symbol": symbol, "timeframe": tf})
                logger.info(f"Keeping {symbol} in current_coins due to open position (timeframe={tf})")

        # If trading is paused, keep ONLY coins with open positions.
        # The LLM may have just set pause_trading = true, so re-read Redis.
        paused_now = await asyncio.to_thread(self.redis.get, "trading:paused")
        if paused_now and paused_now == b"1":
            open_symbols = set(self.positions.keys())
            before_count = len(self.current_coins)
            self.current_coins = [c for c in self.current_coins if c["symbol"] in open_symbols]
            removed = before_count - len(self.current_coins)
            if removed > 0:
                logger.info(
                    "Trading paused: removed %d coin(s) without open positions. "
                    "Remaining: %s",
                    removed,
                    [c["symbol"] for c in self.current_coins],
                )

        # Update coin tenure tracking
        now_ts = time.time()
        new_symbols = {entry["symbol"] for entry in self.current_coins}
        for sym in new_symbols:
            if sym not in self._coin_first_seen:
                self._coin_first_seen[sym] = now_ts
        for sym in list(self._coin_first_seen.keys()):
            if sym not in new_symbols:
                del self._coin_first_seen[sym]

        # Trigger immediate backfill for newly selected coins
        old_symbols = {entry["symbol"] for entry in old_coins}
        for entry in self.current_coins:
            if entry["symbol"] not in old_symbols:
                sym = entry["symbol"]
                tf = entry["timeframe"]
                logger.info(f"Triggering immediate backfill for newly selected coin {sym} ({tf})")
                asyncio.create_task(self._backfill_new_coin(sym, tf))

        # Also trigger immediate news fetch for newly selected coins
        if settings.NEWS_ENABLED:
            for sym in new_symbols:
                logger.info(f"Triggering immediate news fetch for newly selected coin {sym}")
                asyncio.create_task(self._fetch_and_store_news_for_symbol(sym))

        coin_labels = [f"{c['symbol']}({c['timeframe']})" for c in self.current_coins]
        logger.info(f"Selected coins: {coin_labels}")

        # Build a pause/resume message if the LLM provided a decision
        pause_msg = ""
        if isinstance(pause_trading, bool):
            if pause_trading:
                if trading_paused_bool:
                    pause_msg = "⏸️ LLM decided to keep trading paused"
                else:
                    pause_msg = "⏸️ LLM decided to pause trading"
            else:
                if trading_paused_bool:
                    pause_msg = "▶️ LLM decided to resume trading"
                else:
                    pause_msg = "▶️ LLM decided to keep trading active"
            if pause_reason:
                pause_msg += f" – {pause_reason}"

        # Include pause duration if set
        if pause_duration is not None and isinstance(pause_duration, (int, float)) and pause_duration > 0:
            minutes = pause_duration / 60
            if minutes >= 1:
                duration_str = f"{minutes:.0f} min"
            else:
                duration_str = f"{pause_duration:.0f}s"
            if pause_msg:
                pause_msg += f" (auto‑resume in {duration_str})"
            else:
                pause_msg = f"⏱️ LLM set pause duration: {duration_str}"

        if not self.current_coins:
            logger.warning("No coins selected after evaluation. Bot will idle until next cycle.")
            if self.notifier:
                msg = f"⚠️ No coins selected. Bot will idle.\n"
                msg += f"Balance: {base_balance:.2f} {self.base_currency}, "
                msg += f"Per-coin budget: {per_coin_budget:.2f}"
                if pause_msg:
                    msg = pause_msg + "\n" + msg
                await self.notifier.send_notification(
                    msg,
                    summary={
                        "action": "HOLD",
                        "reason": "No coins selected",
                        "base_balance": base_balance,
                        "per_coin_budget": per_coin_budget,
                        "pause_decision": pause_trading if isinstance(pause_trading, bool) else None,
                        "pause_reason": pause_reason,
                        "model_type": "mind",
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                    }
                )
        elif self.notifier:
            coin_reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
            if coin_reasoning:
                msg = f"🔄 Coins updated: {', '.join(coin_labels)}\n💡 {coin_reasoning}"
            else:
                msg = f"🔄 Coins updated: {', '.join(coin_labels)}"
            if pause_msg:
                msg = pause_msg + "\n" + msg
            await self.notifier.send_notification(
                msg,
                summary={
                    "action": "INFO",
                    "reason": "Coins updated",
                    "coins": [c["symbol"] for c in self.current_coins],
                    "coin_reasoning": coin_reasoning,
                    "pause_decision": pause_trading if isinstance(pause_trading, bool) else None,
                    "pause_reason": pause_reason,
                    "model_type": "mind",
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
            )

        await asyncio.to_thread(self.redis.set, last_key, now)

    async def _check_pause_resume_decision(self):
        """When trading is paused, ask the LLM whether to resume (lightweight)."""
        async with self._coin_reeval_lock:
            # Only run if actually paused
            paused_raw = await asyncio.to_thread(self.redis.get, "trading:paused")
            if not paused_raw or paused_raw != b"1":
                return

            # Only handle LLM-initiated pauses. Manual pauses are not subject to auto-resume logic.
            source_raw = await asyncio.to_thread(self.redis.get, "trading:pause_source")
            source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
            if source != "llm":
                logger.debug("Pause/resume check skipped: pause was not initiated by LLM (source=%s).", source or "unknown")
                return

            # Gather minimal market context
            fear_greed = await self._get_fear_greed_index()
            global_market = await self._fetch_global_market_data()
            altcoin_season = await self._fetch_altcoin_season_index()

            # Market breadth from Redis (already computed by background task)
            full_market_breadth = None
            try:
                raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
                if raw:
                    full_market_breadth = json.loads(raw)
            except Exception:
                pass
            market_breadth = getattr(self, '_market_breadth', None)

            # BTC price for context
            btc_price = None
            try:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, "BTC/USDT")
                btc_price = ticker.get("last")
            except Exception:
                pass

            # Current pause reason
            reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
            pause_reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

            # --- Consecutive "keep paused" counter ---
            keep_key = "trading:pause:keep_count"
            keep_count_raw = await asyncio.to_thread(self.redis.get, keep_key)
            try:
                keep_count = int(keep_count_raw) if keep_count_raw else 0
            except (ValueError, TypeError):
                keep_count = 0

            # Build a richer prompt with performance context
            perf = self._compute_performance_metrics()
            daily_pnl = perf["equity_curve"].get("daily_pnl", 0.0)
            total_pnl = perf["equity_curve"].get("total_pnl", 0.0)
            consecutive_losses = perf["equity_curve"].get("consecutive_losses", 0)
            drawdown_pct = perf["equity_curve"].get("drawdown_pct", 0.0)

            prompt_parts = [
                "Trading is currently paused.",
            ]
            if pause_reason:
                prompt_parts.append(f"Pause reason: {pause_reason}")
            prompt_parts.append(f"Account P&L: daily={daily_pnl:.4f}, total={total_pnl:.4f}, drawdown={drawdown_pct:.2f}%")
            if consecutive_losses > 0:
                prompt_parts.append(f"Consecutive losing trades: {consecutive_losses}")
            if fear_greed:
                prompt_parts.append(f"Fear & Greed Index: {fear_greed['value']} ({fear_greed['classification']})")
            if btc_price:
                prompt_parts.append(f"BTC/USDT price: {btc_price}")
            if market_breadth:
                prompt_parts.append(f"Market breadth (top coins): {market_breadth['positive_pct']}% positive")
            if full_market_breadth:
                prompt_parts.append(f"Full market breadth: {full_market_breadth['positive_pct']}% positive")
            if global_market and global_market.get("btc_dominance") is not None:
                prompt_parts.append(f"BTC dominance: {global_market['btc_dominance']}%")
            if altcoin_season:
                prompt_parts.append(f"Altcoin Season Index: {altcoin_season['value']} ({altcoin_season['description']})")

            # Check if this is a recent auto-resume situation
            last_auto_resume_raw = await asyncio.to_thread(self.redis.get, "trading:last_auto_resume")
            if last_auto_resume_raw:
                try:
                    last_auto_resume_ts = float(last_auto_resume_raw)
                    seconds_since = time.time() - last_auto_resume_ts
                    if seconds_since < 3600:  # within the last hour
                        minutes_since = seconds_since / 60
                        prompt_parts.append(
                            f"Trading was auto‑resumed {minutes_since:.1f} minutes ago. "
                            "Market conditions may not have changed significantly. "
                            "Only resume if there is clear, concrete improvement in the data above."
                        )
                except (ValueError, TypeError):
                    pass

            # --- Consecutive keep warning and recovery nudge ---
            max_keep = settings.PAUSE_MAX_CONSECUTIVE_KEEP
            if keep_count > 0:
                prompt_parts.append(
                    f"You have chosen to keep trading paused {keep_count} time(s) in a row. "
                    f"If you keep it paused {max_keep} times consecutively, the engine will "
                    f"force‑resume trading with a reduced global risk multiplier of "
                    f"{settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER} to attempt recovery."
                )

            prompt_parts.append(
                "If the account is in drawdown or has consecutive losses, consider resuming "
                "with a **reduced global risk multiplier** (e.g., 0.3–0.5) instead of staying "
                "paused indefinitely. This allows the bot to cautiously seek small profitable "
                "trades to recover, while limiting downside. You can provide an optional "
                "`global_risk_multiplier` field in your JSON response (0.0–1.0) to set the "
                "risk level upon resume. If you omit it, the current multiplier (or 1.0) will be used."
            )

            prompt = (
                "\n".join(prompt_parts)
                + "\n\nShould we resume trading now? Reply with a JSON object: "
                '{"resume_trading": true/false, "reason": "short explanation", '
                '"global_risk_multiplier": 0.0-1.0 (optional)}'
                + "\n\n**Important:** Only resume if you see specific, high‑confidence opportunities. "
                "If conditions are still poor, you may keep trading paused, but remember that "
                "staying paused forever prevents any recovery. A cautious resume with a low risk "
                "multiplier is often better than doing nothing."
            )

            pause_resume_complexity = self._compute_prompt_complexity(
                num_candidates=0,
                market_breadth=market_breadth,
                fear_greed=fear_greed,
                volatility_percentile=None,
                sentiment_trend_magnitude=None,
                conflicting_signals=False,
                is_critical=False,
            )
            effective_temp = self._get_effective_temperature("actuator", pause_resume_complexity)

            try:
                pause_result = await asyncio.to_thread(
                    get_cached_llm_response, compact_prompt(prompt), COMPACTED_SYSTEM_PROMPT, 120,
                    model_type="actuator",
                    temperature=effective_temp,
                )
                response = pause_result["response"]
                llm_provider = pause_result["provider"]
                llm_model = pause_result["model"]
                decision = json.loads(response)
            except Exception as e:
                logger.warning(f"Pause/resume LLM call failed: {e}")
                # Track consecutive failures in Redis
                fail_key = "trading:pause:llm_fail_count"
                current_fails = await asyncio.to_thread(self.redis.incr, fail_key)
                await asyncio.to_thread(self.redis.expire, fail_key, 3600)
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Could not reach LLM to decide pause/resume (failure #{current_fails}). "
                        f"Auto‑resume will be attempted after {MIN_LLM_PAUSE_DURATION}s if LLM stays silent.",
                        summary={"action": "INFO", "reason": "LLM pause-resume call failed"}
                    )
                # If we failed 3 times in a row, force‑resume (optional but safe)
                if current_fails >= 3:
                    # Double-check source before force-resuming
                    fail_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    if fail_source and (fail_source.decode() if isinstance(fail_source, bytes) else fail_source) != "llm":
                        logger.warning("Force-resume on LLM failure skipped: pause source is not LLM.")
                        return
                    pause_keys = [
                        "trading:paused",
                        "trading:pause_source",
                        "trading:pause_start",
                        "trading:pause_duration",
                        "trading:pause_reason",
                        "trading:llm_pause_time",
                    ]
                    for key in pause_keys:
                        await asyncio.to_thread(self.redis.delete, key)
                    await asyncio.to_thread(self.redis.delete, fail_key)
                    # --- Also reset keep counter and set force‑resume risk multiplier ---
                    await asyncio.to_thread(self.redis.delete, keep_key)
                    await asyncio.to_thread(
                        self.redis.setex, "trading:global_risk_multiplier", 3600,
                        str(settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER)
                    )
                    self._reeval_trigger.set()
                    if self.notifier:
                        await self.notifier.send_notification(
                            "▶️ Trading auto‑resumed because LLM could not be reached for pause decision. "
                            f"Global risk multiplier set to {settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER}.",
                            summary={"action": "RESUME", "reason": "LLM pause-resume failures exceeded limit"}
                        )
                return

            resume_trading = decision.get("resume_trading")
            reason = decision.get("reason", "")

            if resume_trading is True:
                # Source is already verified as "llm" by the early check at the top of this method.

                # Check minimum LLM pause duration
                llm_pause_time_raw = await asyncio.to_thread(self.redis.get, "trading:llm_pause_time")
                if llm_pause_time_raw:
                    try:
                        llm_pause_time = float(llm_pause_time_raw)
                        if time.time() - llm_pause_time < MIN_LLM_PAUSE_DURATION:
                            remaining = MIN_LLM_PAUSE_DURATION - (time.time() - llm_pause_time)
                            logger.info(f"Ignoring LLM resume request: minimum pause duration not elapsed ({remaining:.0f}s remaining).")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⏸️ LLM resume request ignored: minimum pause duration "
                                    f"({MIN_LLM_PAUSE_DURATION}s) not yet elapsed ({remaining:.0f}s remaining).",
                                    summary={"action": "RESUME", "reason": f"LLM resume blocked by minimum pause duration ({MIN_LLM_PAUSE_DURATION}s)", "model_type": "actuator"}
                                )
                            return
                    except (ValueError, TypeError):
                        pass

                # --- Apply optional global_risk_multiplier from LLM ---
                global_mult_raw = decision.get("global_risk_multiplier")
                applied_mult = None
                if global_mult_raw is not None:
                    try:
                        mult_val = float(global_mult_raw)
                        if 0.0 <= mult_val <= 1.0:
                            await asyncio.to_thread(
                                self.redis.setex, "trading:global_risk_multiplier", 3600, str(mult_val)
                            )
                            logger.info(f"LLM set global risk multiplier on resume: {mult_val}")
                            applied_mult = mult_val
                        else:
                            logger.warning(f"Invalid global_risk_multiplier in resume decision: {global_mult_raw}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid global_risk_multiplier value: {global_mult_raw}")

                # Resume trading
                pause_keys = [
                    "trading:paused",
                    "trading:pause_source",
                    "trading:pause_start",
                    "trading:pause_duration",
                    "trading:pause_reason",
                    "trading:llm_pause_time",
                ]
                for key in pause_keys:
                    await asyncio.to_thread(self.redis.delete, key)
                # Reset the keep counter
                await asyncio.to_thread(self.redis.delete, keep_key)
                logger.info("LLM decided to resume trading.")
                self._reeval_trigger.set()
                if self.notifier:
                    reason_text = f" – {reason}" if reason else ""
                    mult_text = f" (risk multiplier: {applied_mult})" if applied_mult is not None else ""
                    await self.notifier.send_notification(
                        f"▶️ Trading resumed by LLM decision{reason_text}{mult_text}",
                        summary={"action": "RESUME", "reason": f"LLM resume request: {reason}" if reason else "LLM resume request", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                    )
            elif resume_trading is False:
                # LLM wants to stay paused – optionally update reason
                if reason:
                    await asyncio.to_thread(self.redis.set, "trading:pause_reason", reason)

                # Increment consecutive keep counter
                new_keep_count = await asyncio.to_thread(self.redis.incr, keep_key)
                # Set a TTL so it doesn't persist forever (e.g., 24h)
                await asyncio.to_thread(self.redis.expire, keep_key, 86400)

                if new_keep_count >= settings.PAUSE_MAX_CONSECUTIVE_KEEP:
                    # Double-check that the pause is still LLM-initiated (should always be true here)
                    current_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    if current_source and (current_source.decode() if isinstance(current_source, bytes) else current_source) != "llm":
                        logger.warning("Force-resume skipped: pause source changed to non-LLM.")
                        return
                    logger.warning(
                        f"LLM kept trading paused {new_keep_count} times consecutively – "
                        f"forcing resume with risk multiplier {settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER}."
                    )
                    # Force resume
                    pause_keys = [
                        "trading:paused",
                        "trading:pause_source",
                        "trading:pause_start",
                        "trading:pause_duration",
                        "trading:pause_reason",
                        "trading:llm_pause_time",
                    ]
                    for key in pause_keys:
                        await asyncio.to_thread(self.redis.delete, key)
                    await asyncio.to_thread(self.redis.delete, keep_key)
                    await asyncio.to_thread(
                        self.redis.setex, "trading:global_risk_multiplier", 3600,
                        str(settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER)
                    )
                    self._reeval_trigger.set()
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"▶️ Trading force‑resumed after {new_keep_count} consecutive pauses. "
                            f"Global risk multiplier set to {settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER}.",
                            summary={
                                "action": "RESUME",
                                "reason": f"Force resume after {new_keep_count} consecutive keep-paused decisions",
                                "model_type": "actuator",
                            }
                        )
                else:
                    logger.info(f"LLM decided to keep trading paused. Reason: {reason} (keep count: {new_keep_count}/{settings.PAUSE_MAX_CONSECUTIVE_KEEP})")
                    if self.notifier:
                        reason_text = f" – {reason}" if reason else ""
                        await self.notifier.send_notification(
                            f"⏸️ LLM decided to keep trading paused{reason_text} "
                            f"({new_keep_count}/{settings.PAUSE_MAX_CONSECUTIVE_KEEP} consecutive keeps)",
                            summary={"action": "PAUSE", "reason": f"LLM keep paused: {reason}" if reason else "LLM keep paused", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                        )
            else:
                logger.warning(f"Invalid resume_trading value in LLM response: {resume_trading}")

    async def _process_coin(self, coin_entry: Dict[str, str], trading_paused: bool = False):
        """Fetch market data, get LLM strategy, validate, and execute."""
        symbol = coin_entry["symbol"]
        assigned_tf = coin_entry["timeframe"]
        tf_seconds = self._timeframe_to_seconds(assigned_tf)

        # --- Maximum coin tenure (per-coin, set by LLM) ---
        max_tenure_hours = coin_entry.get('max_tenure_hours')
        if max_tenure_hours is not None and max_tenure_hours > 0 and 'entry_time' in coin_entry:
            tenure_seconds = max_tenure_hours * 3600
            if time.time() - coin_entry['entry_time'] > tenure_seconds:
                logger.info(f"Max tenure reached for {symbol} ({max_tenure_hours:.1f}h), forcing sell")
                signal = Signal(action="SELL", confidence=1.0, reasoning="Max coin tenure reached")
                await self._execute_signal(symbol, signal, exit_reason="max_tenure")
                return

        # --- Cooldown after a losing trade (LLM-defined) ---
        # Only apply cooldown if there is NO open position for this symbol.
        # An open position must be managed regardless of cooldown.
        if symbol not in self.positions:
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

        # If trading is paused and we have no open position, skip entirely
        if trading_paused and symbol not in self.positions:
            logger.debug(f"Skipping {symbol}: trading paused and no open position.")
            return

        # --- Max hold expired flag ---
        max_hold_expired = False
        max_hold_expired_count = 0
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.get("_max_hold_expired"):
                max_hold_expired = True
                max_hold_expired_count = pos.get("_max_hold_expired_count", 1)

        # --- Stop-loss triggered flag ---
        stop_loss_triggered = False
        stop_loss_review_count = 0
        # --- Take-profit triggered flag ---
        take_profit_triggered = False
        take_profit_review_count = 0
        # --- Partial TP and dust sweep triggers ---
        partial_tp_triggered = False
        partial_tp_review_count = 0
        partial_tp_triggered_levels = []
        dust_sweep_triggered = False
        dust_sweep_review_count = 0
        if symbol in self.positions:
            pos = self.positions[symbol]
            stop_loss_triggered = pos.get("_stop_loss_triggered", False)
            stop_loss_review_count = pos.get("_stop_loss_review_count", 0)
            take_profit_triggered = pos.get("_take_profit_triggered", False)
            take_profit_review_count = pos.get("_take_profit_review_count", 0)
            partial_tp_triggered = pos.get("_partial_tp_triggered", False) or pos.get("_partial_tp_triggered_single", False)
            partial_tp_review_count = pos.get("_partial_tp_review_count", 0) or pos.get("_partial_tp_single_review_count", 0)
            partial_tp_triggered_levels = pos.get("_partial_tp_triggered_levels", [])
            dust_sweep_triggered = pos.get("_dust_sweep_triggered", False)
            dust_sweep_review_count = pos.get("_dust_sweep_review_count", 0)

        try:
            ticker = self.ws_manager.get_ticker(symbol)
            if ticker is None:
                async with self._exchange_semaphore:
                    ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            current_price = ticker['last']
            # Relative strength vs BTC for this coin
            rel_strength_btc = None
            try:
                btc_ticker = self.ws_manager.get_ticker("BTC/USDT")
                if btc_ticker is None:
                    async with self._exchange_semaphore:
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
            order_book = self.ws_manager.get_order_book(symbol)
            if order_book is None:
                async with self._exchange_semaphore:
                    order_book = await asyncio.to_thread(get_order_book, self.exchange, symbol, 20)
            # Fetch recent trades for micro-momentum and liquidity assessment
            recent_trades_raw = self.ws_manager.get_trades(symbol)
            if not recent_trades_raw:
                try:
                    async with self._exchange_semaphore:
                        recent_trades_raw = await asyncio.to_thread(
                            self.exchange.fetch_trades, symbol, limit=20
                        )
                except Exception as e:
                    logger.debug(f"Could not fetch recent trades for {symbol}: {e}")

            # Compute Cumulative Volume Delta (CVD) from recent trades
            cvd = None
            cvd_normalized = None
            if recent_trades_raw:
                buy_vol = sum(t.get('amount', 0) for t in recent_trades_raw if t.get('side') == 'buy')
                sell_vol = sum(t.get('amount', 0) for t in recent_trades_raw if t.get('side') == 'sell')
                total_vol = buy_vol + sell_vol
                cvd = round(buy_vol - sell_vol, 6)
                cvd_normalized = round(cvd / total_vol, 4) if total_vol > 0 else None

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
                    async with self._exchange_semaphore:
                        ohlcv_data = await asyncio.to_thread(
                            get_multi_timeframe_ohlcv, self.exchange, symbol, settings.OHLCV_TIMEFRAMES, limit=100
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
            # Build indicator config from position or defaults
            ind_cfg = self.positions.get(symbol, {}).get('indicator_config') if symbol in self.positions else None

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
                    ind = compute_all_indicators(candles, config=ind_cfg)
                    multi_tf_indicators[tf] = ind
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

            # ATR Percentile (volatility context)
            atr_percentile = None
            if atr is not None and atr > 0:
                atr_percentile_key = f"atr_percentile:{symbol}"
                try:
                    stored_atr = await asyncio.to_thread(self.redis.get, atr_percentile_key)
                    if stored_atr:
                        atr_history = json.loads(stored_atr)
                    else:
                        atr_history = []
                    atr_history.append(atr)
                    atr_history = atr_history[-100:]
                    await asyncio.to_thread(self.redis.setex, atr_percentile_key, 7 * 24 * 3600, json.dumps(atr_history))
                    if len(atr_history) >= 5:
                        sorted_atr = sorted(atr_history)
                        rank = sum(1 for v in sorted_atr if v <= atr)
                        atr_percentile = round(rank / len(sorted_atr) * 100, 1)
                except Exception as e:
                    logger.debug(f"ATR percentile computation failed for {symbol}: {e}")

            # --- Market regime classification (enhanced) ---
            market_regime = self._classify_market_regime(
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                ema_9=ema_9,
                ema_21=ema_21,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle,
                atr=atr,
                atr_percentile=atr_percentile,
                current_price=ticker['last'],
            )

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

                    # --- Gap check: discard historical data if gaps exist ---
                    if len(historical_ohlcv) >= 2:
                        interval_ms = self._timeframe_to_ms(assigned_tf)
                        timestamps = [c[0] for c in historical_ohlcv]
                        has_gap = False
                        for i in range(len(timestamps) - 1):
                            if timestamps[i+1] - timestamps[i] > interval_ms * 1.5:
                                has_gap = True
                                break
                        if has_gap:
                            logger.warning(
                                f"Historical OHLCV for {symbol} {assigned_tf} contains gaps; "
                                f"skipping backtest data for this cycle."
                            )
                            historical_ohlcv = None
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
            order_book_pressure_trend = None
            depth_imbalances = None
            order_book_slope = None
            mid_price_bias = None
            mid = None

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

                # Order book pressure trend (change since last cycle)
                if order_book_pressure is not None:
                    pressure_key = f"ob_pressure:prev:{symbol}"
                    prev_pressure_raw = await asyncio.to_thread(self.redis.get, pressure_key)
                    if prev_pressure_raw is not None:
                        prev_pressure = float(prev_pressure_raw)
                        order_book_pressure_trend = round(order_book_pressure - prev_pressure, 4)
                    await asyncio.to_thread(self.redis.setex, pressure_key, 3600, str(order_book_pressure))

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

            # --- Market impact estimate (price movement per unit of volume) ---
            market_impact_score = None
            if depth_profile and mid is not None and mid > 0:
                ask_vol_01 = depth_profile.get("0.1%", {}).get("ask_volume", 0)
                if ask_vol_01 > 0:
                    price_move_per_unit = (mid * 0.001) / ask_vol_01
                    impact_pct = (price_move_per_unit / mid) * 100 if mid > 0 else 0
                    if impact_pct <= 0.001:
                        market_impact_score = 1.0
                    elif impact_pct >= 0.1:
                        market_impact_score = 0.0
                    else:
                        market_impact_score = round(max(0.0, min(1.0, 1.0 - (impact_pct - 0.001) / 0.099)), 3)

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

                # Composite score (with optional market impact component)
                if market_impact_score is not None:
                    scalping_score = round(0.20 * spread_score + 0.20 * depth_score + 0.20 * freq_score + 0.20 * vol_score + 0.20 * market_impact_score, 3)
                else:
                    scalping_score = round(0.25 * spread_score + 0.25 * depth_score + 0.25 * freq_score + 0.25 * vol_score, 3)

            # Pre-computed slippage estimate for per-coin budget order size
            estimated_slippage_pct = None
            if asks and per_coin_budget > 0:
                desired_quote = per_coin_budget
                remaining_slip = desired_quote
                total_cost_slip = 0.0
                total_base_slip = 0.0
                for ask in asks:
                    price_level = ask[0]
                    volume = ask[1]
                    cost_at_level = price_level * volume
                    if cost_at_level >= remaining_slip:
                        base_filled = remaining_slip / price_level
                        total_cost_slip += remaining_slip
                        total_base_slip += base_filled
                        remaining_slip = 0
                        break
                    else:
                        total_cost_slip += cost_at_level
                        total_base_slip += volume
                        remaining_slip -= cost_at_level
                if total_base_slip > 0 and asks[0][0] > 0:
                    avg_fill_price = total_cost_slip / total_base_slip
                    best_ask_price = asks[0][0]
                    estimated_slippage_pct = round((avg_fill_price - best_ask_price) / best_ask_price * 100, 4)

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
            altcoin_season = await self._fetch_altcoin_season_index()

            # Fetch full market breadth from Redis (computed by background task)
            full_market_breadth = None
            try:
                full_breadth_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
                if full_breadth_raw:
                    full_market_breadth = json.loads(full_breadth_raw)
            except Exception:
                pass
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

            # Fetch current global risk multiplier so the LLM can adjust position sizing
            global_risk_mult = None
            global_mult_raw = await asyncio.to_thread(self.redis.get, "trading:global_risk_multiplier")
            if global_mult_raw:
                try:
                    global_risk_mult = float(global_mult_raw)
                except (ValueError, TypeError):
                    pass

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
                full_market_breadth=full_market_breadth,
                depth_trend=depth_trend,
                parabolic_sar=parabolic_sar,
                keltner_channels=keltner_channels,
                pivot_points=pivot_points,
                donchian_channels=donchian_channels,
                btc_dominance=global_market.get("btc_dominance") if global_market else None,
                total_market_cap=global_market if global_market else None,
                altcoin_season=altcoin_season,
                cvd=cvd,
                cvd_normalized=cvd_normalized,
                order_book_pressure_trend=order_book_pressure_trend,
                estimated_slippage_pct=estimated_slippage_pct,
                atr_percentile=atr_percentile,
                market_impact_score=market_impact_score,
                global_risk_multiplier=global_risk_mult,
                trading_paused=trading_paused,
                max_hold_expired=max_hold_expired,
                max_hold_expired_count=max_hold_expired_count,
                stop_loss_triggered=stop_loss_triggered,
                stop_loss_review_count=stop_loss_review_count,
                take_profit_triggered=take_profit_triggered,
                take_profit_review_count=take_profit_review_count,
                partial_tp_triggered=partial_tp_triggered,
                partial_tp_review_count=partial_tp_review_count,
                partial_tp_triggered_levels=partial_tp_triggered_levels if partial_tp_triggered_levels else None,
                dust_sweep_triggered=dust_sweep_triggered,
                dust_sweep_review_count=dust_sweep_review_count,
                max_partial_tp_reviews=settings.MAX_PARTIAL_TP_REVIEWS,
                max_dust_sweep_reviews=settings.MAX_DUST_SWEEP_REVIEWS,
            )
            logger.debug(f"LLM prompt for {symbol}: {len(prompt)} chars")
            # Build a market snapshot dict for caching (per-coin)
            market_snapshot = {
                "symbol": symbol,
                "ticker": ticker,
                "order_book": order_book,
                "balance": balance,
                "open_positions": open_positions,
                "per_coin_budget": per_coin_budget,
                "max_coins": self.effective_max_coins,
                "performance": perf,
                "ohlcv_data": ohlcv_data,
                "assigned_timeframe": assigned_tf,
                "atr": atr,
                "atr_multi_tf": atr_multi_tf,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "ema_9": ema_9,
                "ema_21": ema_21,
                "stochastic_k": stochastic_k,
                "stochastic_d": stochastic_d,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "obv": obv,
                "mfi": mfi,
                "cci": cci,
                "williams_r": williams_r,
                "ichimoku": ichimoku,
                "donchian_channels": donchian_channels,
                "order_book_imbalance": order_book_imbalance,
                "spread_pct": spread_pct,
                "bid_wall_volume": bid_wall_volume,
                "ask_wall_volume": ask_wall_volume,
                "order_book_pressure": order_book_pressure,
                "depth_imbalances": depth_imbalances,
                "order_book_slope": order_book_slope,
                "mid_price_bias": mid_price_bias,
                "depth_profile": depth_profile,
                "fee_rate": fee_rate,
                "drawdown_pct": perf.get("equity_curve", {}).get("drawdown_pct"),
                "raw_candles": raw_candles,
                "recent_trades": recent_trades_summary,
                "historical_ohlcv": historical_ohlcv,
                "min_order_amount": min_order_amount,
                "min_order_cost": min_order_cost,
                "all_coins": self.current_coins,
                "past_trades": past_trades,
                "aggregate_sentiment": aggregate_sentiment,
                "cycle_spent": self._cycle_spent,
                "remaining_balance": remaining,
                "market_regime": market_regime,
                "recent_trades_data": recent_trades_raw,
                "multi_tf_raw_candles": multi_tf_raw_candles,
                "multi_tf_indicators": multi_tf_indicators,
                "scalping_feasibility_score": scalping_score,
                "fear_greed_index": fear_greed,
                "relative_strength_btc": rel_strength_btc,
                "vwap": vwap,
                "vwap_multi_tf": vwap_multi_tf,
                "session_info": session_info,
                "sentiment_trend": sentiment_trend_val,
                "volume_trend": volume_trend_val,
                "market_breadth": getattr(self, '_market_breadth', None),
                "full_market_breadth": full_market_breadth,
                "depth_trend": depth_trend,
                "parabolic_sar": parabolic_sar,
                "keltner_channels": keltner_channels,
                "pivot_points": pivot_points,
                "btc_dominance": global_market.get("btc_dominance") if global_market else None,
                "total_market_cap": global_market if global_market else None,
                "altcoin_season": altcoin_season,
                "cvd": cvd,
                "cvd_normalized": cvd_normalized,
                "order_book_pressure_trend": order_book_pressure_trend,
                "estimated_slippage_pct": estimated_slippage_pct,
                "atr_percentile": atr_percentile,
                "market_impact_score": market_impact_score,
                "global_risk_multiplier": global_risk_mult,
                "trading_paused": trading_paused,
            }
            market_hash = compute_market_hash(market_snapshot)
            # Determine whether we even need to call the LLM, and if so which model to use
            is_critical = max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered
            has_position = symbol in self.positions

            if self._should_skip_llm_eval(
                symbol=symbol,
                current_price=current_price,
                atr=atr,
                rsi=rsi,
                macd_hist=macd_hist,
                atr_percentile=atr_percentile,
                market_regime=market_regime,
                sentiment_trend_val=sentiment_trend_val,
                timeframe_seconds=tf_seconds,
                has_position=has_position,
                is_critical=is_critical,
            ):
                logger.debug(f"Skipping LLM for {symbol}: market unchanged, no strong signals.")
                # Update snapshot but no LLM call – assume HOLD
                self._update_last_eval_snapshot(symbol, current_price, rsi, macd_hist)
                return

            strategy_model_type = self._choose_model_tier(
                atr_percentile=atr_percentile,
                market_regime=market_regime,
                sentiment_trend_val=sentiment_trend_val,
                rsi=rsi,
                macd_hist=macd_hist,
                macd=macd,
                macd_signal=macd_signal,
                is_critical=is_critical,
            )

            # Compute prompt complexity for temperature selection
            _conflicting = False
            if rsi is not None and macd_hist is not None:
                if (rsi < 30 and macd_hist < 0) or (rsi > 70 and macd_hist > 0):
                    _conflicting = True
            strategy_complexity = self._compute_prompt_complexity(
                num_candidates=len(self.current_coins),
                market_breadth=getattr(self, '_market_breadth', None),
                fear_greed=fear_greed,
                volatility_percentile=atr_percentile,
                sentiment_trend_magnitude=abs(sentiment_trend_val) if sentiment_trend_val is not None else None,
                conflicting_signals=_conflicting,
                is_critical=is_critical,
            )
            effective_temp = self._get_effective_temperature(strategy_model_type, strategy_complexity)

            try:
                strategy_result = await asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(prompt),
                    COMPACTED_SYSTEM_PROMPT,
                    60,
                    market_hash=market_hash,
                    model_type=strategy_model_type,
                    temperature=effective_temp,
                )
                response = strategy_result["response"]
                llm_provider = strategy_result["provider"]
                llm_model = strategy_result["model"]
                # Update snapshot after a real LLM call
                self._update_last_eval_snapshot(symbol, current_price, rsi, macd_hist)
            except asyncio.TimeoutError:
                logger.warning(f"LLM strategy call timed out for {symbol}.")
                # If a critical decision is pending, force a SELL immediately to protect capital.
                if max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered:
                    reason = "LLM timeout"
                    if max_hold_expired:
                        reason = "Max hold expired, LLM timeout"
                    elif stop_loss_triggered:
                        reason = "Stop-loss triggered, LLM timeout"
                    elif take_profit_triggered:
                        reason = "Take-profit triggered, LLM timeout"
                    elif partial_tp_triggered:
                        reason = "Partial TP triggered, LLM timeout"
                    elif dust_sweep_triggered:
                        reason = "Dust sweep triggered, LLM timeout"
                    logger.warning(f"Forcing SELL for {symbol} due to {reason}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏱️ LLM timeout for {symbol} with critical flag – forcing SELL.",
                            summary={"symbol": symbol, "action": "SELL", "reason": reason, "model_type": strategy_model_type}
                        )
                    await self._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning=reason),
                        exit_reason=reason.replace(" ", "_").lower()
                    )
                    return
                # No critical flag – safe to skip this cycle.
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⏱️ LLM timeout for {symbol}, skipping.",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "LLM timeout",
                            "model_type": strategy_model_type,
                        }
                    )
                return

            if response is None:
                logger.warning(f"LLM returned None for {symbol}.")
                # If a critical decision is pending, force a SELL immediately to protect capital.
                if max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered:
                    reason = "LLM returned None"
                    if max_hold_expired:
                        reason = "Max hold expired, LLM returned None"
                    elif stop_loss_triggered:
                        reason = "Stop-loss triggered, LLM returned None"
                    elif take_profit_triggered:
                        reason = "Take-profit triggered, LLM returned None"
                    elif partial_tp_triggered:
                        reason = "Partial TP triggered, LLM returned None"
                    elif dust_sweep_triggered:
                        reason = "Dust sweep triggered, LLM returned None"
                    logger.warning(f"Forcing SELL for {symbol} due to {reason}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ LLM returned empty response for {symbol} with critical flag – forcing SELL.",
                            summary={"symbol": symbol, "action": "SELL", "reason": reason, "model_type": strategy_model_type}
                        )
                    await self._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning=reason),
                        exit_reason=reason.replace(" ", "_").lower()
                    )
                    return
                # No critical flag – safe to skip this cycle.
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ LLM returned empty response for {symbol}, skipping.",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "LLM returned None",
                            "model_type": strategy_model_type,
                        }
                    )
                return

            try:
                strategy = create_strategy_from_llm(response)
            except ValueError as e:
                logger.warning(f"LLM response parse failed for {symbol}: {e}. Retrying with correction prompt.")
                correction_prompt = (
                    "Your previous response was not valid JSON. "
                    "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                    "Here is the original request:\n\n" + prompt
                )
                try:
                    response2 = await asyncio.to_thread(
                        get_cached_llm_response, compact_prompt(correction_prompt), COMPACTED_SYSTEM_PROMPT, 30,
                        model_type="actuator",
                        temperature=effective_temp,
                    )
                    # Update snapshot after retry call
                    self._update_last_eval_snapshot(symbol, current_price, rsi, macd_hist)
                    strategy = create_strategy_from_llm(response2)
                except Exception as e2:
                    logger.error(f"LLM response still invalid after retry for {symbol}: {e2}")
                    strategy = LLMStrategy(Signal(action="HOLD", confidence=0.0, reasoning="Failed to parse LLM response after retry"))
            signal = strategy.generate_signal({})
            signal.model_type = strategy_model_type
            signal.llm_provider = llm_provider
            signal.llm_model = llm_model
            current_price = ticker['last']
            validated = validate_signal(
                signal,
                fee_rate=fee_rate,
                atr=atr,
                price=current_price,
                spread_pct=spread_pct,
                timeframe_seconds=tf_seconds,
            )
            validated.model_type = getattr(signal, 'model_type', None)

            # If the LLM produced a BUY/SELL but the validator rejected it due to a parameter
            # error, give the LLM one chance to fix its own mistake.
            if signal.action != "HOLD" and validated.action == "HOLD":
                logger.warning(
                    f"LLM signal for {symbol} rejected by validator. "
                    f"Original action={signal.action}, confidence={signal.confidence}, "
                    f"reasoning={validated.reasoning}. Raw LLM response: {response}"
                )
                # Only retry if the rejection looks like a parameter mistake (not a market condition)
                if any(kw in validated.reasoning.lower() for kw in [
                    "stop_loss_pct", "take_profit_pct", "trailing_stop_distance",
                    "invalid", "missing", "must be"
                ]):
                    correction_prompt = (
                        f"Your previous trading decision for {symbol} was rejected because: "
                        f"{validated.reasoning}\n"
                        "Please fix the error and output a corrected JSON decision. "
                        "All other market data remains the same."
                    )
                    try:
                        correction_result = await asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(correction_prompt),
                            COMPACTED_SYSTEM_PROMPT,
                            30,
                            model_type="actuator",
                            temperature=effective_temp,
                        )
                        corrected_response = correction_result["response"]
                        llm_provider = correction_result["provider"]
                        llm_model = correction_result["model"]
                        corrected_signal = create_strategy_from_llm(corrected_response).generate_signal({})
                        validated = validate_signal(
                            corrected_signal,
                            fee_rate=fee_rate,
                            atr=atr,
                            price=current_price,
                            spread_pct=spread_pct,
                            timeframe_seconds=tf_seconds,
                        )
                        if validated.action != "HOLD":
                            logger.info(f"LLM corrected its signal for {symbol}: {validated.action}")
                            signal = corrected_signal  # use the corrected signal for logging
                        else:
                            logger.warning(f"LLM correction for {symbol} still invalid: {validated.reasoning}")
                    except Exception as e:
                        logger.error(f"LLM correction call failed for {symbol}: {e}")

            # Log and notify the decision
            logger.info(f"Decision for {symbol}: {validated.action} (confidence: {validated.confidence:.2f})")
            if self.notifier and not (trading_paused and validated.action == "BUY"):
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(validated.action, "❓")
                # Build a short indicator summary
                ind_parts = []
                if rsi is not None:
                    ind_parts.append(f"RSI={rsi:.1f}")
                if macd is not None and macd_signal is not None:
                    ind_parts.append(f"MACD={macd:.4f}/{macd_signal:.4f}")
                    if macd_hist is not None:
                        ind_parts.append(f"Hist={macd_hist:.4f}")
                if bb_upper is not None:
                    ind_parts.append(f"BB={bb_lower:.2f}/{bb_middle:.2f}/{bb_upper:.2f}")
                if ema_9 is not None and ema_21 is not None:
                    ind_parts.append(f"EMA9/21={ema_9:.2f}/{ema_21:.2f}")
                if stochastic_k is not None:
                    ind_parts.append(f"StochK={stochastic_k:.1f}")
                    if stochastic_d is not None:
                        ind_parts.append(f"StochD={stochastic_d:.1f}")
                if adx is not None:
                    ind_parts.append(f"ADX={adx:.1f}")
                    if plus_di is not None and minus_di is not None:
                        ind_parts.append(f"+DI={plus_di:.1f}/-DI={minus_di:.1f}")
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
                    ind_parts.append(f"Cloud={ichimoku['cloud_bottom']:.2f}-{ichimoku['cloud_top']:.2f}")
                if donchian_channels is not None:
                    ind_parts.append(f"Donch={donchian_channels['lower']:.2f}/{donchian_channels['middle']:.2f}/{donchian_channels['upper']:.2f}")
                if vwap is not None:
                    ind_parts.append(f"VWAP={vwap:.4f}")
                if parabolic_sar is not None:
                    ind_parts.append(f"SAR={parabolic_sar:.4f}")
                if keltner_channels is not None:
                    ind_parts.append(f"Kelt={keltner_channels['lower']:.4f}/{keltner_channels['middle']:.4f}/{keltner_channels['upper']:.4f}")
                if pivot_points is not None:
                    ind_parts.append(f"Pivot={pivot_points['pivot']:.4f} R1={pivot_points['r1']:.4f} S1={pivot_points['s1']:.4f}")
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
                    "model_type": getattr(validated, 'model_type', None),
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
                await self.notifier.send_notification(msg, summary=decision_summary)

            # --- LLM‑controlled trade filters ---
            params = signal.strategy_params or {}

            # --- Handle max‑hold‑expired LLM decision ---
            if max_hold_expired and signal.action == "HOLD":
                new_max_hold = params.get("max_hold_time_seconds") if params else None
                if new_max_hold is not None and new_max_hold > 0:
                    # LLM decided to extend – update position and clear the flag
                    logger.info(f"LLM extended max hold time for {symbol} to {new_max_hold}s")
                    if symbol in self.positions:
                        self.positions[symbol]["max_hold_time_seconds"] = new_max_hold
                        # IMPORTANT: Reset the entry timestamp so the new hold period starts now
                        self.positions[symbol]["timestamp"] = int(time.time() * 1000)
                        self.positions[symbol].pop("_max_hold_expired", None)
                        self.positions[symbol].pop("_max_hold_expired_count", None)
                    # Also update current_coins entry_time for consistency
                    for coin_entry in self.current_coins:
                        if coin_entry["symbol"] == symbol:
                            coin_entry["entry_time"] = time.time()
                            break
                    # Notify the user about the extension
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏰ Max hold time for {symbol} extended to {new_max_hold}s by LLM.\n"
                            f"Reasoning: {validated.reasoning}",
                            summary={
                                "symbol": symbol,
                                "action": "HOLD",
                                "reason": validated.reasoning,
                                "new_max_hold_seconds": new_max_hold,
                                "model_type": strategy_model_type,
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                            }
                        )
                    # Also apply any other updated parameters from the LLM
                    self._update_position_params(
                        symbol,
                        params,
                        signal.indicator_config,
                        assigned_tf,
                        current_price,
                        atr,
                    )
                else:
                    # LLM did not provide a new max_hold_time_seconds → treat as SELL
                    logger.warning(
                        f"LLM returned HOLD without new max_hold_time_seconds for {symbol} "
                        f"after max hold expiry – forcing SELL."
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏰ LLM did not extend hold time for {symbol} – closing position.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Max hold expired, LLM did not extend",
                                "exit_reason": "max_hold_time_llm_no_extend",
                                "model_type": strategy_model_type,
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                            }
                        )
                    await self._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning="Max hold expired, LLM did not extend"),
                        exit_reason="max_hold_time_llm_no_extend"
                    )
                    return   # stop further processing for this coin

            # --- Handle stop-loss-triggered LLM decision ---
            if stop_loss_triggered and signal.action == "HOLD":
                # LLM decided to keep holding – must provide a new stop-loss
                new_params = signal.strategy_params or {}
                new_stop_method = new_params.get("stop_loss_method", "fixed")
                new_stop_pct = None
                if new_stop_method == "atr_multiple" and atr is not None and atr > 0:
                    atr_mult = new_params.get("stop_loss_atr_multiple")
                    if atr_mult is not None:
                        new_stop_pct = (atr_mult * atr) / current_price
                else:
                    new_stop_pct = new_params.get("stop_loss_pct")

                if new_stop_pct is not None and new_stop_pct > 0:
                    # LLM provided a new stop – update position and clear the trigger flag
                    logger.info(
                        f"LLM decided to hold {symbol} after stop-loss trigger, "
                        f"new stop_loss_pct={new_stop_pct:.4%}"
                    )
                    if symbol in self.positions:
                        self.positions[symbol]["stop_loss"] = current_price * (1 - new_stop_pct)
                        self.positions[symbol].pop("_stop_loss_triggered", None)
                        self.positions[symbol].pop("_stop_loss_review_count", None)
                        # Also apply any other updated parameters from the LLM
                        self._update_position_params(
                            symbol,
                            new_params,
                            signal.indicator_config,
                            assigned_tf,
                            current_price,
                            atr,
                        )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"🔄 {symbol}: LLM adjusted stop-loss to {new_stop_pct:.4%} – holding.\n"
                            f"Reasoning: {validated.reasoning}",
                            summary={
                                "symbol": symbol,
                                "action": "HOLD",
                                "reason": validated.reasoning,
                                "new_stop_loss_pct": new_stop_pct,
                                "model_type": strategy_model_type,
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                            }
                        )
                    # Skip further processing for this coin (do not execute a trade)
                    return
                else:
                    # LLM returned HOLD but did not provide a new stop-loss → force SELL
                    logger.warning(
                        f"LLM returned HOLD for {symbol} after stop-loss trigger but did not provide "
                        f"a new stop-loss. Forcing SELL."
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⛔ {symbol}: LLM did not provide new stop-loss – selling.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Stop-loss triggered, LLM did not provide new stop",
                                "exit_reason": "stop_loss_llm_no_action",
                                "model_type": strategy_model_type,
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                            }
                        )
                    await self._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered, LLM did not provide new stop"),
                        exit_reason="stop_loss_llm_no_action"
                    )
                    return

            elif stop_loss_triggered and signal.action == "SELL":
                # LLM decided to sell – clear the flag and let the normal SELL execution proceed
                if symbol in self.positions:
                    self.positions[symbol].pop("_stop_loss_triggered", None)
                    self.positions[symbol].pop("_stop_loss_review_count", None)
                # Continue to the normal SELL execution below (do not return)

            # --- Handle take-profit-triggered LLM decision ---
            if take_profit_triggered and signal.action == "HOLD":
                # LLM decided to keep holding – must provide a new take-profit
                new_params = signal.strategy_params or {}
                new_tp_pct = new_params.get("take_profit_pct")
                if new_tp_pct is not None and new_tp_pct > 0:
                    # LLM provided a new take-profit – update position and clear the trigger flag
                    logger.info(
                        f"LLM decided to hold {symbol} after take-profit trigger, "
                        f"new take_profit_pct={new_tp_pct:.4%}"
                    )
                    if symbol in self.positions:
                        self.positions[symbol]["take_profit"] = current_price * (1 + new_tp_pct)
                        self.positions[symbol].pop("_take_profit_triggered", None)
                        self.positions[symbol].pop("_take_profit_review_count", None)
                        # Also apply any other updated parameters from the LLM
                        self._update_position_params(
                            symbol,
                            new_params,
                            signal.indicator_config,
                            assigned_tf,
                            current_price,
                            atr,
                        )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"🔄 {symbol}: LLM adjusted take-profit to {new_tp_pct:.4%} – holding.\n"
                            f"Reasoning: {validated.reasoning}",
                            summary={
                                "symbol": symbol,
                                "action": "HOLD",
                                "reason": validated.reasoning,
                                "new_take_profit_pct": new_tp_pct,
                                "model_type": strategy_model_type,
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                            }
                        )
                    # Skip further processing for this coin
                    return
                else:
                    # LLM returned HOLD but did not provide a new take-profit → force SELL
                    logger.warning(
                        f"LLM returned HOLD for {symbol} after take-profit trigger but did not provide "
                        f"a new take-profit. Forcing SELL."
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"🎯 {symbol}: LLM did not provide new take-profit – selling.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": "Take-profit triggered, LLM did not provide new take-profit",
                                "exit_reason": "take_profit_llm_no_action",
                                "model_type": strategy_model_type,
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                            }
                        )
                    await self._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered, LLM did not provide new take-profit"),
                        exit_reason="take_profit_llm_no_action"
                    )
                    return

            elif take_profit_triggered and signal.action == "SELL":
                # LLM decided to sell – clear the flag and let the normal SELL execution proceed
                if symbol in self.positions:
                    self.positions[symbol].pop("_take_profit_triggered", None)
                    self.positions[symbol].pop("_take_profit_review_count", None)
                # Continue to the normal SELL execution below (do not return)

            # --- Handle partial TP triggered ---
            if partial_tp_triggered and signal.action == "HOLD":
                new_levels = params.get("partial_take_profit_levels") if params else None
                if new_levels is not None:
                    # LLM provided updated levels – apply them and clear triggers
                    self.positions[symbol]["partial_take_profit_levels"] = new_levels
                    self.positions[symbol].pop("_partial_tp_triggered", None)
                    self.positions[symbol].pop("_partial_tp_triggered_single", None)
                    self.positions[symbol].pop("_partial_tp_review_count", None)
                    self.positions[symbol].pop("_partial_tp_single_review_count", None)
                    self.positions[symbol].pop("_partial_tp_triggered_levels", None)
                    self.positions[symbol]["partial_tp_levels_triggered"] = []
                    self.positions[symbol]["partial_tp_depth_wait_start"] = {}
                    logger.info(f"LLM updated partial TP levels for {symbol}")
                    # Also apply any other updated parameters from the LLM
                    self._update_position_params(
                        symbol,
                        params,
                        signal.indicator_config,
                        assigned_tf,
                        current_price,
                        atr,
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"🔄 {symbol}: LLM adjusted partial TP levels – holding.",
                            summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP levels adjusted by LLM", "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model}
                        )
                    return
                else:
                    # LLM did not provide new levels – execute the triggered partial sell(s)
                    logger.info(f"LLM did not update partial TP levels for {symbol}, executing triggered level(s)")
                    if self.positions[symbol].get("_partial_tp_triggered_single"):
                        await self._execute_partial_tp_single(symbol, current_price, None, ticker)
                        self.positions[symbol].pop("_partial_tp_triggered_single", None)
                        self.positions[symbol].pop("_partial_tp_single_review_count", None)
                    if self.positions[symbol].get("_partial_tp_triggered"):
                        for lvl in self.positions[symbol].get("_partial_tp_triggered_levels", []):
                            await self._execute_partial_tp_level(symbol, lvl, current_price, None, ticker)
                        self.positions[symbol].pop("_partial_tp_triggered", None)
                        self.positions[symbol].pop("_partial_tp_review_count", None)
                        self.positions[symbol].pop("_partial_tp_triggered_levels", None)
                    return

            elif partial_tp_triggered and signal.action == "SELL":
                # LLM decided to sell the entire position – clear partial TP flags and let normal SELL proceed
                self.positions[symbol].pop("_partial_tp_triggered", None)
                self.positions[symbol].pop("_partial_tp_triggered_single", None)
                self.positions[symbol].pop("_partial_tp_review_count", None)
                self.positions[symbol].pop("_partial_tp_single_review_count", None)
                self.positions[symbol].pop("_partial_tp_triggered_levels", None)
                # continue to normal SELL execution below

            # --- Handle dust sweep triggered ---
            if dust_sweep_triggered and signal.action == "HOLD":
                self.positions[symbol].pop("_dust_sweep_triggered", None)
                self.positions[symbol].pop("_dust_sweep_review_count", None)
                logger.info(f"LLM decided to hold dust for {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🧹 {symbol}: LLM decided to keep dust – holding.",
                        summary={"symbol": symbol, "action": "HOLD", "reason": "Dust kept by LLM"}
                    )
                return
            elif dust_sweep_triggered and signal.action == "SELL":
                self.positions[symbol].pop("_dust_sweep_triggered", None)
                self.positions[symbol].pop("_dust_sweep_review_count", None)
                logger.info(f"LLM decided to sell dust for {symbol}")
                await self._sweep_dust(symbol)
                return

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

            # --- Pre-check partial TP depth at order placement ---
            partial_levels = params.get("partial_take_profit_levels")
            if partial_levels:
                for i, level in enumerate(partial_levels):
                    min_depth = level.get("min_depth")
                    if min_depth is not None and min_depth > 0:
                        lvl_pct = level["take_profit_pct"]
                        tp_price = current_price * (1 + lvl_pct)
                        asks = order_book.get('asks', [])
                        cum_vol = 0.0
                        for ask in asks:
                            if ask[0] <= tp_price:
                                cum_vol += ask[1]
                            else:
                                break
                        if cum_vol < min_depth:
                            logger.info(
                                f"Skipping BUY {symbol}: partial TP level {i} depth {cum_vol:.4f} "
                                f"below required {min_depth:.4f} at price {tp_price:.4f}"
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⚠️ Skipping BUY {symbol}: insufficient depth for partial TP level {i}",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SKIP",
                                        "reason": f"Insufficient depth for partial TP level {i}",
                                        "depth": cum_vol,
                                        "min_depth": min_depth,
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
                            t = self.ws_manager.get_ticker(sym)
                            if t is None:
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

            # Apply any updated risk parameters from the LLM to the open position
            if symbol in self.positions and signal.strategy_params:
                self._update_position_params(
                    symbol,
                    signal.strategy_params,
                    signal.indicator_config,
                    assigned_tf,
                    current_price,
                    atr,
                )

            if validated.action != "HOLD":
                if trading_paused and validated.action == "BUY":
                    logger.info(f"Ignoring BUY signal for {symbol}: trading is paused.")
                else:
                    # --- Entry condition check (only for BUY) ---
                    if validated.action == "BUY" and validated.entry_condition is not None:
                        etype = validated.entry_condition.get("type")
                        if etype == "delay":
                            # Delay entries are simple time-based waits – schedule directly
                            delay_sec = validated.entry_condition.get("delay_seconds", 0)
                            logger.info(f"Scheduling delayed BUY for {symbol} in {delay_sec}s")
                            asyncio.create_task(
                                self._execute_delayed_entry(symbol, validated, assigned_tf, delay_sec)
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⏳ Delayed entry for {symbol} – executing in {delay_sec}s.",
                                    summary={
                                        "symbol": symbol,
                                        "action": "WAIT",
                                        "reason": "Delay entry scheduled",
                                        "delay_seconds": delay_sec,
                                    }
                                )
                            return  # do NOT execute now

                        timeout = validated.entry_condition.get("timeout_seconds", 300)
                        deadline = time.time() + timeout
                        # Store for background checking – do NOT block the main loop
                        self._pending_entries[symbol] = {
                            "signal": validated,
                            "deadline": deadline,
                            "timeframe": assigned_tf,
                            "condition": validated.entry_condition,
                        }
                        logger.info(
                            f"Queued entry condition for {symbol} (type={etype}, deadline in {timeout}s). "
                            f"Will monitor in background."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏳ Waiting for entry condition on {symbol} "
                                f"(type={etype}, timeout {timeout}s).",
                                summary={
                                    "symbol": symbol,
                                    "action": "WAIT",
                                    "reason": "Entry condition pending",
                                }
                            )
                        return  # do NOT execute now

                    await self._execute_signal(symbol, validated, timeframe=assigned_tf, atr=atr, spread_pct=spread_pct, order_book=order_book)
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
                ticker = self.ws_manager.get_ticker(sym)
                if ticker is None:
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
            "base_currency": self.base_currency,
        }

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return current open positions as trade-like dicts with unrealized P&L."""
        open_trades = []
        for symbol, pos in self.positions.items():
            try:
                ticker = self.ws_manager.get_ticker(symbol)
                if ticker is None:
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
                'timeframe': pos.get('timeframe'),
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

    def get_pause_status(self) -> Dict[str, Any]:
        """Return the current trading pause status, reason, and remaining duration."""
        paused_raw = self.redis.get("trading:paused")
        is_paused = paused_raw is not None and paused_raw == b"1"

        reason_raw = self.redis.get("trading:pause_reason")
        reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

        source_raw = self.redis.get("trading:pause_source")
        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")

        remaining_seconds = None
        if is_paused:
            pause_start_raw = self.redis.get("trading:pause_start")
            pause_duration_raw = self.redis.get("trading:pause_duration")
            if pause_start_raw and pause_duration_raw:
                try:
                    pause_start = float(pause_start_raw)
                    pause_duration = int(pause_duration_raw)
                    elapsed = time.time() - pause_start
                    remaining = pause_duration - elapsed
                    remaining_seconds = max(0, remaining) if remaining > 0 else None
                except (ValueError, TypeError):
                    pass

        return {
            "is_paused": is_paused,
            "reason": reason,
            "remaining_seconds": remaining_seconds,
            "source": source,
        }

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
                ticker = self.ws_manager.get_ticker(pos['symbol'])
                if ticker is None:
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

                ticker = self.ws_manager.get_ticker(symbol)
                if ticker is None:
                    if not self.ws_manager.healthy:
                        # Fallback to REST when WebSocket is down
                        try:
                            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                        except Exception:
                            continue
                    else:
                        continue  # no real-time data yet, skip this check
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
                        if i in pos.get("_partial_tp_triggered_levels", []):
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
                            # --- Depth check with optional timeout ---
                            min_depth = level.get("min_depth")
                            if min_depth is not None and min_depth > 0:
                                try:
                                    ob = self.ws_manager.get_order_book(symbol)
                                    if ob is None:
                                        ob = await asyncio.to_thread(get_order_book, self.exchange, symbol, 20)
                                except Exception as e:
                                    logger.warning(f"Could not fetch order book for depth check on {symbol}: {e}")
                                    ob = None
                                if ob:
                                    asks = ob.get('asks', [])
                                    tp_price = entry_price * (1 + lvl_pct)
                                    cum_vol = 0.0
                                    for ask in asks:
                                        if ask[0] <= tp_price:
                                            cum_vol += ask[1]
                                        else:
                                            break
                                    if cum_vol < min_depth:
                                        depth_timeout = level.get("depth_timeout_seconds")
                                        if depth_timeout is not None and depth_timeout > 0:
                                            now_ts = time.time()
                                            wait_start = pos.setdefault("partial_tp_depth_wait_start", {}).get(i)
                                            if wait_start is None:
                                                pos["partial_tp_depth_wait_start"][i] = now_ts
                                                logger.info(
                                                    f"Partial TP level {i} for {symbol}: depth {cum_vol:.4f} < {min_depth:.4f}, "
                                                    f"waiting up to {depth_timeout}s"
                                                )
                                                continue
                                            else:
                                                elapsed = now_ts - wait_start
                                                if elapsed < depth_timeout:
                                                    logger.debug(
                                                        f"Partial TP level {i} for {symbol}: still waiting for depth "
                                                        f"({elapsed:.1f}s / {depth_timeout}s)"
                                                    )
                                                    continue
                                                else:
                                                    logger.info(
                                                        f"Partial TP level {i} for {symbol}: depth timeout expired "
                                                        f"({depth_timeout}s). Cancelling level."
                                                    )
                                                    triggered.append(i)
                                                    pos["partial_tp_levels_triggered"] = triggered
                                                    if "partial_tp_depth_wait_start" in pos:
                                                        pos["partial_tp_depth_wait_start"].pop(i, None)
                                                    continue
                                        else:
                                            logger.info(
                                                f"Partial TP level {i} for {symbol}: depth {cum_vol:.4f} < {min_depth:.4f} "
                                                f"and no depth_timeout_seconds set. Cancelling level."
                                            )
                                            triggered.append(i)
                                            pos["partial_tp_levels_triggered"] = triggered
                                            continue
                                else:
                                    continue
                            # Clear any depth wait state for this level
                            if "partial_tp_depth_wait_start" in pos:
                                pos["partial_tp_depth_wait_start"].pop(i, None)

                            # --- Instead of executing immediately, set a trigger flag for LLM review ---
                            # Check if we are already waiting for LLM on this level
                            triggered_levels = pos.setdefault("_partial_tp_triggered_levels", [])
                            if i in triggered_levels:
                                continue  # already pending

                            review_count = pos.get("_partial_tp_review_count", 0) + 1
                            if review_count > settings.MAX_PARTIAL_TP_REVIEWS:
                                # Force execute
                                logger.info(f"Partial TP level {i} for {symbol}: max reviews reached, executing.")
                                await self._execute_partial_tp_level(symbol, i, current_price, None, ticker)
                                # After execution, the level is marked triggered; clear the review flags for this level
                                pos.pop("_partial_tp_triggered", None)
                                pos.pop("_partial_tp_review_count", None)
                                pos["_partial_tp_triggered_levels"] = [x for x in pos.get("_partial_tp_triggered_levels", []) if x != i]
                                continue

                            # Set trigger and ask LLM
                            pos["_partial_tp_triggered"] = True
                            pos["_partial_tp_review_count"] = review_count
                            triggered_levels.append(i)
                            self._last_strategy_eval.pop(symbol, None)  # force immediate re‑eval
                            logger.info(f"Partial TP level {i} triggered for {symbol} – asking LLM (review {review_count})")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🔸 Partial TP level {i} triggered for {symbol} – consulting LLM...",
                                    summary={"symbol": symbol, "action": "HOLD", "reason": f"Partial TP level {i} triggered – awaiting LLM"}
                                )
                            break  # only handle one new trigger per cycle; others will be picked up after LLM responds
                        else:
                            # Price dropped below TP level – reset depth wait timer
                            if "partial_tp_depth_wait_start" in pos:
                                pos["partial_tp_depth_wait_start"].pop(i, None)
                else:
                    # Single partial TP – trigger LLM review instead of immediate execution
                    partial_tp_pct = pos.get("partial_take_profit_pct")
                    partial_tp_fraction = pos.get("partial_take_profit_fraction")
                    if (
                        partial_tp_pct is not None
                        and partial_tp_fraction is not None
                        and not pos.get("partial_tp_triggered", False)
                        and not pos.get("_partial_tp_triggered_single")
                    ):
                        entry_price = pos["price"]
                        if current_price >= entry_price * (1 + partial_tp_pct):
                            review_count = pos.get("_partial_tp_single_review_count", 0) + 1
                            if review_count > settings.MAX_PARTIAL_TP_REVIEWS:
                                logger.info(f"Single partial TP for {symbol}: max reviews reached, executing.")
                                await self._execute_partial_tp_single(symbol, current_price, None, ticker)
                                pos.pop("_partial_tp_triggered_single", None)
                                pos.pop("_partial_tp_single_review_count", None)
                            else:
                                pos["_partial_tp_triggered_single"] = True
                                pos["_partial_tp_single_review_count"] = review_count
                                self._last_strategy_eval.pop(symbol, None)
                                logger.info(f"Single partial TP triggered for {symbol} – asking LLM (review {review_count})")
                                if self.notifier:
                                    await self.notifier.send_notification(
                                        f"🔸 Partial TP triggered for {symbol} – consulting LLM...",
                                        summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP triggered – awaiting LLM"}
                                    )

                # --- Dust sweep check (if not already triggered) ---
                if not pos.get("_dust_sweep_triggered"):
                    base = symbol.split("/")[0]
                    quote = symbol.split("/")[1]
                    market = self.exchange.markets.get(symbol, {})
                    limits = market.get("limits", {})
                    min_amount = limits.get("amount", {}).get("min")
                    min_cost = limits.get("cost", {}).get("min")
                    amount = pos["amount"]
                    is_dust = False
                    if min_amount is not None and amount < float(min_amount):
                        is_dust = True
                    elif min_cost is not None and amount * current_price < float(min_cost):
                        is_dust = True

                    if is_dust:
                        review_count = pos.get("_dust_sweep_review_count", 0) + 1
                        if review_count > settings.MAX_DUST_SWEEP_REVIEWS:
                            logger.info(f"Dust sweep max reviews reached for {symbol}, force sweeping.")
                            await self._sweep_dust(symbol)
                        else:
                            pos["_dust_sweep_triggered"] = True
                            pos["_dust_sweep_review_count"] = review_count
                            self._last_strategy_eval.pop(symbol, None)
                            logger.info(f"Dust condition triggered for {symbol} – asking LLM (review {review_count})")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🧹 Dust sweep triggered for {symbol} – consulting LLM...",
                                    summary={"symbol": symbol, "action": "HOLD", "reason": "Dust sweep triggered – awaiting LLM"}
                                )
                else:
                    # If dust was previously triggered but condition no longer holds, clear it
                    base = symbol.split("/")[0]
                    quote = symbol.split("/")[1]
                    market = self.exchange.markets.get(symbol, {})
                    limits = market.get("limits", {})
                    min_amount = limits.get("amount", {}).get("min")
                    min_cost = limits.get("cost", {}).get("min")
                    amount = pos["amount"]
                    is_dust = False
                    if min_amount is not None and amount < float(min_amount):
                        is_dust = True
                    elif min_cost is not None and amount * current_price < float(min_cost):
                        is_dust = True
                    if not is_dust:
                        pos.pop("_dust_sweep_triggered", None)
                        pos.pop("_dust_sweep_review_count", None)
                        logger.info(f"Dust condition cleared for {symbol}")

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

                # --- Max hold time expired → ask LLM instead of auto‑closing ---
                max_hold = pos.get("max_hold_time_seconds")
                if max_hold is not None and max_hold > 0:
                    entry_ts = pos.get("timestamp", 0) / 1000.0  # convert ms to seconds
                    if time.time() - entry_ts > max_hold:
                        # Already waiting for LLM – do not re‑trigger
                        if pos.get("_max_hold_expired"):
                            continue
                        # First expiry – ask LLM
                        expired_count = pos.get("_max_hold_expired_count", 0) + 1
                        pos["_max_hold_expired"] = True
                        pos["_max_hold_expired_count"] = expired_count

                        # Force re‑evaluation on the next main loop tick
                        self._last_strategy_eval.pop(symbol, None)

                        logger.info(
                            f"Max hold time expired for {symbol} (attempt {expired_count}) – asking LLM to decide."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏰ Max hold time expired for {symbol} – asking LLM whether to sell or extend.",
                                summary={
                                    "symbol": symbol,
                                    "action": "HOLD",
                                    "reason": "Max hold time expired – awaiting LLM decision",
                                }
                            )
                        continue   # skip further checks for this symbol in this cycle

                if current_price <= pos["stop_loss"]:
                    # Instead of immediately selling, ask the LLM whether to sell or adjust the stop.
                    review_count = pos.get("_stop_loss_review_count", 0)
                    if review_count >= MAX_STOP_LOSS_REVIEWS:
                        # Fallback: force-sell after too many reviews
                        logger.warning(
                            f"Stop-loss triggered for {symbol} at {current_price} – "
                            f"review count {review_count} >= {MAX_STOP_LOSS_REVIEWS}, forcing SELL."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⛔ Stop‑loss triggered for {symbol} at {current_price:.4f} – "
                                f"max reviews reached, selling.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Stop-loss (max reviews)",
                                    "price": current_price,
                                    "exit_reason": "stop_loss_max_reviews",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Stop-loss (max reviews)"),
                            exit_reason="stop_loss_max_reviews"
                        )
                    else:
                        # First or repeated trigger: set flag and ask LLM
                        if not pos.get("_stop_loss_triggered"):
                            pos["_stop_loss_triggered"] = True
                            pos["_stop_loss_review_count"] = review_count + 1
                            # Force immediate strategy re-evaluation for this coin
                            self._last_strategy_eval.pop(symbol, None)
                            logger.info(
                                f"Stop-loss triggered for {symbol} at {current_price} – "
                                f"asking LLM (review {pos['_stop_loss_review_count']}/{MAX_STOP_LOSS_REVIEWS})."
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⛔ Stop‑loss hit for {symbol} at {current_price:.4f} – consulting LLM...",
                                    summary={
                                        "symbol": symbol,
                                        "action": "HOLD",
                                        "reason": "Stop-loss triggered – awaiting LLM decision",
                                        "price": current_price,
                                    }
                                )
                        else:
                            # Already waiting for LLM; do nothing (avoid re-triggering)
                            logger.debug(
                                f"Stop-loss still triggered for {symbol}, waiting for LLM response "
                                f"(review {review_count}/{MAX_STOP_LOSS_REVIEWS})."
                            )
                elif current_price >= pos["take_profit"]:
                    # Always ask the LLM whether to sell or adjust the take-profit, but cap reviews.
                    review_count = pos.get("_take_profit_review_count", 0)
                    if review_count >= MAX_TAKE_PROFIT_REVIEWS:
                        logger.warning(
                            f"Take-profit triggered for {symbol} at {current_price} – "
                            f"review count {review_count} >= {MAX_TAKE_PROFIT_REVIEWS}, forcing SELL."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🎯 Take‑profit triggered for {symbol} at {current_price:.4f} – "
                                f"max reviews reached, selling.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Take-profit (max reviews)",
                                    "price": current_price,
                                    "exit_reason": "take_profit_max_reviews",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Take-profit (max reviews)"),
                            exit_reason="take_profit_max_reviews"
                        )
                        continue
                    # First or repeated trigger: set flag and ask LLM
                    if not pos.get("_take_profit_triggered"):
                        pos["_take_profit_triggered"] = True
                        pos["_take_profit_review_count"] = review_count + 1
                        # Force immediate strategy re-evaluation for this coin
                        self._last_strategy_eval.pop(symbol, None)
                        logger.info(
                            f"Take-profit triggered for {symbol} at {current_price} – "
                            f"asking LLM (review {pos['_take_profit_review_count']}/{MAX_TAKE_PROFIT_REVIEWS})."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🎯 Take‑profit hit for {symbol} at {current_price:.4f} – consulting LLM...",
                                summary={
                                    "symbol": symbol,
                                    "action": "HOLD",
                                    "reason": "Take-profit triggered – awaiting LLM decision",
                                    "price": current_price,
                                }
                            )
                    else:
                        # Already waiting for LLM; do nothing
                        logger.debug(
                            f"Take-profit still triggered for {symbol}, waiting for LLM response "
                            f"(review {review_count}/{MAX_TAKE_PROFIT_REVIEWS})."
                        )
            except Exception as e:
                logger.error(f"Risk check failed for {symbol}: {e}")

    async def _execute_signal(self, symbol: str, signal, timeframe: str = None, exit_reason: str = None, atr: Optional[float] = None, spread_pct: Optional[float] = None, order_book: Optional[Dict[str, Any]] = None):
        """Execute a BUY or SELL signal."""
        async with self._risk_lock:
            base, quote = symbol.split("/")
            balance = await asyncio.to_thread(self.trader.fetch_balance)

        if signal.action == "BUY":
            # Safety: never buy when trading is paused
            paused = await asyncio.to_thread(self.redis.get, "trading:paused")
            if paused:
                logger.info(f"Ignoring BUY {symbol}: trading is paused (safety check).")
                return
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
                ticker = self.ws_manager.get_ticker(symbol)
                if ticker is None:
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

            # Scale position size by LLM confidence
            desired_amount *= signal.confidence
            logger.info(
                f"Confidence scaling: confidence={signal.confidence:.2f}, "
                f"adjusted desired_amount={desired_amount:.2f} {quote}"
            )

            # Apply max risk per trade cap if provided
            max_risk_pct = params.get("max_risk_per_trade_pct")
            if max_risk_pct is not None and sl_pct > 0:
                total_value = quote_balance
                for sym, pos in self.positions.items():
                    try:
                        t = self.ws_manager.get_ticker(sym)
                        if t is None:
                            t = await asyncio.to_thread(self.exchange.fetch_ticker, sym)
                        total_value += pos['amount'] * t['last']
                    except Exception:
                        pass
                max_risk_amount = total_value * max_risk_pct
                max_allowed_amount = max_risk_amount / sl_pct
                desired_amount = min(desired_amount, max_allowed_amount)
                logger.info(f"Max risk per trade: {max_risk_pct:.2%} of {total_value:.2f} = {max_risk_amount:.2f}, max allowed amount = {max_allowed_amount:.2f}")

            # Apply global risk multiplier if set by LLM in coin selection
            global_mult_raw = await asyncio.to_thread(self.redis.get, "trading:global_risk_multiplier")
            if global_mult_raw:
                try:
                    global_mult = float(global_mult_raw)
                    if 0.0 <= global_mult <= 1.0:
                        desired_amount *= global_mult
                        logger.info(f"Applied global risk multiplier {global_mult}: desired_amount={desired_amount:.2f}")
                except (ValueError, TypeError):
                    pass

            # Apply per-coin position size multiplier if set by LLM in strategy params
            per_coin_mult = params.get("position_size_multiplier")
            if per_coin_mult is not None:
                try:
                    per_coin_mult = float(per_coin_mult)
                    if 0.0 <= per_coin_mult <= 1.0:
                        desired_amount *= per_coin_mult
                        logger.info(f"Applied per-coin position multiplier {per_coin_mult}: desired_amount={desired_amount:.2f}")
                except (ValueError, TypeError):
                    pass

            # --- Minimum absolute profit check (LLM‑defined) ---
            if settings.ENFORCE_MIN_PROFIT_PER_TRADE:
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

            # --- Cap position size to limit slippage (using order book) ---
            if order_book and amount > 0 and settings.MAX_SLIPPAGE_CAP_PCT > 0:
                asks = order_book.get('asks', [])
                if asks:
                    best_ask = asks[0][0]
                    max_slippage = settings.MAX_SLIPPAGE_CAP_PCT / 100.0  # convert percent to decimal
                    max_allowed_cost = 0.0
                    cumulative_cost = 0.0
                    cumulative_base = 0.0
                    for ask in asks:
                        price_level = ask[0]
                        volume = ask[1]
                        # If this level alone would push average price above limit, we may need to partially fill it
                        # Compute average price if we take the whole level
                        new_cost = cumulative_cost + price_level * volume
                        new_base = cumulative_base + volume
                        avg_price = new_cost / new_base if new_base > 0 else best_ask
                        if avg_price > best_ask * (1 + max_slippage):
                            # We can only take a fraction of this level
                            # Solve for the base amount x such that (cumulative_cost + price_level * x) / (cumulative_base + x) = best_ask * (1 + max_slippage)
                            target_avg = best_ask * (1 + max_slippage)
                            if price_level != target_avg:
                                x = (cumulative_cost - target_avg * cumulative_base) / (target_avg - price_level)
                            else:
                                x = float('inf')
                            if x > 0:
                                max_allowed_cost = cumulative_cost + price_level * x
                            break
                        else:
                            cumulative_cost = new_cost
                            cumulative_base = new_base
                            max_allowed_cost = cumulative_cost
                    else:
                        # Entire order book consumed without hitting limit
                        max_allowed_cost = cumulative_cost

                    if max_allowed_cost > 0 and amount > max_allowed_cost:
                        old_amount = amount
                        amount = max_allowed_cost
                        logger.info(
                            f"BUY amount capped from {old_amount:.2f} to {amount:.2f} {quote} "
                            f"to limit slippage to {settings.MAX_SLIPPAGE_CAP_PCT}%"
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ BUY {symbol} capped to {amount:.2f} {quote} (slippage limit {settings.MAX_SLIPPAGE_CAP_PCT}%)",
                                summary={
                                    "symbol": symbol,
                                    "action": "INFO",
                                    "reason": "Position size capped to limit slippage",
                                    "original_amount": old_amount,
                                    "capped_amount": amount,
                                }
                            )

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
                ticker = self.ws_manager.get_ticker(symbol)
                if ticker is None:
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
                        self.positions[symbol]["partial_tp_depth_wait_start"] = {}
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
                        "partial_tp_depth_wait_start": {},
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
                # Persist paper balances immediately
                if settings.TRADING_MODE == "paper":
                    await asyncio.to_thread(save_paper_balances, self.trader.balances)
                await self._save_state()
                if self.notifier:
                    buy_msg = f"🟢 BUY {symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    buy_summary = {
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
                    if signal.model_type:
                        buy_summary["model_type"] = signal.model_type
                    if signal.llm_provider:
                        buy_summary["llm_provider"] = signal.llm_provider
                    if signal.llm_model:
                        buy_summary["llm_model"] = signal.llm_model
                    await self.notifier.send_notification(
                        buy_msg,
                        summary=buy_summary,
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

            # Guard against overselling: cap sell amount to actual balance
            actual_base_balance = balance.get(base, 0.0)
            if pos and gross_amount > actual_base_balance:
                logger.warning(
                    f"Tracked position amount {gross_amount} exceeds actual balance "
                    f"{actual_base_balance} for {symbol}. Capping sell amount to actual balance."
                )
                gross_amount = actual_base_balance

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
                ticker = self.ws_manager.get_ticker(symbol)
                if ticker is None:
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
                # Clear any stop-loss review flags before removing the position
                if pos:
                    pos.pop("_stop_loss_triggered", None)
                    pos.pop("_stop_loss_review_count", None)
                # Remove position
                self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self._pending_entries.pop(symbol, None)
                await self._remove_coin_if_paused(symbol)
                self.trade_history.append(order)
                await asyncio.to_thread(insert_trade, order)
                # Persist paper balances immediately
                if settings.TRADING_MODE == "paper":
                    await asyncio.to_thread(save_paper_balances, self.trader.balances)
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
                    sell_summary = {
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
                    if signal.model_type:
                        sell_summary["model_type"] = signal.model_type
                    if signal.llm_provider:
                        sell_summary["llm_provider"] = signal.llm_provider
                    if signal.llm_model:
                        sell_summary["llm_model"] = signal.llm_model
                    await self.notifier.send_notification(
                        sell_msg,
                        summary=sell_summary,
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

    def _should_skip_llm_eval(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float],
        rsi: Optional[float],
        macd_hist: Optional[float],
        atr_percentile: Optional[float],
        market_regime: str,
        sentiment_trend_val: Optional[float],
        timeframe_seconds: float,
        has_position: bool,
        is_critical: bool,
    ) -> bool:
        """Return True if it’s safe to skip the LLM call and just HOLD."""
        # Never skip critical situations (max hold, stop-loss, take-profit triggered)
        if is_critical:
            return False

        snapshot = self._last_eval_snapshot.get(symbol)
        if snapshot is None:
            # First evaluation – must call
            return False

        now = time.time()
        last_time = snapshot.get("timestamp", 0)
        last_price = snapshot.get("price", 0)

        # Always call if enough time has passed (2× the normal interval)
        if now - last_time > 2 * timeframe_seconds:
            return False

        # Price change since last evaluation
        if last_price > 0:
            price_change_pct = abs(current_price - last_price) / last_price
            # If price moved less than 0.2× ATR (in %), it’s boring
            atr_pct = (atr / current_price) if (atr and atr > 0) else 0.005
            if price_change_pct > atr_pct * 0.5:
                return False   # enough movement to warrant a new look

        # Indicator changes
        last_rsi = snapshot.get("rsi")
        last_macd_hist = snapshot.get("macd_hist")
        if rsi is not None and last_rsi is not None:
            if abs(rsi - last_rsi) > 5:
                return False
        if macd_hist is not None and last_macd_hist is not None:
            if abs(macd_hist - last_macd_hist) > 0.0005:
                return False

        # If we have no open position and nothing is screaming, skip
        if not has_position:
            # Only call if there is a potential entry signal (extreme RSI, MACD crossover, etc.)
            # RSI extreme?
            if rsi is not None and (rsi < 30 or rsi > 70):
                return False
            # MACD histogram direction change? (harder to detect without previous sign – skip for simplicity)
            # Otherwise, no strong signal → skip
            return True

        # Have an open position – skip if price far from stop/tp and indicators calm
        # (the risk management loop will handle stop/tp)
        return True

    async def _check_pending_entries(self):
        """Periodically check pending entry conditions and execute if met."""
        await asyncio.sleep(2)  # short initial delay
        while self._running:
            try:
                now = time.time()
                for symbol in list(self._pending_entries.keys()):
                    entry = self._pending_entries.get(symbol)
                    if entry is None:
                        continue
                    if now >= entry["deadline"]:
                        # Timeout – clear and notify
                        logger.info(f"Entry condition timeout for {symbol}")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏭️ Entry condition timeout for {symbol} – skipping BUY.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Entry condition timeout",
                                }
                            )
                        del self._pending_entries[symbol]
                        continue

                    # Check the condition (non‑blocking)
                    condition_met = await self._check_entry_condition_once(
                        symbol, entry["condition"], entry["timeframe"]
                    )
                    if condition_met:
                        logger.info(f"Entry condition met for {symbol}, executing BUY")
                        # Remove from pending before executing to avoid re‑trigger
                        signal = entry["signal"]
                        del self._pending_entries[symbol]
                        # Check trading pause again (may have changed)
                        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                        if paused:
                            logger.info(f"Ignoring queued BUY {symbol}: trading is now paused.")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⏸️ Queued BUY for {symbol} skipped – trading paused.",
                                    summary={"symbol": symbol, "action": "SKIP", "reason": "Trading paused"}
                                )
                        else:
                            await self._execute_signal(
                                symbol,
                                signal,
                                timeframe=entry["timeframe"],
                                atr=None,
                                spread_pct=None,
                                order_book=None,
                            )
            except Exception as e:
                logger.error(f"Error checking pending entries: {e}", exc_info=True)
            await asyncio.sleep(5)  # check every 5 seconds

    async def _check_entry_condition_once(
        self, symbol: str, condition: Dict[str, Any], timeframe: str
    ) -> bool:
        """Check a single entry condition immediately. Return True if met."""
        etype = condition.get("type")
        if etype == "limit_price":
            target_price = condition["price"]
            ticker = self.ws_manager.get_ticker(symbol)
            if ticker is None:
                try:
                    async with self._exchange_semaphore:
                        ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                except Exception:
                    return False
            current_price = ticker.get("last", 0) if ticker else 0
            return current_price > 0 and current_price <= target_price

        elif etype == "rsi_threshold":
            target_rsi = condition["rsi_below"]
            try:
                async with self._exchange_semaphore:
                    ohlcv = await asyncio.to_thread(
                        get_multi_timeframe_ohlcv, self.exchange, symbol, [timeframe], limit=50
                    )
            except Exception:
                return False
            if timeframe in ohlcv and ohlcv[timeframe]:
                candles = ohlcv[timeframe]
                rsi = compute_rsi([c[4] for c in candles])
                return rsi is not None and rsi <= target_rsi
            return False

        elif etype == "order_book_depth":
            min_vol = condition["min_ask_volume"]
            ob = self.ws_manager.get_order_book(symbol)
            if ob is None:
                try:
                    async with self._exchange_semaphore:
                        ob = await asyncio.to_thread(get_order_book, self.exchange, symbol, 20)
                except Exception:
                    return False
            if ob:
                asks = ob.get("asks", [])
                bids = ob.get("bids", [])
                if asks and bids:
                    mid = (asks[0][0] + bids[0][0]) / 2
                    cum_vol = sum(a[1] for a in asks if a[0] <= mid * 1.01)
                    return cum_vol >= min_vol
            return False

        elif etype == "delay":
            # Delay conditions are handled by _execute_delayed_entry, not the
            # pending-entries system. If we somehow reach here, treat as not met
            # so the deadline handler can deal with it.
            return False

        elif etype == "indicator_combo":
            conditions = condition["conditions"]
            try:
                async with self._exchange_semaphore:
                    ohlcv = await asyncio.to_thread(
                        get_multi_timeframe_ohlcv, self.exchange, symbol, [timeframe], limit=50
                    )
            except Exception:
                return False
            if timeframe in ohlcv and ohlcv[timeframe]:
                candles = ohlcv[timeframe]
                closes = [c[4] for c in candles]
                for cond in conditions:
                    ind = cond["indicator"]
                    thresh = cond["threshold"]
                    direction = cond["direction"]
                    if ind == "rsi":
                        rsi_val = compute_rsi(closes)
                        if rsi_val is None or (direction == "below" and rsi_val > thresh) or (direction == "above" and rsi_val < thresh):
                            return False
                    elif ind == "macd_hist":
                        macd_vals = compute_macd(closes)
                        macd_hist_val = macd_vals[2][-1] if macd_vals and len(macd_vals[2]) > 0 else None
                        if macd_hist_val is None or (direction == "below" and macd_hist_val > thresh) or (direction == "above" and macd_hist_val < thresh):
                            return False
                return True
            return False

        return False

    async def _execute_delayed_entry(self, symbol: str, signal, timeframe: str, delay_seconds: float):
        """Execute a delayed entry after waiting for the specified duration."""
        logger.info(f"Delayed entry: waiting {delay_seconds}s for {symbol}")
        await asyncio.sleep(delay_seconds)
        if not self._running:
            return
        # Check if the symbol already has a position (may have been bought by another path)
        if symbol in self.positions:
            logger.info(f"Skipping delayed BUY for {symbol}: position already exists.")
            return
        # Check trading pause
        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
        if paused:
            logger.info(f"Ignoring delayed BUY {symbol}: trading is now paused.")
            if self.notifier:
                await self.notifier.send_notification(
                    f"⏸️ Delayed BUY for {symbol} skipped – trading paused.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Trading paused"}
                )
            return
        logger.info(f"Delay elapsed for {symbol}, executing BUY")
        await self._execute_signal(
            symbol,
            signal,
            timeframe=timeframe,
            atr=None,
            spread_pct=None,
            order_book=None,
        )

    def _choose_model_tier(
        self,
        atr_percentile: Optional[float],
        market_regime: str,
        sentiment_trend_val: Optional[float],
        rsi: Optional[float],
        macd_hist: Optional[float],
        macd: Optional[float],
        macd_signal: Optional[float],
        is_critical: bool,
    ) -> str:
        """Return "mind" or "actuator" based on market complexity."""
        if is_critical:
            return "mind"

        complexity = 0

        # High or low volatility extremes
        if atr_percentile is not None:
            if atr_percentile > 80 or atr_percentile < 20:
                complexity += 1

        # Turbulent market regime
        if market_regime and ("high volatility" in market_regime or "squeeze" in market_regime):
            complexity += 1

        # Strong sentiment swing
        if sentiment_trend_val is not None and abs(sentiment_trend_val) > 0.2:
            complexity += 1

        # Conflicting technicals (RSI extreme vs MACD direction)
        if rsi is not None and macd_hist is not None:
            if (rsi < 30 and macd_hist < 0) or (rsi > 70 and macd_hist > 0):
                complexity += 1

        # MACD crossover nearby (hist near zero while lines close)
        if macd is not None and macd_signal is not None:
            if abs(macd - macd_signal) < 0.0001 * abs(macd) if macd else 0:
                complexity += 1

        return "mind" if complexity >= 2 else "actuator"

    def _compute_prompt_complexity(
        self,
        num_candidates: int = 0,
        market_breadth: Optional[Dict[str, Any]] = None,
        fear_greed: Optional[Dict[str, Any]] = None,
        volatility_percentile: Optional[float] = None,
        sentiment_trend_magnitude: Optional[float] = None,
        conflicting_signals: bool = False,
        is_critical: bool = False,
    ) -> float:
        """Return a complexity score between 0.0 (simple) and 1.0 (very complex)."""
        score = 0.0
        # More candidates → more complex
        if num_candidates > 20:
            score += 0.2
        elif num_candidates > 10:
            score += 0.1

        # Extreme market breadth (very high or very low) adds complexity
        if market_breadth:
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > 80 or pos_pct < 20:
                score += 0.15

        # Extreme fear/greed
        if fear_greed:
            fg = fear_greed.get("value", 50)
            if fg <= 25 or fg >= 75:
                score += 0.1

        # High volatility percentile
        if volatility_percentile is not None and (volatility_percentile > 80 or volatility_percentile < 20):
            score += 0.1

        # Strong sentiment swing
        if sentiment_trend_magnitude is not None and sentiment_trend_magnitude > 0.2:
            score += 0.1

        # Conflicting technical signals
        if conflicting_signals:
            score += 0.15

        # Critical decision (stop-loss, take-profit, etc.)
        if is_critical:
            score += 0.2

        return min(1.0, score)

    def _get_effective_temperature(self, model_type: str, complexity: float) -> float:
        """Return the temperature to use for a given model_type and complexity score (0-1)."""
        from src.config.settings import Settings
        raw = settings.LLM_MIND_TEMPERATURE if model_type == "mind" else settings.LLM_ACTUATOR_TEMPERATURE
        parsed = Settings.parse_temperature_range(raw)
        if parsed is None:
            # Fall back to global LLM_TEMPERATURE
            return settings.LLM_TEMPERATURE
        lo, hi = parsed
        if lo == hi:
            return lo
        # Map complexity 0→lo, 1→hi
        return lo + (hi - lo) * complexity

    def _update_last_eval_snapshot(self, symbol: str, price: float, rsi: Optional[float], macd_hist: Optional[float]):
        self._last_eval_snapshot[symbol] = {
            "timestamp": time.time(),
            "price": price,
            "rsi": rsi,
            "macd_hist": macd_hist,
        }

    def _update_position_params(
        self,
        symbol: str,
        params: Dict[str, Any],
        indicator_config: Optional[Dict[str, Any]],
        timeframe: str,
        current_price: float,
        atr: Optional[float],
    ):
        """Update risk parameters of an open position from LLM strategy_params."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        # --- Stop-loss (supports fixed pct and ATR multiple) ---
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0:
            atr_mult = params.get("stop_loss_atr_multiple")
            if atr_mult is not None:
                sl_pct = (atr_mult * atr) / current_price
                pos["stop_loss"] = current_price * (1 - sl_pct)
        elif "stop_loss_pct" in params:
            sl_pct = params["stop_loss_pct"]
            try:
                sl_pct = float(sl_pct)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid stop_loss_pct=%r for %s",
                    params["stop_loss_pct"], symbol,
                )
                sl_pct = None
            if sl_pct is not None and sl_pct > 0 and current_price > 0:
                pos["stop_loss"] = current_price * (1 - sl_pct)
            elif sl_pct is not None:
                logger.warning(
                    "Ignoring invalid stop_loss_pct=%s for %s (current_price=%s)",
                    sl_pct, symbol, current_price,
                )

        # --- Take-profit ---
        if "take_profit_pct" in params:
            tp_pct = params["take_profit_pct"]
            try:
                tp_pct = float(tp_pct)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid take_profit_pct=%r for %s",
                    params["take_profit_pct"], symbol,
                )
                tp_pct = None
            if tp_pct is not None and tp_pct > 0 and current_price > 0:
                pos["take_profit"] = current_price * (1 + tp_pct)
            elif tp_pct is not None:
                logger.warning(
                    "Ignoring invalid take_profit_pct=%s for %s (current_price=%s)",
                    tp_pct, symbol, current_price,
                )

        # --- Trailing stop ---
        if "trailing_stop" in params:
            pos["trailing_stop"] = params["trailing_stop"]
        if "trailing_stop_distance_pct" in params:
            pos["trailing_stop_distance_pct"] = params["trailing_stop_distance_pct"]
        if "trailing_stop_activation_pct" in params:
            pos["trailing_stop_activation_pct"] = params["trailing_stop_activation_pct"]

        # --- Trailing take-profit ---
        if "trailing_take_profit" in params:
            pos["trailing_take_profit"] = params["trailing_take_profit"]
        if "trailing_take_profit_distance_pct" in params:
            pos["trailing_take_profit_distance_pct"] = params["trailing_take_profit_distance_pct"]

        # --- Breakeven / lock-profit ---
        if "breakeven_activation_pct" in params:
            pos["breakeven_activation_pct"] = params["breakeven_activation_pct"]
        if "lock_profit_activation_pct" in params:
            pos["lock_profit_activation_pct"] = params["lock_profit_activation_pct"]
        if "lock_profit_level_pct" in params:
            pos["lock_profit_level_pct"] = params["lock_profit_level_pct"]

        # --- Time-based exits ---
        if "max_hold_time_seconds" in params:
            pos["max_hold_time_seconds"] = params["max_hold_time_seconds"]
            # If the LLM explicitly sets a new hold time, clear any expiry flag
            pos.pop("_max_hold_expired", None)
            pos.pop("_max_hold_expired_count", None)

        # --- Cooldown after loss ---
        if "cooldown_after_loss_seconds" in params:
            pos["cooldown_after_loss_seconds"] = params["cooldown_after_loss_seconds"]

        # --- News sentiment exit ---
        if "news_sentiment_exit_threshold" in params:
            pos["news_sentiment_exit_threshold"] = params["news_sentiment_exit_threshold"]

        # --- Max unrealized loss ---
        if "max_unrealized_loss_pct" in params:
            pos["max_unrealized_loss_pct"] = params["max_unrealized_loss_pct"]

        # --- Partial take-profit levels ---
        if "partial_take_profit_levels" in params:
            pos["partial_take_profit_levels"] = params["partial_take_profit_levels"]
            pos["partial_tp_levels_triggered"] = []
            pos["partial_tp_depth_wait_start"] = {}
            # Clear single-level fields to avoid confusion
            pos["partial_take_profit_pct"] = None
            pos["partial_take_profit_fraction"] = None
            pos["partial_tp_triggered"] = None
        else:
            if "partial_take_profit_pct" in params:
                pos["partial_take_profit_pct"] = params["partial_take_profit_pct"]
            if "partial_take_profit_fraction" in params:
                pos["partial_take_profit_fraction"] = params["partial_take_profit_fraction"]
            if "partial_tp_triggered" not in pos:
                pos["partial_tp_triggered"] = False

        # --- Strategy interval ---
        if "strategy_interval_seconds" in params:
            self._strategy_intervals[symbol] = params["strategy_interval_seconds"]

        # --- Indicator config ---
        if indicator_config is not None:
            pos["indicator_config"] = indicator_config

        # --- Timeframe (if changed) ---
        if timeframe:
            pos["timeframe"] = timeframe

        logger.info(f"Updated risk parameters for {symbol} from LLM strategy_params")

    async def _execute_partial_tp_single(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
    ) -> None:
        """Execute a single partial take-profit sell for a position."""
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial TP for {symbol}: no position.")
            return

        fraction = pos.get("partial_take_profit_fraction")
        if fraction is None or fraction <= 0 or fraction >= 1:
            logger.warning(f"Invalid partial_take_profit_fraction for {symbol}: {fraction}")
            return

        sell_amount = pos["amount"] * fraction
        base, quote = symbol.split("/")

        # Check minimum sell size
        market = self.exchange.markets.get(symbol, {})
        limits = market.get("limits", {})
        min_amount = limits.get("amount", {}).get("min")
        min_cost = limits.get("cost", {}).get("min")
        if min_amount is not None and sell_amount < float(min_amount):
            logger.info(f"Partial TP sell amount {sell_amount:.6f} below min {min_amount} for {symbol}, skipping.")
            return
        if min_cost is not None and sell_amount * current_price < float(min_cost):
            logger.info(f"Partial TP sell cost {sell_amount * current_price:.2f} below min {min_cost} for {symbol}, skipping.")
            return

        fee_rate = get_fee_rate(self.exchange, symbol, self.redis)

        try:
            order = await asyncio.to_thread(
                self.trader.create_market_sell_order, symbol, sell_amount
            )
            logger.info(f"Partial TP SELL {symbol}: {sell_amount:.6f} @ {order.get('price', current_price):.4f}")

            # Use actual filled amount from the order
            filled_amount = order.get("amount", sell_amount)

            # Compute fee
            fee = order.get("fee", {})
            fee_cost = float(fee.get("cost", 0.0) or 0.0)
            fee_currency = fee.get("currency", "")
            if fee_cost == 0.0:
                fee_cost = order["cost"] * fee_rate
                fee_currency = quote
                order["fee"] = {"cost": fee_cost, "currency": fee_currency}

            # Prorated cost basis for the sold portion
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (filled_amount / net_base) if net_base > 0 else 0.0

            net_quote = order["cost"] - (fee_cost if fee_currency == quote else 0.0)
            realized_pnl = net_quote - prorated_cost_basis

            order["realized_pnl"] = realized_pnl
            order["cost_basis"] = prorated_cost_basis
            order["exit_reason"] = "partial_take_profit"
            order["strategy_type"] = pos.get("strategy_type", "unknown")
            order["timeframe"] = pos.get("timeframe")
            if "timestamp" in pos:
                order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0

            # Update position: reduce amount, cost_basis, net_base
            remaining_amount = pos["amount"] - filled_amount
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - filled_amount

            if remaining_amount <= 0 or remaining_net_base <= 0:
                # Position fully closed (shouldn't normally happen with partial, but handle gracefully)
                self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self._pending_entries.pop(symbol, None)
                await self._remove_coin_if_paused(symbol)
            else:
                self.positions[symbol]["amount"] = remaining_amount
                self.positions[symbol]["cost_basis"] = remaining_cost_basis
                self.positions[symbol]["net_base"] = remaining_net_base
                self.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                # Clear single partial TP flags
                self.positions[symbol].pop("partial_tp_triggered", None)
                self.positions[symbol].pop("_partial_tp_triggered_single", None)
                self.positions[symbol].pop("_partial_tp_single_review_count", None)

                # Check if remaining amount is dust
                is_dust = False
                if min_amount is not None and remaining_amount < float(min_amount):
                    is_dust = True
                elif min_cost is not None and remaining_amount * current_price < float(min_cost):
                    is_dust = True
                if is_dust:
                    logger.info(f"Remaining {remaining_amount:.6f} {base} is dust after partial TP for {symbol}, sweeping.")
                    await self._sweep_dust(symbol)

            self.trade_history.append(order)
            await asyncio.to_thread(insert_trade, order)
            if settings.TRADING_MODE == "paper":
                await asyncio.to_thread(save_paper_balances, self.trader.balances)
            await self._save_state()

            if self.notifier:
                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                await self.notifier.send_notification(
                    f"🔸 Partial TP SELL {symbol}: {filled_amount:.6f} @ {order.get('price', current_price):.4f} "
                    f"| P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Partial take-profit",
                        "amount": filled_amount,
                        "price": order.get("price", current_price),
                        "realized_pnl": realized_pnl,
                        "exit_reason": "partial_take_profit",
                    }
                )
        except Exception as e:
            logger.error(f"Partial TP sell failed for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_notification(
                    f"❌ Partial TP sell failed for {symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": f"Partial TP sell failed: {e}"[:200]}
                )

    async def _execute_partial_tp_level(
        self,
        symbol: str,
        level_index: int,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
    ) -> None:
        """Execute a partial take-profit sell for a specific level."""
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial TP level for {symbol}: no position.")
            return

        levels = pos.get("partial_take_profit_levels")
        if not levels or level_index >= len(levels):
            logger.warning(f"Invalid partial TP level index {level_index} for {symbol}")
            return

        level = levels[level_index]
        fraction = level.get("fraction")
        if fraction is None or fraction <= 0 or fraction >= 1:
            logger.warning(f"Invalid fraction for partial TP level {level_index} of {symbol}: {fraction}")
            return

        sell_amount = pos["amount"] * fraction
        base, quote = symbol.split("/")

        # Check minimum sell size
        market = self.exchange.markets.get(symbol, {})
        limits = market.get("limits", {})
        min_amount = limits.get("amount", {}).get("min")
        min_cost = limits.get("cost", {}).get("min")
        if min_amount is not None and sell_amount < float(min_amount):
            logger.info(f"Partial TP level {level_index} sell amount {sell_amount:.6f} below min for {symbol}, skipping.")
            return
        if min_cost is not None and sell_amount * current_price < float(min_cost):
            logger.info(f"Partial TP level {level_index} sell cost below min for {symbol}, skipping.")
            return

        fee_rate = get_fee_rate(self.exchange, symbol, self.redis)

        try:
            order = await asyncio.to_thread(
                self.trader.create_market_sell_order, symbol, sell_amount
            )
            logger.info(f"Partial TP level {level_index} SELL {symbol}: {sell_amount:.6f} @ {order.get('price', current_price):.4f}")

            # Use actual filled amount from the order
            filled_amount = order.get("amount", sell_amount)

            # Compute fee
            fee = order.get("fee", {})
            fee_cost = float(fee.get("cost", 0.0) or 0.0)
            fee_currency = fee.get("currency", "")
            if fee_cost == 0.0:
                fee_cost = order["cost"] * fee_rate
                fee_currency = quote
                order["fee"] = {"cost": fee_cost, "currency": fee_currency}

            # Prorated cost basis for the sold portion
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (filled_amount / net_base) if net_base > 0 else 0.0

            net_quote = order["cost"] - (fee_cost if fee_currency == quote else 0.0)
            realized_pnl = net_quote - prorated_cost_basis

            order["realized_pnl"] = realized_pnl
            order["cost_basis"] = prorated_cost_basis
            order["exit_reason"] = f"partial_take_profit_level_{level_index}"
            order["strategy_type"] = pos.get("strategy_type", "unknown")
            order["timeframe"] = pos.get("timeframe")
            if "timestamp" in pos:
                order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0

            # Mark this level as triggered
            if symbol in self.positions:
                triggered = self.positions[symbol].get("partial_tp_levels_triggered", [])
                if level_index not in triggered:
                    triggered.append(level_index)
                    self.positions[symbol]["partial_tp_levels_triggered"] = triggered
                # Clear depth wait state for this level
                if "partial_tp_depth_wait_start" in self.positions[symbol]:
                    self.positions[symbol]["partial_tp_depth_wait_start"].pop(level_index, None)

            # Update position: reduce amount, cost_basis, net_base
            remaining_amount = pos["amount"] - filled_amount
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - filled_amount

            if remaining_amount <= 0 or remaining_net_base <= 0:
                # Position fully closed
                self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self._pending_entries.pop(symbol, None)
                await self._remove_coin_if_paused(symbol)
            else:
                self.positions[symbol]["amount"] = remaining_amount
                self.positions[symbol]["cost_basis"] = remaining_cost_basis
                self.positions[symbol]["net_base"] = remaining_net_base
                self.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                # Clear partial TP review flags for this level
                self.positions[symbol].pop("_partial_tp_triggered", None)
                self.positions[symbol].pop("_partial_tp_review_count", None)
                triggered_levels = self.positions[symbol].get("_partial_tp_triggered_levels", [])
                self.positions[symbol]["_partial_tp_triggered_levels"] = [
                    x for x in triggered_levels if x != level_index
                ]

                # Check if remaining amount is dust
                is_dust = False
                if min_amount is not None and remaining_amount < float(min_amount):
                    is_dust = True
                elif min_cost is not None and remaining_amount * current_price < float(min_cost):
                    is_dust = True
                if is_dust:
                    logger.info(f"Remaining {remaining_amount:.6f} {base} is dust after partial TP for {symbol}, sweeping.")
                    await self._sweep_dust(symbol)

            self.trade_history.append(order)
            await asyncio.to_thread(insert_trade, order)
            if settings.TRADING_MODE == "paper":
                await asyncio.to_thread(save_paper_balances, self.trader.balances)
            await self._save_state()

            if self.notifier:
                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                await self.notifier.send_notification(
                    f"🔸 Partial TP level {level_index} SELL {symbol}: {filled_amount:.6f} @ {order.get('price', current_price):.4f} "
                    f"| P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": f"Partial take-profit level {level_index}",
                        "amount": filled_amount,
                        "price": order.get("price", current_price),
                        "realized_pnl": realized_pnl,
                        "exit_reason": f"partial_take_profit_level_{level_index}",
                        "level_index": level_index,
                    }
                )
        except Exception as e:
            logger.error(f"Partial TP level {level_index} sell failed for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_notification(
                    f"❌ Partial TP level {level_index} sell failed for {symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": f"Partial TP level sell failed: {e}"[:200]}
                )

    async def _sweep_dust(self, symbol: str):
        """Sell any remaining dust balance of a coin after a partial sell."""
        base = symbol.split("/")[0]
        try:
            balance = await asyncio.to_thread(self.trader.get_balance, base)
        except Exception as e:
            logger.warning(f"Dust sweep: could not fetch balance for {base}: {e}")
            return
        if balance <= 0:
            return

        try:
            ticker = self.ws_manager.get_ticker(symbol)
            if ticker is None:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            price = ticker["last"]
        except Exception as e:
            logger.warning(f"Dust sweep: could not fetch price for {symbol}: {e}")
            return

        market = self.exchange.markets.get(symbol, {})
        limits = market.get("limits", {})
        min_amount = limits.get("amount", {}).get("min")
        min_cost = limits.get("cost", {}).get("min")

        if min_amount is not None and balance < float(min_amount):
            logger.info(f"Dust sweep: {balance} {base} below min amount {min_amount}, cannot sell.")
            return
        if min_cost is not None and balance * price < float(min_cost):
            logger.info(f"Dust sweep: notional {balance * price:.4f} below min cost {min_cost}, cannot sell.")
            return

        try:
            order = await asyncio.to_thread(self.trader.create_market_sell_order, symbol, balance)
            logger.info(f"Dust sweep: sold {balance} {base} from {symbol} – order {order.get('id')}")

            # Record the dust sale in trade history for consistency
            fee_rate = get_fee_rate(self.exchange, symbol, self.redis)
            fee = order.get('fee', {})
            fee_cost = float(fee.get('cost', 0.0) or 0.0)
            fee_currency = fee.get('currency', '')
            if fee_cost == 0.0:
                fee_cost = order['cost'] * fee_rate
                fee_currency = symbol.split('/')[1]
                order['fee'] = {'cost': fee_cost, 'currency': fee_currency}

            pos = self.positions.get(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = order['cost'] - (fee_cost if fee_currency == symbol.split('/')[1] else 0.0)
                realized_pnl = net_quote - cost_basis
                order["realized_pnl"] = realized_pnl
                order["cost_basis"] = cost_basis
                order["exit_reason"] = "dust_sweep"
                order["strategy_type"] = pos.get("strategy_type", "unknown")
                order["timeframe"] = pos.get("timeframe")
                if "timestamp" in pos:
                    order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0
                self.trade_history.append(order)
                await asyncio.to_thread(insert_trade, order)

            # Remove the now-empty position
            self.positions.pop(symbol, None)
            self._strategy_intervals.pop(symbol, None)
            self._last_strategy_eval.pop(symbol, None)
            await self._remove_coin_if_paused(symbol)

            if settings.TRADING_MODE == "paper":
                await asyncio.to_thread(save_paper_balances, self.trader.balances)

            if self.notifier:
                await self.notifier.send_notification(
                    f"🧹 Dust sweep: sold remaining {balance} {base} from {symbol}",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Dust sweep",
                        "amount": balance,
                        "exit_reason": "dust_sweep",
                    }
                )
        except Exception as e:
            logger.error(f"Dust sweep failed for {symbol}: {e}")

    def _is_excluded(self, symbol: str, timeframe: str) -> bool:
        """Return True if (symbol, timeframe) is in the EXCLUDED_PAIRS list."""
        for entry in settings.EXCLUDED_PAIRS:
            parts = entry.split("/")
            if len(parts) == 2:
                # "SYMBOL" or "SYMBOL/*" → exclude all timeframes
                if parts[0] == symbol.split("/")[0] and parts[1] == symbol.split("/")[1]:
                    return True
            elif len(parts) == 3:
                # "SYMBOL/TIMEFRAME" → exclude only that specific timeframe
                if (parts[0] == symbol.split("/")[0] and
                    parts[1] == symbol.split("/")[1] and
                    parts[2] == timeframe):
                    return True
        return False

    async def _remove_coin_if_paused(self, symbol: str):
        """If trading is paused, remove the symbol from current_coins to prevent new signals."""
        # Always clear any pending entry for this symbol
        self._pending_entries.pop(symbol, None)
        paused_raw = await asyncio.to_thread(self.redis.get, "trading:paused")
        if paused_raw and paused_raw == b"1":
            self.current_coins = [c for c in self.current_coins if c["symbol"] != symbol]
            logger.info(f"Trading paused: removed {symbol} from current_coins after position closed.")
