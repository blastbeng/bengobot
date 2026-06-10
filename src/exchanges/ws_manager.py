import asyncio
import logging
from typing import Dict, List, Optional, Any
import ccxt.pro as ccxt_pro

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, exchange: ccxt_pro.Exchange, symbols: List[str]):
        self.exchange = exchange
        self.symbols = set(symbols)
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self._ticker_queue = asyncio.Queue()
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self._order_book_tasks: Dict[str, asyncio.Task] = {}
        self._ticker_tasks: Dict[str, asyncio.Task] = {}
        self.trades: Dict[str, List[Dict[str, Any]]] = {}
        self._trade_tasks: Dict[str, asyncio.Task] = {}
        self._use_batch_tickers = True  # will be set to False if batch not supported
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_lock = asyncio.Lock()

    async def start(self):
        """Start the WebSocket watch loop."""
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())

    async def _reconnect(self):
        """Close and recreate the exchange connection, then re-subscribe."""
        async with self._reconnect_lock:
            logger.warning("WebSocket reconnecting...")
            try:
                await self.exchange.close()
            except Exception:
                pass
            # Clear cached data to avoid stale prices
            self.tickers.clear()
            self.order_books.clear()
            self.trades.clear()
            # Re-create the pro exchange
            from src.exchanges.factory import get_pro_exchange
            self.exchange = get_pro_exchange()
            # If using per-symbol tickers, restart those tasks
            if not self._use_batch_tickers:
                for sym in list(self._ticker_tasks.keys()):
                    task = self._ticker_tasks.pop(sym)
                    task.cancel()
                for sym in self.symbols:
                    task = asyncio.create_task(self._watch_ticker(sym))
                    self._ticker_tasks[sym] = task
            # Order book tasks will automatically reconnect on next iteration
            logger.info("WebSocket reconnection complete.")

    async def _watch_ticker(self, symbol: str):
        """Continuously watch the ticker for a single symbol."""
        backoff = 1
        max_backoff = 30
        while self._running and symbol in self.symbols:
            try:
                ticker = await self.exchange.watch_ticker(symbol)
                self.tickers[symbol] = ticker
                await self._ticker_queue.put((symbol, ticker))
                backoff = 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ticker watch error for {symbol}: {e}", exc_info=True)
                await self._reconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def _watch_trades(self, symbol: str):
        """Continuously watch trades for a single symbol."""
        backoff = 1
        max_backoff = 30
        while self._running and symbol in self.symbols:
            try:
                trades = await self.exchange.watch_trades(symbol)
                # Keep only the last 50 trades to limit memory
                self.trades[symbol] = trades[-50:]
                backoff = 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trade watch error for {symbol}: {e}", exc_info=True)
                await self._reconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def stop(self):
        """Stop the watch loop and close the connection."""
        self._running = False
        # Cancel ticker watch task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Cancel all order book tasks
        for task in self._order_book_tasks.values():
            task.cancel()
        await asyncio.gather(*self._order_book_tasks.values(), return_exceptions=True)
        self._order_book_tasks.clear()
        # Cancel all per-symbol ticker tasks
        for task in self._ticker_tasks.values():
            task.cancel()
        await asyncio.gather(*self._ticker_tasks.values(), return_exceptions=True)
        self._ticker_tasks.clear()
        # Cancel all trade watch tasks
        for task in self._trade_tasks.values():
            task.cancel()
        await asyncio.gather(*self._trade_tasks.values(), return_exceptions=True)
        self._trade_tasks.clear()
        await self.exchange.close()

    async def _watch_loop(self):
        """Continuously watch tickers for all subscribed symbols."""
        backoff = 1
        max_backoff = 60
        while self._running:
            if not self.symbols:
                await asyncio.sleep(1)
                continue
            if not self._use_batch_tickers:
                # Per-symbol tasks handle tickers; just keep loop alive
                await asyncio.sleep(1)
                continue
            try:
                tickers = await self.exchange.watch_tickers(list(self.symbols))
                backoff = 1  # reset on success
                for symbol, ticker in tickers.items():
                    self.tickers[symbol] = ticker
                    await self._ticker_queue.put((symbol, ticker))
            except asyncio.CancelledError:
                break
            except (ccxt_pro.NotSupported, ccxt_pro.BadSymbol):
                logger.warning("watch_tickers not supported, falling back to per-symbol watch_ticker")
                self._use_batch_tickers = False
                # Start per-symbol tasks for current symbols
                for sym in list(self.symbols):
                    if sym not in self._ticker_tasks:
                        task = asyncio.create_task(self._watch_ticker(sym))
                        self._ticker_tasks[sym] = task
            except Exception as e:
                logger.error(f"WebSocket watch loop error: {e}", exc_info=True)
                await self._reconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    @property
    def healthy(self) -> bool:
        """Return True if the ticker watch task is alive and the exchange is connected."""
        return self._running and self._task is not None and not self._task.done()

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the latest ticker for a symbol, or None if not available."""
        return self.tickers.get(symbol)

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the latest order book for a symbol, or None if not available."""
        return self.order_books.get(symbol)

    def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Return the latest trades for a symbol, or empty list if not available."""
        return self.trades.get(symbol, [])

    async def _watch_order_book(self, symbol: str):
        """Continuously watch the order book for a single symbol."""
        backoff = 1
        max_backoff = 30
        while self._running and symbol in self.symbols:
            try:
                ob = await self.exchange.watch_order_book(symbol)
                self.order_books[symbol] = ob
                backoff = 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order book watch error for {symbol}: {e}", exc_info=True)
                await self._reconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def update_subscriptions(self, symbols: List[str]):
        """Update the set of symbols to watch (tickers + order books)."""
        new_symbols = set(symbols)
        if new_symbols == self.symbols:
            return

        removed = self.symbols - new_symbols
        added = new_symbols - self.symbols

        # Stop order book watchers for removed symbols
        for sym in removed:
            task = self._order_book_tasks.pop(sym, None)
            if task:
                task.cancel()
            self.order_books.pop(sym, None)

        # Start order book watchers for new symbols
        for sym in added:
            if sym not in self._order_book_tasks:
                task = asyncio.create_task(self._watch_order_book(sym))
                self._order_book_tasks[sym] = task

        # Manage per-symbol ticker tasks if not using batch
        if not self._use_batch_tickers:
            for sym in removed:
                task = self._ticker_tasks.pop(sym, None)
                if task:
                    task.cancel()
                self.tickers.pop(sym, None)
            for sym in added:
                if sym not in self._ticker_tasks:
                    task = asyncio.create_task(self._watch_ticker(sym))
                    self._ticker_tasks[sym] = task

        # Manage trade watchers
        for sym in removed:
            task = self._trade_tasks.pop(sym, None)
            if task:
                task.cancel()
            self.trades.pop(sym, None)
        for sym in added:
            if sym not in self._trade_tasks:
                task = asyncio.create_task(self._watch_trades(sym))
                self._trade_tasks[sym] = task

        self.symbols = new_symbols
        logger.info(f"WebSocket subscriptions updated: {len(self.symbols)} symbols")

    async def wait_for_update(self, timeout: float = 5.0) -> Optional[tuple]:
        """Wait for the next ticker update, or return None after timeout."""
        try:
            return await asyncio.wait_for(self._ticker_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
