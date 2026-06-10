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
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the WebSocket watch loop."""
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self):
        """Stop the watch loop and close the connection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.exchange.close()

    async def _watch_loop(self):
        """Continuously watch tickers for all subscribed symbols."""
        while self._running:
            try:
                tickers = await self.exchange.watch_tickers(list(self.symbols))
                for symbol, ticker in tickers.items():
                    self.tickers[symbol] = ticker
                    await self._ticker_queue.put((symbol, ticker))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket watch loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the latest ticker for a symbol, or None if not available."""
        return self.tickers.get(symbol)

    async def update_subscriptions(self, symbols: List[str]):
        """Update the set of symbols to watch."""
        new_symbols = set(symbols)
        if new_symbols != self.symbols:
            self.symbols = new_symbols
            logger.info(f"WebSocket subscriptions updated: {len(self.symbols)} symbols")
