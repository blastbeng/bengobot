import ccxt
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_available_pairs(exchange: ccxt.Exchange, base_currency: str) -> List[str]:
    """Return list of trading pairs that have the given base currency (e.g., 'USDT')."""
    exchange.load_markets()
    pairs = []
    for symbol, market in exchange.markets.items():
        if market.get('quote') == base_currency and market.get('active', False):
            pairs.append(symbol)
    return pairs

def get_tickers(exchange: ccxt.Exchange, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fetch tickers for given symbols. If symbols is None, fetch all.

    For KuCoin and other exchanges that may 404 on bulk fetch_tickers,
    falls back to fetching each ticker individually.
    """
    params = {}
    if exchange.id == 'kucoin':
        params['type'] = 'spot'
    if symbols:
        try:
            return exchange.fetch_tickers(symbols, params=params)
        except Exception as e:
            logger.warning(
                "fetch_tickers failed for %s (%s); falling back to individual fetch_ticker calls",
                exchange.id, e,
            )
            # Fallback: fetch each ticker individually
            tickers = {}
            for sym in symbols:
                try:
                    tickers[sym] = exchange.fetch_ticker(sym, params=params)
                except Exception as inner_e:
                    logger.warning("Failed to fetch ticker for %s: %s", sym, inner_e)
            return tickers
    else:
        try:
            return exchange.fetch_tickers(params=params)
        except Exception as e:
            logger.warning(
                "fetch_tickers (all) failed for %s (%s); returning empty dict",
                exchange.id, e,
            )
            return {}

def get_order_book(exchange: ccxt.Exchange, symbol: str, limit: int = 20) -> Dict[str, Any]:
    """Fetch order book for a symbol."""
    return exchange.fetch_order_book(symbol, limit)

def get_multi_timeframe_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframes: List[str],
    limit: int = 24
) -> Dict[str, List[List[float]]]:
    """
    Fetch OHLCV data for a symbol across multiple timeframes.
    Returns a dict mapping timeframe -> list of candles.
    """
    result = {}
    for tf in timeframes:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            result[tf] = ohlcv
        except Exception as e:
            logger.warning("Failed to fetch OHLCV for %s %s: %s", symbol, tf, e)
            result[tf] = []
    return result
