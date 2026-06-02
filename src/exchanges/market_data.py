import ccxt
from typing import List, Dict, Any, Optional

def get_available_pairs(exchange: ccxt.Exchange, base_currency: str) -> List[str]:
    """Return list of trading pairs that have the given base currency (e.g., 'USDT')."""
    exchange.load_markets()
    pairs = []
    for symbol, market in exchange.markets.items():
        if market.get('quote') == base_currency and market.get('active', False):
            pairs.append(symbol)
    return pairs

def get_tickers(exchange: ccxt.Exchange, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fetch tickers for given symbols. If symbols is None, fetch all."""
    if symbols:
        return exchange.fetch_tickers(symbols)
    else:
        return exchange.fetch_tickers()

def get_order_book(exchange: ccxt.Exchange, symbol: str, limit: int = 10) -> Dict[str, Any]:
    """Fetch order book for a symbol."""
    return exchange.fetch_order_book(symbol, limit)
