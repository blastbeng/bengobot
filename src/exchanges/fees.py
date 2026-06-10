import logging
from typing import Optional
import redis
import ccxt
from src.utils.retry import retry_on_rate_limit

logger = logging.getLogger(__name__)

FEE_CACHE_TTL = 86400  # 1 day in seconds

# Known default taker fees for exchanges that require API keys for fee lookup.
# These are used when credentials are missing, avoiding noisy warnings.
EXCHANGE_DEFAULT_FEES = {
    "kucoin": 0.001,   # 0.1%
    # Add other exchanges here as needed
}


@retry_on_rate_limit()
def get_fee_rate(
    exchange: ccxt.Exchange,
    symbol: str,
    redis_client: Optional[redis.Redis] = None,
    default: float = 0.001,
) -> float:
    """
    Return the taker fee rate for a symbol.
    Uses Redis cache if available, otherwise fetches from the exchange.
    Falls back to `default` on any error.
    """
    cache_key = f"fee_rate:{symbol}"

    # Try Redis cache first
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached is not None:
                return float(cached)
        except Exception as e:
            logger.warning(f"Redis get failed for fee rate: {e}")

    # Fetch from exchange
    exchange_id = exchange.id.lower() if hasattr(exchange, 'id') else ''
    if exchange_id in EXCHANGE_DEFAULT_FEES and not getattr(exchange, 'apiKey', None):
        # Exchange requires API key but none is set – use known default silently
        rate = EXCHANGE_DEFAULT_FEES[exchange_id]
    else:
        try:
            fees = exchange.fetch_trading_fee(symbol)
            taker = fees.get('taker', fees.get('maker', default))
            rate = float(taker)
        except Exception as e:
            logger.warning(f"Could not fetch trading fee for {symbol}: {e}. Using default {default}")
            rate = default

    # Store in Redis
    if redis_client:
        try:
            redis_client.setex(cache_key, FEE_CACHE_TTL, str(rate))
        except Exception as e:
            logger.warning(f"Redis setex failed for fee rate: {e}")

    logger.debug(f"Fee rate for {symbol}: {rate} (exchange={exchange_id}, cached={cached is not None if redis_client else False})")
    return rate
