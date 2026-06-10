import ccxt
import ccxt.pro as ccxt_pro
from src.config.settings import settings

SUPPORTED_EXCHANGES = {
    "binance": ccxt.binance,
    "kraken": ccxt.kraken,
    "kucoin": ccxt.kucoin,
    "coinbase": ccxt.coinbase,
}

def get_exchange() -> ccxt.Exchange:
    """Return a configured ccxt exchange instance based on settings."""
    exchange_id = settings.EXCHANGE_ID.lower()
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Unsupported exchange: {exchange_id}. "
            f"Supported: {list(SUPPORTED_EXCHANGES.keys())}"
        )

    exchange_class = SUPPORTED_EXCHANGES[exchange_id]
    config = {}

    if settings.TRADING_MODE == "live":
        config["apiKey"] = settings.EXCHANGE_API_KEY
        config["secret"] = settings.EXCHANGE_SECRET
        if settings.EXCHANGE_PASSWORD:
            config["password"] = settings.EXCHANGE_PASSWORD

    exchange = exchange_class(config)
    exchange.timeout = settings.EXCHANGE_TIMEOUT

    if settings.TRADING_MODE == "paper":
        # Enable sandbox mode only if the exchange truly supports it
        if exchange.has.get("sandbox", False):
            exchange.set_sandbox_mode(True)

    exchange.enableRateLimit = True
    return exchange


def get_pro_exchange() -> ccxt_pro.Exchange:
    """Return a configured ccxt.pro exchange instance for WebSocket streams."""
    exchange_id = settings.EXCHANGE_ID.lower()
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Unsupported exchange: {exchange_id}. "
            f"Supported: {list(SUPPORTED_EXCHANGES.keys())}"
        )

    exchange_class = getattr(ccxt_pro, exchange_id)
    config = {}

    if settings.TRADING_MODE == "live":
        config["apiKey"] = settings.EXCHANGE_API_KEY
        config["secret"] = settings.EXCHANGE_SECRET
        if settings.EXCHANGE_PASSWORD:
            config["password"] = settings.EXCHANGE_PASSWORD

    exchange = exchange_class(config)
    exchange.timeout = settings.EXCHANGE_TIMEOUT

    if settings.TRADING_MODE == "paper":
        if exchange.has.get("sandbox", False):
            exchange.set_sandbox_mode(True)

    exchange.enableRateLimit = True
    return exchange
