from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Exchange
    EXCHANGE_ID: str = "binance"
    EXCHANGE_API_KEY: Optional[str] = None
    EXCHANGE_SECRET: Optional[str] = None
    EXCHANGE_PASSWORD: Optional[str] = None

    # Trading mode
    TRADING_MODE: str = "paper"   # "paper" or "live"

    # Base currency
    BASE_CURRENCY: str = "USDT"

    # Max coins to trade
    MAX_COINS: int = 10

    @field_validator("TRADING_MODE")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        if v not in ("paper", "live"):
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
        return v

    @field_validator("MAX_COINS")
    @classmethod
    def validate_max_coins(cls, v: int) -> int:
        if v < 1:
            raise ValueError("MAX_COINS must be at least 1")
        return v

    # OHLCV timeframes for multi-timeframe analysis
    OHLCV_TIMEFRAMES: list[str] = ["5m", "15m", "1h", "4h"]

    @field_validator("OHLCV_TIMEFRAMES")
    @classmethod
    def validate_ohlcv_timeframes(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or not all(isinstance(tf, str) for tf in v):
            raise ValueError("OHLCV_TIMEFRAMES must be a list of strings")
        return v

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        if v not in ("ollama", "openai"):
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'")
        return v

    @field_validator("NEWS_SOURCES")
    @classmethod
    def validate_news_sources(cls, v: list[str]) -> list[str]:
        allowed = {"newsapi", "twitter", "reddit", "facebook", "youtube", "cryptopanic", "coingecko", "cryptocompare", "lunarcrush", "santiment", "messari", "coinmarketcap", "googlenews", "stocktwits", "coinpaprika", "coincodex"}
        for source in v:
            if source not in allowed:
                raise ValueError(f"Invalid news source: {source}. Allowed: {allowed}")
        return v

    # Paper trading
    PAPER_INITIAL_BALANCE: float = 10000.0

    @field_validator("PAPER_INITIAL_BALANCE")
    @classmethod
    def validate_paper_initial_balance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("PAPER_INITIAL_BALANCE must be positive")
        return v

    @model_validator(mode="after")
    def check_credentials(self):
        if self.TRADING_MODE == "live":
            if not self.EXCHANGE_API_KEY or not self.EXCHANGE_SECRET:
                raise ValueError(
                    "EXCHANGE_API_KEY and EXCHANGE_SECRET are required when TRADING_MODE='live'"
                )
        return self

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-v4-flash"
    OLLAMA_API_KEY: Optional[str] = None

    # LLM Provider selection
    LLM_PROVIDER: str = "ollama"   # "ollama" or "openai"

    # OpenAI-compatible API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o"

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Database
    DATABASE_PATH: str = "data/bot.db"

    # News
    NEWS_ENABLED: bool = False
    NEWS_UPDATE_INTERVAL_MINUTES: int = 15

    # News sources
    NEWS_SOURCES: list[str] = ["newsapi"]  # supported: "newsapi", "twitter", "reddit"
    NEWS_API_KEY: Optional[str] = None       # for NewsAPI.org
    TWITTER_BEARER_TOKEN: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "trading-bot/1.0"
    NEWS_MAX_ARTICLES_PER_SYMBOL: int = 5
    NEWS_CACHE_TTL_SECONDS: int = 900        # 15 minutes

    # Facebook (Graph API)
    FACEBOOK_PAGE_ACCESS_TOKEN: Optional[str] = None
    FACEBOOK_PAGE_ID: Optional[str] = None
    FACEBOOK_POST_LIMIT: int = 5

    # RSS Feeds
    RSS_FEEDS: list[str] = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/rss",
        "https://cryptoslate.com/feed/",
        "https://ambcrypto.com/feed/",
        "https://bitcoinmagazine.com/.rss/full/",
        "https://bitcoinist.com/feed/",
        "https://www.newsbtc.com/feed/",
        "https://cryptopotato.com/feed/",
        "https://coinjournal.net/feed/",
        "https://cryptobriefing.com/feed/",
        "https://beincrypto.com/feed/",
        "https://coingape.com/feed/",
        "https://dailyhodl.com/feed/",
        "https://www.cryptonewsz.com/feed/",
        "https://www.cryptopolitan.com/feed/",
    ]

    # YouTube Data API v3
    YOUTUBE_API_KEY: Optional[str] = None
    YOUTUBE_MAX_RESULTS: int = 5

    # CryptoPanic API
    CRYPTOPANIC_API_KEY: Optional[str] = None
    CRYPTOPANIC_MAX_POSTS: int = 5

    # CoinGecko News (free, no API key required)
    COINGECKO_MAX_ARTICLES: int = 5

    # CryptoCompare News API
    CRYPTOCOMPARE_API_KEY: Optional[str] = None
    CRYPTOCOMPARE_MAX_ARTICLES: int = 5

    # LunarCrush API
    LUNARCRUSH_API_KEY: Optional[str] = None
    LUNARCRUSH_MAX_ARTICLES: int = 5

    # Santiment API
    SANTIMENT_API_KEY: Optional[str] = None
    SANTIMENT_MAX_ARTICLES: int = 5

    # Messari API
    MESSARI_API_KEY: Optional[str] = None
    MESSARI_MAX_ARTICLES: int = 5

    # CoinMarketCap API
    COINMARKETCAP_API_KEY: Optional[str] = None
    COINMARKETCAP_MAX_ARTICLES: int = 5

    # Google News RSS (free, no API key)
    GOOGLE_NEWS_MAX_ARTICLES: int = 5

    # StockTwits API
    STOCKTWITS_API_KEY: Optional[str] = None
    STOCKTWITS_MAX_POSTS: int = 5

    # CoinPaprika (free, no API key)
    COINPAPRIKA_MAX_ARTICLES: int = 5

    # CoinCodex (free, no API key)
    COINCODEX_MAX_ARTICLES: int = 5

    # News sentiment risk adjustment
    NEWS_SENTIMENT_RISK_ADJUSTMENT: bool = False
    NEWS_SENTIMENT_NEGATIVE_THRESHOLD: float = -0.5   # compound score below this is very negative
    NEWS_SENTIMENT_POSITIVE_THRESHOLD: float = 0.5    # above this is very positive
    NEWS_SENTIMENT_POSITION_SIZE_MULTIPLIER_NEGATIVE: float = 0.5  # reduce position to 50% if negative
    NEWS_SENTIMENT_POSITION_SIZE_MULTIPLIER_POSITIVE: float = 1.0  # keep normal if positive
    NEWS_SENTIMENT_SKIP_BUY_ON_VERY_NEGATIVE: bool = True  # skip BUY entirely if sentiment below negative threshold

    # News sentiment risk management for open positions
    NEWS_SENTIMENT_EXIT_ON_VERY_NEGATIVE: bool = False   # force close position if sentiment very negative
    NEWS_SENTIMENT_EXIT_THRESHOLD: float = -0.7           # compound score below this triggers forced exit
    NEWS_SENTIMENT_TIGHTEN_STOP: bool = False             # tighten stop-loss when sentiment turns negative
    NEWS_SENTIMENT_TIGHTEN_STOP_THRESHOLD: float = -0.3   # sentiment below this triggers stop tightening
    NEWS_SENTIMENT_TIGHTEN_STOP_MULTIPLIER: float = 0.5   # multiply stop distance by this factor (0.5 = halve distance)

    # News-driven coin discovery
    NEWS_COIN_DISCOVERY_ENABLED: bool = False
    NEWS_COIN_DISCOVERY_MAX_COINS: int = 5          # max new coins to add from news
    NEWS_COIN_DISCOVERY_MIN_SENTIMENT: float = 0.3  # minimum avg compound to consider
    NEWS_COIN_DISCOVERY_MIN_ARTICLES: int = 3       # minimum articles to be considered

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Web
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8083

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
