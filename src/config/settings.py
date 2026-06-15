from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Exchange
    EXCHANGE_ID: str = "binance"
    EXCHANGE_API_KEY: Optional[str] = None
    EXCHANGE_SECRET: Optional[str] = None
    EXCHANGE_PASSWORD: Optional[str] = None
    EXCHANGE_TIMEOUT: int = 30000  # milliseconds – timeout for all exchange API calls

    # CoinMarketCap API key for altcoin season index
    CMC_API_KEY: str = ""

    # Trading mode
    TRADING_MODE: str = "paper"   # "paper" or "live"

    # Risk management check interval (seconds) – stop-loss/take-profit checks
    RISK_CHECK_INTERVAL_SECONDS: int = 15

    # Base currency
    BASE_CURRENCY: str = "USDT"

    # Max coins to trade
    MAX_COINS: int = 10

    # Coin selection
    COIN_SELECTION_MAX_PAIRS: int = 200          # max pairs to include in the LLM prompt
    COIN_SELECTION_MIN_SENTIMENT: float = -1.0   # minimum aggregate sentiment compound to consider a coin (-1.0 = no filter)

    # Limit coin selection to top N by 24h volume (reduces noise)
    COIN_SELECTION_TOP_VOLUME_LIMIT: int = 100

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

    @field_validator("COIN_SELECTION_TOP_VOLUME_LIMIT")
    @classmethod
    def validate_top_volume_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("COIN_SELECTION_TOP_VOLUME_LIMIT must be at least 1")
        return v

    # Maximum number of consecutive "keep paused" LLM decisions before the engine
    # force‑resumes trading with a reduced risk multiplier.
    PAUSE_MAX_CONSECUTIVE_KEEP: int = 3

    @field_validator("PAUSE_MAX_CONSECUTIVE_KEEP")
    @classmethod
    def validate_pause_max_consecutive_keep(cls, v: int) -> int:
        if v < 1:
            raise ValueError("PAUSE_MAX_CONSECUTIVE_KEEP must be at least 1")
        return v

    # Global risk multiplier applied when the engine force‑resumes after
    # PAUSE_MAX_CONSECUTIVE_KEEP consecutive "keep paused" decisions.
    PAUSE_FORCE_RESUME_RISK_MULTIPLIER: float = 0.3

    @field_validator("PAUSE_FORCE_RESUME_RISK_MULTIPLIER")
    @classmethod
    def validate_pause_force_resume_risk_multiplier(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("PAUSE_FORCE_RESUME_RISK_MULTIPLIER must be between 0.0 and 1.0")
        return v

    # Fallback coin selection: minimum 24h quote volume (in base currency) required
    # for a coin to be considered when the LLM returns no coins.
    # Set to 0 to disable the filter.
    FALLBACK_MIN_24H_VOLUME: float = 0.0

    @field_validator("FALLBACK_MIN_24H_VOLUME")
    @classmethod
    def validate_fallback_min_24h_volume(cls, v: float) -> float:
        if v < 0:
            raise ValueError("FALLBACK_MIN_24H_VOLUME must be >= 0")
        return v

    # OHLCV timeframes for multi-timeframe analysis
    OHLCV_TIMEFRAMES: list[str] = ["5m", "15m", "1h", "4h"]

    # Market data download interval (seconds)
    MARKET_DATA_REFRESH_SECONDS: int = 300

    # OHLCV download staggering
    OHLCV_DOWNLOAD_COIN_DELAY_SECONDS: float = 2.0

    # Maximum number of OHLCV candles to insert in a single backfill call.
    # Prevents memory exhaustion and timeouts when backfilling large ranges.
    BACKFILL_MAX_CANDLES_PER_CALL: int = 5000

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

    @model_validator(mode="after")
    def set_database_path(self):
        if "DATABASE_PATH" not in self.model_fields_set:
            if self.TRADING_MODE == "paper":
                self.DATABASE_PATH = "data/paper.db"
            else:
                self.DATABASE_PATH = "data/bengobot.db"
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

    # Mind model (complex reasoning: coin selection, strategy generation)
    OLLAMA_MIND_MODEL: str = "deepseek-v4-flash"
    OPENAI_MIND_MODEL: str = "gpt-4o"

    # Actuator model (fast, time‑critical decisions: stop‑loss/take‑profit reviews, corrections)
    OLLAMA_ACTUATOR_MODEL: str = "deepseek-v4-flash"
    OPENAI_ACTUATOR_MODEL: str = "gpt-4o-mini"

    # Per‑role provider overrides (empty = use global LLM_PROVIDER)
    LLM_MIND_PROVIDER: str = ""
    LLM_ACTUATOR_PROVIDER: str = ""

    # Per‑role OpenAI settings (empty or None = use global OPENAI_*)
    OPENAI_MIND_API_KEY: Optional[str] = None
    OPENAI_ACTUATOR_API_KEY: Optional[str] = None
    OPENAI_MIND_BASE_URL: str = ""
    OPENAI_ACTUATOR_BASE_URL: str = ""

    # Per‑role Ollama settings (empty or None = use global OLLAMA_*)
    OLLAMA_MIND_BASE_URL: str = ""
    OLLAMA_ACTUATOR_BASE_URL: str = ""
    OLLAMA_MIND_API_KEY: Optional[str] = None
    OLLAMA_ACTUATOR_API_KEY: Optional[str] = None

    # LLM temperature (applies to both providers)
    LLM_TEMPERATURE: float = 0.1

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_llm_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")
        return v

    # LLM timeout (seconds) for HTTP requests
    LLM_TIMEOUT: float = 300.0

    @field_validator("LLM_TIMEOUT")
    @classmethod
    def validate_llm_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        return v

    # Maximum slippage allowed when capping buy order size (0.0 = no cap)
    MAX_SLIPPAGE_CAP_PCT: float = 0.1

    @field_validator("MAX_SLIPPAGE_CAP_PCT")
    @classmethod
    def validate_max_slippage_cap_pct(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("MAX_SLIPPAGE_CAP_PCT must be >= 0")
        return v

    # Enforce the LLM's minimum profit per trade check.
    # Set to False to allow trades with very small expected profit.
    ENFORCE_MIN_PROFIT_PER_TRADE: bool = True

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Data directory for logs, database, etc.
    DATA_DIR: str = "data"

    # Database
    DATABASE_PATH: str = "data/bengobot.db"

    # News
    NEWS_ENABLED: bool = False
    NEWS_UPDATE_INTERVAL_MINUTES: int = 60

    # Fast news refresh for currently tracked coins (minutes)
    NEWS_FAST_UPDATE_INTERVAL_MINUTES: int = 5

    NEWS_API_KEY: Optional[str] = None       # for NewsAPI.org
    TWITTER_BEARER_TOKEN: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "trading-bot/1.0"
    NEWS_MAX_ARTICLES_PER_SYMBOL: int = 5
    NEWS_CACHE_TTL_SECONDS: int = 1800       # 30 minutes
    NEWS_HTTP_TIMEOUT_SECONDS: float = 30.0   # timeout for each news source HTTP request
    NEWS_INITIAL_FETCH_TIMEOUT_SECONDS: float = 60.0   # max seconds for initial news fetch on startup
    NEWS_RETENTION_SECONDS: int = 86400   # delete articles older than 24 hours

    # Facebook (Graph API)
    FACEBOOK_PAGE_ACCESS_TOKEN: Optional[str] = None
    FACEBOOK_PAGE_ID: Optional[str] = None
    FACEBOOK_POST_LIMIT: int = 5

    # RSS Feeds
    RSS_FEEDS: list[str] = [
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://cryptoslate.com/feed/",
        "https://bitcoinmagazine.com/feed",
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


    # News-driven coin discovery
    NEWS_COIN_DISCOVERY_ENABLED: bool = False
    NEWS_COIN_DISCOVERY_MAX_COINS: int = 5          # max new coins to add from news
    NEWS_COIN_DISCOVERY_MIN_SENTIMENT: float = 0.3  # minimum avg compound to consider
    NEWS_COIN_DISCOVERY_MIN_ARTICLES: int = 3       # minimum articles to be considered

    # Fear & Greed Index
    FEAR_GREED_ENABLED: bool = True
    FEAR_GREED_CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # Rate limiting for news providers
    NEWS_RATE_LIMIT_ENABLED: bool = True
    NEWS_RATE_LIMIT_PER_SOURCE_SECONDS: float = 1.0   # minimum seconds between requests to the same source

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Web
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8083

    # Logging
    LOG_LEVEL: str = "INFO"

    # Notification log control
    NOTIFICATION_LOG_ENABLED: bool = True

    # Notification verbosity: "all", "errors_only", "trades_only", or "none"
    NOTIFICATION_VERBOSITY: str = "all"

    @field_validator("NOTIFICATION_VERBOSITY")
    @classmethod
    def validate_notification_verbosity(cls, v: str) -> str:
        allowed = {"all", "errors_only", "trades_only", "none"}
        if v not in allowed:
            raise ValueError(f"NOTIFICATION_VERBOSITY must be one of {allowed}")
        return v

    def reload(self):
        """Reload settings from .env file and environment variables."""
        new_settings = self.__class__()
        for field in self.__fields__:
            setattr(self, field, getattr(new_settings, field))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
