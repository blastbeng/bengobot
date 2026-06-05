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
        allowed = {"newsapi", "twitter", "reddit"}
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
        if self.NEWS_ENABLED:
            if "newsapi" in self.NEWS_SOURCES and not self.NEWS_API_KEY:
                raise ValueError("NEWS_API_KEY is required when NEWS_ENABLED and newsapi source is selected")
            if "twitter" in self.NEWS_SOURCES and not self.TWITTER_BEARER_TOKEN:
                raise ValueError("TWITTER_BEARER_TOKEN is required when twitter source is selected")
            if "reddit" in self.NEWS_SOURCES and (not self.REDDIT_CLIENT_ID or not self.REDDIT_CLIENT_SECRET):
                raise ValueError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are required when reddit source is selected")
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
