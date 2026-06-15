import logging
import threading
from typing import List, Dict, Any, Optional
import hashlib
import httpx
import json
import time
import feedparser

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config.settings import settings
from src.database import get_aggregate_sentiment_from_db
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

_sentiment_analyzer = SentimentIntensityAnalyzer()

# Cache for RSS feed content: {url: (timestamp, feed_content)}
_rss_cache = {}
_rss_cache_lock = threading.Lock()
_rss_cache_last_cleanup = 0.0


def _cleanup_rss_cache():
    """Remove RSS cache entries older than 10 minutes."""
    global _rss_cache_last_cleanup
    now = time.time()
    # Throttle: only run cleanup once per 60 seconds
    if now - _rss_cache_last_cleanup < 60:
        return
    with _rss_cache_lock:
        expired = [
            url for url, (ts, _) in _rss_cache.items()
            if now - ts > 600  # 10 minutes
        ]
        for url in expired:
            del _rss_cache[url]
    _rss_cache_last_cleanup = now


class RateLimiter:
    """Thread-safe per-source rate limiter."""
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_request: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, source: str):
        """Block until the required interval has passed since the last request for this source."""
        if not settings.NEWS_RATE_LIMIT_ENABLED:
            return
        with self._lock:
            now = time.time()
            last = self._last_request.get(source, 0.0)
            wait_time = self.min_interval - (now - last)
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()  # re-read after sleep
            self._last_request[source] = now


# Global rate limiter instance, initialized lazily to avoid import order issues.
_rate_limiter: Optional[RateLimiter] = None


def _get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(settings.NEWS_RATE_LIMIT_PER_SOURCE_SECONDS)
    return _rate_limiter


def _get_enabled_sources() -> List[str]:
    """Return a list of source names that are enabled based on configured credentials."""
    sources = []
    if settings.NEWS_API_KEY:
        sources.append("newsapi")
    if settings.TWITTER_BEARER_TOKEN:
        sources.append("twitter")
    if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET:
        sources.append("reddit")
    if settings.FACEBOOK_PAGE_ACCESS_TOKEN and settings.FACEBOOK_PAGE_ID:
        sources.append("facebook")
    if settings.YOUTUBE_API_KEY:
        sources.append("youtube")
    if settings.CRYPTOPANIC_API_KEY:
        sources.append("cryptopanic")
    if settings.CRYPTOCOMPARE_API_KEY:
        sources.append("cryptocompare")
    if settings.LUNARCRUSH_API_KEY:
        sources.append("lunarcrush")
    if settings.SANTIMENT_API_KEY:
        sources.append("santiment")
    if settings.MESSARI_API_KEY:
        sources.append("messari")
    if settings.COINMARKETCAP_API_KEY:
        sources.append("coinmarketcap")
    # Google News is free
    sources.append("googlenews")
    if settings.STOCKTWITS_API_KEY:
        sources.append("stocktwits")
    if settings.RSS_FEEDS:
        sources.append("rss")
    logger.debug(f"News sources auto-enabled: {sources}")
    return sources


def _analyze_sentiment(text: str) -> Dict[str, Any]:
    """Return sentiment label and compound score for a text."""
    scores = _sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": round(compound, 4)}


def _is_relevant(symbol: str, title: str, summary: str) -> bool:
    """Return True if the article is likely relevant to the trading symbol."""
    text = f"{title} {summary}".lower()
    sym_lower = symbol.split("/")[0].lower()
    # Must mention the symbol at least once
    if sym_lower not in text:
        return False
    # Crypto‑specific keywords that indicate relevance
    crypto_keywords = [
        "crypto", "bitcoin", "ethereum", "blockchain", "defi", "nft",
        "altcoin", "token", "exchange", "trading", "bullish", "bearish",
        "price", "market", "volume", "breakout", "support", "resistance",
        "whale", "accumulation", "dump", "pump", "regulation", "sec",
        "binance", "coinbase", "bybit", "kraken", "ftx",
    ]
    # Score: +2 for symbol in title, +1 for each crypto keyword found
    score = 0
    if sym_lower in title.lower():
        score += 2
    for kw in crypto_keywords:
        if kw in text:
            score += 1
    # Require at least 3 points (symbol in title + one keyword, or three keywords)
    return score >= 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_for_symbol(symbol: str) -> List[Dict[str, str]]:
    """
    Fetch news articles for a trading symbol from all enabled sources.
    Returns a list of dicts with keys:
        title, source, url, published_at, summary
    Results are cached in Redis for NEWS_CACHE_TTL_SECONDS.
    """
    if not settings.NEWS_ENABLED:
        return []

    # Use base coin (e.g., "BTC") for caching, not the full pair
    base_coin = symbol.split("/")[0] if "/" in symbol else symbol

    start_time = time.time()
    logger.debug(f"Fetching news for {symbol} (base coin: {base_coin})...")

    redis_client = get_redis_client()
    cache_key = f"news:{base_coin}:{_source_fingerprint()}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            articles = json.loads(cached)
            logger.debug(f"News for {base_coin} served from cache ({len(articles)} articles)")
            return articles
        except Exception:
            pass

    articles: List[Dict[str, str]] = []

    enabled = _get_enabled_sources()
    logger.debug(f"Enabled news sources for {symbol}: {enabled}")
    for source in enabled:
        source_start = time.time()
        if source == "newsapi":
            articles.extend(_fetch_newsapi(symbol))
        elif source == "twitter":
            articles.extend(_fetch_twitter(symbol))
        elif source == "reddit":
            articles.extend(_fetch_reddit(symbol))
        elif source == "facebook":
            articles.extend(_fetch_facebook(symbol))
        elif source == "youtube":
            articles.extend(_fetch_youtube(symbol))
        elif source == "cryptopanic":
            articles.extend(_fetch_cryptopanic(symbol))
        elif source == "cryptocompare":
            articles.extend(_fetch_cryptocompare(symbol))
        elif source == "lunarcrush":
            articles.extend(_fetch_lunarcrush(symbol))
        elif source == "santiment":
            articles.extend(_fetch_santiment(symbol))
        elif source == "messari":
            articles.extend(_fetch_messari(symbol))
        elif source == "coinmarketcap":
            articles.extend(_fetch_coinmarketcap(symbol))
        elif source == "googlenews":
            articles.extend(_fetch_googlenews(symbol))
        elif source == "stocktwits":
            articles.extend(_fetch_stocktwits(symbol))
        elif source == "rss":
            articles.extend(_fetch_rss(symbol))
        source_time = time.time() - source_start
        if source_time > 2.0:
            logger.warning(f"Slow news source '{source}' for {symbol}: {source_time:.2f}s")

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(a)

    # Limit per symbol
    unique = unique[:settings.NEWS_MAX_ARTICLES_PER_SYMBOL]

    # Cache
    try:
        redis_client.setex(cache_key, settings.NEWS_CACHE_TTL_SECONDS, json.dumps(unique))
    except Exception as e:
        logger.warning(f"Failed to cache news for {base_coin}: {e}")

    total_time = time.time() - start_time
    logger.debug(f"News for {symbol}: {len(unique)} articles from {len(enabled)} sources in {total_time:.2f}s")
    if total_time > 5.0:
        logger.warning(f"News fetch for {symbol} took {total_time:.2f}s – consider reducing sources or increasing cache TTL")

    return unique


def get_aggregate_sentiment(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Return an aggregate sentiment summary for a symbol.
    Returns None if no articles are available.
    """
    articles = fetch_news_for_symbol(symbol)
    if not articles:
        return None
    compounds = [a["sentiment"]["compound"] for a in articles if "sentiment" in a]
    if not compounds:
        return None
    avg_compound = sum(compounds) / len(compounds)
    # Count labels
    labels = [a["sentiment"]["label"] for a in articles if "sentiment" in a]
    pos = labels.count("positive")
    neg = labels.count("negative")
    neu = labels.count("neutral")
    return {
        "avg_compound": round(avg_compound, 4),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total_articles": len(articles),
    }


def discover_trending_coins(
    base_currency: str,
    existing_pairs: List[str],
    max_coins: int = 5,
    min_sentiment: float = 0.3,
    min_articles: int = 3,
) -> List[str]:
    """
    Scan recent news for coins not in existing_pairs that have strong positive sentiment.
    Returns a list of trading pair strings (e.g., ["DOGE/USDT", "SHIB/USDT"]).
    """
    if not settings.NEWS_ENABLED:
        return []

    # Fetch top 100 coins by market cap from CoinGecko (free, no key)
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false",
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        coins_data = response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch coin list for discovery: {e}")
        return []

    # Build a set of symbols already in the candidate pool (without the quote currency)
    existing_symbols = {pair.split("/")[0].lower() for pair in existing_pairs}

    candidates = []
    for coin in coins_data:
        symbol = coin.get("symbol", "").upper()
        if not symbol:
            continue
        pair = f"{symbol}/{base_currency}"
        if pair in existing_pairs:
            continue
        if symbol.lower() in existing_symbols:
            continue

        # Check news sentiment for this coin (use base coin for DB lookup)
        base = pair.split("/")[0] if "/" in pair else pair
        agg = get_aggregate_sentiment_from_db(base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if agg and agg["total_articles"] >= min_articles and agg["avg_compound"] >= min_sentiment:
            candidates.append((pair, agg["avg_compound"]))

    # Sort by sentiment descending and take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    discovered = [pair for pair, _ in candidates[:max_coins]]
    if discovered:
        logger.info(f"News-driven coin discovery found: {discovered}")
    return discovered


def _source_fingerprint() -> str:
    """Create a short fingerprint of the current source configuration for cache key."""
    raw = f"{_get_enabled_sources()}:{settings.NEWS_MAX_ARTICLES_PER_SYMBOL}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

def _fetch_newsapi(symbol: str) -> List[Dict[str, str]]:
    if not settings.NEWS_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("newsapi")
        logger.debug(f"Fetching NewsAPI for {symbol}...")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{symbol.split('/')[0]} crypto",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": settings.NEWS_MAX_ARTICLES_PER_SYMBOL,
            "apiKey": settings.NEWS_API_KEY,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for art in data.get("articles", []):
            title = art.get("title", "")
            description = art.get("description", "") or ""
            text = f"{title} {description}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, description):
                continue
            articles.append({
                "title": title,
                "source": art.get("source", {}).get("name", "NewsAPI"),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
                "summary": description[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"NewsAPI returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Twitter (X) via API v2
# ---------------------------------------------------------------------------

def _fetch_twitter(symbol: str) -> List[Dict[str, str]]:
    if not settings.TWITTER_BEARER_TOKEN:
        return []
    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy not installed. Install with: pip install tweepy")
        return []
    try:
        _get_rate_limiter().wait("twitter")
        logger.debug(f"Fetching Twitter for {symbol}...")
        client = tweepy.Client(bearer_token=settings.TWITTER_BEARER_TOKEN, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        query = f"${symbol.split('/')[0]} crypto -is:retweet lang:en"
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(settings.NEWS_MAX_ARTICLES_PER_SYMBOL, 10),
            tweet_fields=["created_at", "text"],
        )
        articles = []
        if tweets.data:
            for tweet in tweets.data:
                sentiment = _analyze_sentiment(tweet.text)
                if not _is_relevant(symbol, tweet.text[:100], tweet.text):
                    continue
                articles.append({
                    "title": tweet.text[:100],
                    "source": "Twitter",
                    "url": f"https://twitter.com/i/web/status/{tweet.id}",
                    "published_at": str(tweet.created_at) if tweet.created_at else "",
                    "summary": tweet.text,
                    "sentiment": sentiment,
                })
        logger.debug(f"Twitter returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Twitter fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def _fetch_reddit(symbol: str) -> List[Dict[str, str]]:
    if not settings.REDDIT_CLIENT_ID or not settings.REDDIT_CLIENT_SECRET:
        return []
    try:
        import praw
    except ImportError:
        logger.warning("praw not installed. Install with: pip install praw")
        return []
    try:
        _get_rate_limiter().wait("reddit")
        logger.debug(f"Fetching Reddit for {symbol}...")
        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
            timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS,
        )
        submissions = reddit.subreddit("all").search(
            f"{symbol.split('/')[0]} crypto",
            sort="relevance",
            time_filter="week",
            limit=settings.NEWS_MAX_ARTICLES_PER_SYMBOL,
        )
        articles = []
        for sub in submissions:
            text = f"{sub.title} {sub.selftext[:300] if sub.selftext else ''}"
            sentiment = _analyze_sentiment(text)
            reddit_summary = sub.selftext[:300] if sub.selftext else sub.title
            if not _is_relevant(symbol, sub.title, reddit_summary):
                continue
            articles.append({
                "title": sub.title,
                "source": f"Reddit r/{sub.subreddit.display_name}",
                "url": f"https://reddit.com{sub.permalink}",
                "published_at": str(sub.created_utc),
                "summary": reddit_summary,
                "sentiment": sentiment,
            })
        logger.debug(f"Reddit returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Reddit fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Facebook (Graph API)
# ---------------------------------------------------------------------------

def _fetch_facebook(symbol: str) -> List[Dict[str, str]]:
    if not settings.FACEBOOK_PAGE_ACCESS_TOKEN or not settings.FACEBOOK_PAGE_ID:
        return []
    try:
        _get_rate_limiter().wait("facebook")
        logger.debug(f"Fetching Facebook for {symbol}...")
        url = f"https://graph.facebook.com/v19.0/{settings.FACEBOOK_PAGE_ID}/posts"
        params = {
            "fields": "message,created_time,permalink_url",
            "limit": settings.FACEBOOK_POST_LIMIT,
            "access_token": settings.FACEBOOK_PAGE_ACCESS_TOKEN,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for post in data.get("data", []):
            message = post.get("message", "")
            if not message:
                continue
            # Simple relevance check: symbol appears in the post
            if symbol.split('/')[0].lower() not in message.lower():
                continue
            sentiment = _analyze_sentiment(message)
            articles.append({
                "title": message[:100],
                "source": "Facebook",
                "url": post.get("permalink_url", ""),
                "published_at": post.get("created_time", ""),
                "summary": message[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"Facebook returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Facebook fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# YouTube Data API v3
# ---------------------------------------------------------------------------

def _fetch_youtube(symbol: str) -> List[Dict[str, str]]:
    if not settings.YOUTUBE_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("youtube")
        logger.debug(f"Fetching YouTube for {symbol}...")
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"{symbol.split('/')[0]} crypto",
            "type": "video",
            "maxResults": settings.YOUTUBE_MAX_RESULTS,
            "order": "date",
            "relevanceLanguage": "en",
            "key": settings.YOUTUBE_API_KEY,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for item in data.get("items", []):
            snippet = item["snippet"]
            title = snippet.get("title", "")
            description = snippet.get("description", "")
            text = f"{title} {description}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, description[:300]):
                continue
            articles.append({
                "title": title,
                "source": "YouTube",
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "published_at": snippet.get("publishedAt", ""),
                "summary": description[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"YouTube returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"YouTube fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# CryptoPanic API
# ---------------------------------------------------------------------------

def _fetch_cryptopanic(symbol: str) -> List[Dict[str, str]]:
    if not settings.CRYPTOPANIC_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("cryptopanic")
        logger.debug(f"Fetching CryptoPanic for {symbol}...")
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": settings.CRYPTOPANIC_API_KEY,
            "currencies": symbol.split("/")[0],  # e.g., BTC from BTC/USDT
            "kind": "news",
            "public": "true",
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for post in data.get("results", [])[:settings.CRYPTOPANIC_MAX_POSTS]:
            title = post.get("title", "")
            summary = post.get("body", "") or post.get("description", "")
            text = f"{title} {summary}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, summary[:300]):
                continue
            articles.append({
                "title": title,
                "source": post.get("source", {}).get("title", "CryptoPanic"),
                "url": post.get("url", ""),
                "published_at": post.get("published_at", ""),
                "summary": summary[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"CryptoPanic returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"CryptoPanic fetch failed for {symbol}: {e}")
        return []



# ---------------------------------------------------------------------------
# CryptoCompare News API
# ---------------------------------------------------------------------------

def _fetch_cryptocompare(symbol: str) -> List[Dict[str, str]]:
    if not settings.CRYPTOCOMPARE_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("cryptocompare")
        logger.debug(f"Fetching CryptoCompare for {symbol}...")
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {
            "lang": "EN",
            "api_key": settings.CRYPTOCOMPARE_API_KEY,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for item in data.get("Data", [])[:settings.CRYPTOCOMPARE_MAX_ARTICLES]:
            title = item.get("title", "")
            body = item.get("body", "")
            text = f"{title} {body}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, body[:300]):
                continue
            articles.append({
                "title": title,
                "source": item.get("source", "CryptoCompare"),
                "url": item.get("url", ""),
                "published_at": item.get("published_on", ""),
                "summary": body[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"CryptoCompare returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"CryptoCompare fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# LunarCrush API
# ---------------------------------------------------------------------------

def _fetch_lunarcrush(symbol: str) -> List[Dict[str, str]]:
    if not settings.LUNARCRUSH_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("lunarcrush")
        logger.debug(f"Fetching LunarCrush for {symbol}...")
        # Extract base currency (e.g., BTC from BTC/USDT)
        base = symbol.split("/")[0]
        url = "https://lunarcrush.com/api/v2"
        params = {
            "data": "assets",
            "key": settings.LUNARCRUSH_API_KEY,
            "symbol": base,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        # The response contains a "data" list with one asset; get its "news" array
        asset_data = data.get("data", [])
        if asset_data:
            news_items = asset_data[0].get("news", [])
            for item in news_items[:settings.LUNARCRUSH_MAX_ARTICLES]:
                title = item.get("title", "")
                summary = item.get("body", "") or item.get("description", "")
                text = f"{title} {summary}"
                sentiment = _analyze_sentiment(text)
                if not _is_relevant(symbol, title, summary[:300]):
                    continue
                articles.append({
                    "title": title,
                    "source": item.get("source", "LunarCrush"),
                    "url": item.get("url", ""),
                    "published_at": item.get("created_at", ""),
                    "summary": summary[:300],
                    "sentiment": sentiment,
                })
        logger.debug(f"LunarCrush returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"LunarCrush fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Santiment API
# ---------------------------------------------------------------------------

def _fetch_santiment(symbol: str) -> List[Dict[str, str]]:
    if not settings.SANTIMENT_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("santiment")
        logger.debug(f"Fetching Santiment for {symbol}...")
        # Santiment uses asset slugs (e.g., "bitcoin", "ethereum"). We'll map common symbols.
        # For simplicity, we'll use the lowercase base currency as slug.
        base = symbol.split("/")[0].lower()
        url = "https://api.santiment.net/news"
        params = {
            "asset": base,
            "limit": settings.SANTIMENT_MAX_ARTICLES,
        }
        headers = {"Authorization": f"Apikey {settings.SANTIMENT_API_KEY}"}
        response = httpx.get(url, params=params, headers=headers, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for item in data:
            title = item.get("title", "")
            summary = item.get("description", "") or item.get("content", "")
            text = f"{title} {summary}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, summary[:300]):
                continue
            articles.append({
                "title": title,
                "source": item.get("source", {}).get("name", "Santiment"),
                "url": item.get("url", ""),
                "published_at": item.get("published_at", ""),
                "summary": summary[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"Santiment returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Santiment fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Messari API
# ---------------------------------------------------------------------------

def _fetch_messari(symbol: str) -> List[Dict[str, str]]:
    if not settings.MESSARI_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("messari")
        logger.debug(f"Fetching Messari for {symbol}...")
        base = symbol.split("/")[0].lower()
        url = f"https://data.messari.io/api/v1/news/{base}"
        headers = {"x-messari-api-key": settings.MESSARI_API_KEY}
        response = httpx.get(url, headers=headers, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for item in data.get("data", [])[:settings.MESSARI_MAX_ARTICLES]:
            title = item.get("title", "")
            summary = item.get("content", "") or item.get("description", "")
            text = f"{title} {summary}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, summary[:300]):
                continue
            articles.append({
                "title": title,
                "source": item.get("source", {}).get("name", "Messari"),
                "url": item.get("url", ""),
                "published_at": item.get("published_at", ""),
                "summary": summary[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"Messari returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Messari fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# CoinMarketCap API
# ---------------------------------------------------------------------------

def _fetch_coinmarketcap(symbol: str) -> List[Dict[str, str]]:
    if not settings.COINMARKETCAP_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("coinmarketcap")
        logger.debug(f"Fetching CoinMarketCap for {symbol}...")
        base = symbol.split("/")[0]
        url = "https://pro-api.coinmarketcap.com/v1/content/latest"
        params = {
            "symbol": base,
            "limit": settings.COINMARKETCAP_MAX_ARTICLES,
        }
        headers = {"X-CMC_PRO_API_KEY": settings.COINMARKETCAP_API_KEY}
        response = httpx.get(url, params=params, headers=headers, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for item in data.get("data", []):
            title = item.get("title", "")
            summary = item.get("content", "") or item.get("description", "")
            text = f"{title} {summary}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, summary[:300]):
                continue
            articles.append({
                "title": title,
                "source": item.get("source", {}).get("name", "CoinMarketCap"),
                "url": item.get("url", ""),
                "published_at": item.get("released_at", ""),
                "summary": summary[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"CoinMarketCap returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"CoinMarketCap fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------

def _fetch_googlenews(symbol: str) -> List[Dict[str, str]]:
    """Fetch news from Google News RSS feed."""
    try:
        _get_rate_limiter().wait("googlenews")
        logger.debug(f"Fetching Google News for {symbol}...")
        base = symbol.split("/")[0]
        url = f"https://news.google.com/rss/search?q={base}+crypto&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:settings.GOOGLE_NEWS_MAX_ARTICLES]:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            text = f"{title} {summary}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, summary[:300]):
                continue
            articles.append({
                "title": title,
                "source": entry.get("source", {}).get("title", "Google News"),
                "url": entry.get("link", ""),
                "published_at": entry.get("published", ""),
                "summary": summary[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"Google News returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Google News fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# StockTwits API
# ---------------------------------------------------------------------------

def _fetch_stocktwits(symbol: str) -> List[Dict[str, str]]:
    if not settings.STOCKTWITS_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("stocktwits")
        logger.debug(f"Fetching StockTwits for {symbol}...")
        base = symbol.split("/")[0]
        # StockTwits uses tickers like BTC.X for crypto
        ticker = f"{base}.X"
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        params = {"access_token": settings.STOCKTWITS_API_KEY, "limit": settings.STOCKTWITS_MAX_POSTS}
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for msg in data.get("messages", []):
            body = msg.get("body", "")
            title = body[:100]
            sentiment_label = msg.get("entities", {}).get("sentiment", {}).get("basic", "")
            # Map StockTwits sentiment to our labels
            if sentiment_label == "Bullish":
                label = "positive"
                compound = 0.5
            elif sentiment_label == "Bearish":
                label = "negative"
                compound = -0.5
            else:
                # Fallback to VADER
                sentiment = _analyze_sentiment(body)
                label = sentiment["label"]
                compound = sentiment["compound"]
            if not _is_relevant(symbol, title, body[:300]):
                continue
            articles.append({
                "title": title,
                "source": "StockTwits",
                "url": f"https://stocktwits.com/{msg.get('user', {}).get('username', '')}/message/{msg.get('id', '')}",
                "published_at": msg.get("created_at", ""),
                "summary": body[:300],
                "sentiment": {"label": label, "compound": compound},
            })
        logger.debug(f"StockTwits returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"StockTwits fetch failed for {symbol}: {e}")
        return []




# ---------------------------------------------------------------------------
# RSS Feeds
# ---------------------------------------------------------------------------

def _fetch_rss(symbol: str) -> List[Dict[str, str]]:
    """Fetch news from configured RSS feeds, filtering for symbol mentions."""
    _cleanup_rss_cache()
    articles = []
    for feed_url in settings.RSS_FEEDS:
        try:
            # Check cache first
            with _rss_cache_lock:
                cached = _rss_cache.get(feed_url)
                if cached and (time.time() - cached[0]) < 300:  # 5-minute TTL
                    feed_content = cached[1]
                else:
                    feed_content = None

            if feed_content is None:
                _get_rate_limiter().wait(feed_url)
                logger.debug(f"Fetching RSS feed: {feed_url}")
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; BengoBot/1.0; +https://github.com/your-repo)"
                }
                # Retry on 429 with exponential backoff
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        resp = httpx.get(
                            feed_url,
                            headers=headers,
                            timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS,
                            follow_redirects=True,
                        )
                        resp.raise_for_status()
                        feed_content = resp.text
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 429 and attempt < max_retries - 1:
                            wait = 2 ** attempt
                            logger.warning(
                                f"RSS feed {feed_url} rate limited, retrying in {wait}s..."
                            )
                            time.sleep(wait)
                        else:
                            raise
                # Cache the successful response
                with _rss_cache_lock:
                    _rss_cache[feed_url] = (time.time(), feed_content)

            feed = feedparser.parse(feed_content)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                combined = f"{title} {summary}".lower()
                if symbol.split("/")[0].lower() not in combined:
                    continue
                text = f"{title} {summary}"
                sentiment = _analyze_sentiment(text)
                if not _is_relevant(symbol, title, summary[:300]):
                    continue
                articles.append({
                    "title": title,
                    "source": feed.feed.get("title", "RSS"),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                    "summary": summary[:300],
                    "sentiment": sentiment,
                })
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"RSS feed not found (404): {feed_url}")
            elif e.response.status_code == 403:
                logger.warning(f"RSS feed access forbidden (403): {feed_url}")
            else:
                logger.warning(f"RSS fetch failed for {feed_url}: {e}")
        except Exception as e:
            logger.warning(f"RSS fetch failed for {feed_url}: {e}")
    logger.debug(f"RSS total articles for {symbol}: {len(articles)}")
    return articles


def test_rss_feeds():
    """Check each configured RSS feed and log whether it is reachable."""
    logger.debug(f"Testing {len(settings.RSS_FEEDS)} RSS feeds...")
    for url in settings.RSS_FEEDS:
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; BengoBot/1.0; +https://github.com/your-repo)"},
                timeout=10,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                logger.debug(f"RSS OK: {url}")
            else:
                logger.warning(f"RSS {url} returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"RSS {url} failed: {e}")
