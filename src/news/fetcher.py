import logging
from typing import List, Dict, Any, Optional
import hashlib
import httpx
import json
import feedparser

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

_sentiment_analyzer = SentimentIntensityAnalyzer()


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
    sym_lower = symbol.lower()
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

    redis_client = get_redis_client()
    cache_key = f"news:{symbol}:{_source_fingerprint()}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    articles: List[Dict[str, str]] = []

    if "newsapi" in settings.NEWS_SOURCES:
        articles.extend(_fetch_newsapi(symbol))

    if "twitter" in settings.NEWS_SOURCES:
        articles.extend(_fetch_twitter(symbol))

    if "reddit" in settings.NEWS_SOURCES:
        articles.extend(_fetch_reddit(symbol))

    if "facebook" in settings.NEWS_SOURCES:
        articles.extend(_fetch_facebook(symbol))

    if settings.RSS_FEEDS:
        articles.extend(_fetch_rss(symbol))

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
        logger.warning(f"Failed to cache news for {symbol}: {e}")

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


def _source_fingerprint() -> str:
    """Create a short fingerprint of the current source configuration for cache key."""
    raw = f"{settings.NEWS_SOURCES}:{settings.NEWS_MAX_ARTICLES_PER_SYMBOL}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

def _fetch_newsapi(symbol: str) -> List[Dict[str, str]]:
    if not settings.NEWS_API_KEY:
        return []
    try:
        from newsapi import NewsApiClient
    except ImportError:
        logger.warning("newsapi-python not installed. Install with: pip install newsapi-python")
        return []
    try:
        client = NewsApiClient(api_key=settings.NEWS_API_KEY)
        query = f"{symbol} crypto"
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=settings.NEWS_MAX_ARTICLES_PER_SYMBOL,
        )
        articles = []
        for art in response.get("articles", []):
            text = f"{art.get('title', '')} {art.get('description', '')}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, art.get("title", ""), art.get("description", "") or ""):
                continue
            articles.append({
                "title": art.get("title", ""),
                "source": art.get("source", {}).get("name", "NewsAPI"),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
                "summary": art.get("description", "") or "",
                "sentiment": sentiment,
            })
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
        client = tweepy.Client(bearer_token=settings.TWITTER_BEARER_TOKEN)
        query = f"${symbol} crypto -is:retweet lang:en"
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
        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
        )
        submissions = reddit.subreddit("all").search(
            f"{symbol} crypto",
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
        url = f"https://graph.facebook.com/v19.0/{settings.FACEBOOK_PAGE_ID}/posts"
        params = {
            "fields": "message,created_time,permalink_url",
            "limit": settings.FACEBOOK_POST_LIMIT,
            "access_token": settings.FACEBOOK_PAGE_ACCESS_TOKEN,
        }
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        articles = []
        for post in data.get("data", []):
            message = post.get("message", "")
            if not message:
                continue
            # Simple relevance check: symbol appears in the post
            if symbol.lower() not in message.lower():
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
        return articles
    except Exception as e:
        logger.warning(f"Facebook fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# RSS Feeds
# ---------------------------------------------------------------------------

def _fetch_rss(symbol: str) -> List[Dict[str, str]]:
    """Fetch news from configured RSS feeds, filtering for symbol mentions."""
    articles = []
    for feed_url in settings.RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                combined = f"{title} {summary}".lower()
                if symbol.lower() not in combined:
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
        except Exception as e:
            logger.warning(f"RSS fetch failed for {feed_url}: {e}")
    return articles
