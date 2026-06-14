import sqlite3
import json
import os
import logging
import time
from typing import Dict, List, Any, Optional
from src.config.settings import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.DATABASE_PATH


def _normalize_symbol(symbol: str) -> str:
    """Extract the base coin from a trading pair (e.g., 'NIM/USDT' -> 'NIM')."""
    return symbol.split("/")[0] if "/" in symbol else symbol


def get_connection() -> sqlite3.Connection:
    """Return a new connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _migrate_db():
    """Add missing columns to existing tables (schema migrations)."""
    conn = get_connection()
    cursor = conn.execute("PRAGMA table_info(trade_history)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    migrations = [
        ("timeframe", "ALTER TABLE trade_history ADD COLUMN timeframe TEXT"),
        ("cost_basis", "ALTER TABLE trade_history ADD COLUMN cost_basis REAL"),
        ("strategy_type", "ALTER TABLE trade_history ADD COLUMN strategy_type TEXT"),
        ("note", "ALTER TABLE trade_history ADD COLUMN note TEXT"),
        ("status", "ALTER TABLE trade_history ADD COLUMN status TEXT"),
    ]

    for column_name, sql in migrations:
        if column_name not in existing_columns:
            try:
                conn.execute(sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists (race with another process)

    conn.close()


def init_db():
    """Create tables if they don't exist, then run migrations."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trading_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS telegram_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            symbol TEXT NOT NULL,
            timeframe TEXT,
            side TEXT NOT NULL,
            type TEXT,
            amount REAL NOT NULL,
            price REAL NOT NULL,
            cost REAL,
            fee_cost REAL,
            fee_currency TEXT,
            realized_pnl REAL,
            cost_basis REAL,
            strategy_type TEXT,
            note TEXT,
            status TEXT,
            timestamp INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trade_history_symbol_timeframe ON trade_history(symbol, timeframe);

        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            title TEXT,
            source TEXT,
            url TEXT,
            published_at TEXT,
            summary TEXT,
            sentiment_label TEXT,
            sentiment_compound REAL,
            fetched_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles(symbol);
        CREATE INDEX IF NOT EXISTS idx_news_fetched_at ON news_articles(fetched_at);

        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_tf_ts ON market_data(symbol, timeframe, timestamp);
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
    """)
    conn.commit()
    conn.close()
    _migrate_db()


def insert_trade(trade: Dict[str, Any]):
    """Insert a completed trade into the trade_history table."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO trade_history (
            order_id, symbol, timeframe, side, type, amount, price, cost,
            fee_cost, fee_currency, realized_pnl, cost_basis,
            strategy_type, note, status, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade.get("id"),
            trade["symbol"],
            trade.get("timeframe"),
            trade["side"],
            trade.get("type"),
            trade["amount"],
            trade["price"],
            trade.get("cost"),
            trade.get("fee", {}).get("cost"),
            trade.get("fee", {}).get("currency"),
            trade.get("realized_pnl"),
            trade.get("cost_basis"),
            trade.get("strategy_type"),
            trade.get("note"),
            trade.get("status", "closed"),
            trade["timestamp"],
        ),
    )
    conn.commit()
    conn.close()


# ---------- Trading state helpers ----------

def load_trading_state() -> Dict[str, Any]:
    """Load all trading state from the database."""
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM trading_state").fetchall()
    conn.close()
    state = {}
    for row in rows:
        try:
            state[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            state[row["key"]] = row["value"]
    return state


def save_trading_state(key: str, value: Any):
    """Insert or update a single trading state key."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO trading_state (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
    conn.commit()
    conn.close()


def delete_trading_state(key: str):
    """Remove a trading state key."""
    conn = get_connection()
    conn.execute("DELETE FROM trading_state WHERE key = ?", (key,))
    conn.commit()
    conn.close()


# ---------- Paper balance helpers ----------

def save_paper_balances(balances: Dict[str, float]):
    """Persist the paper simulator's balances dict."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO trading_state (key, value) VALUES (?, ?)",
        ("paper_balances", json.dumps(balances))
    )
    conn.commit()
    conn.close()


def load_paper_balances() -> Dict[str, float]:
    """Load the paper simulator's balances dict. Returns empty dict if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT value FROM trading_state WHERE key = 'paper_balances'"
    ).fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


# ---------- Telegram state helpers ----------

def get_telegram_chat_id() -> Optional[int]:
    """Retrieve the stored Telegram chat ID."""
    conn = get_connection()
    row = conn.execute("SELECT value FROM telegram_state WHERE key = 'chat_id'").fetchone()
    conn.close()
    if row:
        try:
            return int(row["value"])
        except (ValueError, TypeError):
            return None
    return None

def cleanup_old_ohlcv(retention_days: int = 30):
    """Delete OHLCV candles older than retention_days for all symbols and timeframes."""
    conn = get_connection()
    cutoff_ms = int((time.time() - retention_days * 24 * 60 * 60) * 1000)
    deleted = conn.execute(
        "DELETE FROM market_data WHERE timestamp < ?",
        (cutoff_ms,)
    ).rowcount
    conn.commit()
    conn.close()
    if deleted:
        logger.info(f"Cleaned up {deleted} old OHLCV candles (older than {retention_days} days)")
    return deleted


def get_performance() -> Dict[str, Any]:
    """Return performance summary grouped by coin and timeframe, plus a TOTAL row."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            symbol,
            timeframe,
            COUNT(*) AS trade_count,
            SUM(realized_pnl) AS total_profit,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS losses,
            SUM(cost_basis) AS total_cost_basis
        FROM trade_history
        WHERE side = 'sell' AND realized_pnl IS NOT NULL
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """).fetchall()
    conn.close()

    performance = []
    total_trades = 0
    total_profit = 0.0
    total_wins = 0
    total_losses = 0
    total_cost_basis = 0.0

    for row in rows:
        symbol = row["symbol"]
        timeframe = row["timeframe"] or "N/A"
        trade_count = row["trade_count"]
        profit = row["total_profit"] or 0.0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        cost_basis = row["total_cost_basis"] or 0.0

        profit_pct = (profit / cost_basis * 100) if cost_basis else 0.0
        win_rate = (wins / trade_count * 100) if trade_count else 0.0

        performance.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "trade_count": trade_count,
            "profit": round(profit, 4),
            "profit_pct": round(profit_pct, 2),
            "win_rate": round(win_rate, 2),
        })

        total_trades += trade_count
        total_profit += profit
        total_wins += wins
        total_losses += losses
        total_cost_basis += cost_basis

    total_profit_pct = (total_profit / total_cost_basis * 100) if total_cost_basis else 0.0
    total_win_rate = (total_wins / total_trades * 100) if total_trades else 0.0

    total_row = {
        "symbol": "TOTAL",
        "timeframe": "",
        "trade_count": total_trades,
        "profit": round(total_profit, 4),
        "profit_pct": round(total_profit_pct, 2),
        "win_rate": round(total_win_rate, 2),
    }

    return {
        "rows": performance,
        "total": total_row,
    }


def set_telegram_chat_id(chat_id: int):
    """Store the Telegram chat ID."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO telegram_state (key, value) VALUES (?, ?)",
        ("chat_id", str(chat_id))
    )
    conn.commit()
    conn.close()


def store_news_articles(symbol: str, articles: List[Dict[str, Any]]):
    """Replace all stored articles for a symbol with a fresh batch."""
    symbol = _normalize_symbol(symbol)
    conn = get_connection()
    now = time.time()
    # Delete old articles for this symbol
    conn.execute("DELETE FROM news_articles WHERE symbol = ?", (symbol,))
    # Insert new articles
    for art in articles:
        sentiment = art.get("sentiment", {})
        conn.execute(
            """
            INSERT INTO news_articles (
                symbol, title, source, url, published_at, summary,
                sentiment_label, sentiment_compound, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                art.get("title", ""),
                art.get("source", ""),
                art.get("url", ""),
                art.get("published_at", ""),
                art.get("summary", ""),
                sentiment.get("label", ""),
                sentiment.get("compound", 0.0),
                now,
            ),
        )
    conn.commit()
    conn.close()


def get_news_for_symbol(symbol: str, max_age_seconds: int = 900) -> List[Dict[str, Any]]:
    """Retrieve recent news articles for a symbol from the database."""
    symbol = _normalize_symbol(symbol)
    conn = get_connection()
    cutoff = time.time() - max_age_seconds
    rows = conn.execute(
        """
        SELECT title, source, url, published_at, summary, sentiment_label, sentiment_compound
        FROM news_articles
        WHERE symbol = ? AND fetched_at >= ?
        ORDER BY fetched_at DESC
        """,
        (symbol, cutoff),
    ).fetchall()
    conn.close()
    articles = []
    for row in rows:
        articles.append({
            "title": row["title"],
            "source": row["source"],
            "url": row["url"],
            "published_at": row["published_at"],
            "summary": row["summary"],
            "sentiment": {
                "label": row["sentiment_label"],
                "compound": row["sentiment_compound"],
            },
        })
    return articles


def get_aggregate_sentiment_from_db(symbol: str, max_age_seconds: int = 900) -> Optional[Dict[str, Any]]:
    """Return aggregate sentiment for a symbol from the database."""
    symbol = _normalize_symbol(symbol)
    articles = get_news_for_symbol(symbol, max_age_seconds)
    if not articles:
        return None
    compounds = [a["sentiment"]["compound"] for a in articles if "sentiment" in a]
    if not compounds:
        return None
    avg_compound = sum(compounds) / len(compounds)
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


def cleanup_old_news(retention_seconds: int):
    """Delete news articles older than retention_seconds."""
    conn = get_connection()
    cutoff = time.time() - retention_seconds
    deleted = conn.execute("DELETE FROM news_articles WHERE fetched_at < ?", (cutoff,)).rowcount
    conn.commit()
    conn.close()
    if deleted:
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaned up {deleted} old news articles.")


def insert_ohlcv_batch(symbol: str, timeframe: str, candles: List[List]):
    """Insert OHLCV candles into the market_data table, ignoring duplicates."""
    if not candles:
        return
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT OR IGNORE INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (symbol, timeframe, c[0], c[1], c[2], c[3], c[4], c[5])
                for c in candles
            ],
        )
        conn.commit()
        logger.debug(f"Inserted {len(candles)} OHLCV candles for {symbol} {timeframe}")
    finally:
        conn.close()


def get_ohlcv(symbol: str, timeframe: str, since_ms: int = None, limit: int = 500) -> List[Dict[str, Any]]:
    """Retrieve OHLCV candles from the market_data table."""
    conn = get_connection()
    query = "SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = ? AND timeframe = ?"
    params: list = [symbol, timeframe]
    if since_ms is not None:
        query += " AND timestamp >= ?"
        params.append(since_ms)
    query += " ORDER BY timestamp ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [
        {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        for row in rows
    ]


def get_latest_ohlcv_timestamp(symbol: str, timeframe: str) -> Optional[int]:
    """Return the latest timestamp for a symbol/timeframe, or None if no data exists."""
    conn = get_connection()
    row = conn.execute(
        "SELECT MAX(timestamp) AS ts FROM market_data WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    ).fetchone()
    conn.close()
    if row and row["ts"] is not None:
        return row["ts"]
    return None
