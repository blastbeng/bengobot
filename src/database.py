import sqlite3
import json
import os
from typing import Dict, List, Any, Optional
from src.config.settings import settings

DB_PATH = settings.DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """Return a new connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
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
    """)
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


def set_telegram_chat_id(chat_id: int):
    """Store the Telegram chat ID."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO telegram_state (key, value) VALUES (?, ?)",
        ("chat_id", str(chat_id))
    )
    conn.commit()
    conn.close()
