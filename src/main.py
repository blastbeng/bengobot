import asyncio
import logging
import uvicorn
from src.web.app import app
from src.config.settings import settings
from src.database import init_db, get_telegram_chat_id, set_telegram_chat_id
from src.utils.redis_client import get_redis_client
from src.trading.engine import TradingEngine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def _seed_telegram_chat_id():
    """If TELEGRAM_CHAT_ID is set in env and no chat_id is stored, store it."""
    if settings.TELEGRAM_CHAT_ID:
        existing = get_telegram_chat_id()
        if existing is None:
            try:
                chat_id = int(settings.TELEGRAM_CHAT_ID)
                set_telegram_chat_id(chat_id)
                logging.info(f"Seeded Telegram chat ID from env: {chat_id}")
            except ValueError:
                logging.warning("TELEGRAM_CHAT_ID in .env is not a valid integer")


def _cleanup_redis_state():
    """Remove old trading state keys from Redis (now stored in SQLite)."""
    redis = get_redis_client()
    keys_to_delete = [
        "trading:current_coins",
        "trading:positions",
        "trading:trade_history",
        "trading:initial_balance",
    ]
    for key in keys_to_delete:
        redis.delete(key)


async def main():
    init_db()
    _seed_telegram_chat_id()
    _cleanup_redis_state()
    engine = TradingEngine()
    logging.info("Trading engine initialized.")
    from src.web.app import set_engine
    set_engine(engine)
    # Set up Telegram notifier before starting the engine
    if settings.TELEGRAM_BOT_TOKEN:
        from src.telegram.bot import TelegramBot
        telegram_bot = TelegramBot(engine)
        engine.set_notifier(telegram_bot)

        # Initialize the bot so we can send a startup message immediately
        await telegram_bot.initialize()
        await telegram_bot.send_notification("🤖 Bengobot started! Use the buttons below to control me.")

        # Start polling in the background
        asyncio.create_task(telegram_bot.run())

    # Now start the trading engine
    asyncio.create_task(engine.run())
    # Run the web server
    config = uvicorn.Config(
        app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
