import asyncio
import logging
import uvicorn
from src.web.app import app
from src.config.settings import settings
from src.trading.engine import TradingEngine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

async def main():
    engine = TradingEngine()
    # Start engine as a background task
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
