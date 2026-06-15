import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config.settings import settings
from src.utils.redis_client import get_redis_client, check_redis_connection
from src.llm.prompts import get_cached_news_summary

app = FastAPI(title="Crypto Trading Bot")

logger = logging.getLogger(__name__)

# Serve static files (dashboard)
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global engine reference
_engine = None

def set_engine(engine):
    global _engine
    _engine = engine
    logger.info("Trading engine attached to web server")

def get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine

@app.get("/")
async def root():
    return FileResponse("src/web/static/index.html")

@app.get("/health")
async def health():
    redis_ok = check_redis_connection()
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
    }

@app.get("/api/status")
def status():
    engine = get_engine()
    redis = get_redis_client()
    paused = redis.get("trading:paused") == "1"
    return {
        "current_coins": engine.current_coins,
        "positions": engine.positions,
        "balances": engine.trader.fetch_balance(),
        "paused": paused,
    }

@app.get("/api/trades")
def trades(limit: int = 0):
    engine = get_engine()
    # limit is ignored – open trades are always all current positions
    return {"trades": engine.get_open_trades()}

@app.get("/api/profit")
def profit():
    engine = get_engine()
    return engine.get_profit_summary()

@app.get("/api/performance")
def performance():
    engine = get_engine()
    return engine.get_performance_summary()

@app.get("/api/risk")
def risk():
    engine = get_engine()
    return engine.get_risk_metrics()

@app.get("/api/news")
async def news():
    engine = get_engine()
    coins = engine.current_coins
    result = {}
    for entry in coins:
        symbol = entry["symbol"]
        try:
            news_data = await run_in_threadpool(get_cached_news_summary, symbol)
            result[symbol] = news_data["summary"]
        except Exception:
            result[symbol] = "Could not generate summary."
    return result

@app.get("/api/history")
def history(limit: int = 50):
    engine = get_engine()
    trades = engine.trade_history[-limit:]
    return trades

@app.post("/api/pause")
def pause():
    engine = get_engine()
    redis = engine.redis
    redis.set("trading:paused", "1")
    redis.set("trading:pause_source", "manual")
    redis.delete("trading:pause_start")
    redis.delete("trading:pause_duration")
    redis.delete("trading:pause_reason")
    redis.delete("trading:llm_pause_time")
    return {"status": "paused"}

@app.post("/api/resume")
def resume():
    engine = get_engine()
    redis = engine.redis
    keys = [
        "trading:paused",
        "trading:pause_source",
        "trading:pause_start",
        "trading:pause_duration",
        "trading:pause_reason",
        "trading:llm_pause_time",
    ]
    for key in keys:
        redis.delete(key)
    return {"status": "resumed"}

@app.post("/api/sell")
def sell(symbol: str = None):
    engine = get_engine()
    if symbol:
        asyncio.create_task(engine.sell_position(symbol))
        return {"status": f"selling {symbol}"}
    else:
        asyncio.create_task(engine.sell_all_positions())
        return {"status": "selling all"}

@app.post("/api/reload")
def reload():
    settings.reload()
    return {"status": "reloaded"}

@app.get("/api/config")
def config():
    mind_provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    actuator_provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER
    if mind_provider == "ollama":
        mind_model = settings.OLLAMA_MIND_MODEL
    else:
        mind_model = settings.OPENAI_MIND_MODEL
    if actuator_provider == "ollama":
        actuator_model = settings.OLLAMA_ACTUATOR_MODEL
    else:
        actuator_model = settings.OPENAI_ACTUATOR_MODEL

    return {
        "exchange_id": settings.EXCHANGE_ID,
        "trading_mode": settings.TRADING_MODE,
        "base_currency": settings.BASE_CURRENCY,
        "max_coins": settings.MAX_COINS,
        "llm_mind_provider": mind_provider,
        "llm_mind_model": mind_model,
        "llm_actuator_provider": actuator_provider,
        "llm_actuator_model": actuator_model,
        "web_port": settings.WEB_PORT,
    }

@app.get("/api/ohlcv/{symbol:path}")
async def ohlcv(symbol: str, timeframe: str = "1h", limit: int = 24):
    engine = get_engine()
    exchange = engine.exchange
    try:
        ohlcv_data = await asyncio.to_thread(
            exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )
        result = []
        for candle in ohlcv_data:
            result.append({
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            })
        return {"symbol": symbol, "timeframe": timeframe, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ticker/{symbol:path}")
async def ticker(symbol: str):
    engine = get_engine()
    exchange = engine.exchange
    try:
        ticker_data = await asyncio.to_thread(exchange.fetch_ticker, symbol)
        return {
            "symbol": symbol,
            "last": ticker_data.get("last"),
            "bid": ticker_data.get("bid"),
            "ask": ticker_data.get("ask"),
            "change_24h": ticker_data.get("percentage"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                engine = get_engine()
                redis = get_redis_client()
                data = await asyncio.to_thread(lambda: {
                    "current_coins": engine.current_coins,
                    "positions": engine.positions,
                    "balances": engine.trader.fetch_balance(),
                    "trades": engine.trade_history,
                    "profit": engine.get_profit_summary(),
                    "performance": engine.get_performance_summary(),
                    "paused": redis.get("trading:paused") == "1",
                })
                await websocket.send_text(json.dumps(data))
            except HTTPException:
                # Engine not ready yet, send empty
                await websocket.send_text(json.dumps({"status": "initializing"}))
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
