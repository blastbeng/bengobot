import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config.settings import settings
from src.utils.redis_client import get_redis_client, check_redis_connection

app = FastAPI(title="Crypto Trading Bot")

# Serve static files (dashboard)
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global engine reference
_engine = None

def set_engine(engine):
    global _engine
    _engine = engine

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
        "coin_timeframes": getattr(engine, 'coin_timeframes', {}),
        "positions": engine.positions,
        "balances": engine.trader.fetch_balance(),
        "paused": paused,
    }

@app.get("/api/trades")
def trades(limit: int = 20):
    engine = get_engine()
    trades = engine.trade_history[-limit:]
    return {"trades": trades}

@app.get("/api/profit")
def profit():
    engine = get_engine()
    return engine.get_profit_summary()

@app.get("/api/config")
def config():
    return {
        "exchange_id": settings.EXCHANGE_ID,
        "trading_mode": settings.TRADING_MODE,
        "base_currency": settings.BASE_CURRENCY,
        "max_coins": settings.MAX_COINS,
        "ollama_model": settings.OLLAMA_MODEL,
        "web_port": settings.WEB_PORT,
    }

@app.get("/api/ohlcv/{symbol}")
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                engine = get_engine()
                redis = get_redis_client()
                data = await asyncio.to_thread(lambda: {
                    "current_coins": engine.current_coins,
                    "positions": engine.positions,
                    "balances": engine.trader.fetch_balance(),
                    "trades": engine.trade_history[-20:],
                    "profit": engine.get_profit_summary(),
                    "paused": redis.get("trading:paused") == "1",
                })
                await websocket.send_text(json.dumps(data))
            except HTTPException:
                # Engine not ready yet, send empty
                await websocket.send_text(json.dumps({"status": "initializing"}))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
