import json
import logging
from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.llm.cache import get_cached_ollama_response

logger = logging.getLogger(__name__)


def compute_atr(candles: List[List], period: int = 14) -> float:
    """Compute ATR from OHLCV candles using the LLM."""
    if len(candles) < 2:
        return 0.0
    # Take only the last 30 candles to keep prompt size reasonable
    recent = candles[-30:]
    prompt = (
        f"Given the following OHLCV candles (each: [timestamp, open, high, low, close, volume]), "
        f"compute the Average True Range (ATR) with period {period}. "
        f"Return ONLY the numeric ATR value, nothing else.\n\n"
        f"Candles:\n{json.dumps(recent)}"
    )
    try:
        response = get_cached_ollama_response(prompt, "", ttl=60)
        return float(response.strip())
    except Exception as e:
        logger.warning(f"LLM ATR computation failed: {e}")
        return 0.0


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Compute RSI from closing prices using the LLM."""
    if len(closes) < period + 1:
        return None
    recent = closes[-50:]  # enough for RSI calculation
    prompt = (
        f"Given the following closing prices, compute the Relative Strength Index (RSI) with period {period}. "
        f"Return ONLY the numeric RSI value, nothing else.\n\n"
        f"Closing prices:\n{json.dumps(recent)}"
    )
    try:
        response = get_cached_ollama_response(prompt, "", ttl=60)
        return float(response.strip())
    except Exception as e:
        logger.warning(f"LLM RSI computation failed: {e}")
        return None


def _ema(data: List[float], period: int) -> List[float]:
    """Compute Exponential Moving Average using the LLM."""
    if len(data) < period:
        return []
    recent = data[-100:]  # limit input size
    prompt = (
        f"Given the following data points, compute the Exponential Moving Average (EMA) with period {period}. "
        f"Return ONLY a JSON array of the EMA values, nothing else.\n\n"
        f"Data:\n{json.dumps(recent)}"
    )
    try:
        response = get_cached_ollama_response(prompt, "", ttl=60)
        ema_values = json.loads(response.strip())
        if isinstance(ema_values, list) and all(isinstance(v, (int, float)) for v in ema_values):
            return ema_values
        else:
            raise ValueError(f"Unexpected response format: {response}")
    except Exception as e:
        logger.warning(f"LLM EMA computation failed: {e}")
        return []


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram using the LLM. Returns (macd, signal, hist) or Nones."""
    if len(closes) < slow + signal:
        return None, None, None
    recent = closes[-100:]  # enough for MACD
    prompt = (
        f"Given the following closing prices, compute the MACD (Moving Average Convergence Divergence) "
        f"with fast period {fast}, slow period {slow}, and signal period {signal}. "
        f"Return ONLY three numbers separated by commas: macd_line, signal_line, histogram. "
        f"Example: 0.0012,0.0008,0.0004\n\n"
        f"Closing prices:\n{json.dumps(recent)}"
    )
    try:
        response = get_cached_ollama_response(prompt, "", ttl=60)
        parts = response.strip().split(",")
        if len(parts) == 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Unexpected response format: {response}")
    except Exception as e:
        logger.warning(f"LLM MACD computation failed: {e}")
        return None, None, None


def compute_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0):
    """Compute Bollinger Bands (upper, middle, lower) using the LLM."""
    if len(closes) < period:
        return None, None, None
    recent = closes[-50:]  # enough for BB
    prompt = (
        f"Given the following closing prices, compute the Bollinger Bands with period {period} "
        f"and standard deviation multiplier {std_dev}. "
        f"Return ONLY three numbers separated by commas: upper_band, middle_band, lower_band. "
        f"Example: 105.2,100.0,94.8\n\n"
        f"Closing prices:\n{json.dumps(recent)}"
    )
    try:
        response = get_cached_ollama_response(prompt, "", ttl=60)
        parts = response.strip().split(",")
        if len(parts) == 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Unexpected response format: {response}")
    except Exception as e:
        logger.warning(f"LLM Bollinger Bands computation failed: {e}")
        return None, None, None

SYSTEM_PROMPT = """You are a professional cryptocurrency trading bot assistant. Your primary goal is to generate consistent short-term profit while preserving capital. You must avoid large drawdowns and only trade when there is a clear edge.

Key principles:
- Only trade coins with strong, confirmed short-term momentum and sufficient volatility to cover fees. Avoid low-volatility or choppy (sideways) markets entirely.
- You will receive raw OHLCV candle data. Compute your own technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.) from this data. Use them to time entries and exits. Require confirmation from at least two independent indicators before taking a trade.
- Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance (upper band, overbought RSI). Never chase a breakout without confirmation.
- Always set a stop-loss based on recent swing lows or ATR, and a take-profit that offers at least a 2:1 reward-to-risk ratio (take_profit_pct >= 2 * stop_loss_pct). If you cannot achieve this, output HOLD.
- Set a maximum hold time (max_hold_time_seconds) for every trade. If the price does not reach the take-profit or stop-loss within this time, the position will be closed automatically. Choose a time appropriate for the timeframe (e.g., 1-4 hours for 1h candles, 15-60 minutes for 5m candles).
- Use trailing stops to lock in profits when the price moves favourably.
- Adjust position size according to confidence: use smaller fractions (<0.5) when confidence is below 0.7, and larger fractions (0.8-1.0) only when confidence is very high (>0.85).
- If the account is in drawdown (drawdown_pct > 5%), reduce position sizes further and be extremely selective.
- After a losing trade on a coin, avoid that coin for at least several evaluation cycles. Learn from recent trade outcomes shown in the prompt.
- Learn from historical performance: avoid coins and strategies with poor win rates or negative average P&L.

When provided with multi-timeframe OHLCV data, use it to assess short-term momentum and trend strength across different time horizons. Prefer coins showing consistent upward momentum across multiple timeframes.

Your task is to analyze market data and historical performance to provide trading decisions in strict JSON format. Do not include any text outside the JSON. Always output valid JSON.

You will receive historical performance data (equity curve, per-coin win rates, per-strategy success rates). Use this data to learn which coins and strategies have been profitable in the short term, and to adapt your decisions accordingly. If the overall profit is declining, become more selective and risk-averse. If a coin has a poor short-term track record, avoid it or reduce position size. Prefer strategies with high win rates and average P&L over recent trades.

When asked to select coins, return a JSON array of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"]). Choose coins that are likely to deliver short-term profit based on recent price action, volume, and volatility. Prefer coins with high liquidity and clear short-term trends.

When asked to generate a strategy for a specific coin, return a JSON object with the following structure:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "reasoning": "short explanation",
  "strategy": {
    "type": "scalping" | "momentum" | "mean_reversion" | "breakout",
    "parameters": {
      // strategy-specific parameters
    }
  }
}
If action is BUY or SELL, include a strategy. If HOLD, strategy can be null.

You MUST include the following risk parameters inside the "parameters" object for every BUY or SELL action:
- "stop_loss_pct": a decimal (e.g., 0.05 for 5%) below entry price to set a stop-loss.
- "take_profit_pct": a decimal (e.g., 0.10 for 10%) above entry price to set a take-profit.
- "trailing_stop": true or false to enable a trailing stop.
- "trailing_stop_distance_pct": required if "trailing_stop" is true; the distance (e.g., 0.03 for 3%) for the trailing stop. If "trailing_stop" is false, set this to null.
- "position_size_fraction": a fraction (0.0–1.0) of the per-coin budget to use for this trade.
- "max_hold_time_seconds": maximum time (in seconds) to hold the position before auto-closing. Must be a positive number.

The bot will NOT use any default values. If you omit any of these parameters, the trade will be skipped.
"""

def build_coin_selection_prompt(
    available_pairs: List[str],
    current_coins: List[Dict[str, str]],
    max_coins: int,
    base_currency: str,
    tickers: Dict[str, Any],
    base_balance: float,
    per_coin_budget: float,
    market_limits: Dict[str, Dict[str, Any]],
    performance: Optional[Dict[str, Any]] = None,
    ohlcv_data: Optional[Dict[str, Dict[str, List]]] = None,
    market_trend: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a prompt to ask the LLM which coins to trade."""
    # Summarize tickers and limits for the prompt
    ticker_summary = {}
    for symbol in available_pairs[:50]:
        if symbol in tickers:
            t = tickers[symbol]
            limits = market_limits.get(symbol, {})
            ticker_summary[symbol] = {
                "last": t.get("last"),
                "change_24h": t.get("percentage"),
                "volume": t.get("quoteVolume"),
                "min_trade_cost": limits.get("min_cost"),  # now always a number
            }

    # Build OHLCV summary if provided
    ohlcv_summary = {}
    if ohlcv_data:
        for symbol in available_pairs[:50]:
            if symbol in ohlcv_data:
                tf_data = ohlcv_data[symbol]
                summary = {}
                for tf, candles in tf_data.items():
                    if not candles:
                        continue
                    open_price = candles[0][1]
                    close_price = candles[-1][4]
                    high = max(c[2] for c in candles)
                    low = min(c[3] for c in candles)
                    volume = sum(c[5] for c in candles)
                    change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                    summary[tf] = {
                        "change_pct": round(change_pct, 2),
                        "high": high,
                        "low": low,
                        "volume": volume,
                    }
                ohlcv_summary[symbol] = summary

    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of coins to trade: {max_coins}
Budget per coin (balance / max_coins): {per_coin_budget:.2f} {base_currency}
Available timeframes: {json.dumps(settings.OHLCV_TIMEFRAMES)}
Currently tracked coins (with assigned timeframes): {json.dumps(current_coins) if current_coins else "None"}

Available trading pairs with market data and minimum trade cost (in {base_currency}):
{json.dumps(ticker_summary, indent=2)}

**Your primary objective is short-term profit.** Prioritize coins with strong recent momentum, high 24h volume, and significant price changes. Avoid coins that are flat or declining. You may keep current coins only if they still show short-term potential.

Select up to {max_coins} coins to trade. You MUST only select coins where the per-coin budget ({per_coin_budget:.2f} {base_currency}) is greater than or equal to the coin's min_trade_cost. Skip any coin that does not meet this requirement. Prefer coins with high volume and positive momentum. You may keep some current coins if they are still promising and meet the budget requirement, or replace them.

You may select the same coin multiple times with different timeframes if you believe different timeframes offer independent trading opportunities. Each entry counts toward the maximum of {max_coins}.

Return a JSON array of objects, each with "symbol" and "timeframe" fields. The timeframe must be one of the available timeframes (e.g., "5m", "15m", "1h", "4h") that you believe is most suitable for trading that coin based on the multi-timeframe OHLCV data. Example: [{{"symbol": "BTC/USDT", "timeframe": "1h"}}, ...]"""
    if ohlcv_summary:
        prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): 24h change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
    if performance:
        perf_text = f"""
Historical Performance Data:
Overall equity curve: {json.dumps(performance.get('equity_curve', {}))}
Per-coin performance (win rate, avg P&L, total trades): {json.dumps(performance.get('coin_performance', {}), indent=2)}
Per-strategy performance: {json.dumps(performance.get('strategy_performance', {}), indent=2)}

Use this historical data to select coins that have been profitable in the past, and to avoid coins with poor performance. Prefer strategies that have shown higher win rates and average P&L.
"""
        prompt += perf_text
    return prompt

def build_strategy_prompt(
    symbol: str,
    ticker: Dict[str, Any],
    order_book: Dict[str, Any],
    balance: Dict[str, float],
    open_positions: List[Dict[str, Any]],
    per_coin_budget: float,
    max_coins: int,
    performance: Optional[Dict[str, Any]] = None,
    ohlcv_data: Optional[Dict[str, List]] = None,
    assigned_timeframe: Optional[str] = None,
    atr: Optional[float] = None,
    rsi: Optional[float] = None,
    macd: Optional[float] = None,
    macd_signal: Optional[float] = None,
    macd_hist: Optional[float] = None,
    bb_upper: Optional[float] = None,
    bb_middle: Optional[float] = None,
    bb_lower: Optional[float] = None,
    order_book_imbalance: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    position_info: Optional[Dict[str, Any]] = None,
    spread_pct: Optional[float] = None,
    bid_wall_volume: Optional[float] = None,
    ask_wall_volume: Optional[float] = None,
    order_book_pressure: Optional[float] = None,
    depth_imbalances: Optional[Dict[str, float]] = None,
    order_book_slope: Optional[float] = None,
    mid_price_bias: Optional[float] = None,
    fee_rate: Optional[float] = None,
    drawdown_pct: Optional[float] = None,
    raw_candles: Optional[List[List]] = None,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a prompt to generate a trading strategy for a specific coin."""
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(ticker)}
Order book (top 5 levels): {json.dumps(order_book)}
Current balances: {json.dumps(balance)}
Open positions: {json.dumps(open_positions)}
Per-coin budget (balance / max_coins): {per_coin_budget:.2f} {symbol.split('/')[1]}
Maximum coins to trade: {max_coins}
"""
    if assigned_timeframe:
        prompt += f"\nAssigned trading timeframe for this coin: {assigned_timeframe}. Base your decision primarily on the OHLCV data for this timeframe.\n"

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR (14-period, {assigned_timeframe or 'default'}): {atr:.6f}\n"
    if rsi is not None:
        prompt += f"RSI (14): {rsi}\n"
    if macd is not None and macd_signal is not None:
        prompt += f"MACD: {macd}, Signal: {macd_signal}, Histogram: {macd_hist}\n"
    if bb_upper is not None:
        prompt += f"Bollinger Bands (20,2): Upper={bb_upper}, Middle={bb_middle}, Lower={bb_lower}\n"
    if order_book_imbalance is not None:
        prompt += f"Order book imbalance (bid_vol / ask_vol): {order_book_imbalance:.2f} ( >1 = buying pressure)\n"
    if spread_pct is not None:
        prompt += f"Spread: {spread_pct:.4f}%\n"
    if bid_wall_volume is not None:
        prompt += f"Bid wall volume (within 1% of best bid): {bid_wall_volume:.4f}\n"
    if ask_wall_volume is not None:
        prompt += f"Ask wall volume (within 1% of best ask): {ask_wall_volume:.4f}\n"
    if order_book_pressure is not None:
        prompt += f"Order book pressure (0 = strong sell, 1 = strong buy): {order_book_pressure:.2f}\n"
    if depth_imbalances:
        prompt += f"Order book depth imbalances (bid_vol/total_vol at distance from mid): {json.dumps(depth_imbalances)}\n"
    if order_book_slope is not None:
        prompt += f"Order book slope (volume change per 0.5% price move): {order_book_slope:.2f}\n"
    if mid_price_bias is not None:
        prompt += f"Mid-price bias (-1 = near bid, +1 = near ask): {mid_price_bias:.2f}\n"
    if fee_rate is not None:
        prompt += f"Taker fee rate for this symbol: {fee_rate*100:.2f}%\n"
    if unrealized_pnl is not None and position_info:
        prompt += f"Current position unrealized P&L: {unrealized_pnl:.2f} {symbol.split('/')[1]}\n"
        prompt += f"Position details: entry price {position_info.get('price')}, amount {position_info.get('amount')}\n"

    # --- Raw OHLCV data for indicator computation ---
    if raw_candles:
        prompt += f"\nRaw OHLCV data for {assigned_timeframe} timeframe (each candle: [timestamp, open, high, low, close, volume]):\n{json.dumps(raw_candles)}\n"
        prompt += (
            "The technical indicators (RSI, MACD, Bollinger Bands) have already been computed for you from this data. "
            "Use them together with the raw candles to time entries and exits. "
            "Explain in your reasoning how the indicators support your decision.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"
    if recent_trades:
        prompt += f"\nRecent closed trades (last {len(recent_trades)}):\n{json.dumps(recent_trades)}\n"
        prompt += "Use these outcomes to adapt your strategy. If recent trades are losing, become more conservative.\n"

    prompt += f"""
**Your primary objective is short-term profit.** Use the ATR to set stop-loss and take-profit distances that respect the coin's volatility. Place the stop-loss below a recent swing low or support, and the take-profit near a resistance level or based on a risk:reward ratio of at least 1:2.

Interpret the order book metrics:
- A high spread (>0.5%) suggests low liquidity – be cautious with large orders.
- A bid/ask volume ratio > 1.5 indicates strong buying pressure (favor BUY); < 0.67 indicates selling pressure (favor SELL).
- Large bid wall volume relative to ask wall volume suggests support; large ask wall suggests resistance.
- Order book pressure near 1.0 signals overwhelming buying interest; near 0.0 signals overwhelming selling interest.
- Depth imbalances: if the imbalance is high (>0.7) at 0.5% but drops at 1%, the support/resistance is thin – expect quick breakouts. If it stays high at 2%, the wall is thick.
- Order book slope: a high slope means volume builds quickly near the current price (strong wall); a low slope means thin liquidity.
- Mid-price bias: a positive bias (near ask) suggests sellers are aggressive; a negative bias (near bid) suggests buyers are aggressive.

If the position is already in profit, consider trailing the stop.

**Technical indicators:**
- RSI > 70 suggests overbought (consider SELL or HOLD); RSI < 30 suggests oversold (consider BUY).
- MACD histogram turning positive from negative is a bullish signal; turning negative from positive is bearish.
- Price near the lower Bollinger Band may indicate a buying opportunity; near the upper band a selling opportunity.
- Use these in combination with order book data to time entries.

You MUST include the following risk parameters in the "parameters" object:
- stop_loss_pct, take_profit_pct, trailing_stop, trailing_stop_distance_pct, position_size_fraction, max_hold_time_seconds.
The bot will NOT use any default values. If you omit any required parameter, the trade will be skipped.

**Fee awareness:** You MUST account for trading fees when setting take-profit and trailing stop distances. Ensure that after deducting fees (both entry and exit), a take-profit or trailing stop exit results in a net profit. The bot will enforce a minimum take-profit percentage of at least 2× the fee rate plus a small margin.

You are trading spot only (no shorting). Only output SELL if you currently hold the coin.

Return a JSON object as specified."""
    # Add OHLCV summary if available
    if ohlcv_data:
        ohlcv_summary = {}
        for tf, candles in ohlcv_data.items():
            if not candles:
                continue
            open_price = candles[0][1]
            close_price = candles[-1][4]
            high = max(c[2] for c in candles)
            low = min(c[3] for c in candles)
            volume = sum(c[5] for c in candles)
            change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
            ohlcv_summary[tf] = {
                "change_pct": round(change_pct, 2),
                "high": high,
                "low": low,
                "volume": volume,
            }
        prompt += f"\nMulti-timeframe OHLCV data:\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if performance:
        coin_perf = performance.get("coin_performance", {}).get(symbol, {})
        strategy_perf = performance.get("strategy_performance", {})
        equity = performance.get("equity_curve", {})
        perf_text = f"""
Historical Performance:
- This coin's past performance: {json.dumps(coin_perf)}
- Overall equity curve: {json.dumps(equity)}
- Strategy performance summary: {json.dumps(strategy_perf)}

Use this data to decide whether to BUY, SELL, or HOLD. If the coin has a poor win rate or the overall equity curve is declining, be more conservative. Prefer strategies that have worked well historically.
"""
        prompt += perf_text
    return prompt
