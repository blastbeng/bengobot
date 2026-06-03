import json
from typing import List, Dict, Any, Optional
from src.config.settings import settings


def compute_atr(candles: List[List], period: int = 14) -> float:
    """Compute ATR from OHLCV candles (each candle: [timestamp, open, high, low, close, volume])."""
    if len(candles) < 2:
        return 0.0
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        prev_close = candles[i - 1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if not true_ranges:
        return 0.0
    period = min(period, len(true_ranges))
    return sum(true_ranges[-period:]) / period


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Compute RSI for a list of closing prices."""
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def _ema(data: List[float], period: int) -> List[float]:
    """Exponential moving average."""
    if len(data) < period:
        return []
    ema = [sum(data[:period]) / period]
    multiplier = 2 / (period + 1)
    for price in data[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram. Returns (macd, signal, hist) or Nones."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    # Align lengths
    macd_line = macd_line[-len(signal_line):]
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return round(macd_line[-1], 6), round(signal_line[-1], 6), round(hist[-1], 6)


def compute_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0):
    """Return (upper, middle, lower) Bollinger Bands for the most recent candle."""
    if len(closes) < period:
        return None, None, None
    recent = closes[-period:]
    middle = sum(recent) / period
    variance = sum((x - middle) ** 2 for x in recent) / period
    std = variance ** 0.5
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return round(upper, 6), round(middle, 6), round(lower, 6)

SYSTEM_PROMPT = """You are a professional cryptocurrency trading bot assistant. Your primary goal is to generate consistent short-term profit while preserving capital. You must avoid large drawdowns and only trade when there is a clear edge.

Key principles:
- Only trade coins with strong, confirmed short-term momentum and sufficient volatility to cover fees.
- Avoid trading in choppy, sideways, or low-volume markets. If the overall market (e.g., BTC) is flat or declining, be more selective.
- You will receive raw OHLCV candle data. Compute your own technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.) from this data. Use them to time entries and exits. Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance (upper band, overbought RSI).
- Always set a stop-loss based on recent swing lows or ATR, and a take-profit that offers at least a 2:1 reward-to-risk ratio (take_profit_pct >= 2 * stop_loss_pct). If you cannot achieve this, output HOLD.
- Use trailing stops to lock in profits when the price moves favourably.
- Adjust position size according to confidence: use smaller fractions (<0.5) when confidence is below 0.7, and larger fractions (0.8-1.0) only when confidence is very high (>0.85).
- If the account is in drawdown (drawdown_pct > 5%), reduce position sizes further and be extremely selective.
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
            "You MUST compute your own technical indicators from this raw data. "
            "Common indicators include RSI (14), MACD (12,26,9), Bollinger Bands (20,2), "
            "moving averages, etc. Use these computed indicators to time entries and exits. "
            "Explain in your reasoning how the indicators support your decision.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"

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
- stop_loss_pct, take_profit_pct, trailing_stop, trailing_stop_distance_pct, position_size_fraction.
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
