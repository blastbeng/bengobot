import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
from src.database import get_news_for_symbol, get_aggregate_sentiment_from_db

logger = logging.getLogger(__name__)


def compute_atr(candles: List[List], period: int = 14) -> float:
    """Compute Average True Range from OHLCV candles using Wilder's smoothing."""
    if len(candles) < period + 1:
        return 0.0
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        prev_close = candles[i - 1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    if len(tr_values) < period:
        return 0.0
    # Wilder's smoothing: seed with SMA of first `period` values
    atr = sum(tr_values[:period]) / period
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period
    return atr


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Compute RSI from closing prices."""
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    # Use smoothed averages for the remaining data
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0)
        loss = -min(diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_ema(data: List[float], period: int) -> List[float]:
    """Compute Exponential Moving Average."""
    if len(data) < period:
        return []
    ema_values = []
    # SMA for the first value
    sma = sum(data[:period]) / period
    ema_values.append(sma)
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(data)):
        ema = (data[i] - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(ema)
    return ema_values


def compute_stochastic(
    highs: List[float], lows: List[float], closes: List[float],
    period: int = 14, smooth_k: int = 3
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Stochastic Oscillator %K and %D."""
    if len(closes) < period:
        return None, None
    recent_high = max(highs[-period:])
    recent_low = min(lows[-period:])
    if recent_high == recent_low:
        return 50.0, 50.0
    fast_k = ((closes[-1] - recent_low) / (recent_high - recent_low)) * 100
    k_values = []
    for i in range(-period, 0):
        start = i - period + 1
        end = i + 1 if i + 1 != 0 else None
        h = max(highs[start:end])
        l = min(lows[start:end])
        if h == l:
            k_values.append(50.0)
        else:
            k_values.append(((closes[i] - l) / (h - l)) * 100)
    if len(k_values) < smooth_k:
        return fast_k, None
    slow_d = sum(k_values[-smooth_k:]) / smooth_k
    return fast_k, slow_d


def compute_adx(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute ADX, +DI, -DI using Wilder's smoothing."""
    if len(closes) < period + 1:
        return None, None, None
    tr = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i-1]
        prev_low = lows[i-1]
        prev_close = closes[i-1]
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        up_move = high - prev_high
        down_move = prev_low - low
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)
    if len(tr) < period:
        return None, None, None
    atr = sum(tr[:period]) / period
    smoothed_plus_dm = sum(plus_dm[:period]) / period
    smoothed_minus_dm = sum(minus_dm[:period]) / period
    dx_values = []
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
        smoothed_plus_dm = (smoothed_plus_dm * (period - 1) + plus_dm[i]) / period
        smoothed_minus_dm = (smoothed_minus_dm * (period - 1) + minus_dm[i]) / period
        if atr == 0:
            dx = 0.0
        else:
            plus_di = (smoothed_plus_dm / atr) * 100
            minus_di = (smoothed_minus_dm / atr) * 100
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0.0
        dx_values.append(dx)
    if not dx_values:
        return None, None, None
    adx = sum(dx_values[:period]) / period if len(dx_values) >= period else dx_values[-1]
    for i in range(period, len(dx_values)):
        adx = (adx * (period - 1) + dx_values[i]) / period
    if atr == 0:
        plus_di = 0.0
        minus_di = 0.0
    else:
        plus_di = (smoothed_plus_dm / atr) * 100
        minus_di = (smoothed_minus_dm / atr) * 100
    return adx, plus_di, minus_di


def compute_obv(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Compute On-Balance Volume (latest value)."""
    if len(closes) < 2 or len(volumes) < 2:
        return None
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
    return obv


def compute_mfi(
    highs: List[float], lows: List[float], closes: List[float],
    volumes: List[float], period: int = 14
) -> Optional[float]:
    """Compute Money Flow Index."""
    if len(closes) < period + 1:
        return None
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    raw_money_flow = [tp * vol for tp, vol in zip(typical_prices, volumes)]
    recent_tp = typical_prices[-period-1:]
    recent_rmf = raw_money_flow[-period-1:]
    pos = 0.0
    neg = 0.0
    for i in range(1, len(recent_tp)):
        if recent_tp[i] > recent_tp[i-1]:
            pos += recent_rmf[i]
        else:
            neg += recent_rmf[i]
    if neg == 0:
        return 100.0
    return 100 - (100 / (1 + pos / neg))


def compute_cci(
    highs: List[float], lows: List[float], closes: List[float], period: int = 20
) -> Optional[float]:
    """Compute Commodity Channel Index."""
    if len(closes) < period:
        return None
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    recent_tp = typical_prices[-period:]
    sma = sum(recent_tp) / period
    mean_deviation = sum(abs(tp - sma) for tp in recent_tp) / period
    if mean_deviation == 0:
        return 0.0
    cci = (recent_tp[-1] - sma) / (0.015 * mean_deviation)
    return cci


def compute_williams_r(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> Optional[float]:
    """Compute Williams %R."""
    if len(closes) < period:
        return None
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    if highest_high == lowest_low:
        return -50.0
    wr = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
    return wr


def compute_macd(
    closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute MACD line, signal line, and histogram."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    if not ema_fast or not ema_slow:
        return None, None, None
    # Align lengths: MACD line = fast EMA - slow EMA (use the shorter length)
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(min_len)]
    signal_line = compute_ema(macd_line, signal)
    if not signal_line:
        return None, None, None
    # Return the latest values
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    hist = macd_val - signal_val
    return macd_val, signal_val, hist


def compute_bollinger_bands(
    closes: List[float], period: int = 20, std_dev: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    if len(closes) < period:
        return None, None, None
    recent = closes[-period:]
    middle = sum(recent) / period
    variance = sum((x - middle) ** 2 for x in recent) / period
    std = math.sqrt(variance)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def _format_news_for_prompt(articles: list) -> str:
    """Format a list of news articles into a compact string for the LLM prompt."""
    if not articles:
        return "No recent news available."
    lines = []
    for i, art in enumerate(articles, 1):
        sentiment = art.get("sentiment", {})
        label = sentiment.get("label", "unknown")
        compound = sentiment.get("compound", 0.0)
        lines.append(
            f"{i}. [{art.get('source', 'Unknown')}] {art.get('title', '')} "
            f"({art.get('published_at', '')}) - Sentiment: {label} ({compound:.2f}) - {art.get('summary', '')[:200]}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a professional cryptocurrency trading bot assistant. Your primary goal is to generate consistent profit across short, medium, and long timeframes. Prioritize positions where you find the most profit potential, regardless of timeframe, while preserving capital. You must avoid large drawdowns and only trade when there is a clear edge.

Key principles:
- **Confidence is your directional conviction, not a trade gate.**
  Set confidence between 0.0 and 1.0 to reflect how sure you are about the price direction.
  - 0.0 → no conviction (should be HOLD).
  - 0.5 → moderate belief.
  - 1.0 → absolute certainty.
  **You must set `position_size_fraction` yourself to reflect your confidence, risk level, and any other factors.**
  The engine will NOT scale the position size automatically – it will use exactly the fraction you provide.
  Therefore, if you have low confidence, set a smaller `position_size_fraction`; if high confidence, you may set a larger one.
  Only output HOLD when you have no directional edge at all.
- Only trade coins with strong, confirmed short-term momentum and sufficient volatility to cover fees. Avoid low-volatility or choppy (sideways) markets entirely.
- You will receive raw OHLCV candle data. Compute your own technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.) from this data. Use them to time entries and exits. Require confirmation from at least two independent indicators before taking a trade.
- Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance (upper band, overbought RSI). Never chase a breakout without confirmation.
- Set a stop-loss based on recent swing lows, support levels, or ATR. Use the ATR to gauge volatility and choose a stop distance that gives the trade enough room to breathe while limiting risk. You decide the appropriate multiplier and reward:risk ratio based on current market conditions, volatility, and your confidence. The bot enforces a minimum stop distance of 0.5× ATR. Your stop_loss_pct (or the effective stop from atr_multiple) must be at least 0.5 * (ATR / current_price). If you set a tighter stop, the trade will be rejected.
Example: If ATR=50 and current price=5000, a 2× ATR stop distance is 100, so stop_loss_pct = 100/5000 = 0.02 (2%). Place the stop at 4900. If the nearest swing low is at 4920, use that as the stop level (distance 80, which is 1.6× ATR, still acceptable).
- Set a take-profit that you believe is achievable given the current trend, volatility, and order‑book depth. The reward:risk ratio is entirely your decision; you may accept lower ratios if the probability of success is high, or demand higher ratios in uncertain markets.
- Set a maximum hold time (max_hold_time_seconds) for every trade. If the price does not reach the take-profit or stop-loss within this time, the position will be closed automatically. Choose a time appropriate for the timeframe (e.g., 1-4 hours for 1h candles, 15-60 minutes for 5m candles).
- Use trailing stops to lock in profits when the price moves favourably.
- Adjust position size according to your confidence, risk level, account drawdown, and portfolio exposure. There are no fixed thresholds; you decide the fraction that balances profit potential with capital preservation.
- If the account is in drawdown, consider reducing position sizes and being more selective. The severity of the reduction is your decision based on the drawdown percentage and recent performance.
- After a losing trade on a coin, avoid that coin for at least several evaluation cycles. Learn from recent trade outcomes shown in the prompt.
- Learn from historical performance: avoid coins and strategies with poor win rates or negative average P&L.
- **Learn from past trade outcomes for each coin.** The prompt will include a list of recent closed trades for the current coin. Use this to avoid repeating mistakes and to reinforce successful patterns. If a coin has a string of losses, be more cautious or avoid it.
- You must set a cooldown duration for every BUY. After a losing trade on a coin, the bot will skip that coin for the duration you specify.

You will receive recent news headlines with sentiment scores for each coin. Use this information to gauge market sentiment and potential catalysts.
- Strong positive sentiment may justify higher confidence, larger position sizes, and longer max hold times.
- Strong negative sentiment should make you more cautious: reduce position size, tighten stops, shorten max hold time, or avoid the coin entirely.
- Neutral or mixed sentiment should not override technical signals, but can be used as a tie‑breaker.
- If news sentiment conflicts with technical indicators, give more weight to the indicators, but explain your reasoning.

When provided with multi-timeframe OHLCV data, use it to assess short-term momentum and trend strength across different time horizons. Prefer coins showing consistent upward momentum across multiple timeframes.

Your task is to analyze market data and historical performance to provide trading decisions in strict JSON format. Do not include any text outside the JSON. Always output valid JSON.

You will receive historical performance data (equity curve, per-coin win rates, per-strategy success rates). Use this data to learn which coins and strategies have been profitable in the short term, and to adapt your decisions accordingly. If the overall profit is declining, become more selective and risk-averse. If a coin has a poor short-term track record, avoid it or reduce position size. Prefer strategies with high win rates and average P&L over recent trades.

When selecting coins, consider the provided technical indicators (RSI, MACD, Bollinger Bands, EMAs, Stochastic, ADX, OBV, MFI, CCI, Williams %R) to identify coins with strong momentum, oversold/overbought conditions, and trend strength. Prefer coins with bullish indicator alignments.

You may optionally include an "indicator_config" object in your strategy JSON to customize the indicator parameters for future cycles. If omitted, default parameters will be used. The object can contain any of the following keys (all optional):
- rsi_period (int, default 14)
- macd_fast (int, default 12)
- macd_slow (int, default 26)
- macd_signal (int, default 9)
- bb_period (int, default 20)
- bb_std (float, default 2.0)
- ema_fast (int, default 9)
- ema_slow (int, default 21)
- stoch_k_period (int, default 14)
- stoch_d_period (int, default 3)
- adx_period (int, default 14)
- mfi_period (int, default 14)
- cci_period (int, default 20)
- willr_period (int, default 14)

You may optionally include a "backtest_summary" field (string) when historical OHLCV data is provided. This should be a concise summary of your backtest analysis, e.g., "Simulated 5 trades over 30 days: 3 wins, 2 losses, net +2.3%". Include it only if you performed a backtest.

When asked to select coins, return a JSON array of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"]). Choose coins that are likely to deliver short-term profit based on recent price action, volume, and volatility. Prefer coins with high liquidity and clear short-term trends.

When asked to generate a strategy for a specific coin, return a JSON object with the following structure:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,   # directional conviction (0 = no edge, 1 = certain). Used to scale position size.
  "reasoning": "short explanation",
  "risk_level": "low" | "medium" | "high",
  "strategy": {
    "type": "scalping" | "momentum" | "mean_reversion" | "breakout",
    "parameters": {
      // strategy-specific parameters
    }
  }
}
The "risk_level" field controls overall risk appetite for this trade:
- "low": use smaller position sizes, wider stops, only trade when very confident.
- "medium": normal risk (default).
- "high": aggressive, larger position sizes, tighter stops (only when market conditions are extremely favourable).

If action is BUY or SELL, include a strategy. If HOLD, strategy can be null.

You MUST include the following risk parameters inside the "parameters" object for every BUY or SELL action. All numeric values must be numbers, not strings.

- "stop_loss_pct": a decimal between 0.001 and 0.5 (e.g., 0.02 for 2%). Must be greater than 0 and less than 1.0. This is required unless you use the "atr_multiple" stop-loss method (see below).
- "take_profit_pct": a decimal between 0.005 and 2.0 (e.g., 0.05 for 5%). Must be greater than stop_loss_pct and at least 2× the fee rate.
- "trailing_stop": true or false to enable a trailing stop.
- "trailing_stop_distance_pct": required if "trailing_stop" is true; a decimal between 0.001 and 0.1 (e.g., 0.01 for 1%). Must be less than stop_loss_pct. If "trailing_stop" is false, set this to null.
- "position_size_fraction": a decimal between 0.1 and 1.0 representing the fraction of your **total available quote currency balance** to allocate to this trade (e.g., 0.5 for 50% of your entire quote balance). Must be > 0 and ≤ 1. The sum of this fraction across all coins you trade should not exceed 1.0, so leave enough capital for other opportunities.
- "max_hold_time_seconds": a positive integer number of seconds (e.g., 3600 for 1 hour). Must be > 0.
- "cooldown_after_loss_seconds": a non-negative integer (0 or more). If the trade results in a loss, the bot will avoid this coin for this many seconds before considering it again. Set 0 to allow immediate re-entry.

You may also include the following optional parameters to fine-tune risk management:

- "stop_loss_method": "fixed" (default) or "atr_multiple". If "atr_multiple", the stop distance is computed as stop_loss_atr_multiple × ATR, and "stop_loss_pct" is optional (if provided, it will be ignored). Use this to set a volatility-based stop.
- "stop_loss_atr_multiple": required if stop_loss_method is "atr_multiple". A positive float (e.g., 2.0 for 2× ATR). The stop distance will be (multiplier × ATR) / current_price.
- "trailing_stop_activation_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2%). The trailing stop will only start updating once the price has moved in your favor by at least this percentage from the entry price. If omitted, the trailing stop is active immediately.
- "breakeven_activation_pct": an optional decimal between 0 and 1.0 (e.g., 0.005 for 0.5%). If set, once the price rises by this percentage above your entry, the stop‑loss will be moved to the exact break‑even price (covering the exit fee). This locks in a risk‑free trade. Use this for scalping or when you want to protect a small gain.
- "lock_profit_activation_pct": an optional decimal between 0 and 1.0 (e.g., 0.005 for 0.5%). If set, once the price rises by this percentage above your entry, the stop‑loss will be moved to a guaranteed profit level (see lock_profit_level_pct). Use this to scalp small gains.
- "lock_profit_level_pct": required if lock_profit_activation_pct is set. A decimal between 0 and lock_profit_activation_pct (e.g., 0.003 for 0.3%). The new stop‑loss level will be set to entry_price * (1 + lock_profit_level_pct). This locks in a minimum profit even if the price reverses.
- "max_risk_per_trade_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2% of portfolio). The position size will be limited so that the potential loss (entry - stop) does not exceed this fraction of your total portfolio value. If omitted, position sizing uses only position_size_fraction.

The bot will NOT use any default values for required parameters. If you omit any required parameter, the trade will be skipped. Optional parameters are not required; if omitted, the bot will use its standard behavior.
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
    news_sentiment: Optional[Dict[str, Dict[str, Any]]] = None,
    coin_indicators: Optional[Dict[str, Dict[str, Any]]] = None,
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
            if settings.NEWS_ENABLED:
                base = symbol.split("/")[0] if "/" in symbol else symbol
                agg = get_aggregate_sentiment_from_db(base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                if agg:
                    ticker_summary[symbol]["sentiment"] = agg

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

    # --- News section ---
    news_section = ""
    if settings.NEWS_ENABLED:
        news_lines = []
        pairs_to_check = available_pairs[:20]
        for pair in pairs_to_check:
            base = pair.split("/")[0] if "/" in pair else pair
            articles = get_news_for_symbol(base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if articles:
                formatted = _format_news_for_prompt(articles)
                news_lines.append(f"**{pair}**\n{formatted}")
        if news_lines:
            news_section = "Recent news for top coins:\n\n" + "\n\n".join(news_lines)

    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of coins to trade: {max_coins}
Budget per coin (balance / max_coins): {per_coin_budget:.2f} {base_currency}
Available timeframes: {json.dumps(settings.OHLCV_TIMEFRAMES)}
Currently tracked coins (with assigned timeframes): {json.dumps(current_coins) if current_coins else "None"}

Available trading pairs with market data and minimum trade cost (in {base_currency}):
{json.dumps(ticker_summary, indent=2)}

**Your primary objective is profit across short, medium, and long timeframes. Prioritize coins where you find the most profit potential, regardless of timeframe.** Prioritize coins with strong momentum, high volume, and clear trends on multiple timeframes. Avoid coins that are flat or declining on all timeframes. You may keep current coins only if they still show potential on at least one timeframe.

Select between 1 and {max_coins} coins to trade. You decide the exact number based on how many high‑quality opportunities you see. If market conditions are poor, you may choose fewer coins (even just 1) to concentrate capital on the best setup. If many strong setups exist, you may select up to {max_coins}. You MUST only select coins where the per-coin budget ({per_coin_budget:.2f} {base_currency}) is greater than or equal to the coin's min_trade_cost. Skip any coin that does not meet this requirement. Prefer coins with high volume and positive momentum. You may keep some current coins if they are still promising and meet the budget requirement, or replace them.

**Use the historical performance data to guide your selection.** Prefer coins that have a positive average P&L and a win rate above 50% in recent trades. Avoid coins that have a string of losses or a negative average P&L, unless there is a strong technical or news‑driven reason to include them.

Each symbol can only appear once in your selection. Choose the single best timeframe for each coin based on the multi-timeframe OHLCV data.

Return a JSON object with two fields:
- "coins": a JSON array of objects, each with "symbol" and "timeframe" (the timeframe must be one of the available timeframes, e.g., "5m", "15m", "1h", "4h").
- "max_coins": an integer between 1 and {max_coins} indicating how many coins you actually want to trade. This must equal the length of the "coins" array.

Example: {{"coins": [{{"symbol": "BTC/USDT", "timeframe": "1h"}}, {{"symbol": "ETH/USDT", "timeframe": "15m"}}], "max_coins": 2}}"""
    if ohlcv_summary:
        prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if coin_indicators:
        prompt += "\nTechnical indicators for candidate coins:\n"
        for sym, tf_indicators in coin_indicators.items():
            lines = [f"{sym}:"]
            for tf, ind in tf_indicators.items():
                lines.append(f"  [{tf}]")
                if ind.get('rsi') is not None:
                    lines.append(f"    RSI(14)={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    lines.append(f"    MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    lines.append(f"    BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    lines.append(f"    EMA9={ind['ema_9']:.4f} EMA21={ind['ema_21']:.4f}")
                if ind.get('stochastic_k') is not None:
                    d_str = f"{ind['stochastic_d']:.2f}" if ind['stochastic_d'] is not None else "N/A"
                    lines.append(f"    Stoch %K={ind['stochastic_k']:.2f} %D={d_str}")
                if ind.get('adx') is not None:
                    lines.append(f"    ADX(14)={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    lines.append(f"    OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    lines.append(f"    MFI(14)={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    lines.append(f"    CCI(20)={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    lines.append(f"    Williams %R(14)={ind['williams_r']:.2f}")
            prompt += "\n".join(lines) + "\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): 24h change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
    if news_sentiment:
        prompt += "\n## News Sentiment\n"
        prompt += "Aggregate sentiment from recent news articles (compound score -1 to +1, higher = more positive):\n"
        for sym in available_pairs:
            base = sym.split("/")[0] if "/" in sym else sym
            if base in news_sentiment:
                ns = news_sentiment[base]
                prompt += (
                    f"- {sym}: compound={ns['avg_compound']}, "
                    f"positive={ns['positive']}, negative={ns['negative']}, "
                    f"neutral={ns['neutral']}, total_articles={ns['total_articles']}\n"
                )
        prompt += "\n"
    if news_section:
        prompt += f"\n{news_section}\n"
    if performance:
        perf_text = f"""
Historical Performance Data:
Overall equity curve: {json.dumps(performance.get('equity_curve', {}))}
Per-coin performance (win rate, avg P&L, total trades): {json.dumps(performance.get('coin_performance', {}), indent=2)}
Per-strategy performance: {json.dumps(performance.get('strategy_performance', {}), indent=2)}

Use this historical data to select coins that have been profitable in the past, and to avoid coins with poor performance. Prefer strategies that have shown higher win rates and average P&L.
"""
        prompt += perf_text
    prompt += (
        "\n**Important:** The engine will use your parameters exactly as you provide them. "
        "No additional scaling, clamping, or overrides will be applied. You are fully responsible "
        "for setting stop_loss_pct, take_profit_pct, position_size_fraction, trailing_stop, "
        "max_hold_time_seconds, cooldown_after_loss_seconds, and all optional parameters. Make sure they are appropriate for "
        "the current market conditions, your confidence, and the account's risk profile.\n"
    )
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
    atr_multi_tf: Optional[Dict[str, float]] = None,
    rsi: Optional[float] = None,
    macd: Optional[float] = None,
    macd_signal: Optional[float] = None,
    macd_hist: Optional[float] = None,
    bb_upper: Optional[float] = None,
    bb_middle: Optional[float] = None,
    bb_lower: Optional[float] = None,
    ema_9: Optional[float] = None,
    ema_21: Optional[float] = None,
    stochastic_k: Optional[float] = None,
    stochastic_d: Optional[float] = None,
    adx: Optional[float] = None,
    plus_di: Optional[float] = None,
    minus_di: Optional[float] = None,
    obv: Optional[float] = None,
    mfi: Optional[float] = None,
    cci: Optional[float] = None,
    williams_r: Optional[float] = None,
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
    historical_ohlcv: Optional[List[List]] = None,
    min_order_amount: Optional[float] = None,
    min_order_cost: Optional[float] = None,
    all_coins: Optional[List[Dict[str, str]]] = None,
    past_trades: Optional[List[Dict[str, Any]]] = None,
    aggregate_sentiment: Optional[Dict[str, Any]] = None,
    cycle_spent: Optional[float] = None,
    remaining_balance: Optional[float] = None,
    market_regime: Optional[str] = None,
) -> str:
    """Build a prompt to generate a trading strategy for a specific coin."""
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(ticker)}
Order book (top 5 levels): {json.dumps(order_book)}
Current balances: {json.dumps(balance)}
"""
    # --- Portfolio context: total base balance and all tracked coins ---
    base_currency = symbol.split('/')[1]
    base_balance = balance.get(base_currency, 0.0)
    prompt += f"\nTotal {base_currency} balance available: {base_balance:.2f}\n"
    if all_coins:
        other_coins = [c for c in all_coins if c["symbol"] != symbol]
        if other_coins:
            coin_list_str = ", ".join(f"{c['symbol']}({c['timeframe']})" for c in other_coins)
            prompt += f"Other coins being traded (you must leave budget for them): {coin_list_str}\n"
        else:
            prompt += "This is the only coin being traded; you may use the full budget.\n"
    prompt += f"""Open positions: {json.dumps(open_positions)}
Your total available {base_currency} balance: {base_balance:.2f}
Suggested equal share per coin (balance / max_coins): {per_coin_budget:.2f} {base_currency}
Maximum coins to trade: {max_coins}
"""
    if cycle_spent is not None and remaining_balance is not None:
        prompt += (
            f"Amount already allocated to other coins in this cycle: {cycle_spent:.2f} {base_currency}\n"
            f"Remaining available for this coin: {remaining_balance:.2f} {base_currency}\n"
            "Your position_size_fraction must not require more than the remaining balance. "
            "If the remaining balance is low, reduce your fraction accordingly or output HOLD.\n"
        )
    prompt += (
        f"**position_size_fraction** now represents a fraction of your **total {base_currency} balance** (0.1 to 1.0). "
        f"You may allocate more than the equal share for high‑confidence/high‑profit opportunities, and less for riskier ones. "
        f"**Important:** The sum of position_size_fraction across all coins you intend to trade must not exceed 1.0, "
        f"so that you leave enough capital for other coins. Plan your allocations accordingly.\n"
    )
    base_coin = symbol.split('/')[0]
    quote_coin = symbol.split('/')[1]
    if min_order_amount is not None or min_order_cost is not None:
        prompt += f"\nMinimum order size for {symbol}:"
        if min_order_amount is not None:
            prompt += f" {min_order_amount} {base_coin}"
        if min_order_cost is not None:
            prompt += f" (or {min_order_cost} {quote_coin} cost)"
        prompt += (
            ". Your position_size_fraction must result in an order that meets both the minimum amount "
            "and the minimum cost. Use the current price to convert between amount and cost.\n"
        )
    if assigned_timeframe:
        prompt += f"\nAssigned trading timeframe for this coin: {assigned_timeframe}. Base your decision primarily on the OHLCV data for this timeframe.\n"
    if market_regime:
        prompt += f"\nMarket regime: {market_regime}\n"
        prompt += (
            "Use this regime to adjust your strategy:\n"
            "- Trending: use wider stops (to avoid being shaken out) and larger position sizes if trend is strong.\n"
            "- Ranging: use tighter stops and smaller positions; prefer mean‑reversion strategies.\n"
            "- High volatility: reduce position size and widen stops.\n"
            "- Low volatility: you may tighten stops but beware of false breakouts.\n"
        )

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR (14-period, {assigned_timeframe or 'default'}): {atr:.6f}\n"
        prompt += (
            "Use the ATR to set your stop-loss distance. Convert the chosen distance into a percentage "
            "of the current price for the stop_loss_pct parameter. You decide the appropriate multiplier "
            "based on current volatility and your risk assessment.\n"
            "The bot requires the stop distance to be at least 0.5× ATR. Ensure your stop_loss_pct (or atr_multiple) meets this minimum.\n"
        )
    if atr_multi_tf:
        prompt += f"ATR across timeframes: {json.dumps(atr_multi_tf)}\n"
        prompt += (
            "Use the higher-timeframe ATR to gauge overall volatility and the lower-timeframe ATR for precise stop-loss placement. "
            "If the higher‑timeframe ATR is large, widen your stop accordingly to avoid being stopped out by normal swings.\n"
        )
    if rsi is not None:
        prompt += f"RSI (14): {rsi}\n"
    if macd is not None and macd_signal is not None:
        prompt += f"MACD: {macd}, Signal: {macd_signal}, Histogram: {macd_hist}\n"
    if bb_upper is not None:
        prompt += f"Bollinger Bands (20,2): Upper={bb_upper}, Middle={bb_middle}, Lower={bb_lower}\n"
    if ema_9 is not None:
        prompt += f"EMA (9): {ema_9}\n"
    if ema_21 is not None:
        prompt += f"EMA (21): {ema_21}\n"
    if stochastic_k is not None:
        d_str = f"{stochastic_d:.2f}" if stochastic_d is not None else "N/A"
        prompt += f"Stochastic Oscillator: %K={stochastic_k:.2f}, %D={d_str}\n"
    if adx is not None:
        prompt += f"ADX(14): {adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}\n"
    if obv is not None:
        prompt += f"On-Balance Volume (OBV): {obv:.2f}\n"
    if mfi is not None:
        prompt += f"Money Flow Index (MFI 14): {mfi:.2f}\n"
    if cci is not None:
        prompt += f"Commodity Channel Index (CCI 20): {cci:.2f}\n"
    if williams_r is not None:
        prompt += f"Williams %R (14): {williams_r:.2f}\n"
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
            "The technical indicators (RSI, MACD, Bollinger Bands, EMA) have already been computed for you from this data. "
            "Use them together with the raw candles to time entries and exits. "
            "Explain in your reasoning how the indicators support your decision.\n"
        )
    if historical_ohlcv:
        # Limit to last 500 candles to keep prompt size manageable
        limited_hist = historical_ohlcv[-500:] if len(historical_ohlcv) > 500 else historical_ohlcv
        prompt += f"\nHistorical OHLCV data for the last {len(limited_hist)} candles ({assigned_timeframe} timeframe):\n{json.dumps(limited_hist)}\n"
        prompt += (
            "You have been provided with historical OHLCV data covering up to the last 30 days (or the available period). "
            "Use this data to perform a backtest analysis: simulate potential trades based on your strategy, evaluate profitability, "
            "and use the insights to inform your current decision. You may choose a subset of this period for your backtest "
            "(default is the full period). Explain in your reasoning how the backtest results influenced your decision.\n"
            "Include a 'backtest_summary' field in your JSON output with a short summary of the backtest results.\n"
        )
        prompt += (
            "Your backtest analysis must directly influence your current decision. "
            "If the backtest shows poor performance for your intended strategy, adjust your parameters "
            "(e.g., wider stop, smaller position, different entry timing) or output HOLD. "
            "Explain in your reasoning how the backtest results affected your choices.\n"
        )
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"
    if recent_trades:
        prompt += f"\nRecent closed trades (last {len(recent_trades)}):\n{json.dumps(recent_trades)}\n"
        prompt += "Use these outcomes to adapt your strategy. If recent trades are losing, become more conservative.\n"

    # --- Past trades for this specific coin ---
    if past_trades:
        prompt += f"\nPast closed trades for {symbol} (last {len(past_trades)}):\n"
        for t in past_trades:
            entry_price = t.get("price", 0.0)
            exit_price = t.get("exit_price", 0.0)
            amount = t.get("amount", 0.0)
            pnl = t.get("realized_pnl", 0.0)
            exit_reason = t.get("exit_reason", "unknown")
            hold_time = t.get("hold_time_seconds", None)
            strategy = t.get("strategy_type", "unknown")
            cost_basis = t.get("cost_basis", amount * entry_price)
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            hold_str = f"{hold_time:.0f}s" if hold_time is not None else "N/A"
            prompt += (
                f"- Entry: {entry_price:.4f}, Exit: {exit_price:.4f}, Amount: {amount:.6f}, "
                f"P&L: {pnl:+.4f} ({pnl_pct:+.2f}%), Reason: {exit_reason}, "
                f"Hold: {hold_str}, Strategy: {strategy}\n"
            )
        prompt += "Use these past outcomes to avoid repeating mistakes and to reinforce successful patterns.\n"

    # --- Aggregate sentiment summary ---
    if aggregate_sentiment:
        prompt += (
            f"\nAggregate news sentiment for {symbol}:\n"
            f"  Compound score: {aggregate_sentiment['avg_compound']:.2f}  (range -1 to +1)\n"
            f"  Positive articles: {aggregate_sentiment['positive']}\n"
            f"  Negative articles: {aggregate_sentiment['negative']}\n"
            f"  Neutral articles: {aggregate_sentiment['neutral']}\n"
            f"  Total articles: {aggregate_sentiment['total_articles']}\n"
        )
        prompt += (
            "Use this aggregate sentiment to adjust your confidence, position size, and risk parameters. "
            "Strong positive sentiment may justify higher confidence and larger positions; "
            "strong negative sentiment should make you more cautious or even skip the trade.\n"
        )

    # --- News section (detailed articles) ---
    news_section = ""
    if settings.NEWS_ENABLED:
        base = symbol.split("/")[0] if "/" in symbol else symbol
        articles = get_news_for_symbol(base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if articles:
            news_section = "Recent news articles for this coin:\n" + _format_news_for_prompt(articles)
    if news_section:
        prompt += f"\n{news_section}\n"
        prompt += "Consider the detailed news headlines above when setting your confidence, position size, and max hold time. "
        prompt += "If sentiment is very negative, reduce max hold time to limit exposure.\n"

    prompt += f"""
**Your primary objective is profit across short, medium, and long timeframes. Prioritize positions where you find the most profit potential, regardless of timeframe.** Use the ATR to set stop-loss and take-profit distances that respect the coin's volatility. Place the stop-loss below a recent swing low or support, and the take-profit near a resistance level or based on your own risk:reward assessment. You have full freedom to choose the stop distance and reward:risk ratio that you believe will maximise profitability while managing risk.

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
- EMA(9) crossing above EMA(21) is a bullish signal (golden cross); crossing below is bearish (death cross).
- Use these in combination with order book data to time entries.

**Additional technical indicators:**
- Stochastic Oscillator (%K, %D): values above 80 indicate overbought, below 20 oversold. Look for bullish cross (%K crossing above %D) near oversold for BUY signals, bearish cross near overbought for SELL.
- ADX: measures trend strength. ADX > 25 indicates a strong trend; ADX < 20 suggests a ranging market. Use +DI and -DI crossovers to determine trend direction (+DI > -DI = uptrend, -DI > +DI = downtrend).
- OBV: confirms price trends. Rising OBV with rising price confirms uptrend; divergence (price up, OBV down) warns of weakness.
- MFI: volume-weighted RSI. Overbought > 80, oversold < 20. Divergences can signal reversals.
- CCI: measures deviation from average. Values above +100 suggest overbought, below -100 oversold. Use for timing entries/exits.
- Williams %R: similar to Stochastic, ranges -100 to 0. Values above -20 overbought, below -80 oversold.

You MUST include the following risk parameters in the "parameters" object:
- stop_loss_pct (required unless using stop_loss_method="atr_multiple"), take_profit_pct, trailing_stop, trailing_stop_distance_pct, position_size_fraction, max_hold_time_seconds, cooldown_after_loss_seconds.
You may also include optional parameters: stop_loss_method, stop_loss_atr_multiple, trailing_stop_activation_pct, breakeven_activation_pct, lock_profit_activation_pct, lock_profit_level_pct, max_risk_per_trade_pct, entry_confidence_threshold. See the system prompt for details.
The bot will NOT use any default values. If you omit any required parameter, the trade will be skipped.

**Fee awareness:** You MUST account for trading fees when setting take-profit and trailing stop distances. Ensure that after deducting fees (both entry and exit), a take-profit or trailing stop exit results in a net profit. You must ensure that the take-profit percentage is high enough to cover both entry and exit fees and still yield a net profit. The bot will not enforce any minimum; it is entirely your responsibility.

You are trading spot only (no shorting). Only output SELL if you currently hold the coin.

**Execution Decision:**
You must decide whether to actually execute this trade right now. Output a boolean field `"execute"` in your JSON.
- Set `"execute": true` only if you are confident that entering this trade immediately will be profitable, considering all provided data (price, order book, balance, open positions, sentiment, technical indicators, fees, etc.).
- Set `"execute": false` if you believe the trade should be skipped – for example, if the risk/reward is insufficient, the market is too choppy, or there is no clear edge. The engine will honour this and not place the trade.
The `action`, `confidence`, and all other fields must still be provided as before, but the trade will only be executed when `execute` is true.

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
- This coin's past performance: {json.dumps(coin_perf)} (stop_loss_hits = number of times stop-loss was triggered; avg_hold_time_seconds = average trade duration)
- Overall equity curve: {json.dumps(equity)}
- Strategy performance summary: {json.dumps(strategy_perf)}

Use this data to decide whether to BUY, SELL, or HOLD. If the coin has a poor win rate or the overall equity curve is declining, be more conservative. Prefer strategies that have worked well historically.
"""
        perf_text += (
            "Use this performance data to calibrate your parameters:\n"
            "- If the coin has a low win rate or negative average P&L, reduce position_size_fraction, "
            "widen the stop (to avoid being stopped out prematurely), and shorten max_hold_time_seconds.\n"
            "- If the coin has a high win rate and positive average P&L, you may increase position size "
            "and use tighter stops to lock in profits.\n"
            "- If stop_loss_hits is high, consider using a wider stop (larger stop_loss_pct or higher ATR multiplier) "
            "or switching to a longer timeframe.\n"
            "- Use avg_hold_time_seconds to set a realistic max_hold_time_seconds – do not set it far below "
            "the average unless you have a specific reason.\n"
        )
        prompt += perf_text
    return prompt
