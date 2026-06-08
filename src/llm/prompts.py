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


def compute_vwap(candles: List[List]) -> Optional[float]:
    """Compute Volume Weighted Average Price from OHLCV candles.

    Uses typical price = (high + low + close) / 3.
    Returns None if no volume data.
    """
    if not candles:
        return None
    total_volume = 0.0
    total_value = 0.0
    for c in candles:
        high, low, close, volume = c[2], c[3], c[4], c[5]
        typical_price = (high + low + close) / 3.0
        total_value += typical_price * volume
        total_volume += volume
    if total_volume == 0:
        return None
    return total_value / total_volume


def compute_ichimoku(
    highs: List[float], lows: List[float], closes: List[float],
    tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52,
) -> Optional[Dict[str, Optional[float]]]:
    """Compute Ichimoku Cloud components.

    Returns dict with tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
    chikou_span, cloud_top, cloud_bottom. Returns None if insufficient data.
    """
    if len(closes) < senkou_b_period:
        return None

    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 over tenkan_period
    tenkan_high = max(highs[-tenkan_period:])
    tenkan_low = min(lows[-tenkan_period:])
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line): (highest high + lowest low) / 2 over kijun_period
    kijun_high = max(highs[-kijun_period:])
    kijun_low = min(lows[-kijun_period:])
    kijun_sen = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2

    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 over senkou_b_period
    senkou_b_high = max(highs[-senkou_b_period:])
    senkou_b_low = min(lows[-senkou_b_period:])
    senkou_span_b = (senkou_b_high + senkou_b_low) / 2

    # Chikou Span (Lagging Span): current close
    chikou_span = closes[-1]

    # Cloud boundaries
    cloud_top = max(senkou_span_a, senkou_span_b)
    cloud_bottom = min(senkou_span_a, senkou_span_b)

    return {
        "tenkan_sen": round(tenkan_sen, 8),
        "kijun_sen": round(kijun_sen, 8),
        "senkou_span_a": round(senkou_span_a, 8),
        "senkou_span_b": round(senkou_span_b, 8),
        "chikou_span": round(chikou_span, 8),
        "cloud_top": round(cloud_top, 8),
        "cloud_bottom": round(cloud_bottom, 8),
    }


def compute_parabolic_sar(
    highs: List[float], lows: List[float],
    acceleration: float = 0.02, max_acceleration: float = 0.2
) -> Optional[float]:
    """Compute Parabolic SAR (latest value).

    Uses the standard Wilder's method. Returns the current SAR value,
    or None if insufficient data.
    """
    if len(highs) < 2:
        return None

    # Initialise
    sar = lows[0]  # start with first low (assume uptrend)
    ep = highs[0]  # extreme point
    af = acceleration
    trend = 1  # 1 = uptrend, -1 = downtrend

    for i in range(1, len(highs)):
        prev_sar = sar
        prev_ep = ep
        prev_af = af
        prev_trend = trend

        # Update SAR
        sar = prev_sar + prev_af * (prev_ep - prev_sar)

        # Ensure SAR is below the low of the prior two bars in uptrend
        if trend == 1:
            if i >= 2:
                sar = min(sar, lows[i-1], lows[i-2])
            else:
                sar = min(sar, lows[i-1])
        else:
            if i >= 2:
                sar = max(sar, highs[i-1], highs[i-2])
            else:
                sar = max(sar, highs[i-1])

        # Check for reversal
        if trend == 1:
            if lows[i] < sar:
                trend = -1
                sar = ep  # new SAR is the previous extreme point
                ep = lows[i]
                af = acceleration
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + acceleration, max_acceleration)
        else:
            if highs[i] > sar:
                trend = 1
                sar = ep
                ep = highs[i]
                af = acceleration
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + acceleration, max_acceleration)

    return round(sar, 8)


def compute_keltner_channels(
    closes: List[float], highs: List[float], lows: List[float],
    ema_period: int = 20, atr_mult: float = 2.0, atr_period: int = 10
) -> Optional[Dict[str, float]]:
    """Compute Keltner Channels (upper, middle, lower).

    Middle line = EMA of closes.
    Upper/Lower = middle ± atr_mult * ATR.
    Returns dict with 'upper', 'middle', 'lower', or None if insufficient data.
    """
    if len(closes) < max(ema_period, atr_period + 1):
        return None

    # Middle line: EMA of closes
    ema_vals = compute_ema(closes, ema_period)
    if not ema_vals:
        return None
    middle = ema_vals[-1]

    # ATR from the provided highs, lows, closes
    tr_values = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_values.append(tr)
    if len(tr_values) < atr_period:
        return None
    atr = sum(tr_values[:atr_period]) / atr_period
    for i in range(atr_period, len(tr_values)):
        atr = (atr * (atr_period - 1) + tr_values[i]) / atr_period

    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr

    return {
        "upper": round(upper, 8),
        "middle": round(middle, 8),
        "lower": round(lower, 8),
    }


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
- Set a stop-loss based on recent swing lows, support levels, or ATR. Use the ATR to gauge volatility and choose a stop distance that gives the trade enough room to breathe while limiting risk. You decide the appropriate multiplier and reward:risk ratio based on current market conditions, volatility, and your confidence. You have full freedom to choose the stop distance. Use the ATR to gauge volatility and set a stop that gives the trade enough room while limiting risk. There is no hardcoded minimum – you decide what is appropriate.
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
- If the daily realized P&L is deeply negative or market conditions are poor, you may select 0 coins in the coin selection step. This will pause trading until the next evaluation cycle.

You may also set a daily profit target by including the optional field `"daily_profit_target_pct"` in your coin selection JSON. This is a decimal between 0 and 1.0 (e.g., 0.02 for 2%). If the day's realized P&L reaches or exceeds this percentage of the initial balance, the bot will pause trading until the next day. Use this to lock in profits and avoid overtrading after a successful day.

You may also set a global coin re-evaluation interval by including the optional field `"coin_revaluation_interval_seconds"` in your coin selection JSON. This controls how often the bot re-evaluates the entire coin list. Set a shorter interval (e.g., 120-300s) for fast scalping, or a longer interval (e.g., 900-1800s) for slower markets. Minimum 60 seconds. If omitted, the previous value (or default 900s) is kept.

You will receive recent news headlines with sentiment scores for each coin. **Sentiment is a primary factor in coin selection.** Use this information to gauge market sentiment and potential catalysts. Prefer coins with strong positive sentiment; avoid coins with negative sentiment unless technicals are exceptionally bullish.
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
- ichimoku_tenkan (int, default 9)
- ichimoku_kijun (int, default 26)
- ichimoku_senkou_b (int, default 52)

You may optionally include a "backtest_summary" field (string) when historical OHLCV data is provided. This should be a concise summary of your backtest analysis, e.g., "Simulated 5 trades over 30 days: 3 wins, 2 losses, net +2.3%". Include it only if you performed a backtest.

When asked to select coins, return a JSON array of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"]). Choose coins that are likely to deliver short-term profit based on recent price action, volume, and volatility. Prefer coins with high liquidity and clear short-term trends.

When selecting coins, you will see a "scalping suitability score" (0-1) for each candidate, along with spread and depth metrics. Coins with very low spread (<0.1%) and high depth are ideal for scalping tiny percentages (e.g., 0.1-0.5% take-profit). Use this data to pick coins where you can reliably capture small gains.

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
- "trailing_take_profit": an optional boolean (default false). If true, the take‑profit price will trail the current price upward by a fixed percentage (`trailing_take_profit_distance_pct`). The take‑profit never moves down. This allows you to capture more profit in trending moves while still scalping small percentages.
- "trailing_take_profit_distance_pct": required if `trailing_take_profit` is true. A decimal between 0.001 and 0.1 (e.g., 0.002 for 0.2%). The take‑profit will be set to `current_price * (1 + trailing_take_profit_distance_pct)` whenever the price rises, but it will never decrease.
- "breakeven_activation_pct": an optional decimal between 0 and 1.0 (e.g., 0.005 for 0.5%). If set, once the price rises by this percentage above your entry, the stop‑loss will be moved to the exact break‑even price (covering the exit fee). This locks in a risk‑free trade. Use this for scalping or when you want to protect a small gain.
- "lock_profit_activation_pct": an optional decimal between 0 and 1.0 (e.g., 0.005 for 0.5%). If set, once the price rises by this percentage above your entry, the stop‑loss will be moved to a guaranteed profit level (see lock_profit_level_pct). Use this to scalp small gains.
- "lock_profit_level_pct": required if lock_profit_activation_pct is set. A decimal between 0 and lock_profit_activation_pct (e.g., 0.003 for 0.3%). The new stop‑loss level will be set to entry_price * (1 + lock_profit_level_pct). This locks in a minimum profit even if the price reverses.
- "partial_take_profit_pct": an optional decimal between 0 and 1.0 (e.g., 0.003 for 0.3%). If set, the bot will sell a fraction of the position when the price rises by this percentage above the entry. Use this to scalp a quick small profit while holding the rest for a larger move.
- "partial_take_profit_fraction": required if partial_take_profit_pct is set. A decimal between 0 and 1.0 (e.g., 0.5 for 50%). The fraction of the current position to sell at the partial take‑profit level.
- "partial_take_profit_levels": an optional array of objects, each with:
    - "take_profit_pct": a decimal between 0 and 1.0 (e.g., 0.002 for 0.2%). The price increase from entry that triggers this partial sale.
    - "fraction": a decimal between 0 and 1.0 (e.g., 0.25 for 25%). The fraction of the **original** position to sell at this level.
    - "min_depth": (optional) a positive number in base currency. If set, the bot will check that the cumulative ask volume from the current mid price up to the take‑profit price is at least this value before executing the partial sale. If the depth is insufficient, the level is skipped (not triggered) and the bot will re‑evaluate on the next cycle.
    - "max_time_seconds": (optional) a positive integer. If the position has been open longer than this many seconds and this level has not yet triggered, the level is cancelled (marked as triggered). Use this to abandon a scalp target that hasn't been reached in time.
  Levels must be sorted by increasing take_profit_pct. The sum of all fractions must be ≤ 1.0. Each level is triggered only once. If this array is provided, the single `partial_take_profit_pct` and `partial_take_profit_fraction` are ignored. Use this to scale out of a position gradually, locking in profits at multiple small targets.
- "max_risk_per_trade_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2% of portfolio). The position size will be limited so that the potential loss (entry - stop) does not exceed this fraction of your total portfolio value. If omitted, position sizing uses only position_size_fraction.
- "min_profit_per_trade": an optional non-negative number (in quote currency, e.g., 0.5 for 0.5 USDT). If set, the bot will skip the trade if the expected gross profit (position size × take_profit_pct) is below this value. Use this to avoid trades that would yield only a negligible gain.
- "min_risk_reward_ratio": an optional positive number (e.g., 1.5). If set, the validator will reject the trade unless take_profit_pct / stop_loss_pct >= this value. Use this to enforce a minimum reward for the risk you are taking.
- "max_spread_pct": an optional positive number (e.g., 0.5 for 0.5%). If set, the bot will skip the trade if the current bid‑ask spread (as a percentage of mid price) exceeds this value. Use this to avoid illiquid coins. **Do not set this below 0.01 (0.01%) – values below this threshold are unrealistically strict and will be rejected.**
- "min_depth_at_take_profit": an optional positive number (in base currency, e.g., 0.5 for 0.5 BTC). If set, the bot will check the cumulative ask volume from the current mid price up to the take‑profit price. If that volume is less than this value, the trade will be skipped because the take‑profit may not fill without moving the price. Use this to ensure your scalp targets are reachable.
- "max_slippage_pct": an optional positive number (e.g., 0.1 for 0.1%). If set, the bot will compute the expected average fill price for a market buy order of the intended size by walking the order book. If the average fill price exceeds the best ask by more than this percentage, the trade will be skipped. Use this to avoid excessive slippage on illiquid coins, which is essential for scalping very small percentages.
- "max_unrealized_loss_pct": an optional decimal between 0 and 1.0 (e.g., 0.002 for 0.2%). If set, the bot will monitor the unrealized loss of the position. If the current price falls below `entry_price * (1 - max_unrealized_loss_pct)`, the position will be closed immediately, regardless of the stop‑loss. Use this as a soft stop to cut losses quickly when scalping tiny percentages. Must be less than `stop_loss_pct`.
- "min_confidence": an optional decimal between 0.0 and 1.0 (e.g., 0.6). If set, the bot will skip the trade if your confidence is below this threshold. Use this to enforce a minimum conviction level.

You will also receive a summary of the most recent individual trades (last 20). Use this to gauge very short‑term momentum and whether the market is active enough for scalping. A high number of small trades with balanced buy/sell pressure and a tight price range suggests a liquid market suitable for capturing tiny percentages.
- "news_sentiment_exit_threshold": an optional float between -1.0 and 1.0 (e.g., -0.5). If set, the bot will monitor the aggregate news sentiment for this coin. If the compound score drops below this threshold while the position is open, the position will be closed immediately. Use this to exit on strongly negative news.
- "strategy_interval_seconds": an optional positive integer (e.g., 60, 120, 300). If set, the bot will re‑evaluate the strategy for this coin every N seconds instead of the default interval. Use shorter intervals (60‑120s) for scalping very small percentages, and longer intervals (300‑600s) for swing trades. If omitted, the global default applies.

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
    daily_pnl: Optional[float] = None,
    coin_scores: Optional[Dict[str, float]] = None,
    coin_spreads: Optional[Dict[str, float]] = None,
    coin_depths: Optional[Dict[str, float]] = None,
    historical_ohlcv_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    fear_greed_index: Optional[Dict[str, Any]] = None,
    relative_strength_btc: Optional[Dict[str, Dict[str, Any]]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    sentiment_trend: Optional[Dict[str, Optional[float]]] = None,
    volume_trends: Optional[Dict[str, Optional[float]]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
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

Select between 0 and {max_coins} coins to trade. If market conditions are extremely unfavorable (e.g., high losses, poor momentum, negative sentiment), you may select 0 coins to pause trading until the next evaluation. You decide the exact number based on how many high‑quality opportunities you see. If market conditions are poor, you may choose fewer coins (even 0 or 1) to concentrate capital on the best setup. If many strong setups exist, you may select up to {max_coins}. You MUST only select coins where the per-coin budget ({per_coin_budget:.2f} {base_currency}) is greater than or equal to the coin's min_trade_cost. Skip any coin that does not meet this requirement. Prefer coins with high volume and positive momentum. You may keep some current coins if they are still promising and meet the budget requirement, or replace them.

**Use the historical performance data to guide your selection.** Prefer coins that have a positive average P&L and a win rate above 50% in recent trades. Avoid coins that have a string of losses or a negative average P&L, unless there is a strong technical or news‑driven reason to include them.

Each symbol can only appear once in your selection. Choose the single best timeframe for each coin based on the multi-timeframe OHLCV data.

Return a JSON object with two fields:
- "coins": a JSON array of objects, each with "symbol" and "timeframe" (the timeframe must be one of the available timeframes, e.g., "5m", "15m", "1h", "4h").
- "max_coins": an integer between 0 and {max_coins} indicating how many coins you actually want to trade. Set to 0 to pause trading. This must equal the length of the "coins" array.

You may optionally include "coin_revaluation_interval_seconds" (integer >= 60) to change how often the bot re-evaluates the coin list.

You may optionally include "daily_profit_target_pct" (float between 0 and 1) to set a daily profit target. If reached, trading pauses until the next day.

Example: {{"coins": [{{"symbol": "BTC/USDT", "timeframe": "1h"}}, {{"symbol": "ETH/USDT", "timeframe": "15m"}}], "max_coins": 2, "coin_revaluation_interval_seconds": 300, "daily_profit_target_pct": 0.02}}"""
    if coin_scores:
        prompt += "\nScalping suitability scores (0-1, higher = better for quick small profits):\n"
        for sym in available_pairs[:50]:
            if sym in coin_scores:
                prompt += f"  {sym}: {coin_scores[sym]:.3f}\n"
        prompt += (
            "Prioritise coins with higher scores, but use your own judgement. "
            "The score combines volume, volatility, spread, and momentum.\n"
        )
    if coin_spreads or coin_depths:
        prompt += "\nOrder book metrics for top coins (lower spread & higher depth = better for scalping):\n"
        for sym in available_pairs[:50]:
            parts = []
            if sym in coin_spreads:
                parts.append(f"spread={coin_spreads[sym]:.3f}%")
            if sym in coin_depths:
                parts.append(f"depth={coin_depths[sym]:.2f}")
            if parts:
                prompt += f"  {sym}: {', '.join(parts)}\n"
        prompt += "Prefer coins with spread < 0.2% and high depth for scalping very small percentages.\n"
    if ohlcv_summary:
        prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if correlation_matrix:
        # Trim to only include pairs that appear in the candidate list
        trimmed = {}
        for sym_a, row in correlation_matrix.items():
            trimmed[sym_a] = {sym_b: v for sym_b, v in row.items()}
        prompt += (
            "\nPairwise correlation matrix (Pearson correlation of daily returns, range -1 to +1):\n"
            f"{json.dumps(trimmed, indent=2)}\n"
            "Use this to diversify your selection. Coins with correlation > 0.7 move very similarly – "
            "avoid selecting too many highly correlated coins, as they concentrate risk. "
            "Prefer coins with low or negative correlation to your existing selections to spread risk.\n"
        )
    if historical_ohlcv_summary:
        prompt += (
            "\nHistorical OHLCV summary from database (up to 30 days, price change %, high, low, volume, candle count):\n"
            f"{json.dumps(historical_ohlcv_summary, indent=2)}\n"
            "Use this longer-term data to assess sustained trends and avoid coins in prolonged decline. "
            "Prefer coins with consistent upward momentum over the full period.\n"
        )
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
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    lines.append(f"    Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
            prompt += "\n".join(lines) + "\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): 24h change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
    if fear_greed_index:
        prompt += (
            f"\nCrypto Fear & Greed Index: {fear_greed_index['value']} "
            f"({fear_greed_index['classification']})\n"
            "This index reflects overall market sentiment (0 = Extreme Fear, 100 = Extreme Greed). "
            "Use it to gauge the general mood: extreme fear may present buying opportunities, "
            "extreme greed may signal a market top. Adjust your coin selection and risk parameters accordingly.\n"
        )
    if session_info:
        prompt += (
            f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
            "Use this to gauge typical market activity: Asian session often has lower volatility, "
            "European and US sessions have higher volume and volatility. Adjust your coin selection "
            "and risk parameters accordingly.\n"
        )
    if relative_strength_btc:
        prompt += "\nRelative strength vs BTC (ratio = coin_price / btc_price; relative_24h_pct = outperformance vs BTC over 24h):\n"
        for sym, data in relative_strength_btc.items():
            rel_str = f"{data['relative_24h_pct']:+.2f}%" if data['relative_24h_pct'] is not None else "N/A"
            prompt += f"  {sym}: ratio={data['ratio']:.8f}, 24h rel={rel_str}\n"
        prompt += (
            "A rising ratio means the coin is outperforming BTC, which is a bullish signal. "
            "A falling ratio means it's underperforming. Use this to identify coins with strong relative momentum. "
            "Prefer coins with positive relative 24h performance, but use your own judgement.\n"
        )
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
    if sentiment_trend:
        prompt += "\nSentiment trend (change in compound score since last cycle):\n"
        for base, delta in sentiment_trend.items():
            if delta is not None:
                prompt += f"  {base}: {delta:+.4f}\n"
        prompt += (
            "A positive delta means sentiment is improving; a negative delta means it is deteriorating. "
            "Use this to gauge whether the narrative is strengthening or weakening. "
            "Improving sentiment may justify higher confidence; deteriorating sentiment may warrant caution.\n"
        )
    if volume_trends:
        prompt += "\nVolume trend (24h volume relative to recent average):\n"
        for sym in available_pairs[:50]:
            if sym in volume_trends and volume_trends[sym] is not None:
                prompt += f"  {sym}: {volume_trends[sym]:.2f}x\n"
        prompt += (
            "A ratio > 1.0 means current 24h volume is above the recent average; "
            "> 2.0 suggests a significant spike that often precedes large moves. "
            "Use this to identify coins with unusual activity. "
            "Prefer coins with elevated volume when looking for breakout or momentum trades; "
            "be cautious with low-volume coins as moves may lack conviction.\n"
        )
    if market_breadth:
        prompt += (
            f"\nMarket breadth: {market_breadth['positive_pct']}% of {market_breadth['total_count']} "
            f"candidate coins have a positive 24h change ({market_breadth['positive_count']} positive).\n"
            "High breadth (>70%) indicates broad market strength (risk-on); low breadth (<30%) indicates weakness (risk-off). "
            "Use this to gauge overall market participation and adjust your coin selection and risk parameters accordingly.\n"
        )
    if news_section:
        prompt += f"\n{news_section}\n"
    prompt += (
        "\n**Sentiment is a critical factor in coin selection.** "
        "Prefer coins with a positive aggregate sentiment (compound > 0.1). "
        "Avoid coins with strongly negative sentiment (compound < -0.2) unless there is overwhelming technical evidence. "
        "Use sentiment to gauge market hype and potential short‑term momentum.\n"
    )
    if performance:
        perf_text = f"""
Historical Performance Data:
Overall equity curve: {json.dumps(performance.get('equity_curve', {}))}
Per-coin performance (win rate, avg P&L, total trades): {json.dumps(performance.get('coin_performance', {}), indent=2)}
Per-strategy performance: {json.dumps(performance.get('strategy_performance', {}), indent=2)}

Use this historical data to select coins that have been profitable in the past, and to avoid coins with poor performance. Prefer strategies that have shown higher win rates and average P&L.
"""
        prompt += perf_text
        if daily_pnl is not None:
            prompt += f"Today's realized P&L: {daily_pnl:.4f} {base_currency}\n"
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
    depth_profile: Optional[Dict[str, Dict[str, float]]] = None,
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
    recent_trades_data: Optional[List[Dict[str, Any]]] = None,
    multi_tf_raw_candles: Optional[Dict[str, List[List]]] = None,
    multi_tf_indicators: Optional[Dict[str, Dict[str, Any]]] = None,
    scalping_feasibility_score: Optional[float] = None,
    fear_greed_index: Optional[Dict[str, Any]] = None,
    relative_strength_btc: Optional[Dict[str, Any]] = None,
    vwap: Optional[float] = None,
    vwap_multi_tf: Optional[Dict[str, float]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    sentiment_trend: Optional[float] = None,
    volume_trend: Optional[float] = None,
    ichimoku: Optional[Dict[str, Optional[float]]] = None,
    market_breadth: Optional[Dict[str, Any]] = None,
    depth_trend: Optional[float] = None,
    parabolic_sar: Optional[float] = None,
    keltner_channels: Optional[Dict[str, float]] = None,
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
    if fear_greed_index:
        prompt += (
            f"\nCrypto Fear & Greed Index: {fear_greed_index['value']} "
            f"({fear_greed_index['classification']})\n"
            "This index reflects overall market sentiment (0 = Extreme Fear, 100 = Extreme Greed). "
            "Use it to gauge the general mood: extreme fear may present buying opportunities, "
            "extreme greed may signal a market top. Adjust your coin selection and risk parameters accordingly.\n"
        )
    if relative_strength_btc:
        rel_str = f"{relative_strength_btc['relative_24h_pct']:+.2f}%" if relative_strength_btc.get('relative_24h_pct') is not None else "N/A"
        prompt += (
            f"\nRelative strength vs BTC: ratio={relative_strength_btc['ratio']:.8f}, "
            f"24h relative performance={rel_str}\n"
            "A positive relative performance means this coin is outperforming BTC, which may indicate strong momentum. "
            "Use this to adjust your confidence and position size.\n"
        )

    if vwap is not None:
        prompt += f"\nVWAP ({assigned_timeframe or 'current'}): {vwap:.6f}\n"
        prompt += (
            "VWAP (Volume Weighted Average Price) is the average price weighted by volume. "
            "It acts as a fair value benchmark: price above VWAP suggests bullish sentiment, "
            "price below VWAP suggests bearish sentiment. Use it as a dynamic support/resistance level. "
            "A break above VWAP with volume can confirm an uptrend; a rejection at VWAP may signal a reversal.\n"
        )
    if vwap_multi_tf:
        prompt += f"VWAP across timeframes: {json.dumps(vwap_multi_tf)}\n"
        prompt += (
            "Compare VWAP across timeframes: if the price is above VWAP on all timeframes, "
            "the trend is strongly bullish. Divergences (e.g., above on 5m but below on 1h) "
            "may indicate a short‑term bounce within a larger downtrend.\n"
        )
    if session_info:
        prompt += (
            f"\nCurrent UTC hour: {session_info['utc_hour']} ({session_info['session']} session)\n"
            "Use this to gauge typical market activity: Asian session often has lower volatility, "
            "European and US sessions have higher volume and volatility. Adjust your coin selection "
            "and risk parameters accordingly.\n"
        )

    # --- Volatility, order book imbalance, and position P&L context ---
    if atr is not None:
        prompt += f"ATR (14-period, {assigned_timeframe or 'default'}): {atr:.6f}\n"
        prompt += (
            "Use the ATR to set your stop-loss distance. Convert the chosen distance into a percentage "
            "of the current price for the stop_loss_pct parameter. You decide the appropriate multiplier "
            "based on current volatility and your risk assessment.\n"
            "You may set any stop distance you believe is appropriate. Use the ATR to inform your decision, but there is no enforced minimum.\n"
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
    if ichimoku is not None:
        prompt += (
            f"Ichimoku Cloud: Tenkan-sen={ichimoku['tenkan_sen']:.4f}, "
            f"Kijun-sen={ichimoku['kijun_sen']:.4f}, "
            f"Senkou Span A={ichimoku['senkou_span_a']:.4f}, "
            f"Senkou Span B={ichimoku['senkou_span_b']:.4f}, "
            f"Cloud: {ichimoku['cloud_bottom']:.4f} - {ichimoku['cloud_top']:.4f}\n"
        )
        prompt += (
            "Interpretation: Price above the cloud confirms an uptrend; below confirms a downtrend. "
            "Tenkan-sen crossing above Kijun-sen is a bullish signal; crossing below is bearish. "
            "The cloud (between Span A and Span B) acts as dynamic support/resistance – "
            "a thick cloud means strong support/resistance; a thin cloud means it can be easily broken. "
            "Chikou Span (current close) above past prices confirms bullish momentum.\n"
        )
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
    if depth_profile:
        prompt += "\nOrder book depth profile (cumulative volume at distance from mid):\n"
        for dist, vols in depth_profile.items():
            prompt += f"  {dist}: bid={vols['bid_volume']:.4f}, ask={vols['ask_volume']:.4f}\n"
        prompt += (
            "Use this depth profile to set take‑profit levels that are likely to be filled. "
            "If the ask volume at a certain distance is thin, a small take‑profit may be filled quickly. "
            "If it's thick, you may need a larger move or a smaller position.\n"
        )
    if recent_trades_data:
        # Summarise last 20 trades: count buys vs sells, average size, price range
        buys = [t for t in recent_trades_data if t.get('side') == 'buy']
        sells = [t for t in recent_trades_data if t.get('side') == 'sell']
        avg_buy_size = sum(t['amount'] for t in buys) / len(buys) if buys else 0
        avg_sell_size = sum(t['amount'] for t in sells) / len(sells) if sells else 0
        prices = [t['price'] for t in recent_trades_data]
        price_range = (min(prices), max(prices)) if prices else (0, 0)
        prompt += (
            f"\nRecent trade activity (last {len(recent_trades_data)} trades):\n"
            f"  Buys: {len(buys)}, avg size: {avg_buy_size:.4f}\n"
            f"  Sells: {len(sells)}, avg size: {avg_sell_size:.4f}\n"
            f"  Price range: {price_range[0]:.4f} - {price_range[1]:.4f}\n"
        )
        prompt += (
            "Use this to assess micro‑momentum and liquidity. "
            "A high frequency of small trades with tight spreads is ideal for scalping.\n"
        )
    if scalping_feasibility_score is not None:
        prompt += f"\nScalping feasibility score: {scalping_feasibility_score:.3f} (0-1, higher = better for very small take‑profits)\n"
        prompt += (
            "This score combines spread, order book depth at 0.1%, trade frequency, and volatility. "
            "A score above 0.7 suggests the coin is highly suitable for scalping tiny percentages (e.g., 0.1-0.5% take‑profit). "
            "Use this to decide whether to employ a scalping strategy and how tight to set your take‑profit and stop‑loss.\n"
        )
    if fee_rate is not None:
        prompt += f"Taker fee rate for this symbol: {fee_rate*100:.2f}%\n"
        prompt += (
            "You must set take_profit_pct high enough to cover round‑trip fees and the spread. "
            "The engine will not enforce any minimum – it trusts your calculation.\n"
        )
    if unrealized_pnl is not None and position_info:
        prompt += f"Current position unrealized P&L: {unrealized_pnl:.2f} {symbol.split('/')[1]}\n"
        prompt += f"Position details: entry price {position_info.get('price')}, amount {position_info.get('amount')}\n"

    # --- Multi-timeframe raw OHLCV and indicators ---
    if multi_tf_raw_candles:
        prompt += "\nMulti-timeframe raw OHLCV data (each candle: [timestamp, open, high, low, close, volume]):\n"
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_raw_candles:
                candles = multi_tf_raw_candles[tf]
                prompt += f"\n{tf} timeframe ({len(candles)} candles):\n{json.dumps(candles)}\n"
        prompt += (
            "Use the raw candles from all timeframes to assess short‑term momentum, support/resistance, "
            "and overall trend. The lower timeframes (5m, 15m) are ideal for timing scalping entries and exits; "
            "the higher timeframes (1h, 4h) show the larger trend.\n"
        )
    if multi_tf_indicators:
        prompt += "\nComputed technical indicators per timeframe:\n"
        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in multi_tf_indicators:
                ind = multi_tf_indicators[tf]
                lines = [f"[{tf}]"]
                if ind.get('rsi') is not None:
                    lines.append(f"  RSI={ind['rsi']:.2f}")
                if ind.get('macd') is not None:
                    lines.append(f"  MACD={ind['macd']:.4f} Signal={ind['macd_signal']:.4f} Hist={ind['macd_hist']:.4f}")
                if ind.get('bb_upper') is not None:
                    lines.append(f"  BB Upper={ind['bb_upper']:.4f} Middle={ind['bb_middle']:.4f} Lower={ind['bb_lower']:.4f}")
                if ind.get('ema_9') is not None:
                    lines.append(f"  EMA9={ind['ema_9']:.4f} EMA21={ind['ema_21']:.4f}")
                if ind.get('stochastic_k') is not None:
                    lines.append(f"  Stoch %K={ind['stochastic_k']:.2f} %D={ind['stochastic_d']:.2f}")
                if ind.get('adx') is not None:
                    lines.append(f"  ADX={ind['adx']:.2f} +DI={ind['plus_di']:.2f} -DI={ind['minus_di']:.2f}")
                if ind.get('obv') is not None:
                    lines.append(f"  OBV={ind['obv']:.2f}")
                if ind.get('mfi') is not None:
                    lines.append(f"  MFI={ind['mfi']:.2f}")
                if ind.get('cci') is not None:
                    lines.append(f"  CCI={ind['cci']:.2f}")
                if ind.get('williams_r') is not None:
                    lines.append(f"  Williams %R={ind['williams_r']:.2f}")
                if ind.get('ichimoku') is not None:
                    ich = ind['ichimoku']
                    lines.append(f"  Ichimoku: Tenkan={ich['tenkan_sen']:.4f} Kijun={ich['kijun_sen']:.4f} SpanA={ich['senkou_span_a']:.4f} SpanB={ich['senkou_span_b']:.4f} Cloud={ich['cloud_bottom']:.4f}-{ich['cloud_top']:.4f}")
                prompt += "\n".join(lines) + "\n"
        prompt += (
            "Use these indicators across timeframes to confirm signals. "
            "For scalping, focus on 5m/15m RSI, MACD, and Bollinger Bands for entry timing, "
            "while ensuring the 1h/4h trend supports the direction.\n"
        )
    elif raw_candles:
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
    if sentiment_trend is not None:
        prompt += f"\nSentiment trend (change in compound score since last cycle): {sentiment_trend:+.4f}\n"
        prompt += (
            "A positive delta means sentiment is improving; a negative delta means it is deteriorating. "
            "Use this to adjust your confidence and risk parameters: improving sentiment may justify a larger position, "
            "while deteriorating sentiment may warrant a smaller position or tighter stops.\n"
        )
    if volume_trend is not None:
        prompt += f"\nVolume trend: {volume_trend:.2f}x (current 24h volume relative to recent average)\n"
        prompt += (
            "A ratio > 1.0 means volume is above average; > 2.0 suggests a significant spike. "
            "Elevated volume confirms the strength of a price move and increases the reliability of technical signals. "
            "Low volume during a breakout may signal a fakeout – reduce position size or wait for confirmation. "
            "Use this to adjust your confidence and position size accordingly.\n"
        )
    if market_breadth:
        prompt += (
            f"\nMarket breadth: {market_breadth['positive_pct']}% of {market_breadth['total_count']} "
            f"candidate coins have a positive 24h change ({market_breadth['positive_count']} positive).\n"
            "High breadth (>70%) indicates broad market strength (risk-on); low breadth (<30%) indicates weakness (risk-off). "
            "Use this to gauge overall market participation and adjust your coin selection and risk parameters accordingly.\n"
        )
    if depth_trend is not None:
        prompt += f"\nOrder book depth trend (change in total depth within 1% of mid since last cycle): {depth_trend:+.4f}\n"
        prompt += (
            "A positive delta means depth is increasing (growing liquidity and conviction); "
            "a negative delta means depth is decreasing (thinning liquidity). "
            "Increasing depth supports larger positions and tighter stops; decreasing depth warrants caution.\n"
        )
    if parabolic_sar is not None:
        prompt += f"\nParabolic SAR: {parabolic_sar:.6f}\n"
        prompt += (
            "Parabolic SAR is a trailing stop/reversal indicator. "
            "When the price is above the SAR, the trend is up; when below, the trend is down. "
            "The SAR can be used as a dynamic stop‑loss level: place your stop just below the SAR in an uptrend, "
            "or just above in a downtrend. A flip of the SAR relative to price signals a potential trend reversal.\n"
        )
    if keltner_channels:
        prompt += (
            f"\nKeltner Channels (20 EMA, 2× ATR): "
            f"Upper={keltner_channels['upper']:.6f}, "
            f"Middle={keltner_channels['middle']:.6f}, "
            f"Lower={keltner_channels['lower']:.6f}\n"
        )
        prompt += (
            "Keltner Channels are volatility‑based envelopes. "
            "Price near the upper band suggests overbought conditions; near the lower band suggests oversold. "
            "A breakout above the upper band with expanding ATR signals strong momentum; "
            "a squeeze (bands narrowing) indicates low volatility and often precedes a large move. "
            "Use the middle line as dynamic support/resistance. "
            "Combine with other indicators to confirm entries and exits.\n"
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
- Ichimoku Cloud: provides trend direction, support/resistance, and momentum in one system. Price above the cloud = uptrend; below = downtrend; inside = ranging/uncertain. Tenkan-sen crossing above Kijun-sen is bullish (golden cross); crossing below is bearish (death cross). The cloud (between Senkou Span A and B) acts as dynamic support/resistance. A thick cloud means strong S/R; a thin cloud is easily broken. Chikou Span (current close) above past prices confirms bullish momentum; below confirms bearish.

You MUST include the following risk parameters in the "parameters" object:
- stop_loss_pct (required unless using stop_loss_method="atr_multiple"), take_profit_pct, trailing_stop, trailing_stop_distance_pct, position_size_fraction, max_hold_time_seconds, cooldown_after_loss_seconds.
You may also include optional parameters: stop_loss_method, stop_loss_atr_multiple, trailing_stop_activation_pct, trailing_take_profit, trailing_take_profit_distance_pct, breakeven_activation_pct, lock_profit_activation_pct, lock_profit_level_pct, partial_take_profit_pct, partial_take_profit_fraction, partial_take_profit_levels, max_risk_per_trade_pct, min_profit_per_trade, min_risk_reward_ratio, max_spread_pct, min_depth_at_take_profit, max_slippage_pct, max_unrealized_loss_pct, min_confidence, entry_confidence_threshold, news_sentiment_exit_threshold, strategy_interval_seconds. See the system prompt for details.
The bot will NOT use any default values. If you omit any required parameter, the trade will be skipped.

**Fee awareness:** You are solely responsible for ensuring that every trade is profitable after fees and spread. The bot provides you with the taker fee rate and the current spread. You must set take_profit_pct (and partial_take_profit_pct if used) high enough to cover both entry and exit fees plus the spread, and still leave a net profit. There is no engine‑side minimum – if you set a take‑profit that is too low, the trade will lose money. Use the formula: minimum take_profit_pct = 1/(1-fee)^2 - 1 + spread_decimal. Add a buffer for safety.

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
