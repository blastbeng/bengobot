import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from src.config.settings import settings
try:
    from src.news.fetcher import fetch_news_for_symbol, get_aggregate_sentiment
except ImportError:
    fetch_news_for_symbol = None
    get_aggregate_sentiment = None

logger = logging.getLogger(__name__)


def compute_atr(candles: List[List], period: int = 14) -> float:
    """Compute Average True Range from OHLCV candles."""
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
    recent_tr = tr_values[-period:]
    return sum(recent_tr) / period


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


SYSTEM_PROMPT = """You are a professional cryptocurrency trading bot assistant. Your primary goal is to generate consistent short-term profit while preserving capital. You must avoid large drawdowns and only trade when there is a clear edge.

Key principles:
- Only trade coins with strong, confirmed short-term momentum and sufficient volatility to cover fees. Avoid low-volatility or choppy (sideways) markets entirely.
- You will receive raw OHLCV candle data. Compute your own technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.) from this data. Use them to time entries and exits. Require confirmation from at least two independent indicators before taking a trade.
- Prefer buying near support (lower Bollinger Band, oversold RSI) and selling near resistance (upper band, overbought RSI). Never chase a breakout without confirmation.
- Always set a stop-loss based on recent swing lows or ATR. Prefer ATR‑based stops because they adapt to current volatility. A stop distance of 1.5–2.5× ATR is usually appropriate. Never use a stop tighter than 1× ATR. Place the stop just below a recent swing low or support level, but ensure the distance is at least 1× ATR.
Example: If ATR=50 and current price=5000, a 2× ATR stop distance is 100, so stop_loss_pct = 100/5000 = 0.02 (2%). Place the stop at 4900. If the nearest swing low is at 4920, use that as the stop level (distance 80, which is 1.6× ATR, still acceptable). Never use a stop tighter than 1× ATR (50 points, 1%).
- Set a take-profit that offers at least a 2:1 reward-to-risk ratio (take_profit_pct >= 2 * stop_loss_pct). If you cannot achieve this, output HOLD.
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
- "position_size_fraction": a decimal between 0.1 and 1.0 (e.g., 0.5 for 50% of budget). Must be > 0 and ≤ 1.
- "max_hold_time_seconds": a positive integer number of seconds (e.g., 3600 for 1 hour). Must be > 0.

You may also include the following optional parameters to fine-tune risk management:

- "stop_loss_method": "fixed" (default) or "atr_multiple". If "atr_multiple", the stop distance is computed as stop_loss_atr_multiple × ATR, and "stop_loss_pct" is optional (if provided, it will be ignored). Use this to set a volatility-based stop.
- "stop_loss_atr_multiple": required if stop_loss_method is "atr_multiple". A positive float (e.g., 2.0 for 2× ATR). The stop distance will be (multiplier × ATR) / current_price.
- "trailing_stop_activation_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2%). The trailing stop will only start updating once the price has moved in your favor by at least this percentage from the entry price. If omitted, the trailing stop is active immediately.
- "max_risk_per_trade_pct": a decimal between 0 and 1.0 (e.g., 0.02 for 2% of portfolio). The position size will be limited so that the potential loss (entry - stop) does not exceed this fraction of your total portfolio value. If omitted, position sizing uses only position_size_fraction.
- "entry_confidence_threshold": a decimal between 0 and 1.0. Overrides the global minimum confidence for this specific trade. Use this to raise the bar for risky trades or lower it for high-conviction setups.

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
            if settings.NEWS_ENABLED and get_aggregate_sentiment is not None:
                agg = get_aggregate_sentiment(symbol)
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
    if settings.NEWS_ENABLED and fetch_news_for_symbol is not None:
        news_lines = []
        pairs_to_check = available_pairs[:20]
        for pair in pairs_to_check:
            articles = fetch_news_for_symbol(pair)
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

**Your primary objective is short-term profit.** Prioritize coins with strong recent momentum, high 24h volume, and significant price changes. Avoid coins that are flat or declining. You may keep current coins only if they still show short-term potential.

Select up to {max_coins} coins to trade. You MUST only select coins where the per-coin budget ({per_coin_budget:.2f} {base_currency}) is greater than or equal to the coin's min_trade_cost. Skip any coin that does not meet this requirement. Prefer coins with high volume and positive momentum. You may keep some current coins if they are still promising and meet the budget requirement, or replace them.

Each symbol can only appear once in your selection. Choose the single best timeframe for each coin based on the multi-timeframe OHLCV data.

Return a JSON array of objects, each with "symbol" and "timeframe" fields. The timeframe must be one of the available timeframes (e.g., "5m", "15m", "1h", "4h") that you believe is most suitable for trading that coin based on the multi-timeframe OHLCV data. Example: [{{"symbol": "BTC/USDT", "timeframe": "1h"}}, ...]"""
    if ohlcv_summary:
        prompt += f"\nMulti-timeframe OHLCV summary (price change %, high, low, volume):\n{json.dumps(ohlcv_summary, indent=2)}\n"
    if market_trend:
        prompt += f"\nOverall market trend ({market_trend['symbol']}): 24h change {market_trend.get('change_24h')}%, last price {market_trend.get('last')}\n"
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
        prompt += (
            "Use the ATR to set your stop-loss distance. A good stop distance is 1.5–2.5× ATR. "
            "Convert that distance into a percentage of the current price for the stop_loss_pct parameter. "
            "For example, if ATR=50 and price=5000, 2× ATR = 100, so stop_loss_pct = 100/5000 = 0.02 (2%). "
            "Never use a stop tighter than 1× ATR.\n"
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
    if drawdown_pct is not None:
        prompt += f"Current account drawdown: {drawdown_pct}%\n"
    if recent_trades:
        prompt += f"\nRecent closed trades (last {len(recent_trades)}):\n{json.dumps(recent_trades)}\n"
        prompt += "Use these outcomes to adapt your strategy. If recent trades are losing, become more conservative.\n"

    # --- News section ---
    news_section = ""
    if settings.NEWS_ENABLED and fetch_news_for_symbol is not None:
        articles = fetch_news_for_symbol(symbol)
        if articles:
            news_section = "Recent news for this coin:\n" + _format_news_for_prompt(articles)
    if news_section:
        prompt += f"\n{news_section}\n"

    prompt += f"""
**Your primary objective is short-term profit.** Use the ATR to set stop-loss and take-profit distances that respect the coin's volatility. Place the stop-loss below a recent swing low or support, and the take-profit near a resistance level or based on a risk:reward ratio of at least 1:2. **Crucially, your stop distance must be at least 1× ATR, and preferably 1.5–2.5× ATR, to avoid being stopped out by normal market noise.**

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

You MUST include the following risk parameters in the "parameters" object:
- stop_loss_pct (required unless using stop_loss_method="atr_multiple"), take_profit_pct, trailing_stop, trailing_stop_distance_pct, position_size_fraction, max_hold_time_seconds.
You may also include optional parameters: stop_loss_method, stop_loss_atr_multiple, trailing_stop_activation_pct, max_risk_per_trade_pct, entry_confidence_threshold. See the system prompt for details.
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
- This coin's past performance: {json.dumps(coin_perf)} (stop_loss_hits = number of times stop-loss was triggered; avg_hold_time_seconds = average trade duration)
- Overall equity curve: {json.dumps(equity)}
- Strategy performance summary: {json.dumps(strategy_perf)}

Use this data to decide whether to BUY, SELL, or HOLD. If the coin has a poor win rate or the overall equity curve is declining, be more conservative. Prefer strategies that have worked well historically.
"""
        prompt += perf_text
    return prompt
