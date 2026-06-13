import math
from typing import List, Dict, Optional, Tuple, Any


def compute_atr(candles: List[List], period: int = 14) -> Optional[float]:
    """Compute Average True Range from OHLCV candles using Wilder's smoothing.

    Returns None if insufficient data.
    """
    if len(candles) < period + 1:
        return None
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        prev_close = candles[i - 1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    if len(tr_values) < period:
        return None
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


def compute_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Compute classic Pivot Points from the previous period's high, low, close.

    Returns dict with 'pivot', 'r1', 'r2', 's1', 's2'.
    """
    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {
        "pivot": round(pivot, 8),
        "r1": round(r1, 8),
        "r2": round(r2, 8),
        "s1": round(s1, 8),
        "s2": round(s2, 8),
    }


def compute_donchian_channels(
    highs: List[float], lows: List[float], period: int = 20
) -> Optional[Dict[str, float]]:
    """Compute Donchian Channels (upper, middle, lower).

    Upper = highest high over N periods
    Lower = lowest low over N periods
    Middle = (upper + lower) / 2

    Returns dict with 'upper', 'middle', 'lower', or None if insufficient data.
    """
    if len(highs) < period or len(lows) < period:
        return None

    upper = max(highs[-period:])
    lower = min(lows[-period:])
    middle = (upper + lower) / 2.0

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


def compute_all_indicators(
    candles: List[List],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute all technical indicators for a list of OHLCV candles.
    candles: list of [timestamp, open, high, low, close, volume]
    config: optional dict with custom periods (e.g., {'rsi_period': 14, ...})
    Returns a dict with keys like 'atr', 'rsi', 'macd', etc.
    Missing indicators are set to None.
    """
    if config is None:
        config = {}

    ind = {}
    if len(candles) < 2:
        return ind

    # ATR
    ind['atr'] = compute_atr(candles)

    if len(candles) >= 26:
        closes = [c[4] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        volumes = [c[5] for c in candles]

        rsi_period = config.get('rsi_period', 14)
        macd_fast = config.get('macd_fast', 12)
        macd_slow = config.get('macd_slow', 26)
        macd_signal_period = config.get('macd_signal', 9)
        bb_period = config.get('bb_period', 20)
        bb_std = config.get('bb_std', 2.0)
        ema_fast = config.get('ema_fast', 9)
        ema_slow = config.get('ema_slow', 21)
        stoch_k_period = config.get('stoch_k_period', 14)
        stoch_d_period = config.get('stoch_d_period', 3)
        adx_period = config.get('adx_period', 14)
        mfi_period = config.get('mfi_period', 14)
        cci_period = config.get('cci_period', 20)
        willr_period = config.get('willr_period', 14)
        ichimoku_tenkan = config.get('ichimoku_tenkan', 9)
        ichimoku_kijun = config.get('ichimoku_kijun', 26)
        ichimoku_senkou_b = config.get('ichimoku_senkou_b', 52)
        donchian_period = config.get('donchian_period', 20)

        ind['rsi'] = compute_rsi(closes, period=rsi_period)
        macd_val, macd_sig, macd_hist = compute_macd(closes, fast=macd_fast, slow=macd_slow, signal=macd_signal_period)
        ind['macd'] = macd_val
        ind['macd_signal'] = macd_sig
        ind['macd_hist'] = macd_hist
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes, period=bb_period, std_dev=bb_std)
        ind['bb_upper'] = bb_upper
        ind['bb_middle'] = bb_middle
        ind['bb_lower'] = bb_lower
        ema_9_list = compute_ema(closes, ema_fast)
        ema_21_list = compute_ema(closes, ema_slow)
        ind['ema_9'] = ema_9_list[-1] if ema_9_list else None
        ind['ema_21'] = ema_21_list[-1] if ema_21_list else None
        stoch_k, stoch_d = compute_stochastic(highs, lows, closes, period=stoch_k_period, smooth_k=stoch_d_period)
        ind['stochastic_k'] = stoch_k
        ind['stochastic_d'] = stoch_d
        adx_val, plus_di, minus_di = compute_adx(highs, lows, closes, period=adx_period)
        ind['adx'] = adx_val
        ind['plus_di'] = plus_di
        ind['minus_di'] = minus_di
        ind['obv'] = compute_obv(closes, volumes)
        ind['mfi'] = compute_mfi(highs, lows, closes, volumes, period=mfi_period)
        ind['cci'] = compute_cci(highs, lows, closes, period=cci_period)
        ind['williams_r'] = compute_williams_r(highs, lows, closes, period=willr_period)
        ind['ichimoku'] = compute_ichimoku(highs, lows, closes, tenkan_period=ichimoku_tenkan, kijun_period=ichimoku_kijun, senkou_b_period=ichimoku_senkou_b)
        ind['donchian_channels'] = compute_donchian_channels(highs, lows, period=donchian_period)

    return ind
