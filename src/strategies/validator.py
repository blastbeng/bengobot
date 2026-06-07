from .base import Signal
from typing import Dict, Any, Optional

VALID_STRATEGY_TYPES = {"scalping", "momentum", "mean_reversion", "breakout"}


def validate_signal(
    signal: Signal,
    market_data: Optional[Dict[str, Any]] = None,
    fee_rate: Optional[float] = None,
    atr: Optional[float] = None,
    price: Optional[float] = None,
    spread_pct: Optional[float] = None,
) -> Signal:
    """
    Validate a trading signal.
    - If action is HOLD, return as-is.
    - Validate strategy_type and required risk parameters.
    - Enforce risk/reward and ATR-based stop rules.
    Confidence is NOT used to reject trades; it will be used later for position sizing.
    """
    if signal.action == "HOLD":
        return signal

    if signal.strategy_type and signal.strategy_type not in VALID_STRATEGY_TYPES:
        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid strategy type: {signal.strategy_type}")

    # Require risk parameters for BUY/SELL
    if signal.action in ("BUY", "SELL"):
        params = signal.strategy_params or {}
        # Determine stop-loss method (default "fixed")
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method not in ("fixed", "atr_multiple"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_method")

        if stop_method == "atr_multiple":
            # stop_loss_pct is optional; stop_loss_atr_multiple is required
            if "stop_loss_atr_multiple" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing stop_loss_atr_multiple for atr_multiple method")
            atr_mult = params["stop_loss_atr_multiple"]
            if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_atr_multiple")
            # We still allow stop_loss_pct if present, but it's not required
            sl = params.get("stop_loss_pct")
            if sl is not None and (not isinstance(sl, (int, float)) or not (0 < sl < 1.0)):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")
        else:  # "fixed"
            if "stop_loss_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing required parameter: stop_loss_pct")
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")

        # The rest of the required parameters remain unchanged
        required = ["take_profit_pct", "trailing_stop", "position_size_fraction", "max_hold_time_seconds"]
        for key in required:
            if key not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Missing required parameter: {key}")
        tp = params["take_profit_pct"]
        if not isinstance(tp, (int, float)) or not (0 < tp < 10.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid take_profit_pct")
        # Enforce minimum take-profit to cover fees AND spread
        if fee_rate is not None and fee_rate > 0:
            # Base minimum from fees: need (1+tp)*(1-fee)^2 > 1  => tp > 1/(1-f)^2 - 1
            min_tp_pct = (1.0 / ((1.0 - fee_rate) ** 2)) - 1.0
            # Add spread cost if available (spread_pct is in percent, e.g. 0.05 = 0.05%)
            if spread_pct is not None and spread_pct > 0:
                spread_decimal = spread_pct / 100.0
                min_tp_pct += spread_decimal
            # Add a small buffer (0.1%) to ensure net profit
            min_tp_pct += 0.001
            if tp <= min_tp_pct:
                return Signal(
                    action="HOLD",
                    confidence=0.0,
                    reasoning=f"take_profit_pct ({tp:.4%}) too low to cover fees+spread (min {min_tp_pct:.4%})"
                )
        # Enforce minimum stop distance based on ATR (if available)
        if atr is not None and price is not None and price > 0 and atr > 0:
            min_stop_pct = 0.5 * (atr / price)   # at least 0.5× ATR
            if stop_method == "atr_multiple":
                expected_sl_pct = params["stop_loss_atr_multiple"] * atr / price
                if expected_sl_pct < min_stop_pct:
                    return Signal(
                        action="HOLD",
                        confidence=0.0,
                        reasoning=f"ATR-based stop distance ({expected_sl_pct:.4%}) is too tight (min {min_stop_pct:.4%})"
                    )
            else:
                if sl < min_stop_pct:
                    return Signal(
                        action="HOLD",
                        confidence=0.0,
                        reasoning=f"stop_loss_pct ({sl:.4%}) is too tight relative to ATR (min {min_stop_pct:.4%})"
                    )
        trailing = params["trailing_stop"]
        if not isinstance(trailing, bool):
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop must be boolean")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            if tsd is None or not isinstance(tsd, (int, float)) or not (0 < tsd < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid or missing trailing_stop_distance_pct")
        psf = params["position_size_fraction"]
        if not isinstance(psf, (int, float)) or not (0 < psf <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_fraction")
        mht = params["max_hold_time_seconds"]
        if not isinstance(mht, (int, float)) or mht <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_hold_time_seconds")

        if "cooldown_after_loss_seconds" not in params:
            return Signal(action="HOLD", confidence=0.0, reasoning="Missing required parameter: cooldown_after_loss_seconds")
        cd = params["cooldown_after_loss_seconds"]
        if not isinstance(cd, (int, float)) or cd < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid cooldown_after_loss_seconds")

        # Optional new parameters
        if "trailing_stop_activation_pct" in params:
            tsa = params["trailing_stop_activation_pct"]
            if not isinstance(tsa, (int, float)) or not (0 <= tsa <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid trailing_stop_activation_pct")
        if "breakeven_activation_pct" in params:
            bap = params["breakeven_activation_pct"]
            if not isinstance(bap, (int, float)) or not (0 < bap <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid breakeven_activation_pct")
        if "lock_profit_activation_pct" in params:
            lpa = params["lock_profit_activation_pct"]
            if not isinstance(lpa, (int, float)) or not (0 < lpa <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid lock_profit_activation_pct")
            if "lock_profit_level_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing lock_profit_level_pct")
            lpl = params["lock_profit_level_pct"]
            if not isinstance(lpl, (int, float)) or not (0 < lpl < lpa):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid lock_profit_level_pct (must be < activation)")
        if "max_risk_per_trade_pct" in params:
            mrp = params["max_risk_per_trade_pct"]
            if not isinstance(mrp, (int, float)) or not (0 < mrp <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_risk_per_trade_pct")

        # Logical consistency checks (no hardcoded values)
        if sl is not None and tp <= sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct must be greater than stop_loss_pct")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            if tsd is not None and sl is not None and tsd >= sl:
                return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop_distance_pct must be less than stop_loss_pct")

    return signal
