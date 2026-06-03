from .base import Signal
from typing import Dict, Any, Optional

MIN_CONFIDENCE = 0.65
VALID_STRATEGY_TYPES = {"scalping", "momentum", "mean_reversion", "breakout"}


def validate_signal(signal: Signal, market_data: Optional[Dict[str, Any]] = None) -> Signal:
    """
    Validate a trading signal.
    - If action is HOLD, return as-is.
    - If confidence < MIN_CONFIDENCE, return HOLD.
    - If strategy_type is set but not in the allowed set, return HOLD.
    - Validate known parameters inside strategy_params if provided.
    Otherwise return the original signal.
    """
    if signal.action == "HOLD":
        return signal

    if signal.confidence < MIN_CONFIDENCE:
        return Signal(action="HOLD", confidence=0.0, reasoning="Confidence too low")

    if signal.strategy_type and signal.strategy_type not in VALID_STRATEGY_TYPES:
        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid strategy type: {signal.strategy_type}")

    # Require risk parameters for BUY/SELL
    if signal.action in ("BUY", "SELL"):
        params = signal.strategy_params or {}
        required = ["stop_loss_pct", "take_profit_pct", "trailing_stop", "position_size_fraction", "max_hold_time_seconds"]
        for key in required:
            if key not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Missing required parameter: {key}")
        # Validate each
        sl = params["stop_loss_pct"]
        if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")
        tp = params["take_profit_pct"]
        if not isinstance(tp, (int, float)) or not (0 < tp < 10.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid take_profit_pct")
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

        # Logical consistency checks (no hardcoded values)
        if tp <= sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct must be greater than stop_loss_pct")
        if tp < 2 * sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct must be at least 2x stop_loss_pct")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            if tsd is not None and tsd >= sl:
                return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop_distance_pct must be less than stop_loss_pct")

    return signal
