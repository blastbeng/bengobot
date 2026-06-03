from .base import Signal
from typing import Dict, Any, Optional

MIN_CONFIDENCE = 0.5
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

    # Validate known parameters if provided
    params = signal.strategy_params or {}
    if "stop_loss_pct" in params:
        sl = params["stop_loss_pct"]
        if not (0 < sl < 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")
    if "take_profit_pct" in params:
        tp = params["take_profit_pct"]
        if not (0 < tp < 10.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid take_profit_pct")
    if params.get("trailing_stop") and "trailing_stop_distance_pct" not in params:
        return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop requires trailing_stop_distance_pct")
    if "trailing_stop_distance_pct" in params:
        tsd = params["trailing_stop_distance_pct"]
        if not (0 < tsd < 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid trailing_stop_distance_pct")
    if "position_size_fraction" in params:
        psf = params["position_size_fraction"]
        if not (0 < psf <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_fraction")

    return signal
