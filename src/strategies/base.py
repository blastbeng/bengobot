from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Signal:
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    strategy_type: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None   # LLM-defined parameters
    risk_level: Optional[str] = None   # "low", "medium", "high"
    indicator_config: Optional[Dict[str, Any]] = None   # LLM-defined indicator parameters
    backtest_summary: Optional[str] = None   # LLM-provided backtest summary
    # --- dynamic trading parameters from LLM ---
    stop_loss: Optional[float] = None        # percentage below entry (e.g., 0.05 for 5%)
    take_profit: Optional[float] = None      # percentage above entry (e.g., 0.10 for 10%)
    position_size: Optional[float] = None    # fraction of per-coin budget (0.0 - 1.0)
    trailing_stop: Optional[bool] = False    # whether to use a trailing stop
    max_hold_minutes: Optional[int] = None   # maximum time to hold the position
    reason: Optional[str] = None             # LLM's explanation (for logging)


class Strategy:
    """Abstract base for trading strategies."""
    def generate_signal(self, market_data: Dict[str, Any]) -> Signal:
        raise NotImplementedError


class LLMStrategy(Strategy):
    """A strategy that wraps a pre-computed LLM decision."""
    def __init__(self, signal: Signal):
        self.signal = signal

    def generate_signal(self, market_data: Dict[str, Any]) -> Signal:
        # For now, returns the static decision; can be extended later.
        return self.signal
