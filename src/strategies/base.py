from dataclasses import dataclass
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
