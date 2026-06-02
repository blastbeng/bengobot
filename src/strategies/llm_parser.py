import json
from .base import Signal


def parse_llm_response(response_text: str) -> Signal:
    """
    Parse the LLM's JSON response into a Signal.
    If parsing fails, returns a HOLD signal with zero confidence.
    """
    try:
        data = json.loads(response_text)
        action = data.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "")
        strategy = data.get("strategy")
        strategy_type = None
        parameters = None
        if isinstance(strategy, dict):
            strategy_type = strategy.get("type")
            parameters = strategy.get("parameters")
        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=strategy_type,
            parameters=parameters,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return Signal(action="HOLD", confidence=0.0, reasoning="Failed to parse LLM response")
