import json
from .base import Signal, LLMStrategy


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
        strategy_params = None
        if isinstance(strategy, dict):
            strategy_type = strategy.get("type")
            strategy_params = strategy.get("parameters")   # can be None or a dict

        risk_level = data.get("risk_level")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        indicator_config = data.get("indicator_config")
        if not isinstance(indicator_config, dict):
            indicator_config = None

        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            risk_level=risk_level,
            indicator_config=indicator_config,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return Signal(action="HOLD", confidence=0.0, reasoning="Failed to parse LLM response")


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
