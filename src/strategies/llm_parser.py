import json
import re
from .base import Signal, LLMStrategy


def parse_llm_response(response_text: str) -> Signal:
    """
    Parse the LLM's JSON response into a Signal.
    Supports JSON wrapped in ```json ... ``` code blocks or raw JSON.
    If parsing fails, returns a HOLD signal with zero confidence.
    """
    try:
        # Try to extract JSON from a markdown code block first
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(response_text)

        action = data.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        reasoning = data.get("reasoning", "")

        strategy = data.get("strategy")
        strategy_type = None
        strategy_params = None
        if isinstance(strategy, dict):
            strategy_type = strategy.get("type")
            strategy_params = strategy.get("parameters")

        risk_level = data.get("risk_level")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        indicator_config = data.get("indicator_config")
        if not isinstance(indicator_config, dict):
            indicator_config = None

        backtest_summary = data.get("backtest_summary")
        if not isinstance(backtest_summary, str):
            backtest_summary = None

        # --- dynamic trading parameters ---
        stop_loss = data.get("stop_loss")
        take_profit = data.get("take_profit")
        position_size = data.get("position_size")
        if position_size is not None:
            position_size = max(0.0, min(1.0, float(position_size)))
        trailing_stop = bool(data.get("trailing_stop", False))
        max_hold_minutes = data.get("max_hold_minutes")
        reason = data.get("reason", "")

        return Signal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            risk_level=risk_level,
            indicator_config=indicator_config,
            backtest_summary=backtest_summary,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            trailing_stop=trailing_stop,
            max_hold_minutes=max_hold_minutes,
            reason=reason,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return Signal(action="HOLD", confidence=0.0, reasoning="Failed to parse LLM response")


def create_strategy_from_llm(response_text: str) -> LLMStrategy:
    """
    Parse the LLM response and return an LLMStrategy instance.
    """
    signal = parse_llm_response(response_text)
    return LLMStrategy(signal)
