import json
from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a professional cryptocurrency trading bot assistant. Your task is to analyze market data and provide trading decisions in strict JSON format. Do not include any text outside the JSON. Always output valid JSON.

When asked to select coins, return a JSON array of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"]). Choose coins that are likely to be profitable based on recent price action, volume, and volatility. Prefer coins with high liquidity and clear trends.

When asked to generate a strategy for a specific coin, return a JSON object with the following structure:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "reasoning": "short explanation",
  "strategy": {
    "type": "scalping" | "momentum" | "mean_reversion" | "breakout",
    "parameters": {
      // strategy-specific parameters
    }
  }
}
If action is BUY or SELL, include a strategy. If HOLD, strategy can be null.
"""

def build_coin_selection_prompt(
    available_pairs: List[str],
    current_coins: List[str],
    max_coins: int,
    base_currency: str,
    tickers: Dict[str, Any],
    base_balance: float,
    per_coin_budget: float,
    market_limits: Dict[str, Dict[str, Any]]
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
                "min_trade_cost": limits.get("min_cost"),      # in quote currency
                "min_trade_amount": limits.get("min_amount"),  # in base currency
            }

    prompt = f"""Current base currency: {base_currency}
Your available {base_currency} balance: {base_balance:.2f}
Maximum number of coins to trade: {max_coins}
Budget per coin (balance / max_coins): {per_coin_budget:.2f} {base_currency}
Currently traded coins: {json.dumps(current_coins)}

Available trading pairs with market data and minimum trade requirements:
{json.dumps(ticker_summary, indent=2)}

Select up to {max_coins} coins to trade. You MUST only select coins where the per-coin budget ({per_coin_budget:.2f} {base_currency}) is greater than or equal to the coin's min_trade_cost (if provided). If min_trade_cost is not available, use min_trade_amount multiplied by the last price as an estimate. Skip any coin that does not meet this requirement. Prefer coins with high volume and positive momentum. You may keep some current coins if they are still promising and meet the budget requirement, or replace them.

Return a JSON array of symbols."""
    return prompt

def build_strategy_prompt(
    symbol: str,
    ticker: Dict[str, Any],
    order_book: Dict[str, Any],
    balance: Dict[str, float],
    open_positions: List[Dict[str, Any]],
    per_coin_budget: float,
    max_coins: int
) -> str:
    """Build a prompt to generate a trading strategy for a specific coin."""
    prompt = f"""Symbol: {symbol}
Current ticker: {json.dumps(ticker)}
Order book (top 5 levels): {json.dumps(order_book)}
Current balances: {json.dumps(balance)}
Open positions: {json.dumps(open_positions)}
Per-coin budget (balance / max_coins): {per_coin_budget:.2f} {symbol.split('/')[1]}
Maximum coins to trade: {max_coins}

Based on the above, decide whether to BUY, SELL, or HOLD. Consider the per-coin budget: only BUY if the budget is sufficient to meet the minimum trade size and the position size is meaningful. If the budget is too small, prefer HOLD. Provide a strategy if action is BUY or SELL. Return a JSON object as specified."""
    return prompt
