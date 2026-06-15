import logging

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


def _get_ollama_response(prompt: str, system_prompt: str = "", model: str = None,
                         base_url: str = None, api_key: str = None) -> str:
    """Send a prompt to the configured Ollama model and return the response text."""
    url = f"{(base_url or settings.OLLAMA_BASE_URL).rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    effective_api_key = api_key or settings.OLLAMA_API_KEY
    if effective_api_key:
        headers["Authorization"] = f"Bearer {effective_api_key}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": settings.LLM_TEMPERATURE,
    }

    try:
        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e


def _get_openai_response(prompt: str, system_prompt: str = "", model: str = None,
                         base_url: str = None, api_key: str = None) -> str:
    """Send a prompt to the configured OpenAI-compatible API and return the response text."""
    url = f"{(base_url or settings.OPENAI_BASE_URL).rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    effective_api_key = api_key or settings.OPENAI_API_KEY
    if effective_api_key:
        headers["Authorization"] = f"Bearer {effective_api_key}"

    messages = []
    if system_prompt:
        msg = {"role": "system", "content": system_prompt}
        # Enable OpenAI prompt caching to reduce costs for repeated system messages
        msg["cache_control"] = {"type": "ephemeral"}
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or settings.OPENAI_MODEL,
        "messages": messages,
        "temperature": settings.LLM_TEMPERATURE,
    }

    try:
        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e


def get_llm_response(prompt: str, system_prompt: str = "", model_type: str = "actuator") -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Uses Redis caching with a 5-minute TTL (keyed by prompt + system prompt).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.

    Note: get_cached_llm_response now returns a dict with "response", "provider", "model".
    This function returns only the response text for backward compatibility.
    """
    from src.llm.cache import get_cached_llm_response  # local import to avoid circular dependency at module level

    result = get_cached_llm_response(prompt, system_prompt, ttl=300, model_type=model_type)
    if result is None:
        # This should not happen because the underlying raw call raises on failure,
        # but guard against unexpected None.
        raise RuntimeError("LLM returned an empty response")
    return result["response"]
