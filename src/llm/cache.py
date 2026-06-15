import hashlib
import json
import logging
from typing import Optional
from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

def get_cached_llm_response(
    prompt: str,
    system_prompt: str = "",
    ttl: int = 300,
    market_hash: str = None,
    model_type: str = "actuator",
) -> Optional[dict]:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    Returns a dict with keys: "response" (str), "provider" (str), "model" (str).
    If market_hash is provided, the cache key is based on that hash
    (representing the market snapshot). Otherwise, the key is based on
    the prompt and system prompt.
    ttl: time-to-live in seconds (default 5 minutes).
    model_type: "mind" for complex reasoning, "actuator" for fast time‑critical decisions.

    When the primary provider is "ollama" and the call fails, automatically
    falls back to the OpenAI-compatible endpoint (which can be OpenRouter)
    using the per-role settings, but only if the OpenAI API key or base URL
    for that role is configured.
    """
    redis_client = get_redis_client()

    # Determine effective provider and model for the primary choice
    if model_type == "mind":
        provider = settings.LLM_MIND_PROVIDER or settings.LLM_PROVIDER
    else:
        provider = settings.LLM_ACTUATOR_PROVIDER or settings.LLM_PROVIDER

    if provider == "openai":
        model = settings.OPENAI_MIND_MODEL if model_type == "mind" else settings.OPENAI_ACTUATOR_MODEL
        base_url = (settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL) if model_type == "mind" else (settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL)
        api_key = (settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY) if model_type == "mind" else (settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY)
    else:  # ollama
        model = settings.OLLAMA_MIND_MODEL if model_type == "mind" else settings.OLLAMA_ACTUATOR_MODEL
        base_url = (settings.OLLAMA_MIND_BASE_URL or settings.OLLAMA_BASE_URL) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_BASE_URL or settings.OLLAMA_BASE_URL)
        api_key = (settings.OLLAMA_MIND_API_KEY or settings.OLLAMA_API_KEY) if model_type == "mind" else (settings.OLLAMA_ACTUATOR_API_KEY or settings.OLLAMA_API_KEY)

    # Build cache key (unchanged logic)
    if market_hash:
        cache_key = f"llm:{model_type}:market:{market_hash}"
    else:
        key_data = json.dumps(
            {"prompt": prompt, "system": system_prompt, "model_type": model_type}, sort_keys=True
        )
        cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"

    # Try cache
    cached = redis_client.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            if isinstance(data, dict) and "response" in data:
                logger.debug("LLM cache hit for key %s", cache_key[:32])
                return data
        except (json.JSONDecodeError, TypeError):
            pass  # fall through to re-fetch

    # --- Primary call ---
    response_text = None
    used_provider = provider
    used_model = model

    try:
        if provider == "openai":
            from src.llm.llm_client import _get_openai_response
            response_text = _get_openai_response(prompt, system_prompt, model=model, base_url=base_url, api_key=api_key)
        else:
            from src.llm.llm_client import _get_ollama_response
            response_text = _get_ollama_response(prompt, system_prompt, model=model, base_url=base_url, api_key=api_key)
    except Exception as e:
        if provider == "ollama":
            # --- Fallback to OpenAI-compatible provider ---
            fallback_model = settings.OPENAI_MIND_MODEL if model_type == "mind" else settings.OPENAI_ACTUATOR_MODEL
            fallback_base_url = (settings.OPENAI_MIND_BASE_URL or settings.OPENAI_BASE_URL) if model_type == "mind" else (settings.OPENAI_ACTUATOR_BASE_URL or settings.OPENAI_BASE_URL)
            fallback_api_key = (settings.OPENAI_MIND_API_KEY or settings.OPENAI_API_KEY) if model_type == "mind" else (settings.OPENAI_ACTUATOR_API_KEY or settings.OPENAI_API_KEY)

            if fallback_api_key or fallback_base_url:
                logger.warning(
                    "Ollama call failed (%s). Falling back to OpenAI-compatible provider "
                    "for %s role (model=%s).", e, model_type, fallback_model
                )
                try:
                    from src.llm.llm_client import _get_openai_response
                    response_text = _get_openai_response(
                        prompt, system_prompt,
                        model=fallback_model,
                        base_url=fallback_base_url,
                        api_key=fallback_api_key,
                    )
                    used_provider = "openai"
                    used_model = fallback_model
                except Exception as fallback_e:
                    logger.error("OpenAI fallback also failed: %s", fallback_e)
                    raise
            else:
                logger.warning(
                    "Ollama call failed and no OpenAI fallback credentials configured. "
                    "Original error: %s", e
                )
                raise
        else:
            raise

    if response_text is None:
        logger.warning("LLM returned None response; not caching.")
        return None

    # Store in cache as JSON
    cache_data = json.dumps({
        "response": response_text,
        "provider": used_provider,
        "model": used_model,
    })
    redis_client.setex(cache_key, ttl, cache_data)
    logger.debug("LLM cache miss – stored response for key %s (provider=%s, model=%s)", cache_key[:32], used_provider, used_model)
    return {
        "response": response_text,
        "provider": used_provider,
        "model": used_model,
    }


def _stringify_keys(obj):
    """Recursively convert all dict keys to strings for JSON-safe sorting."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(item) for item in obj]
    return obj


def compute_market_hash(data: dict) -> str:
    """Return a SHA-256 hex digest of the JSON-serialised market data."""
    # sort_keys ensures deterministic output
    safe_data = _stringify_keys(data)
    serialized = json.dumps(safe_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
