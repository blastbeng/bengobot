import hashlib
import json
from src.llm.ollama_client import get_ollama_response
from src.utils.redis_client import get_redis_client

def get_cached_ollama_response(prompt: str, system_prompt: str = "", ttl: int = 300) -> str:
    """
    Get an LLM response, using Redis cache to avoid duplicate calls.
    Cache key is based on the prompt and system prompt.
    ttl: time-to-live in seconds (default 5 minutes).
    """
    redis_client = get_redis_client()
    # Create a deterministic cache key
    key_data = json.dumps({"prompt": prompt, "system": system_prompt}, sort_keys=True)
    cache_key = f"llm:{hashlib.sha256(key_data.encode()).hexdigest()}"

    # Try to get from cache
    cached = redis_client.get(cache_key)
    if cached:
        return cached

    # Not cached, call LLM
    response = get_ollama_response(prompt, system_prompt)

    # Store in cache with TTL
    redis_client.setex(cache_key, ttl, response)
    return response
