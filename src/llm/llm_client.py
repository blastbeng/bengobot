import logging

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


def get_ollama_response(prompt: str, system_prompt: str = "") -> str:
    """Send a prompt to the configured Ollama model and return the response text."""
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    if settings.OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e


def get_openai_response(prompt: str, system_prompt: str = "") -> str:
    """Send a prompt to the configured OpenAI-compatible API and return the response text."""
    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OPENAI_API_KEY}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": messages,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e


def get_llm_response(prompt: str, system_prompt: str = "") -> str:
    """Send a prompt to the configured LLM provider and return the response text."""
    if settings.LLM_PROVIDER == "openai":
        logger.debug("Using OpenAI-compatible LLM provider")
        return get_openai_response(prompt, system_prompt)
    else:
        logger.debug("Using Ollama LLM provider")
        return get_ollama_response(prompt, system_prompt)
