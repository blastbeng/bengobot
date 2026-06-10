import asyncio
import time
import logging
from functools import wraps
import ccxt

logger = logging.getLogger(__name__)

def retry_on_rate_limit(max_retries=3, base_delay=1.0):
    """
    Decorator that retries a function if it raises ccxt.RateLimitExceeded.
    Uses exponential backoff: delay = base_delay * (2 ** attempt).
    Works for both sync and async functions.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit exceeded in {func.__name__}, "
                            f"retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit exceeded in {func.__name__}, "
                            f"retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
