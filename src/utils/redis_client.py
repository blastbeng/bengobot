import redis
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

def get_redis_client() -> redis.Redis:
    """Return a Redis client configured from settings."""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_timeout=5,           # seconds – max time for any Redis command
        socket_connect_timeout=5,   # seconds – max time to establish connection
    )

def check_redis_connection() -> bool:
    """Test if Redis is reachable. Returns True if successful, False otherwise."""
    try:
        r = get_redis_client()
        r.ping()
        return True
    except redis.ConnectionError as e:
        logger.warning("Redis connection failed: %s", e)
        return False
