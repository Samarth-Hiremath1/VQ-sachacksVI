import redis
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Create Redis connection (lazy initialization)
redis_client = None

def get_redis():
    """Get Redis client instance"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return redis_client


def check_redis_health():
    """Check if Redis connection is healthy"""
    try:
        client = get_redis()
        client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False