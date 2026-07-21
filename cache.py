import redis
import json
from config import REDIS_URL

TTL = 600  # 10 minutes


def get_redis():
    """Get Redis connection — returns None if unavailable."""
    try:
        if not REDIS_URL or REDIS_URL == "your_url":
            return None
        r = redis.from_url(REDIS_URL)
        r.ping()
        return r
    except Exception as e:
        print(f"[Cache] Redis unavailable: {str(e)}")
        return None


def get_cached(query: str):
    return None

def set_cache(query: str, results: dict):
    pass
