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
    """
    Fetch cached search results for a query.
    Returns the deserialized dict if found, None otherwise.
    """
    r = get_redis()
    if not r:
        return None
    try:
        key  = f"pricehunt:search:{query.strip().lower()}"
        data = r.get(key)
        if data:
            print(f"[Cache] HIT for query: '{query}'")
            return json.loads(data)
        print(f"[Cache] MISS for query: '{query}'")
        return None
    except Exception as e:
        print(f"[Cache] get_cached failed: {str(e)}")
        return None


def set_cache(query: str, results: dict):
    """
    Store search results in Redis with a 10-minute TTL.
    Silently skips if Redis is unavailable.
    """
    r = get_redis()
    if not r:
        return
    try:
        key = f"pricehunt:search:{query.strip().lower()}"
        r.setex(key, TTL, json.dumps(results))
        print(f"[Cache] SET for query: '{query}' (TTL: {TTL}s)")
    except Exception as e:
        print(f"[Cache] set_cache failed: {str(e)}")

# import redis
# import json
# from config import REDIS_URL

# TTL = 600  # 10 minutes


# def get_redis():
#     """Get Redis connection — returns None if unavailable."""
#     try:
#         if not REDIS_URL or REDIS_URL == "your_url":
#             return None
#         r = redis.from_url(REDIS_URL)
#         r.ping()
#         return r
#     except Exception as e:
#         print(f"[Cache] Redis unavailable: {str(e)}")
#         return None


# def get_cached(query: str):
#     return None

# def set_cache(query: str, results: dict):
#     pass
