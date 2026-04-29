"""
modules/cache.py
================
Advanced Feature – Caching

Purpose
-------
* Avoid redundant API calls for identical queries  → saves cost & time
* Uses a simple file-based JSON cache (no Redis/DB dependency)
* TTL (time-to-live) controlled via .env (CACHE_TTL_HOURS)
* Cache key = SHA-256 hash of (query + engine) for collision-free storage

Cache structure (one JSON file per query)
-----------------------------------------
{
  "query":      "original query string",
  "engine":     "duckduckgo",
  "timestamp":  1714300000.0,
  "ttl_hours":  24,
  "data":       { ... }         ← any serialisable payload
}
"""

import os
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional, Dict

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR      = os.getenv("CACHE_DIR", ".cache")
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "24"))


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _cache_key(query: str, namespace: str = "default") -> str:
    """SHA-256 hash of (namespace + query) → filename-safe hex string."""
    raw = f"{namespace}::{query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    """Full path to a cache file."""
    path = Path(CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.json"


def _is_expired(timestamp: float, ttl_hours: float) -> bool:
    """Return True if the cached entry has exceeded its TTL."""
    age_hours = (time.time() - timestamp) / 3600
    return age_hours > ttl_hours


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def cache_get(query: str, namespace: str = "default") -> Optional[Any]:
    """
    Retrieve a cached result for the given query.

    Parameters
    ----------
    query     : The original query string.
    namespace : Logical bucket (e.g. 'search', 'rag', 'multi').

    Returns
    -------
    The cached data payload, or None if not found / expired.
    """
    key  = _cache_key(query, namespace)
    path = _cache_path(key)

    if not path.exists():
        logger.debug(f"[Cache] MISS  — {namespace}:'{query[:60]}'")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[Cache] Corrupt entry, deleting: {exc}")
        path.unlink(missing_ok=True)
        return None

    ttl = entry.get("ttl_hours", CACHE_TTL_HOURS)
    if _is_expired(entry["timestamp"], ttl):
        logger.info(f"[Cache] EXPIRED — {namespace}:'{query[:60]}'")
        path.unlink(missing_ok=True)
        return None

    age_min = (time.time() - entry["timestamp"]) / 60
    logger.info(f"[Cache] HIT ✓  — {namespace}:'{query[:60]}' (age {age_min:.1f} min)")
    return entry["data"]


def cache_set(
    query: str,
    data: Any,
    namespace: str = "default",
    ttl_hours: float = CACHE_TTL_HOURS,
) -> None:
    """
    Store a result in the cache.

    Parameters
    ----------
    query     : The original query string.
    data      : Any JSON-serialisable payload to cache.
    namespace : Logical bucket.
    ttl_hours : How long this entry is valid (hours).
    """
    key  = _cache_key(query, namespace)
    path = _cache_path(key)

    entry = {
        "query":     query,
        "namespace": namespace,
        "timestamp": time.time(),
        "ttl_hours": ttl_hours,
        "data":      data,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        logger.info(f"[Cache] STORED — {namespace}:'{query[:60]}' (TTL {ttl_hours}h)")
    except OSError as exc:
        logger.error(f"[Cache] Failed to write: {exc}")


def cache_clear(namespace: Optional[str] = None) -> int:
    """
    Delete cached entries.

    Parameters
    ----------
    namespace : If given, only delete entries for this namespace.
                If None, clear the entire cache directory.

    Returns
    -------
    Number of files deleted.
    """
    path = Path(CACHE_DIR)
    if not path.exists():
        return 0

    deleted = 0
    for f in path.glob("*.json"):
        if namespace is None:
            f.unlink()
            deleted += 1
        else:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    entry = json.load(fh)
                if entry.get("namespace") == namespace:
                    f.unlink()
                    deleted += 1
            except Exception:
                pass

    logger.info(f"[Cache] Cleared {deleted} entr(ies) (namespace={namespace or 'ALL'})")
    return deleted


def cache_stats() -> Dict:
    """Return statistics about the current cache."""
    path = Path(CACHE_DIR)
    if not path.exists():
        return {"total": 0, "expired": 0, "valid": 0, "size_kb": 0.0}

    total = expired = valid = 0
    size  = 0.0
    for f in path.glob("*.json"):
        total += 1
        size  += f.stat().st_size / 1024
        try:
            with open(f, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            if _is_expired(entry["timestamp"], entry.get("ttl_hours", CACHE_TTL_HOURS)):
                expired += 1
            else:
                valid += 1
        except Exception:
            expired += 1

    return {"total": total, "expired": expired, "valid": valid, "size_kb": round(size, 2)}


# ─────────────────────────────────────────────────────────────
# Decorator for easy caching of any function
# ─────────────────────────────────────────────────────────────

def cached(namespace: str = "default", ttl_hours: float = CACHE_TTL_HOURS):
    """
    Decorator that wraps any function with cache-get / cache-set.

    The first positional argument of the decorated function is used
    as the cache key (query string).

    Usage
    -----
    @cached(namespace="search", ttl_hours=12)
    def my_search(query, ...):
        ...
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(query, *args, **kwargs):
            cached_result = cache_get(query, namespace=namespace)
            if cached_result is not None:
                return cached_result
            result = func(query, *args, **kwargs)
            cache_set(query, result, namespace=namespace, ttl_hours=ttl_hours)
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[Cache] Stats before:", cache_stats())

    cache_set("test query", {"results": [1, 2, 3]}, namespace="test")
    result = cache_get("test query", namespace="test")
    print("[Cache] Retrieved:", result)

    print("[Cache] Stats after:", cache_stats())
    cache_clear(namespace="test")
    print("[Cache] Stats cleared:", cache_stats())
