"""
modules/web_search.py
=====================
Step 1 – Web Search

Uses DuckDuckGo (free, no API key required) as the primary search engine.
Returns the top N results with title, snippet, and URL.

Swap guides
-----------
* Google SerpAPI  → set SERPAPI_API_KEY in .env and call search_serpapi()
* Bing Search API → set BING_API_KEY in .env and call search_bing()
"""

import os
import time
import logging
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))


# ─────────────────────────────────────────────────────────────
# PRIMARY: DuckDuckGo (free, no key)
# ─────────────────────────────────────────────────────────────

def search_duckduckgo(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Search DuckDuckGo and return the top N results.

    Parameters
    ----------
    query       : The search query string.
    max_results : How many results to return (default: env MAX_SEARCH_RESULTS).

    Returns
    -------
    List of dicts with keys: title, snippet, url, rank
    """
    try:
        from ddgs import DDGS
    except ImportError:
        raise ImportError(
            "ddgs is not installed. "
            "Run: pip install ddgs"
        )

    results: List[Dict] = []
    logger.info(f"[WebSearch] DuckDuckGo query: '{query}' (max={max_results})")

    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            for rank, item in enumerate(raw, start=1):
                results.append({
                    "rank":    rank,
                    "title":   item.get("title", "No title"),
                    "snippet": item.get("body", "No snippet available"),
                    "url":     item.get("href", ""),
                })
    except Exception as exc:
        logger.error(f"[WebSearch] DuckDuckGo error: {exc}")

    logger.info(f"[WebSearch] Retrieved {len(results)} results.")
    return results


# ─────────────────────────────────────────────────────────────
# OPTIONAL: Google SerpAPI (requires SERPAPI_API_KEY)
# ─────────────────────────────────────────────────────────────

def search_serpapi(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Search via Google SerpAPI.
    Requires: pip install google-search-results
    Requires: SERPAPI_API_KEY in .env
    """
    try:
        from serpapi import GoogleSearch
    except ImportError:
        raise ImportError(
            "SerpAPI client not installed. Run: pip install google-search-results"
        )

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise EnvironmentError("SERPAPI_API_KEY not set in .env")

    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": api_key,
    }
    search = GoogleSearch(params)
    raw    = search.get_dict().get("organic_results", [])
    results = []
    for rank, item in enumerate(raw[:max_results], start=1):
        results.append({
            "rank":    rank,
            "title":   item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url":     item.get("link", ""),
        })
    return results


# ─────────────────────────────────────────────────────────────
# OPTIONAL: Bing Search API (requires BING_API_KEY)
# ─────────────────────────────────────────────────────────────

def search_bing(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Search via Bing Web Search API.
    Requires: BING_API_KEY in .env
    """
    import requests

    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        raise EnvironmentError("BING_API_KEY not set in .env")

    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers  = {"Ocp-Apim-Subscription-Key": api_key}
    params   = {"q": query, "count": max_results}

    response = requests.get(endpoint, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    results = []
    for rank, item in enumerate(
        response.json().get("webPages", {}).get("value", [])[:max_results],
        start=1,
    ):
        results.append({
            "rank":    rank,
            "title":   item.get("name", ""),
            "snippet": item.get("snippet", ""),
            "url":     item.get("url", ""),
        })
    return results


# ─────────────────────────────────────────────────────────────
# Public façade – always call this
# ─────────────────────────────────────────────────────────────

def web_search(
    query: str,
    max_results: int = MAX_RESULTS,
    engine: str = "duckduckgo",
) -> List[Dict]:
    """
    Unified web-search interface.

    Parameters
    ----------
    query       : The search query.
    max_results : Number of results to return.
    engine      : 'duckduckgo' | 'serpapi' | 'bing'

    Returns
    -------
    List of result dicts: {rank, title, snippet, url}
    """
    engines = {
        "duckduckgo": search_duckduckgo,
    }
    fn = engines.get(engine.lower())
    if fn is None:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {list(engines)}")

    return fn(query, max_results=max_results)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    results = web_search("LangChain RAG tutorial 2024")
    print(json.dumps(results, indent=2))
