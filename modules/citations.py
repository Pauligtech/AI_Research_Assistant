"""
modules/citations.py
====================
Step 4 – Professional Citation Tracking

Generates APA 7th-edition style citations from web search results
and content metadata, as used in academic research papers.

Citation format
---------------
Author(s). (Year, Month Day). Title of page. Site Name. URL

When author / date are unavailable, sensible fallbacks are applied.
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Core citation builder
# ─────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Return a clean domain name from a URL (e.g., 'en.wikipedia.org')."""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "Unknown Source"


def _title_case(title: str) -> str:
    """Capitalise only the first letter of the title (sentence case for APA)."""
    return title.strip().capitalize() if title else "Untitled page"


def build_apa_citation(
    title: str,
    url: str,
    author: Optional[str] = None,
    date: Optional[str] = None,
    site_name: Optional[str] = None,
) -> str:
    """
    Build an APA 7th-edition web citation.

    Parameters
    ----------
    title     : Page title (required).
    url       : Full URL (required).
    author    : Author name(s). Defaults to site domain.
    date      : Publication/access date string. Defaults to today.
    site_name : Website name. Defaults to domain extracted from URL.

    Returns
    -------
    Formatted APA citation string.
    """
    # Author
    author_str = author if author else _extract_domain(url)

    # Date
    if date:
        date_str = date
    else:
        date_str = datetime.now().strftime("%Y, %B %d")

    # Site name
    site = site_name if site_name else _extract_domain(url)

    # Clean title
    clean_title = _title_case(title)

    return f"{author_str}. ({date_str}). {clean_title}. *{site}*. {url}"


# ─────────────────────────────────────────────────────────────
# Bulk citation generation from search results
# ─────────────────────────────────────────────────────────────

def generate_citations(search_results: List[Dict]) -> List[Dict]:
    """
    Generate APA citations for a list of search results.

    Parameters
    ----------
    search_results : List of dicts with keys: title, url, snippet, rank.

    Returns
    -------
    List of dicts: {rank, citation, url, title}
    """
    citations = []
    for result in search_results:
        url   = result.get("url", "")
        title = result.get("title", "Untitled")
        rank  = result.get("rank", len(citations) + 1)

        if not url:
            continue

        apa = build_apa_citation(title=title, url=url)
        citations.append({
            "rank":     rank,
            "title":    title,
            "url":      url,
            "citation": apa,
        })

    logger.info(f"[Citations] Generated {len(citations)} APA citation(s).")
    return citations


# ─────────────────────────────────────────────────────────────
# Format citation block (ready to paste into a report)
# ─────────────────────────────────────────────────────────────

def format_reference_list(citations: List[Dict]) -> str:
    """
    Return a formatted References section (APA style).

    Parameters
    ----------
    citations : Output of generate_citations().

    Returns
    -------
    Multi-line string, ready to append to a research report.
    """
    if not citations:
        return "\n## References\n\nNo sources retrieved.\n"

    lines = ["\n## References\n"]
    for i, c in enumerate(citations, start=1):
        lines.append(f"[{i}] {c['citation']}\n")

    return "\n".join(lines)


def format_inline_ref(rank: int) -> str:
    """Return an inline citation marker, e.g. [3]."""
    return f"[{rank}]"


# ─────────────────────────────────────────────────────────────
# Citation index (for tracking which sources were actually used)
# ─────────────────────────────────────────────────────────────

class CitationTracker:
    """
    Keeps a live registry of sources cited during a research session.

    Usage
    -----
    tracker = CitationTracker()
    tracker.add(title="...", url="...", snippet="...")
    print(tracker.reference_list())
    """

    def __init__(self):
        self._registry: List[Dict] = []
        self._url_index: Dict[str, int] = {}

    def add(
        self,
        title: str,
        url: str,
        snippet: str = "",
        author: Optional[str] = None,
        date: Optional[str] = None,
    ) -> int:
        """
        Register a source and return its citation number.
        Deduplicates by URL – same URL always gets the same number.
        """
        if url in self._url_index:
            return self._url_index[url]

        apa = build_apa_citation(title=title, url=url, author=author, date=date)
        rank = len(self._registry) + 1
        self._registry.append({
            "rank":     rank,
            "title":    title,
            "url":      url,
            "snippet":  snippet,
            "citation": apa,
        })
        self._url_index[url] = rank
        logger.debug(f"[Citations] Added [{rank}]: {url}")
        return rank

    def add_from_search_results(self, results: List[Dict]) -> List[int]:
        """Bulk-add from web search result list. Returns list of citation numbers."""
        ranks = []
        for r in results:
            rank = self.add(
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
            )
            ranks.append(rank)
        return ranks

    def reference_list(self) -> str:
        """Return the formatted APA reference list."""
        return format_reference_list(self._registry)

    def get_all(self) -> List[Dict]:
        """Return the full citation registry."""
        return list(self._registry)

    def __len__(self):
        return len(self._registry)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_results = [
        {
            "rank": 1,
            "title": "What is Retrieval-Augmented Generation?",
            "url": "https://aws.amazon.com/what-is/retrieval-augmented-generation/",
            "snippet": "RAG is a technique that combines...",
        },
        {
            "rank": 2,
            "title": "LangChain RAG Tutorial",
            "url": "https://python.langchain.com/docs/tutorials/rag/",
            "snippet": "This tutorial walks you through...",
        },
    ]

    tracker = CitationTracker()
    tracker.add_from_search_results(sample_results)
    print(tracker.reference_list())
