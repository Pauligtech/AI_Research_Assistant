"""
modules/__init__.py
===================
Package initialiser — exposes the public API of every module.
"""

from .web_search       import web_search
from .content_retrieval import retrieve_and_index, fetch_documents, chunk_documents
from .rag_summariser   import summarise, multi_perspective_research, get_llm
from .citations        import CitationTracker, generate_citations, format_reference_list
from .cache            import cache_get, cache_set, cache_clear, cache_stats, cached
from .streaming        import stream_to_console, stream_to_callback, stream_generator

__all__ = [
    "web_search",
    "retrieve_and_index",
    "fetch_documents",
    "chunk_documents",
    "summarise",
    "multi_perspective_research",
    "get_llm",
    "CitationTracker",
    "generate_citations",
    "format_reference_list",
    "cache_get",
    "cache_set",
    "cache_clear",
    "cache_stats",
    "cached",
    "stream_to_console",
    "stream_to_callback",
    "stream_generator",
]
