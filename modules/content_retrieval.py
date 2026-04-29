"""
modules/content_retrieval.py
============================
Step 2 – Content Retrieval

Pipeline
--------
1. Fetch HTML from URLs   → WebBaseLoader
2. Chunk text              → RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. Embed chunks            → HuggingFace sentence-transformers (free)
4. Store in vector DB      → ChromaDB (local, persistent)
"""

# Set USER_AGENT before any langchain import to suppress the warning
import os
os.environ.setdefault("USER_AGENT", "AI-Research-Assistant/1.0")

import logging
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHROMA_DIR    = os.getenv("CHROMA_DIR", ".chroma_db")

# HuggingFace embedding model (runs locally, no API key needed)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────
# 1.  Fetch & extract text from URLs
# ─────────────────────────────────────────────────────────────

def fetch_documents(urls: List[str]) -> List:
    """
    Load web pages and return LangChain Document objects.

    Parameters
    ----------
    urls : List of page URLs to scrape.

    Returns
    -------
    List of LangChain Document objects with page_content and metadata.
    """
    from langchain_community.document_loaders import WebBaseLoader

    valid_urls = [u for u in urls if u.startswith("http")]
    if not valid_urls:
        logger.warning("[ContentRetrieval] No valid URLs provided.")
        return []

    logger.info(f"[ContentRetrieval] Fetching {len(valid_urls)} URL(s)…")
    docs = []
    for url in valid_urls:
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            docs.extend(loaded)
            logger.info(f"  ✓ {url} → {len(loaded)} doc(s)")
        except Exception as exc:
            logger.warning(f"  ✗ Failed to load {url}: {exc}")

    logger.info(f"[ContentRetrieval] Total documents fetched: {len(docs)}")
    return docs


# ─────────────────────────────────────────────────────────────
# 2.  Chunk documents
# ─────────────────────────────────────────────────────────────

def chunk_documents(docs: List, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """
    Split documents into smaller chunks for embedding.

    Parameters
    ----------
    docs          : List of LangChain Document objects.
    chunk_size    : Characters per chunk (default from .env).
    chunk_overlap : Overlap between chunks (default from .env).

    Returns
    -------
    List of chunked LangChain Document objects.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # fallback for older versions

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        f"[ContentRetrieval] Split into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ─────────────────────────────────────────────────────────────
# 3 & 4.  Embed and store in ChromaDB
# ─────────────────────────────────────────────────────────────

def _get_embeddings():
    """Return a HuggingFace embedding instance (cached in memory)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List, collection_name: str = "research") -> object:
    """
    Embed chunks and store them in a local ChromaDB collection.

    Parameters
    ----------
    chunks          : Chunked LangChain Document objects.
    collection_name : ChromaDB collection identifier.

    Returns
    -------
    A LangChain Chroma vectorstore instance (ready for similarity search).
    """
    from langchain_community.vectorstores import Chroma

    if not chunks:
        logger.warning("[ContentRetrieval] No chunks to embed.")
        return None

    embeddings = _get_embeddings()
    logger.info(f"[ContentRetrieval] Embedding {len(chunks)} chunks → ChromaDB…")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    logger.info(f"[ContentRetrieval] Vector store saved to '{CHROMA_DIR}'.")
    return vector_store


def load_vector_store(collection_name: str = "research") -> Optional[object]:
    """
    Load an existing ChromaDB collection from disk.

    Returns None if the collection does not exist.
    """
    from langchain_community.vectorstores import Chroma

    if not os.path.exists(CHROMA_DIR):
        logger.info("[ContentRetrieval] No existing ChromaDB found.")
        return None

    embeddings = _get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    logger.info(f"[ContentRetrieval] Loaded existing ChromaDB from '{CHROMA_DIR}'.")
    return vector_store


# ─────────────────────────────────────────────────────────────
# Convenience: full pipeline in one call
# ─────────────────────────────────────────────────────────────

def retrieve_and_index(
    urls: List[str],
    collection_name: str = "research",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Optional[object]:
    """
    End-to-end: fetch → chunk → embed → store.

    Returns a ready-to-query ChromaDB vectorstore, or None on failure.
    """
    docs   = fetch_documents(urls)
    if not docs:
        return None
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    return build_vector_store(chunks, collection_name)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_urls = ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    vs = retrieve_and_index(test_urls)
    if vs:
        results = vs.similarity_search("What is RAG?", k=3)
        for r in results:
            print(r.page_content[:200], "\n---")
