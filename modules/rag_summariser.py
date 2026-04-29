"""
modules/rag_summariser.py
=========================
Step 3 – RAG + Summarisation

Uses HuggingFace Inference API directly (via requests) as the LLM backend.
This bypasses langchain-huggingface's provider routing which breaks on free tier.

Pipeline
--------
1. Retrieve top-K relevant chunks from vector store
2. Build a context-aware prompt
3. Call HF Inference API directly
4. Return structured research output
"""

import os
import logging
import requests
from typing import List, Dict, Optional, Any, Iterator

from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

load_dotenv()
logger = logging.getLogger(__name__)

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "google/flan-t5-large")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
# ─────────────────────────────────────────────────────────────
# API-based LLM — calls HF Inference router (confirmed working)
# Model: facebook/bart-large-cnn (summarization, free tier, status 200 ✓)
# ─────────────────────────────────────────────────────────────

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"


class ApiLLM(LLM):
    """
    LangChain-compatible LLM that calls the HuggingFace Inference
    router API. Confirmed working with facebook/bart-large-cnn.
    """
    model_id:       str = "facebook/bart-large-cnn"
    api_token:      str = ""
    max_length:     int = 250

    @property
    def _llm_type(self) -> str:
        return "hf_api"

    def _call(
        self,
        prompt: str,
        stop=None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        import time

        url     = f"{HF_ROUTER_BASE}/{self.model_id}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type":  "application/json",
        }

        # Extract the context text portion for BART summarisation
        if "Context:" in prompt:
            text = prompt.split("Context:")[-1].strip()
        else:
            text = prompt

        # BART max input ~1024 tokens — truncate safely
        text = text[:3500]
        if len(text) < 30:
            return "Insufficient context to summarise."

        payload = {"inputs": text}

        # Retry up to 3 times with exponential backoff (handles timeouts & 502/503)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logger.info(f"[RAG] POST {self.model_id} (attempt {attempt}/{max_retries})")
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(f"API request failed after {max_retries} attempts: {e}")
                wait = 2 ** attempt
                print(f"  ⚠  Network/Timeout error, retrying in {wait}s…")
                time.sleep(wait)
                continue

            if resp.status_code in (502, 503):
                if attempt == max_retries:
                    raise RuntimeError(
                        f"HuggingFace server error ({resp.status_code}) after "
                        f"{max_retries} attempts. Try again in a minute."
                    )
                wait = 2 ** attempt
                print(f"  ⚠  Server {resp.status_code}, retrying in {wait}s…")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list) and result:
                return result[0].get("summary_text", str(result[0]))
            return str(result)

        return "Summary unavailable — HuggingFace server did not respond."


def get_llm(streaming: bool = False) -> ApiLLM:
    """Return an ApiLLM using facebook/bart-large-cnn via HF router."""
    if not HF_API_TOKEN:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN is not set in .env"
        )
    llm = ApiLLM(model_id=HF_MODEL_ID, api_token=HF_API_TOKEN)
    logger.info(f"[RAG] API LLM ready: {llm.model_id}")
    return llm


# ─────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────

RESEARCH_PROMPT_TEMPLATE = """You are a research assistant. Answer the question below clearly and concisely based on the context provided.

Question: {query}

Context:
{context}

Provide a structured answer covering:
1. Summary
2. Key Findings
3. Limitations
4. Next Steps"""


def build_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Assemble the RAG prompt with retrieved context.

    Parameters
    ----------
    query          : Original research query.
    context_chunks : List of relevant text chunks from ChromaDB.

    Returns
    -------
    Full prompt string ready for the LLM.
    """
    context = "\n\n---\n\n".join(context_chunks)
    return RESEARCH_PROMPT_TEMPLATE.format(query=query, context=context)


# ─────────────────────────────────────────────────────────────
# Retrieval helper
# ─────────────────────────────────────────────────────────────

def retrieve_context(
    vector_store,
    query: str,
    k: int = 5,
) -> List[Dict]:
    """
    Retrieve the top-K most relevant chunks from the vector store.

    Parameters
    ----------
    vector_store : ChromaDB/LangChain vectorstore instance.
    query        : Search query.
    k            : Number of chunks to retrieve.

    Returns
    -------
    List of dicts: {content, source, score}
    """
    if vector_store is None:
        logger.warning("[RAG] No vector store provided – skipping retrieval.")
        return []

    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    chunks = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "unknown"),
            "score":   round(score, 4),
        })
    logger.info(f"[RAG] Retrieved {len(chunks)} relevant chunks (k={k})")
    return chunks


# ─────────────────────────────────────────────────────────────
# Summarisation (batch mode)
# ─────────────────────────────────────────────────────────────

def summarise(
    query: str,
    vector_store,
    k: int = 5,
    llm=None,
) -> Dict:
    """
    Run RAG: retrieve context → build prompt → generate summary.

    Parameters
    ----------
    query        : Research question.
    vector_store : ChromaDB vectorstore with indexed content.
    k            : Chunks to retrieve.
    llm          : Optional pre-built LLM (creates one if None).

    Returns
    -------
    Dict with keys: summary, context_chunks, query
    """
    if llm is None:
        llm = get_llm(streaming=False)

    context_chunks = retrieve_context(vector_store, query, k=k)
    context_texts  = [c["content"] for c in context_chunks]

    if not context_texts:
        logger.warning("[RAG] No context retrieved – generating without RAG.")
        context_texts = ["No relevant web content was retrieved for this query."]

    prompt  = build_prompt(query, context_texts)
    logger.info("[RAG] Generating summary…")
    summary = llm.invoke(prompt)

    return {
        "query":          query,
        "summary":        summary,
        "context_chunks": context_chunks,
    }


# ─────────────────────────────────────────────────────────────
# Multi-perspective research
# ─────────────────────────────────────────────────────────────

PERSPECTIVES = [
    ("Scientific / Technical",   "Explain {query} from a scientific and technical perspective."),
    ("Practical / Applications",  "What are the real-world applications of {query}?"),
    ("Critical / Limitations",    "What are the main limitations and criticisms of {query}?"),
    ("Future / Trends",           "What future developments and trends are expected for {query}?"),
]


def multi_perspective_research(
    query: str,
    vector_store,
    k: int = 4,
    llm=None,
) -> Dict:
    """
    Generate research from multiple viewpoints.

    Returns
    -------
    Dict mapping perspective name → summary text
    """
    if llm is None:
        llm = get_llm(streaming=False)

    results = {}
    for perspective_name, template in PERSPECTIVES:
        sub_query = template.format(query=query)
        logger.info(f"[RAG] Perspective: {perspective_name}")
        result = summarise(sub_query, vector_store, k=k, llm=llm)
        results[perspective_name] = result["summary"]

    return {
        "query":        query,
        "perspectives": results,
    }


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[Test] RAG Summariser module loaded successfully.")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Token : {'set ✓' if HF_API_TOKEN else 'NOT SET ✗'}")
