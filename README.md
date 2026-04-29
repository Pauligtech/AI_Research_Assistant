# 🔬 AI Research Assistant

An **autonomous, multi-step AI research pipeline** that accepts any research query and produces structured, cited outputs — entirely using free tools.

---

## Architecture

```
Query
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Step 1: Web Search (DuckDuckGo)                        │
│          → top-5 results (title, snippet, URL)          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Step 2: Content Retrieval                               │
│          → WebBaseLoader (fetch HTML)                    │
│          → RecursiveCharacterTextSplitter (1000/200)     │
│          → HuggingFace Embeddings → ChromaDB             │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3: RAG + Summarisation                             │
│          → Retrieve top-K chunks from ChromaDB           │
│          → Build structured prompt                       │
│          → HuggingFace LLM (Mistral-7B, free)            │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Step 4: Citation Tracking                               │
│          → APA 7th-edition format                        │
│          → Deduplication by URL                          │
└────────────────────────┬─────────────────────────────────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
        Advanced Cache      Advanced Streaming
        (SHA-256 keys,      (token-by-token,
         JSON + TTL)         callbacks, generators)
```

---

## Project Structure

```
AI_Research_Assistant/
├── core/
│   ├── main.py               # Master orchestrator (CLI entry point)
│   └── requirements.txt      # All dependencies
├── modules/
│   ├── __init__.py           # Package exports
│   ├── web_search.py         # Step 1 — DuckDuckGo / SerpAPI / Bing
│   ├── content_retrieval.py  # Step 2 — Fetch → Chunk → Embed → ChromaDB
│   ├── rag_summariser.py     # Step 3 — RAG + HuggingFace LLM
│   ├── citations.py          # Step 4 — APA citation tracking
│   ├── cache.py              # Advanced — File-based caching
│   └── streaming.py          # Advanced — Real-time token streaming
├── setup_env.py              # One-click environment setup
├── .env.example              # Template for environment variables
└── .gitignore
```

---

## Setup (3 steps)

### 1. Get a Free HuggingFace Token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens**
3. Click **New token** → select **Read** → copy the token

### 2. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Open .env and paste your token:
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### 3. Run Setup Script

```bash
python setup_env.py
```

This installs all packages and validates your environment automatically.

---

## Usage

### Interactive Mode (recommended for learning)

```bash
python core/main.py
```

You will be prompted to enter a query and choose a mode.

### Command-Line Mode

```bash
# Standard RAG summary
python core/main.py --query "What is Retrieval-Augmented Generation?"

# Multi-perspective research (4 viewpoints)
python core/main.py --query "Climate change solutions" --mode multi

# Real-time streaming output
python core/main.py --query "Future of AI agents" --mode stream

# Skip cache for a fresh result
python core/main.py --query "Latest AI news" --no-cache

# Use Bing instead of DuckDuckGo
python core/main.py --query "Quantum computing" --engine bing
```

---

## Modes

| Mode | Description |
|------|-------------|
| `rag` | Standard RAG pipeline → structured 5-section summary |
| `multi` | 4 perspectives: Scientific, Practical, Critical, Future |
| `stream` | Real-time token-by-token output (no waiting) |

---

## Features

| Feature | Detail |
|---------|--------|
| **Search** | DuckDuckGo (free) · SerpAPI · Bing (pluggable) |
| **Chunking** | 1000 chars, 200 overlap (configurable via .env) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| **Vector DB** | ChromaDB (local, persistent) |
| **LLM** | `mistralai/Mistral-7B-Instruct-v0.2` via HuggingFace Hub (free) |
| **Citations** | APA 7th-edition · deduplication · inline markers |
| **Cache** | SHA-256 keyed JSON · TTL · `@cached` decorator |
| **Streaming** | Console · custom callbacks · generator pattern |
| **Multi-perspective** | 4 simultaneous research viewpoints |

---

## Swapping the LLM

Edit `.env`:

```env
# Other free models on HuggingFace
HF_MODEL_ID=google/flan-t5-large
HF_MODEL_ID=HuggingFaceH4/zephyr-7b-beta
HF_MODEL_ID=tiiuae/falcon-7b-instruct
```

---

## Evaluation Metrics (from brief)

| Metric | Implementation |
|--------|---------------|
| ≥ 80% multi-step task completion | Full pipeline: search → chunk → RAG → cite |
| Tool chaining | `web_search` → `retrieve_and_index` → `summarise` → `CitationTracker` |
| QA, summarization, planning | `rag` mode (QA/summary) + `multi` mode (planning) |
| Minimal manual intervention | Single command: `python core/main.py --query "..."` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACEHUB_API_TOKEN` | — | **Required.** Your HF token |
| `HF_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | LLM model |
| `MAX_SEARCH_RESULTS` | `5` | Results per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `CACHE_DIR` | `.cache` | Cache directory |
| `CACHE_TTL_HOURS` | `24` | Cache expiry (hours) |
| `CHROMA_DIR` | `.chroma_db` | ChromaDB directory |
