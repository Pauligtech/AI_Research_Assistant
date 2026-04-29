"""
core/main.py
============
AI Research Assistant — Master Orchestrator
===========================================

Chains all steps into a single, cohesive pipeline:

  Step 1  →  Web Search (DuckDuckGo)
  Step 2  →  Content Retrieval + ChromaDB indexing
  Step 3  →  RAG Summarisation (HuggingFace LLM)
  Step 4  →  APA Citation Tracking
  Advanced → Caching  (repeat queries served instantly)
  Advanced → Streaming (real-time token output)
  Advanced → Multi-perspective Research

Usage (interactive CLI)
-----------------------
  python core/main.py

Usage (single query, non-interactive)
--------------------------------------
  python core/main.py --query "What is Retrieval-Augmented Generation?" --mode rag

Flags
-----
  --query   Query string
  --mode    rag | multi | stream   (default: rag)
  --engine  duckduckgo | serpapi | bing  (default: duckduckgo)
  --results Number of search results (default: 5)
  --no-cache  Bypass cache for this run
"""
# Import libraries 
import sys
import os
import logging
import argparse
import time
from pathlib import Path

# ── Ensure project root is on the path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel   import Panel
from rich.rule    import Rule
from rich.table   import Table
from rich         import print as rprint

# ── Project modules ──────────────────────────────────────────────────────────
from modules.web_search        import web_search
from modules.content_retrieval import retrieve_and_index
from modules.rag_summariser    import summarise, multi_perspective_research, get_llm, build_prompt, retrieve_context
from modules.citations         import CitationTracker
from modules.cache             import cache_get, cache_set, cache_stats
from modules.streaming         import stream_to_console

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # Suppress noisy library logs
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("ai_research_assistant")

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         🔬  AI Research Assistant  v1.0                 ║
║   Search → Retrieve → RAG → Summarise → Cite           ║
╚══════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_research(
    query: str,
    mode: str = "rag",
    engine: str = "duckduckgo",
    max_results: int = 5,
    use_cache: bool = True,
    stream: bool = False,
) -> dict:
    """
    Execute the full AI research pipeline.

    Parameters
    ----------
    query       : Research question / topic.
    mode        : 'rag' | 'multi' | 'stream'
    engine      : Search engine to use.
    max_results : Number of search results.
    use_cache   : Whether to use the file-based cache.
    stream      : If True, stream the final answer to console.

    Returns
    -------
    dict with keys: query, summary (or perspectives), citations, metadata
    """
    start = time.time()
    cache_ns = f"{mode}_{engine}"

    # ── Cache check ───────────────────────────────────────────────────────
    if use_cache:
        cached = cache_get(query, namespace=cache_ns)
        if cached:
            console.print(Panel(
                "[bold green]⚡ Cache HIT — returning instant result![/]",
                border_style="green",
            ))
            return cached

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker = CitationTracker()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Web Search
    # ════════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Step 1 · Web Search[/]"))
    console.print(f"  Engine : [yellow]{engine}[/]  |  Max results : [yellow]{max_results}[/]")

    results = web_search(query, max_results=max_results, engine=engine)

    if not results:
        console.print("[red]No search results found. Check your internet connection.[/]")
        return {"error": "No search results returned."}

    # Show results table
    table = Table(title="Search Results", show_lines=True)
    table.add_column("#",       style="cyan",  width=4)
    table.add_column("Title",   style="white", width=40)
    table.add_column("URL",     style="blue",  width=55)

    for r in results:
        table.add_row(str(r["rank"]), r["title"][:40], r["url"][:55])
        tracker.add(title=r["title"], url=r["url"], snippet=r["snippet"])

    console.print(table)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Content Retrieval & Indexing
    # ════════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Step 2 · Content Retrieval & Indexing[/]"))

    urls         = [r["url"] for r in results]
    vector_store = retrieve_and_index(urls)

    if not vector_store:
        console.print("[yellow]Warning: No content indexed. RAG will run without context.[/]")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — RAG Summarisation / Multi-perspective
    # ════════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Step 3 · RAG Summarisation[/]"))

    llm = get_llm(streaming=(mode == "stream"))
    output = {}

    if mode == "multi":
        console.print("  Mode : [magenta]Multi-Perspective Research[/]")
        mp_result = multi_perspective_research(query, vector_store, llm=llm)
        for perspective, text in mp_result["perspectives"].items():
            console.print(Panel(
                text,
                title=f"[bold magenta]{perspective}[/]",
                border_style="magenta",
            ))
        output = mp_result

    elif mode == "stream":
        console.print("  Mode : [green]Real-Time Streaming[/]")
        context_chunks = retrieve_context(vector_store, query, k=5) if vector_store else []
        context_texts  = [c["content"] for c in context_chunks] or [
            "No retrieved context available."
        ]
        prompt = build_prompt(query, context_texts)
        summary = stream_to_console(prompt)
        output = {"query": query, "summary": summary}

    else:  # default: rag
        console.print("  Mode : [blue]RAG + Summarisation[/]")
        rag_result = summarise(query, vector_store, llm=llm)
        console.print(Panel(
            rag_result["summary"],
            title="[bold blue]Research Summary[/]",
            border_style="blue",
            padding=(1, 2),
        ))
        output = rag_result

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — Citations
    # ════════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold cyan]Step 4 · Citations[/]"))
    ref_list = tracker.reference_list()
    console.print(ref_list)

    # ── Assemble final result ─────────────────────────────────────────────
    elapsed = round(time.time() - start, 2)
    final = {
        **output,
        "citations":  tracker.get_all(),
        "references": ref_list,
        "metadata": {
            "engine":       engine,
            "mode":         mode,
            "results_used": len(results),
            "elapsed_sec":  elapsed,
        },
    }

    # ── Cache the result ──────────────────────────────────────────────────
    if use_cache:
        cache_set(query, final, namespace=cache_ns)
        console.print(f"  [dim]Result cached for future queries.[/]")

    console.print(Rule())
    console.print(f"  ✅ Done in [bold]{elapsed}s[/] | Sources: [bold]{len(tracker)}[/]")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI loop
# ─────────────────────────────────────────────────────────────────────────────

def interactive_cli():
    """Run the assistant in interactive mode."""
    console.print(BANNER, style="bold blue")

    # Show cache stats
    stats = cache_stats()
    console.print(
        f"  Cache: [green]{stats['valid']} valid[/] | "
        f"[red]{stats['expired']} expired[/] | "
        f"[dim]{stats['size_kb']} KB[/]\n"
    )

    while True:
        console.print(Rule("[dim]New Query[/]"))
        query = console.input("  [bold]🔍 Enter your research query[/] (or 'quit'): ").strip()

        if query.lower() in ("quit", "exit", "q"):
            console.print("\n  Goodbye! 👋\n", style="bold green")
            break

        if not query:
            continue

        # Mode selection
        console.print(
            "\n  [bold]Select mode:[/]\n"
            "    [1] RAG + Summary  (default)\n"
            "    [2] Multi-Perspective\n"
            "    [3] Streaming (real-time)\n"
        )
        mode_map = {"1": "rag", "2": "multi", "3": "stream", "": "rag"}
        choice = console.input("  Choice [1/2/3]: ").strip()
        mode   = mode_map.get(choice, "rag")

        run_research(query=query, mode=mode)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser (non-interactive / scripted usage)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Research Assistant — Chain: search → retrieve → RAG → cite"
    )
    parser.add_argument("--query",    type=str, default=None,
                        help="Research question (omit for interactive mode)")
    parser.add_argument("--mode",     type=str, default="rag",
                        choices=["rag", "multi", "stream"],
                        help="Operation mode (default: rag)")
    parser.add_argument("--engine",   type=str, default="duckduckgo",
                        choices=["duckduckgo", "serpapi", "bing"],
                        help="Search engine (default: duckduckgo)")
    parser.add_argument("--results",  type=int, default=5,
                        help="Number of search results (default: 5)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass cache for this run")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.query:
        # Non-interactive: run once and exit
        console.print(BANNER, style="bold blue")
        run_research(
            query=args.query,
            mode=args.mode,
            engine=args.engine,
            max_results=args.results,
            use_cache=not args.no_cache,
        )
    else:
        # Interactive CLI loop
        interactive_cli()
