"""
modules/streaming.py
====================
Advanced Feature – Real-Time Streaming

Streams LLM output token-by-token to the console (or any callback)
so the user sees results as they are generated — no waiting.

Two modes
---------
1. stream_to_console  : prints tokens directly as they arrive
2. stream_to_callback : calls a user-supplied function with each token
                        (useful for web UIs, websockets, etc.)
"""

import os
import sys
import logging
from typing import Callable, Optional, Generator

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


# ─────────────────────────────────────────────────────────────
# LangChain streaming callback
# ─────────────────────────────────────────────────────────────

class StreamPrinter:
    """
    A simple streaming handler that prints tokens as they arrive.
    Compatible with LangChain's streaming interface.
    """

    def __init__(self, end_marker: str = "\n\n[Stream complete]\n"):
        self.end_marker  = end_marker
        self.full_output = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called by LangChain for every new token."""
        print(token, end="", flush=True)
        self.full_output += token

    def on_llm_end(self, *args, **kwargs) -> None:
        print(self.end_marker, flush=True)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        print(f"\n[Stream ERROR]: {error}", flush=True)
        logger.error(f"[Streaming] LLM error: {error}")


# ─────────────────────────────────────────────────────────────
# Streaming LLM factory
# ─────────────────────────────────────────────────────────────

def get_streaming_llm(callback_handler=None):
    """
    Build a streaming HuggingFaceEndpoint LLM.

    Parameters
    ----------
    callback_handler : Optional LangChain callback. If None, uses StreamPrinter.

    Returns
    -------
    (llm, printer) – the LLM and the StreamPrinter instance.
    """
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    if not HF_API_TOKEN:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN not set. "
            "See .env.example for instructions."
        )

    printer = StreamPrinter()

    # Use LangChain's built-in stdout handler for simplicity
    handler = callback_handler or StreamingStdOutCallbackHandler()

    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        huggingfacehub_api_token=HF_API_TOKEN,
        max_new_tokens=512,
        temperature=0.4,
        repetition_penalty=1.3,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        streaming=True,
        callbacks=[handler],
        task="text2text-generation",
    )
    return llm, printer


# ─────────────────────────────────────────────────────────────
# Stream to console
# ─────────────────────────────────────────────────────────────

def stream_to_console(prompt: str) -> str:
    """
    Stream the LLM response for a given prompt directly to stdout.

    Parameters
    ----------
    prompt : Full prompt string to send to the LLM.

    Returns
    -------
    The complete generated text (collected from the stream).
    """
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    llm, printer = get_streaming_llm()

    print("\n" + "═" * 60)
    print("  🔴 LIVE STREAM  —  generating response…")
    print("═" * 60 + "\n")

    result = llm.invoke(prompt)

    print("\n" + "═" * 60)
    return result


# ─────────────────────────────────────────────────────────────
# Stream with custom callback (for GUI / websocket integration)
# ─────────────────────────────────────────────────────────────

def stream_to_callback(
    prompt: str,
    on_token: Callable[[str], None],
    on_complete: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Stream LLM tokens, calling `on_token(token)` for each new token.

    Parameters
    ----------
    prompt      : Full prompt string.
    on_token    : Called with each token string as it arrives.
    on_complete : Called with the full output when streaming ends.

    Returns
    -------
    The complete generated text.
    """
    from langchain.callbacks.base import BaseCallbackHandler

    class CustomCallback(BaseCallbackHandler):
        def __init__(self):
            self.full_output = ""

        def on_llm_new_token(self, token: str, **kwargs):
            self.full_output += token
            on_token(token)

        def on_llm_end(self, *args, **kwargs):
            if on_complete:
                on_complete(self.full_output)

    cb = CustomCallback()
    llm, _ = get_streaming_llm(callback_handler=cb)
    llm.invoke(prompt)
    return cb.full_output


# ─────────────────────────────────────────────────────────────
# Generator-based streaming (for async / iterative consumption)
# ─────────────────────────────────────────────────────────────

def stream_generator(prompt: str) -> Generator[str, None, None]:
    """
    Yield tokens one-by-one (generator pattern).

    Usage
    -----
    for token in stream_generator(prompt):
        print(token, end="", flush=True)
    """
    tokens = []

    def collect(token: str):
        tokens.append(token)

    # We use a thread + callback trick to convert callback → generator
    import threading

    done_event = threading.Event()
    full_output = {"text": ""}

    def run():
        result = stream_to_callback(
            prompt,
            on_token=collect,
            on_complete=lambda t: full_output.update({"text": t}),
        )
        done_event.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    idx = 0
    while not done_event.is_set() or idx < len(tokens):
        if idx < len(tokens):
            yield tokens[idx]
            idx += 1
        else:
            import time
            time.sleep(0.01)

    thread.join()


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[Streaming] Module loaded.")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Token : {'set ✓' if HF_API_TOKEN else 'NOT SET ✗'}")
    if HF_API_TOKEN:
        stream_to_console("In one sentence, what is Retrieval-Augmented Generation?")
