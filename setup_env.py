"""
setup_env.py
============
One-click environment setup script for AI Research Assistant.

Run this ONCE after cloning / downloading the project:
  python setup_env.py

What it does
------------
1. Checks Python version (≥3.9 required)
2. Installs all packages from core/requirements.txt
3. Creates a .env file from .env.example (if one doesn't exist)
4. Validates the HuggingFace token
5. Verifies key imports
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_python():
    major, minor = sys.version_info[:2]
    print(f"  Python {major}.{minor} detected.")
    if major < 3 or (major == 3 and minor < 9):
        print("  ✗ Python 3.9+ is required.")
        sys.exit(1)
    print("  ✓ Python version OK.")


def install_packages():
    req_file = ROOT / "core" / "requirements.txt"
    print(f"\n  Installing packages from {req_file} …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  ✗ pip install failed. See output above.")
        sys.exit(1)
    print("  ✓ All packages installed.")


def create_env_file():
    env_file     = ROOT / ".env"
    env_example  = ROOT / ".env.example"

    if env_file.exists():
        print(f"\n  .env already exists — skipping creation.")
        return

    if env_example.exists():
        shutil.copy(env_example, env_file)
        print(f"\n  ✓ Created .env from .env.example")
        print("  ⚠️  IMPORTANT: Open .env and add your HUGGINGFACEHUB_API_TOKEN!")
    else:
        print(f"\n  ✗ .env.example not found. Creating minimal .env …")
        env_file.write_text(
            "HUGGINGFACEHUB_API_TOKEN=your_token_here\n"
            "HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2\n"
            "MAX_SEARCH_RESULTS=5\n"
            "CHUNK_SIZE=1000\n"
            "CHUNK_OVERLAP=200\n"
            "CACHE_DIR=.cache\n"
            "CACHE_TTL_HOURS=24\n"
            "CHROMA_DIR=.chroma_db\n",
            encoding="utf-8",
        )


def validate_token():
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

    print()
    if not token or token == "your_token_here" or token == "your_huggingface_token_here":
        print(
            "  ⚠️  HuggingFace token NOT configured.\n"
            "      1. Go to https://huggingface.co/settings/tokens\n"
            "      2. Create a free Read token\n"
            "      3. Paste it into .env as HUGGINGFACEHUB_API_TOKEN=hf_xxxx\n"
        )
    else:
        masked = token[:8] + "…" + token[-4:]
        print(f"  ✓ HUGGINGFACEHUB_API_TOKEN found ({masked})")


def verify_imports():
    print("\n  Verifying key imports…")
    checks = [
        ("langchain",              "langchain"),
        ("langchain_community",    "langchain-community"),
        ("langchain_huggingface",  "langchain-huggingface"),
        ("duckduckgo_search",      "duckduckgo-search"),
        ("chromadb",               "chromadb"),
        ("sentence_transformers",  "sentence-transformers"),
        ("rich",                   "rich"),
        ("dotenv",                 "python-dotenv"),
    ]
    all_ok = True
    for module, package in checks:
        try:
            __import__(module)
            print(f"    ✓ {package}")
        except ImportError:
            print(f"    ✗ {package}  ← run: pip install {package}")
            all_ok = False

    if not all_ok:
        print("\n  Some packages are missing. Re-run: pip install -r core/requirements.txt")
    else:
        print("\n  ✓ All imports verified.")


def main():
    print("\n" + "═" * 60)
    print("  AI Research Assistant — Environment Setup")
    print("═" * 60)

    check_python()
    install_packages()
    create_env_file()
    validate_token()
    verify_imports()

    print("\n" + "═" * 60)
    print("  Setup complete! Next steps:")
    print("  1. Add your HuggingFace token to .env")
    print("  2. Run:  python core/main.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
