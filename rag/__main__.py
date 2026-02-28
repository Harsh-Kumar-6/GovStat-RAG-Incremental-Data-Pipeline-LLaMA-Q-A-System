"""RAG CLI — python -m rag [serve|ingest]"""
from __future__ import annotations

import argparse
import json
import sys

from scraper.logger import get_logger

log = get_logger(__name__)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from scraper.config import settings
    uvicorn.run(
        "rag.api:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    from rag.indexer import get_store
    store = get_store()
    store.build()
    print(json.dumps({"status": "ok", "vectors": store.index.ntotal}))


def cmd_ask(args: argparse.Namespace) -> None:
    """Quick CLI query — useful for smoke-testing without the UI."""
    from rag.indexer import get_store
    from rag.llm import ask_llm

    store = get_store()
    store.load()
    chunks = store.search(args.question, k=args.k)
    answer = ask_llm(args.question, chunks)
    print("\nAnswer:", answer)
    print("\nSources:")
    for c in chunks:
        print(f"  - {c.get('title','?')}  {c.get('url','')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MoSPI RAG CLI")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start FastAPI server")
    serve_p.add_argument("--host",   default=None)
    serve_p.add_argument("--port",   type=int, default=None)
    serve_p.add_argument("--reload", action="store_true")

    sub.add_parser("ingest", help="Build / rebuild FAISS index")

    ask_p = sub.add_parser("ask", help="Ask a question from the CLI")
    ask_p.add_argument("question", type=str)
    ask_p.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()