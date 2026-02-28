"""CLI entry points for scraper commands."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from scraper.config import settings
from scraper.crawl import crawl
from scraper.db import get_conn, init_db
from scraper.logger import get_logger
from scraper.parse import parse_all_pdfs

log = get_logger(__name__)


def cmd_crawl(args: argparse.Namespace) -> None:
    seed_urls: List[str] = args.seed_url or settings.seed_urls
    max_pages: int = args.max_pages or settings.max_pages_per_run
    docs = crawl(seed_urls=seed_urls, max_pages=max_pages)
    print(json.dumps({"status": "ok", "documents_scraped": len(docs)}))


def cmd_parse(args: argparse.Namespace) -> None:
    parse_all_pdfs()
    print(json.dumps({"status": "ok"}))


def cmd_report(args: argparse.Namespace) -> None:
    db_path = settings.db_path
    init_db(db_path)
    with get_conn(db_path) as conn:
        n_docs   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_files  = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_pdfs   = conn.execute("SELECT COUNT(*) FROM files WHERE file_type='pdf'").fetchone()[0]
        n_tables = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        by_cat   = conn.execute(
            "SELECT category, COUNT(*) AS cnt FROM documents GROUP BY category ORDER BY cnt DESC LIMIT 10"
        ).fetchall()

    report = {
        "documents":        n_docs,
        "files":            n_files,
        "pdfs":             n_pdfs,
        "extracted_tables": n_tables,
        "top_categories":   [{"category": r[0], "count": r[1]} for r in by_cat],
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MoSPI scraper CLI")
    sub = parser.add_subparsers(dest="command")

    crawl_p = sub.add_parser("crawl", help="Crawl MoSPI pages")
    crawl_p.add_argument("--seed-url", nargs="*", help="Override seed URLs")
    crawl_p.add_argument("--max-pages", type=int, help="Max listing pages per seed")

    sub.add_parser("parse",  help="Download PDFs and extract text/tables")
    sub.add_parser("report", help="Show run summary")

    args = parser.parse_args()

    if args.command == "crawl":
        cmd_crawl(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()