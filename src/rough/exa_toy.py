"""Toy with the Exa Search API.

Usage:
    uv run python src/rough/exa_toy.py "latest One Piece chapter spoilers"
    uv run python src/rough/exa_toy.py "best lens for portrait photography" --since "" --count 5
    uv run python src/rough/exa_toy.py "query" --domains reddit.com --text

Needs EXA_API_KEY in .env (https://dashboard.exa.ai -> API Keys).
Docs: https://exa.ai/docs/reference/search
"""
import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "https://api.exa.ai/search"


def search(query: str, count: int = 10, search_type: str = "instant",
           domains: list[str] | None = None, since: str = "",
           with_text: bool = False) -> dict:
    key = os.getenv("EXA_API_KEY", "")
    if not key:
        raise SystemExit("EXA_API_KEY not set (check .env)")
    body: dict = {
        "query": query,
        "type": search_type,
        "numResults": max(1, min(100, count)),
    }
    if domains:
        body["includeDomains"] = domains
    if since:
        body["startPublishedDate"] = since  # ISO 8601, e.g. 2026-09-20
    if with_text:
        body["contents"] = {"text": True}
    r = httpx.post(
        ENDPOINT,
        headers={"x-api-key": key, "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description="Exa search toy")
    ap.add_argument("query", help="search query")
    ap.add_argument("--count", type=int, default=10, help="results 1-100 (default 10)")
    ap.add_argument("--type", default="instant",
                    help="instant|fast|auto|deep-lite|deep (default instant: cheapest/fastest)")
    ap.add_argument("--domains", default="reddit.com",
                    help="comma-separated includeDomains (default reddit.com; empty = no filter)")
    ap.add_argument("--since", default="",
                    help="only results published after this ISO date, e.g. 2026-09-20 (default none)")
    ap.add_argument("--text", action="store_true", help="fetch full page text (costs more)")
    ap.add_argument("--raw", action="store_true", help="dump raw JSON response")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    result = search(args.query, count=args.count, search_type=args.type,
                    domains=domains, since=args.since, with_text=args.text)

    if args.raw:
        print(json.dumps(result, indent=2)[:4000])
        return

    print(f"cost: {result.get('costDollars', {})}")
    results = result.get("results", []) or []
    print(f"{len(results)} results (type={args.type}, domains={domains or 'any'}, since={args.since or 'none'})")
    for i, r in enumerate(results, 1):
        body = r.get("text") or " | ".join(r.get("highlights", []) or [])
        print(f"\n[{i}] {r.get('title', '')[:90]}")
        print(f"    {r.get('url', '')}")
        print(f"    date={r.get('publishedDate') or 'unknown'}")
        if body:
            print(f"    {body[:280].replace(chr(10), ' ')}")


if __name__ == "__main__":
    sys.exit(main())
