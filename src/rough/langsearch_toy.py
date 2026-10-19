"""Toy with the LangSearch Web Search API.

Usage:
    uv run python src/rough/langsearch_toy.py "latest One Piece chapter spoilers"
    uv run python src/rough/langsearch_toy.py "best lens for portrait photography" --freshness noLimit --count 5
    uv run python src/rough/langsearch_toy.py "query" --domains reddit.com --text

Needs LANGSEARCH_API_KEY in .env (Dashboard -> API keys).
Docs: https://docs.langsearch.com/reference/search-api-guide
"""
import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "https://api.langsearch.com/v1/web-search"


def search(query: str, count: int = 10, freshness: str = "oneMonth",
           domains: list[str] | None = None, with_text: bool = False,
           max_chars: int = 3000) -> dict:
    key = os.getenv("LANGSEARCH_API_KEY", "")
    if not key:
        raise SystemExit("LANGSEARCH_API_KEY not set (check .env)")
    body: dict = {"query": query, "count": count, "freshness": freshness}
    if domains:
        body["includeDomains"] = domains
    if with_text:
        body["contents"] = {"text": {"maxCharacters": max_chars}}
    r = httpx.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description="LangSearch toy")
    ap.add_argument("query", help="search query")
    ap.add_argument("--count", type=int, default=10, help="results 1-50 (default 10)")
    ap.add_argument("--freshness", default="oneMonth",
                    help="noLimit|oneDay|oneWeek|oneMonth|oneYear|YYYY-MM-DD|range (default oneMonth)")
    ap.add_argument("--domains", default="reddit.com",
                    help="comma-separated includeDomains (default reddit.com; empty = no filter)")
    ap.add_argument("--text", action="store_true", help="fetch full page text (costs more tokens)")
    ap.add_argument("--raw", action="store_true", help="dump raw JSON response")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    result = search(args.query, count=args.count, freshness=args.freshness,
                    domains=domains, with_text=args.text)

    if args.raw:
        print(json.dumps(result, indent=2)[:4000])
        return

    usage = result.get("usage", {})
    print(f"tokens in={usage.get('input_tokens')} out={usage.get('output_tokens')}")
    pages = (result.get("data", {}).get("webPages", {}).get("value", []) or [])
    print(f"{len(pages)} results (freshness={args.freshness}, domains={domains or 'any'})")
    for i, p in enumerate(pages, 1):
        body = p.get("text") or p.get("snippet", "")
        print(f"\n[{i}] {p.get('name', '')[:90]}")
        print(f"    {p.get('url', '')}")
        print(f"    date={p.get('datePublished') or 'unknown'}")
        print(f"    {body[:280].replace(chr(10), ' ')}")


if __name__ == "__main__":
    sys.exit(main())
