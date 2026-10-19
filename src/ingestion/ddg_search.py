from ddgs import DDGS
import logging
import re
from typing import List, Dict


def subreddit_from_url(url: str) -> str:
    m = re.search(r"reddit\.com/r/([^/]+)", url or "")
    return m.group(1) if m else "?"


class DDGSSearcher:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search DuckDuckGo and return results."""
        with DDGS() as ddgs:
            results = ddgs.text(f"site:reddit.com {query}", max_results=max_results)


            # filtering valid urls
            valid_results = {}
            raw_count = 0
            for _ in results:
                raw_count += 1
                # Must be a specific thread (has /comments/), not just a subreddit page
                if "reddit.com/r/" in _["href"] and "/comments/" in _["href"]:
                    valid_results[_["href"]] = _
            threads = list(valid_results.values())
            logging.info(f"DDGS: {raw_count} raw hits -> {len(threads)} reddit threads")
            for t in threads:
                logging.info(f"DDGS thread: r/{subreddit_from_url(t['href'])} | {t.get('title', '')[:80]} | {t['href']}")
            return threads


if __name__ == "__main__":
    query = "do androids dream of electric sheep"

    res = DDGSSearcher().search(query, max_results=10)
