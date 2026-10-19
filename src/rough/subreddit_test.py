"""Strict reddit-only inference test: query -> subreddits -> threads in subreddits.

Usage:
    uv run python -m src.rough.subreddit_test   # edit QUERY below to try another topic

Needs Reddit creds in .env (same as the app). No DDGS, no web search.
"""
import logging
import re

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.praw_fetcher import RedditFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

QUERY = "latest chapter spoilers for onepiece"
N_SUBS = 5
N_THREADS = 5

# Words that describe the *question*, not the *topic*. Stripped before
# subreddit discovery so generic manga-sub descriptions can't match on them.
STOPWORDS = frozenset(
    "whats what which where when who whom whose is are was were the a an "
    "latest newest recent current new just released this week "
    "about on for of to in and or vs best good top buy buying "
    "chapter episode episodes spoiler spoilers leak leaks plot summary thread "
    "manga anime show series".split()
)


def topic_of(query: str) -> str:
    words = [
        w for w in re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
        if w not in STOPWORDS and len(w) > 2
    ]
    return " ".join(words)


def search_subreddit(query: str, n_subs: int = N_SUBS) -> list[str]:
    """Resolve a query to subreddits via title+description match. Returns names."""
    topic = topic_of(query)
    logging.info(f"--- search_subreddit (topic={topic!r}) ---")
    fetcher = RedditFetcher()
    subs = list(fetcher.reddit.subreddits.search(topic))
    logging.info(f"{len(subs)} subreddits found for {query!r}")
    # Keep PRAW's relevance order: sorting by size surfaces generic giants
    # (r/Naruto, r/southpark) above the actually topical subs.
    ranked = subs[:n_subs]
    for s in ranked:
        desc = (s.public_description or "").replace("\n", " ")[:100]
        logging.info(f"r/{s.display_name} | subs={s.subscribers} | {desc}")
    return [s.display_name for s in ranked]


def search_in_subreddit(query: str, subreddits: list[str] | str, n_threads: int = N_THREADS) -> None:
    """Search fresh threads within the given subreddits, newest first."""
    if isinstance(subreddits, str):
        subreddits = [subreddits]  # iterating a bare string yields characters
    logging.info("--- search_in_subreddit ---")
    fetcher = RedditFetcher()
    for sub in subreddits:
        logging.info(f"--- r/{sub} sorted=new ---")
        try:
            hits = list(
                fetcher.reddit.subreddit(sub).search(
                    query, sort="new", time_filter="month", limit=n_threads
                )
            )
        except Exception as e:  # noqa: BLE001 - toy, log and move on
            logging.warning(f"search failed: {e}")
            continue
        if not hits:
            logging.info("(no threads)")
        for h in hits:
            logging.info(f"{h.title[:75]} | https://www.reddit.com{h.permalink}")


if __name__ == "__main__":

    QUERY = "real madrid last game"
    names = search_subreddit(QUERY)
    names
    search_in_subreddit(QUERY, "soccer")
