import praw
import datetime
import logging
from typing import List, Dict
from src.config.settings import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_USERNAME,
    REDDIT_PASSWORD,
)
from src.ingestion.ddg_search import DDGSSearcher
import json
import re


SUB_RESOLVE_LIMIT = 3
SUGGEST_PER_QUERY = 2
MAX_SUBS_TOTAL = 4
SUB_SEARCH_LIMIT = 100
SUB_COARSE_LIMIT = 20


class RedditFetcher:
    """Fetches comments from Reddit posts via PRAW."""

    def __init__(self, comment_limit: int = 20, min_score: int = 1, min_length: int = 25):
        self.comment_limit = comment_limit
        self.min_score = min_score
        self.min_length = min_length
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
        )

    def _is_valid_comment(self, comment) -> bool:
        return (
            not comment.stickied
            and comment.body not in ("[deleted]", "[removed]")
            and comment.score >= self.min_score
            and len(comment.body.strip()) >= self.min_length
        )

    def fetch_thread(self, url: str) -> Dict:
        """Fetch full thread context: title, post body, subreddit, stats + top comments.

        Comments alone lose the context that makes them meaningful
        ("that fixed it for me" resolves nothing without the thread title),
        so ingest judges and indexes at thread level, not comment level.
        """
        try:
            submission = self.reddit.submission(url=url)
            submission.comment_sort = "best"
            try:
                created = int(getattr(submission, "created_utc", 0) or 0)
                date = (
                    datetime.datetime.fromtimestamp(created, datetime.timezone.utc).strftime("%Y-%m-%d")
                    if created else ""
                )
                thread = {
                    "title": submission.title or "",
                    "body": submission.selftext or "",
                    "subreddit": submission.subreddit.display_name if submission.subreddit else "",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "date": date,
                }
                logging.info(
                    f"PRAW thread: r/{thread['subreddit']} | {date or 'undated'} "
                    f"| score={thread['score']} comments={thread['num_comments']} "
                    f"| {thread['title'][:80]}"
                )
            except Exception:
                thread = {"title": "", "body": "", "subreddit": "", "score": 0, "num_comments": 0, "date": ""}
            submission.comments.replace_more(limit=0)
            results = []
            for c in submission.comments.list():
                if not self._is_valid_comment(c):
                    continue
                try:
                    permalink = f"https://www.reddit.com{c.permalink}"
                except Exception:
                    permalink = url
                results.append({"body": c.body.strip(), "url": permalink})
                if len(results) >= self.comment_limit:
                    break
            thread["comments"] = results
            return thread
        except Exception as e:
            logging.warning(f"Failed to fetch thread {url}: {e}")
            return {"title": "", "body": "", "subreddit": "", "score": 0, "num_comments": 0, "date": "", "comments": []}

    def fetch(self, url: str) -> List[Dict]:
        """Fetch top comments as {body, url} dicts (url = comment permalink)."""
        return self.fetch_thread(url)["comments"]

    def resolve_subreddits(self, query: str, limit: int = SUB_RESOLVE_LIMIT) -> List[str]:
        """Pick subreddits via exhaustive PRAW discovery + Jev two-pass selection.

        1. `subreddits.search(query)` returns up to SUB_SEARCH_LIMIT real
           candidates by name/title/description — kept in full, no truncation.
        2. A proper-noun probe extracts named tokens from the query (e.g.
           "Oshawa", "One Piece") and appends any that resolve to a real
           subreddit, since whole-phrase search misses location/franchise subs.
        3. Jev (TypeSafe classifier) scores the exhaustive list in one batched
           Noul call, keeps the top SUB_COARSE_LIMIT, re-asks on that shortlist,
           and keeps the top `limit`. On any Jev failure it falls back to
           PRAW's relevance order.
        """
        try:
            found = list(self.reddit.subreddits.search(query, limit=SUB_SEARCH_LIMIT))
        except Exception as e:
            logging.warning(f"Subreddit discovery failed for {query!r}: {e}")
            return []
        cands, seen = [], set()
        for s in found:
            try:
                name = s.display_name
            except Exception:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                subs_count = s.subscribers
            except Exception:
                subs_count = None
            if subs_count is None:
                logging.info(f"Sub resolve: skip r/{name} (no subscriber count, likely inaccessible)")
                continue
            cands.append({
                "name": name,
                "title": s.title or "",
                "desc": (s.public_description or "")[:300],
                "subs": subs_count,
            })
        for name in self._probe_named_subs(query, seen):
            try:
                s = self.reddit.subreddit(name)
                subs_count = s.subscribers
                display = s.display_name or name
            except Exception:
                continue
            if subs_count is None or display.lower() in seen:
                continue
            seen.add(display.lower())
            cands.append({
                "name": display,
                "title": s.title or display,
                "desc": (s.public_description or "")[:300],
                "subs": subs_count,
            })
            logging.info(f"Sub resolve: probed r/{display} (subs={subs_count})")
        if not cands:
            return []
        picked = self._jev_pick(query, cands, limit)
        kept = {n.lower() for n in picked}
        for name in picked:
            logging.info(f"Sub resolve: KEEP r/{name} (jev)")
        for c in cands:
            if c["name"].lower() not in kept:
                logging.info(f"Sub resolve: DROP r/{c['name']}")
        return picked

    def _probe_named_subs(self, query: str, seen: set) -> List[str]:
        """Capitalised runs (up to 3 words) that might name a subreddit."""
        runs = re.findall(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,2}\b", query)
        out, dedup = [], set(seen)
        for run in runs:
            key = run.strip().lower()
            if key and key not in dedup:
                dedup.add(key)
                out.append(run.strip())
        return out

    def _jev_pick(self, query: str, cands: List[Dict], limit: int) -> List[str]:
        """Two-pass Jev selection: exhaustive list -> top SUB_COARSE_LIMIT -> top `limit`.

        Pass 1 scores every candidate for broad relevance and keeps the top
        SUB_COARSE_LIMIT. Pass 2 re-asks on the shortlist with a stricter
        subject-specificity question and keeps only those clearing
        JUDGE_THRESHOLD (up to `limit`) — so a query with one true home
        returns one sub instead of padding to three with wrong-city lookalikes.
        Fails open to PRAW relevance order on any Jev error or empty scores.
        """
        fallback = [c["name"] for c in cands[:limit]]
        try:
            from src.graph.jev import batch_noul
            from src.config.settings import JUDGE_THRESHOLD

            def _score(subset: List[Dict], template: str) -> dict:
                state = {"query": query, "candidates": subset}
                questions = {
                    c["name"]: template.format(
                        name=c["name"], title=c["title"][:120], desc=c["desc"][:200]
                    )
                    for c in subset
                }
                return batch_noul(state, questions)

            probs1 = _score(
                cands,
                "Consider the subreddit r/{name} (title: {title}). Description: {desc}. "
                "Does it host discussion that could help answer the user's question?",
            )
            if not probs1:
                raise ValueError("pass 1 returned no scores")
            coarse = sorted(
                cands, key=lambda c: probs1.get(c["name"], -1.0), reverse=True
            )[:SUB_COARSE_LIMIT]

            probs2 = _score(
                coarse,
                "Of the candidate subreddits, is r/{name} (title: {title}) about the "
                "specific subject the question names — the exact place, franchise, team, "
                "or product — and not merely the general topic? A sub about the same "
                "topic but a different specific subject is NOT a match.",
            )
            shortlist = sorted(
                coarse, key=lambda c: probs2.get(c["name"], -1.0), reverse=True
            )
            picked = [
                c["name"]
                for c in shortlist
                if probs2.get(c["name"], 0.0) >= JUDGE_THRESHOLD
            ][:limit]
            if not picked:
                picked = [c["name"] for c in shortlist[:limit]]
            return picked
        except Exception as e:
            logging.warning(f"Sub pick Jev fail-open ({e}): using PRAW relevance order")
            return fallback

    def search_new(self, subreddits: List[str], query: str, limit_per_sub: int = 3,
                   sort: str = "relevance", time_filter: str = "all") -> List[Dict]:
        """Search subreddits directly for threads matching the query.

        Defaults: Reddit's own relevance ranking, no time window (no
        hardcoded recency bias — the judge arbitrates freshness from thread
        content instead). Returns unenriched stubs ({href, title,
        subreddit}) for enrich() to flesh out via fetch_thread.
        """
        stubs, seen = [], set()
        for sub in subreddits:
            try:
                for s in self.reddit.subreddit(sub).search(
                    query, sort=sort, time_filter=time_filter, limit=limit_per_sub
                ):
                    href = f"https://www.reddit.com{s.permalink}"
                    if href in seen:
                        continue
                    seen.add(href)
                    stubs.append({"href": href, "title": s.title or "", "subreddit": sub})
                logging.info(f"PRAW sub search: r/{sub} ({sort}/{time_filter}) -> {sum(1 for x in stubs if x['subreddit'] == sub)} threads")
            except Exception as e:
                logging.warning(f"PRAW fresh search failed on r/{sub}: {e}")
        return stubs

    def enrich(self, posts: List[Dict]) -> List[Dict]:
        for i, post in enumerate(posts, 1):
            logging.info(f"[{i}/{len(posts)}] Fetching: {post.get('title', '')[:60]} | {post.get('href', '')}")
            thread = self.fetch_thread(post["href"])
            # PRAW's real title beats the search stub title (often just "Link to reddit.com").
            if thread["title"]:
                post["title"] = thread["title"]
            if thread["subreddit"] and not post.get("subreddit"):
                post["subreddit"] = thread["subreddit"]
            post["body"] = thread["body"]
            post["score"] = thread["score"]
            post["num_comments"] = thread["num_comments"]
            post["date"] = thread["date"]
            post["comments"] = thread["comments"]
            logging.info(f"[{i}/{len(posts)}] Got {len(post['comments'])} usable comments (min_score={self.min_score}, limit={self.comment_limit})")
        total = sum(len(p.get("comments", [])) for p in posts)
        logging.info(f"PRAW summary: {total} comments from {len(posts)} threads ({sum(1 for p in posts if p.get('comments'))} threads non-empty)")
        return posts


if __name__ == "__main__":
    query = "do androids dream of electric sheep"
    searcher = DDGSSearcher()
    fetcher = RedditFetcher()
    dir(fetcher)
    posts = searcher.search(query, max_results=5)
    enriched_posts = fetcher.enrich(posts)
    # json loads indent=2 for readability
    print(json.dumps(enriched_posts, indent=2))

    