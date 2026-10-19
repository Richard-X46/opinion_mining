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
import math
import re


SUB_RESOLVE_LIMIT = 3
SUB_ENTITY_LIMIT = 4
SUB_SEMANTIC_LIMIT = 10
SUB_HARVEST_LIMIT = 15
SUB_MIN_SCORE = 0.32

# Generic query filler — not content. Used only to decide which phrases are
# worth probing as entity names. Reddit remains the source of truth.
_GENERIC = frozenset(
    "what whats which where when who whom whose why how "
    "is are was were be been being am do does did doing "
    "the a an this that these those there here it its "
    "i you your my me we our us they them their he she his her "
    "of to in on for with about from by as at into over after before "
    "and or but not no yes so if then than just also very really "
    "any some can could should would will may might must "
    "please thanks thank help useful use using tips tip advice recommend recommendations "
    "latest newest recent currently current today now new news "
    "best top good great worth buy buying "
    "strategy strategies solution solutions approach ways "
    "game games gaming".split()
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _normalize_name(text: str) -> str:
    """Reddit names are alphanumeric; strip case and punctuation for comparison."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if len(w) > 1 and w not in _GENERIC]


def entity_candidates(texts: list[str], limit: int = SUB_ENTITY_LIMIT) -> list[str]:
    """Phrases from the question worth probing against real subreddit names.

    Delimited chunks (the part after a dash/comma) are the strongest signal:
    "there is this game - the farmer was replaced, any tips?" gives the exact
    game name without an LLM. Sliding n-grams cover queries with no
    punctuation. Nothing is assumed about the subject; Reddit verifies it.
    """
    priority: list[tuple[float, str]] = []
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()

    def add(phrase: str, bonus: float = 0.0, strong: bool = False) -> None:
        phrase = re.sub(r"\s+", " ", (phrase or "")).strip(" .,!?:;-\"'")
        if not phrase or len(phrase) > 80:
            return
        norm = _normalize_name(phrase)
        if len(norm) < 3 or norm in seen:
            return
        words = _words(phrase)
        content = [w for w in words if len(w) > 1 and w not in _GENERIC]
        if not content:
            return
        seen.add(norm)
        score = (
            2.0 * len(content)
            + 0.5 * min(len(words), 6)
            + 0.1 * min(len(max(content, key=len)), 12)
            + bonus
        )
        (priority if strong else scored).append((score, phrase))

    for text in texts:
        if not text:
            continue
        for quoted in re.findall(r'"([^"]{2,80})"', text):
            add(quoted, bonus=3.0, strong=True)
        for segment in re.split(r"[-,;:!?()\[\]{}|/\n]+", text):
            # Dashes/commas usually isolate the actual subject from the
            # question boilerplate, so keep those phrases ahead of n-grams.
            add(segment, bonus=2.0, strong=True)
        words = _words(text)
        for n in range(min(5, len(words)), 0, -1):
            for i in range(0, len(words) - n + 1):
                add(" ".join(words[i:i + n]))

    priority.sort(key=lambda item: (-item[0], -len(item[1])))
    scored.sort(key=lambda item: (-item[0], -len(item[1])))
    return [phrase for _, phrase in (priority + scored)[:limit]]


def keyword_queries(raw_query: str, rewritten_query: str,
                    primary_entity: str = "", limit: int = 2) -> list[str]:
    """Compact, entity-free terms for searching inside a known subreddit.

    Reddit search is lexical: a long rewrite like "fertilizer effect in The
    Farmer Was Replaced" returns unrelated threads, while "fertilizer" finds
    the right ones. Remove the already-resolved entity and the question
    filler, keep the specific object terms.
    """
    excluded = set(_words(primary_entity)) if primary_entity else set()

    def terms(text: str) -> list[str]:
        return [w for w in _content_words(text) if w not in excluded]

    out: list[str] = []

    def add(words: list[str]) -> None:
        words = list(dict.fromkeys(words))[:4]
        if not words:
            return
        query = " ".join(words)
        if query not in out:
            out.append(query)

    raw_terms = terms(raw_query)
    rewritten_terms = terms(rewritten_query)
    add(raw_terms[-4:])
    add(rewritten_terms[-4:])
    all_terms = list(dict.fromkeys(raw_terms + rewritten_terms))
    if all_terms:
        # One high-recall term: "fertilizer", "pumpkin", "spawned".
        add([max(all_terms, key=len)])
    return out[:limit]


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

    def resolve_subreddits_detailed(
        self,
        query: str,
        raw_query: str = "",
        limit: int = SUB_RESOLVE_LIMIT,
    ) -> Dict:
        """Resolve a question to subreddits using Reddit itself as ground truth.

        Candidates come from three independent signals:
        1. Exact/prefix name probes for entity phrases ("the farmer was
           replaced" -> r/TheFarmerWasReplaced).
        2. Reddit's subreddit search (title + description), which handles
           aliases and communities whose name is not the subject ("man
           united" -> r/reddevils).
        3. A bounded site-wide thread harvest, weighted by result position
           rather than raw hit count, so prolific generic subs cannot drown
           out a niche one.

        Scoring is deterministic; no LLM gets veto power over an exact name
        match. Subscriber count is only a weak tiebreak.
        """
        entities = entity_candidates([raw_query, query])
        cands: dict[str, Dict] = {}

        for entity in entities:
            self._probe_exact_name(entity, cands)
        # Semantic search is noisier, so spend it on the strongest phrases only.
        for entity in entities[:2]:
            self._search_subreddits(entity, cands)
        self._search_subreddits(query, cands)
        self._harvest_threads(query, cands)
        if len(cands) < 2 and raw_query and raw_query != query:
            self._harvest_threads(raw_query, cands)

        # If a phrase literally names a subreddit, that phrase becomes the
        # topic anchor. Long diluted rewrites ("steam farm simulation game
        # farmer replaced tips") must not lend their generic tokens ("steam")
        # the same weight as the exact name.
        exact_entities: list[str] = []
        for c in cands.values():
            for entity in c.get("exact_entities", []):
                if entity not in exact_entities:
                    exact_entities.append(entity)
        primary = (
            min(exact_entities, key=lambda e: (len(_words(e)), len(_normalize_name(e))))
            if exact_entities else ""
        )
        scoring_entities = [primary] if primary else entities

        query_words = set(_content_words(query) or _words(query))
        for c in cands.values():
            c["name_score"] = self._name_score(c["name"], scoring_entities)
            c["source_score"] = self._source_score(c)
            c["desc_score"] = 0.0
            c["score"] = (
                0.50 * c["name_score"]
                + 0.25 * c["source_score"]
                + 0.10 * self._subscriber_score(c["subs"])
            )

        # Fetch descriptions only where they can change the ranking.
        for c in sorted(cands.values(), key=lambda x: x["score"], reverse=True)[:8]:
            self._enrich_candidate(c)
            c["desc_score"] = self._desc_score(c, scoring_entities, query_words)
            c["score"] += 0.15 * c["desc_score"]

        ranked = sorted(cands.values(), key=lambda x: x["score"], reverse=True)
        best = ranked[0]["score"] if ranked else 0.0
        # With an exact anchor, hold secondary candidates to a relevance
        # relative to it. Without one, use the absolute floor.
        min_score = SUB_MIN_SCORE if not primary else max(SUB_MIN_SCORE, 0.45 * best)
        picked: List[Dict] = []
        picked_keys: set[str] = set()

        def take(c: Dict) -> None:
            key = _normalize_name(c["name"])
            if key in picked_keys:
                return
            picked_keys.add(key)
            picked.append(c)

        # Exact name matches are anchors: they cannot be voted out by a
        # classifier, subscriber count, or a generic sub's popularity.
        for c in ranked:
            if c.get("exact_entities") and len(picked) < limit:
                take(c)
        for c in ranked:
            if len(picked) >= limit:
                break
            if c["score"] >= min_score:
                take(c)
        if not picked and ranked:
            take(ranked[0])

        for c in ranked[:10]:
            state = "KEEP" if _normalize_name(c["name"]) in picked_keys else "drop"
            logging.info(
                f"Sub resolve: {state} r/{c['name']} score={c['score']:.2f} "
                f"(name={c['name_score']:.2f} src={c['source_score']:.2f} "
                f"desc={c['desc_score']:.2f} subs={c['subs']}) via {','.join(c['sources'])}"
            )
        return {
            "subreddits": [c["name"] for c in picked],
            "entities": entities,
            "primary_entity": primary,
            "candidates": ranked,
        }

    def resolve_subreddits(self, query: str, raw_query: str = "",
                           limit: int = SUB_RESOLVE_LIMIT) -> List[str]:
        """Backward-compatible name-only wrapper."""
        return self.resolve_subreddits_detailed(query, raw_query=raw_query, limit=limit)["subreddits"]

    def _add_candidate(self, cands: dict, sub, source: str, rank: int) -> None:
        try:
            name = sub.display_name
            subs = sub.subscribers
        except Exception:
            return
        if not name or subs is None:
            return
        key = _normalize_name(name)
        if not key:
            return
        c = cands.get(key)
        if c is None:
            c = {
                "name": name,
                "subs": subs,
                "sources": [],
                "best_rank": rank,
                "title": "",
                "desc": "",
                "exact_entities": [],
                "_sub": sub,
            }
            cands[key] = c
        c["best_rank"] = min(c["best_rank"], rank)
        if source not in c["sources"]:
            c["sources"].append(source)

    def _probe_exact_name(self, phrase: str, cands: dict) -> None:
        """Ask Reddit whether this phrase is literally a subreddit name."""
        norm = _normalize_name(phrase)
        if len(norm) < 3:
            return
        try:
            found = self.reddit.subreddits.search_by_name(norm, exact=True)
        except Exception:
            return
        for s in found:
            self._add_candidate(cands, s, "exact_name", 0)
            key = _normalize_name(getattr(s, "display_name", ""))
            if key in cands and phrase not in cands[key]["exact_entities"]:
                cands[key]["exact_entities"].append(phrase)

    def _search_subreddits(self, phrase: str, cands: dict) -> None:
        try:
            found = list(self.reddit.subreddits.search(phrase, limit=SUB_SEMANTIC_LIMIT))
        except Exception as e:
            logging.warning(f"Sub resolve: subreddit search failed for {phrase!r}: {e}")
            return
        for rank, s in enumerate(found):
            self._add_candidate(cands, s, "subreddit_search", rank)

    def _harvest_threads(self, query: str, cands: dict) -> None:
        """Rank subs by where matching threads actually live (position-weighted)."""
        try:
            stubs = list(self.reddit.subreddit("all").search(query, limit=SUB_HARVEST_LIMIT))
        except Exception as e:
            logging.warning(f"Sub resolve: thread harvest failed: {e}")
            return
        for rank, s in enumerate(stubs):
            try:
                sub = s.subreddit
            except Exception:
                continue
            self._add_candidate(cands, sub, "thread_harvest", rank)

    @staticmethod
    def _name_score(name: str, entities: list[str]) -> float:
        n = _normalize_name(name)
        if not n:
            return 0.0
        best = 0.0
        for entity in entities:
            e = _normalize_name(entity)
            if not e:
                continue
            if n == e:
                score = 1.0
            elif e in n or n in e:
                score = 0.70
            else:
                et = set(_content_words(entity))
                nt = set(_content_words(name))
                score = len(et & nt) / len(et) if et else 0.0
            best = max(best, score)
        return best

    @staticmethod
    def _source_score(c: Dict) -> float:
        bases = {
            "exact_name": 1.0,
            "subreddit_search": 0.82,
            "thread_harvest": 0.62,
        }
        best = max((bases.get(s, 0.4) for s in c["sources"]), default=0.0)
        return max(0.0, best - min(c["best_rank"], 10) * 0.02)

    @staticmethod
    def _subscriber_score(subs: int) -> float:
        return min(1.0, math.log10(max(int(subs or 0), 1)) / 7.0)

    def _enrich_candidate(self, c: Dict) -> None:
        sub = c.get("_sub")
        if sub is None:
            return
        try:
            c["title"] = sub.title or ""
        except Exception:
            pass
        try:
            c["desc"] = (sub.public_description or "")[:500]
        except Exception:
            pass

    @staticmethod
    def _desc_score(c: Dict, entities: list[str], query_words: set[str]) -> float:
        text = f"{c.get('title', '')} {c.get('desc', '')}"
        target = set(_content_words(" ".join(entities))) | set(query_words)
        if not target:
            return 0.0
        return len(set(_content_words(text)) & target) / len(target)

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
    fetcher = RedditFetcher()
    query = "farmer replaced game tips"
    print(fetcher.resolve_subreddits_detailed(query))