from src.ingestion.praw_fetcher import RedditFetcher, keyword_queries
from src.ingestion.judge import judge_threads
from src.graph.store import collection
from src.ingestion.embedder import get_embedder
import logging
import re


SUB_RESOLVE_LIMIT = 3
PER_SUB_LIMIT = 5
MAX_THREADS = 20


POST_SNIPPET_CHARS = 300


def thread_doc(title: str, body: str, comment: str) -> str:
    """One indexable unit: the comment WITH its thread attached.

    A bare comment ("that fixed it for me") never matches the query by
    cosine similarity; prefixed with its thread title + post body, the
    embedding carries the context a human reader would use.
    """
    return clean_text(f"Thread: {title}\nPost: {body[:POST_SNIPPET_CHARS]}\nComment: {comment}")


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def ingest(query: str, session_id: str = "", subreddits: list[str] | None = None,
           anchor_subs: list[str] | None = None, raw_query: str = "",
           topic_id: str = "", topic: str = "") -> tuple[list[dict], list[str]]:
    """Strict reddit-only ingest: query -> subreddits -> threads -> judge -> index.

    Discovery is PRAW-native and deterministic: entity phrases are probed
    against real subreddit names, Reddit's own subreddit search supplies
    semantic candidates, and a position-weighted thread harvest adds evidence.
    No web search and no LLM veto over an exact name match.

    `subreddits`, when given, skips discovery entirely (user-picked subs).
    `anchor_subs` are the active topic's subs: follow-ups reuse them instead
    of drifting to a freshly discovered community.
    Returns (posts, subs_used).
    """
    query = (query or "").strip()[:150]
    if not query:
        logging.warning("Ingest called with empty query, skipping.")
        return [], []
    logging.info(f"Ingesting posts for query: '{query}'")
    fetcher = RedditFetcher()
    primary = ""
    if subreddits:
        subs = [s for s in subreddits if s]
        logging.info(f"Ingest using caller-provided subs (skipping discovery): {subs}")
    elif anchor_subs:
        subs = [s for s in anchor_subs if s]
        logging.info(f"Ingest reusing active topic anchor: {subs}")
    else:
        resolved = fetcher.resolve_subreddits_detailed(
            query, raw_query=raw_query, limit=SUB_RESOLVE_LIMIT
        )
        subs = resolved["subreddits"]
        primary = resolved.get("primary_entity", "")
    if not subs:
        logging.warning(f"No subreddits resolved for '{query}', skipping thread search.")
        return [], []
    # Reddit search is lexical. Inside a known sub, compact keywords
    # ("fertilizer") beat a long rewrite ("fertilizer effect in The Farmer
    # Was Replaced"); try those first, then the rewrite for coverage.
    search_queries: list[str] = []
    seen_queries: set[str] = set()

    def add_query(q: str) -> None:
        q = (q or "").strip()
        key = q.lower()
        if q and key not in seen_queries:
            seen_queries.add(key)
            search_queries.append(q)

    for q in keyword_queries(raw_query or query, query, primary_entity=primary):
        add_query(q)
    add_query(query)
    if primary:
        add_query(primary)

    posts, seen_hrefs = [], set()
    for q in search_queries:
        stubs = fetcher.search_new(subs, q, limit_per_sub=PER_SUB_LIMIT)
        added = 0
        for s in stubs:
            href = s.get("href")
            if not href or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            posts.append(s)
            added += 1
        logging.info(f"Thread search: '{q}' -> {added} new threads")
        if len(posts) >= MAX_THREADS:
            break
    posts = posts[:MAX_THREADS]
    logging.info(f"Thread search: {len(posts)} threads from {subs} to enrich")

    posts = fetcher.enrich(posts)

    # Thread-level relevance gate: keep only threads that could resolve the
    # query (title + body + sample comments judged together). Fail-open —
    # on judge error every thread is kept, i.e. the old behavior.
    verdicts = judge_threads(query, posts, topic=primary or topic)

    def _flatten(posts, verdicts):
        """Flatten kept threads (each comment carries its thread context)."""
        comments = []
        kept_threads = 0
        for post, verdict in zip(posts, verdicts):
            title = post.get("title", "")
            body = post.get("body", "") or ""
            fetched = len(post.get("comments", []))
            if not verdict["resolves"]:
                logging.info(f"Thread '{title[:60]}': DROPPED by judge — {verdict['reason']} (fetched={fetched})")
                continue
            kept_threads += 1
            kept = 0
            for comment in post.get("comments", []):
                raw = comment.get("body", "") if isinstance(comment, dict) else str(comment)
                url = comment.get("url", "") if isinstance(comment, dict) else post.get("href", "")
                text = thread_doc(title, body, raw)
                if len(text) > 20:
                    kept += 1
                    comments.append({
                        "text": text,
                        "subreddit": post.get("subreddit", ""),
                        "title": title,
                        "url": url or post.get("href", ""),
                        "thread_url": post.get("href", ""),
                    })
            logging.info(f"Thread '{title[:60]}': PRAW fetched={fetched}, kept after clean={kept} [{verdict['reason']}]")
        return comments, kept_threads

    comments, kept_threads = _flatten(posts, verdicts)

    if not comments:
        # Second chance: relevance ranking buries niche threads under popular
        # generic ones, so re-run the same query with recency order and let
        # the judge arbitrate the fresh batch instead.
        logging.info("Judge kept nothing — retrying thread search with sort='new'")
        seen_hrefs = {p.get("href") for p in posts}
        retry = []
        for q in search_queries:
            for p in fetcher.search_new(subs, q, limit_per_sub=PER_SUB_LIMIT, sort="new"):
                href = p.get("href")
                if href and href not in seen_hrefs:
                    seen_hrefs.add(href)
                    retry.append(p)
        if retry:
            posts = fetcher.enrich(retry)
            comments, kept_threads = _flatten(posts, judge_threads(query, posts, topic=primary or topic))

    if not comments:
        logging.warning(f"No comments to index ({kept_threads}/{len(posts)} threads kept by judge).")
        return posts, subs

    texts = [c["text"] for c in comments]
    embeddings = get_embedder().encode(texts, show_progress_bar=True).tolist()

    # offset IDs so repeated ingests don't collide
    offset = collection.count()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"subreddit": c["subreddit"], "title": c["title"], "url": c["url"], "thread_url": c.get("thread_url", ""), "session_id": session_id, "topic_id": topic_id} for c in comments],
        ids=[f"{session_id or 'global'}_{offset + i}" for i in range(len(comments))],
    )
    logging.info(f"Indexed {len(comments)} comments into ChromaDB (session={session_id or 'global'})")
    return posts, subs
