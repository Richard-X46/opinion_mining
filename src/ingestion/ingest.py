from src.ingestion.praw_fetcher import RedditFetcher
from src.ingestion.judge import judge_threads
from src.graph.store import collection
from src.ingestion.embedder import get_embedder
import logging
import re


SUB_RESOLVE_LIMIT = 3
PER_SUB_LIMIT = 3
MAX_THREADS = 12


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


def ingest(query: str, session_id: str = "") -> list[dict]:
    """Strict reddit-only ingest: query -> subreddits -> threads -> judge -> index.

    Discovery is PRAW-native: the topic (question words stripped) resolves
    subreddits via title+description match, then each sub is searched with
    the full rewritten query (Reddit relevance ranking, no time window —
    the judge arbitrates freshness from thread content). No web search.
    """
    query = (query or "").strip()[:150]
    if not query:
        logging.warning("Ingest called with empty query, skipping.")
        return []
    logging.info(f"Ingesting posts for query: '{query}'")
    fetcher = RedditFetcher()
    # No keyword extraction: the full rewritten query goes to subreddit
    # discovery, and the LLM picks from real candidates (see resolve_subreddits).
    subs = fetcher.resolve_subreddits(query, limit=SUB_RESOLVE_LIMIT)
    if not subs:
        logging.warning(f"No subreddits resolved for '{query}', skipping thread search.")
        return []
    posts = fetcher.search_new(subs, query, limit_per_sub=PER_SUB_LIMIT)
    posts = posts[:MAX_THREADS]
    logging.info(f"Thread search: {len(posts)} threads from {subs} to enrich")

    posts = fetcher.enrich(posts)

    # Thread-level relevance gate: keep only threads that could resolve the
    # query (title + body + sample comments judged together). Fail-open —
    # on judge error every thread is kept, i.e. the old behavior.
    verdicts = judge_threads(query, posts)

    # flatten kept threads (each comment carries its thread context)
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
                })
        logging.info(f"Thread '{title[:60]}': PRAW fetched={fetched}, kept after clean={kept} [{verdict['reason']}]")

    if not comments:
        logging.warning(f"No comments to index ({kept_threads}/{len(posts)} threads kept by judge).")
        return posts

    texts = [c["text"] for c in comments]
    embeddings = get_embedder().encode(texts, show_progress_bar=True).tolist()

    # offset IDs so repeated ingests don't collide
    offset = collection.count()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"subreddit": c["subreddit"], "title": c["title"], "url": c["url"], "session_id": session_id} for c in comments],
        ids=[f"{session_id or 'global'}_{offset + i}" for i in range(len(comments))],
    )
    logging.info(f"Indexed {len(comments)} comments into ChromaDB (session={session_id or 'global'})")
    return posts