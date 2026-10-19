"""Thread-level relevance gate.

Cosine similarity between the query and an isolated comment cannot tell
whether the comment *resolves* the query — the context that makes a comment
meaningful lives in its thread (title + post body + sibling comments).
So before indexing, one cheap batched LLM call judges each thread:
does this thread contain material that could help answer the question?

Fail-open by design: any error (LLM down, bad JSON, disabled) keeps every
thread, i.e. today's behavior. A wrongly dropped thread is unrecoverable;
a wrongly kept thread just costs embedding + retrieval noise.
"""
import datetime
import json
import logging
import re
from typing import List, Dict

from src.config.settings import JUDGE_MODEL, JUDGE_ENABLED, DECISION_BACKEND, JUDGE_THRESHOLD

MAX_JUDGE_COMMENTS = 8
JUDGE_COMMENT_CHARS = 150
JUDGE_BODY_CHARS = 500


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def thread_brief(index: int, post: Dict) -> str:
    """Compact thread summary sized for a batched judge call."""
    lines = [
        f"Thread {index}:",
        f"Date: {post.get('date') or 'unknown'}",
        f"Title: {(post.get('title') or '')[:200]}",
        f"Subreddit: r/{post.get('subreddit') or '?'} "
        f"(score={post.get('score', 0)}, comments={post.get('num_comments', 0)})",
        f"Post: {((post.get('body') or '')[:JUDGE_BODY_CHARS])}",
        "Top comments:",
    ]
    for c in (post.get("comments") or [])[:MAX_JUDGE_COMMENTS]:
        body = c.get("body", "") if isinstance(c, dict) else str(c)
        lines.append(f"- {body[:JUDGE_COMMENT_CHARS]}")
    return "\n".join(lines)


JUDGE_SYSTEM = (
    "You are a relevance filter for a Reddit opinion-mining pipeline. "
    "Reply with a JSON array ONLY, no prose, no code fences."
)


def judge_prompt(query: str, briefs: List[str], topic: str = "") -> str:
    joined = "\n\n".join(briefs)
    topic_line = f"Active topic/entity: {topic}\n" if topic else ""
    return f"""Today is {_today()}.

{topic_line}User question (standalone, pronouns already resolved): {query}

Below are {len(briefs)} Reddit threads. For EACH thread, decide: does it contain opinions, experiences, facts, or summaries that could help answer the question? A thread about a different chapter, episode, product, or topic does NOT resolve it, even within the same franchise. If the question asks for the latest/current/today's anything, a thread that is weeks or months old does NOT resolve it — drop it.

Reply with a JSON array ONLY — one object per thread, in order:
[{{"resolves": true, "reason": "<=15 words"}}, ...]

Threads:
{joined}"""


def parse_verdicts(text: str, n: int) -> List[Dict] | None:
    """Parse the judge reply; None if unusable (caller fails open)."""
    try:
        m = re.search(r"\[.*\]", text.strip(), re.DOTALL)
        if not m:
            return None
        verdicts = json.loads(m.group(0))
        if not isinstance(verdicts, list) or len(verdicts) != n:
            return None
        out = []
        for v in verdicts:
            out.append({
                "resolves": bool(v.get("resolves", True)),
                "reason": str(v.get("reason", ""))[:120],
            })
        return out
    except Exception:
        return None


def _fail_open(posts: List[Dict], reason: str) -> List[Dict]:
    logging.warning(f"Thread judge fail-open ({reason}): keeping all {len(posts)} threads")
    return [{"resolves": True, "reason": reason} for _ in posts]


def _judge_jev(query: str, posts: List[Dict], topic: str = "") -> List[Dict]:
    """Jev path: one batched Noul per thread, dates + today in the state."""
    from src.graph.jev import batch_noul
    today = _today()
    threads = []
    for p in posts:
        comments = []
        for c in (p.get("comments") or [])[:MAX_JUDGE_COMMENTS]:
            body = c.get("body", "") if isinstance(c, dict) else str(c)
            comments.append(body[:JUDGE_COMMENT_CHARS])
        threads.append({
            "date": p.get("date") or "unknown",
            "title": (p.get("title") or "")[:200],
            "post": (p.get("body") or "")[:JUDGE_BODY_CHARS],
            "comments": comments,
        })
    state = {"today": today, "query": query, "topic": topic, "threads": threads}
    questions = {}
    for i, p in enumerate(posts):
        title = (p.get("title") or "")[:120]
        date = p.get("date") or "unknown"
        topic_hint = f"The active topic/entity is \"{topic}\". " if topic else ""
        questions[f"keep_{i}"] = (
            f"Consider only the thread titled \"{title}\" (dated {date}). "
            f"{topic_hint}"
            "Does it contain opinions, facts, or experiences that help answer the "
            "user's question? Judge topic match first; a thread about a different "
            "entity does not resolve it. On recency: ONLY penalize an older thread "
            "if the question explicitly asks for the latest, current, or today's "
            "information. A what/who/which question is answered just as well by "
            "last year's thread — do not downvote it for age."
        )
    probs = batch_noul(state, questions)
    verdicts = []
    for i, post in enumerate(posts):
        prob = probs.get(f"keep_{i}")
        if prob is None:
            verdicts.append({"resolves": True, "reason": "jev miss — fail open"})
        else:
            keep = prob >= JUDGE_THRESHOLD
            verdicts.append({"resolves": keep, "reason": f"jev noul={prob:.2f}"})
            logging.info(
                f"Judge: {'KEEP' if keep else 'DROP'} '{(post.get('title') or '')[:60]}' "
                f"[{post.get('date') or 'undated'}] — noul={prob:.2f}"
            )
    return verdicts


def judge_threads(query: str, posts: List[Dict], topic: str = "") -> List[Dict]:
    """Judge each thread against the query. Always returns one verdict per post."""
    if not posts:
        return []
    logging.info(f"Judging {len(posts)} threads against query: '{(query or '')[:150]}'")
    if not JUDGE_ENABLED:
        return _fail_open(posts, "judge disabled")
    if DECISION_BACKEND == "jev":
        try:
            return _judge_jev(query, posts, topic=topic)
        except Exception as e:
            return _fail_open(posts, f"jev error: {e}")
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=JUDGE_MODEL)
        briefs = [thread_brief(i, p) for i, p in enumerate(posts)]
        raw = llm.invoke([
            ("system", JUDGE_SYSTEM),
            ("user", judge_prompt(query, briefs, topic=topic)),
        ]).content
        verdicts = parse_verdicts(raw or "", len(posts))
        if verdicts is None:
            return _fail_open(posts, "unparseable reply")
        for (post, v) in zip(posts, verdicts):
            logging.info(
                f"Judge: {'KEEP' if v['resolves'] else 'DROP'} "
                f"'{(post.get('title') or '')[:60]}' — {v['reason']}"
            )
        return verdicts
    except Exception as e:
        return _fail_open(posts, f"error: {e}")
