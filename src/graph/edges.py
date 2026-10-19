from src.graph.state import RAGState


def route_input_safety(state: RAGState) -> str:
    return "blocked" if state.blocked else "rewrite"


FOLLOWUP_RETRY_FLOOR = 0.45
_RECENCY_WORDS = frozenset(
    "latest newest recent recently current currently today now update updates "
    "new this week this month live breaking".split()
)


def route_intent(state: RAGState) -> str:
    """Follow-ups retrieve from the active topic; new topics always ingest."""
    if (state.intent or "") == "followup":
        return "retrieve"
    return "ingest"


def route_after_retrieve(state: RAGState) -> str:
    """Weak follow-up retrieval gets one re-search of the SAME anchor subs.

    This is the conversational repair path: a follow-up can be on-topic but
    not covered by what was already indexed. Searching fresh subreddits here
    would break the topic; searching the current anchor keeps continuity.
    """
    if (state.intent or "") == "followup" and state.ingest_attempts == 0 and state.session_subs:
        weak = (not state.docs) or state.best_score < FOLLOWUP_RETRY_FLOOR
        query = (state.query or "").lower()
        wants_fresh = any(word in query for word in _RECENCY_WORDS)
        if weak or wants_fresh:
            return "ingest"
    return "generate_prep"


def route_output_safety(state: RAGState) -> str:
    if not state.blocked:
        return "done"
    if state.retries < 2:
        return "retry"
    return "blocked"