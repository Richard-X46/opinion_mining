from src.graph.state import RAGState


def route_input_safety(state: RAGState) -> str:
    return "blocked" if state.blocked else "rewrite"


def route_corpus(state: RAGState) -> str:
    # V1 drift-lite: (re-)ingest when the collection is empty or this query
    # hasn't been ingested yet in the session. Adaptive drift detection is
    # post-V1 (see docs/mission.md non-goals).
    try:
        from src.graph.store import collection
        if collection.count() == 0:
            return "ingest"
    except Exception:
        return "ingest"
    return "ingest" if state.rewritten_query not in state.corpus_queries else "retrieve"


def route_intent(state: RAGState) -> str:
    """Follow-ups skip ingest and answer from the session corpus.

    Anything else falls through to the corpus check (ingest if this exact
    query hasn't been ingested yet, else retrieve). The honesty guard in
    app.py remains the backstop for empty retrievals.
    """
    if (state.intent or "") == "followup":
        return "retrieve"
    return route_corpus(state)


def route_output_safety(state: RAGState) -> str:
    if not state.blocked:
        return "done"
    if state.retries < 2:
        return "retry"
    return "blocked"