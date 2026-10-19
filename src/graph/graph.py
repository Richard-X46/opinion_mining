from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes import (
    node_input_safety,
    node_rewrite,
    node_intent,
    node_retrieve,
    node_generate_prep,
)
from src.graph.edges import (
    route_input_safety,
    route_intent,
    route_after_retrieve,
)
from src.ingestion.ingest import ingest
import logging

logging.basicConfig(level=logging.INFO)

def node_ingest(state: RAGState) -> RAGState:
    from src.graph.store import collection
    logging.info(f"Ingesting for query: {state.query}")
    before = collection.count()
    forced = [s for s in (state.subreddits or []) if s]
    # Follow-ups search the active topic's anchor, never a freshly discovered
    # community. New topics and user-pinned subs resolve/replace the anchor.
    reuse_anchor = (
        list(state.session_subs or [])
        if not forced and (state.intent or "") == "followup"
        else None
    )
    posts, subs_used = ingest(
        state.rewritten_query,
        session_id=state.session_id,
        subreddits=forced or None,
        anchor_subs=reuse_anchor,
        raw_query=state.query,
        topic_id=state.topic_id,
        topic=state.topic_label,
    )
    fresh = collection.count() - before
    logging.info(f"Fresh comments indexed this turn: {fresh}")
    corpus_queries = state.corpus_queries + [state.rewritten_query]
    titles = [p.get("title", "") for p in posts or []]
    corpus_threads = state.corpus_threads + [t for t in titles if t]
    if reuse_anchor:
        session_subs = list(reuse_anchor)
    else:
        session_subs = [s for s in subs_used if s]
    return {
        **state.model_dump(),
        "corpus_queries": corpus_queries,
        "corpus_threads": corpus_threads,
        "fresh_indexed": fresh,
        "subreddits": subs_used,
        "session_subs": session_subs,
        "ingest_attempts": state.ingest_attempts + 1,
    }


builder = StateGraph(RAGState)

builder.add_node("input_safety", node_input_safety)
builder.add_node("rewrite", node_rewrite)
builder.add_node("intent", node_intent)
builder.add_node("ingest", node_ingest)
builder.add_node("retrieve", node_retrieve)
builder.add_node("generate_prep", node_generate_prep)

builder.set_entry_point("input_safety")

builder.add_conditional_edges(
    "input_safety",
    route_input_safety,
    {"rewrite": "rewrite", "blocked": END},
)
builder.add_edge("rewrite", "intent")
builder.add_conditional_edges(
    "intent",
    route_intent,
    {"ingest": "ingest", "retrieve": "retrieve"},
)
builder.add_edge("ingest", "retrieve")
builder.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {"ingest": "ingest", "generate_prep": "generate_prep"},
)
builder.add_edge("generate_prep", END)

graph = builder.compile()


def run_graph(query: str, state: RAGState | None = None, session_id: str = "") -> RAGState:
    return RAGState(**graph.invoke(build_initial(query, state, session_id)))


class Cancelled(Exception):
    """Raised when a stop request aborts a run between graph nodes."""


def build_initial(query: str, state: RAGState | None = None, session_id: str = "",
                  subreddits: list[str] | None = None, reseed: bool = False) -> dict:
    initial = state.model_dump() if state else {}
    # Carry over conversation memory (messages, corpus_queries) but reset
    # everything computed fresh each turn — otherwise a new query (or a
    # refreshed page with a new session) inherits stale docs/sources.
    # subreddits is per-turn: explicit user picks pin the search, otherwise
    # auto-resolve (never inherit last turn's subs into a new topic).
    # reseed ("none of these") wipes the session anchor and counts the round —
    # after two failed reseeds the app stops guessing and says sorry.
    if reseed:
        initial["session_subs"] = []
        initial["topic_id"] = ""
        initial["topic_label"] = ""
        initial["anchor_attempts"] = (state.anchor_attempts if state else 0) + 1
        initial["subreddits"] = []
    initial.update({
        "query": query,
        "session_id": session_id or initial.get("session_id", ""),
        "rewritten_query": "",
        "docs": [],
        "sources": [],
        "source_details": [],
        "subreddits": list(subreddits) if subreddits else [],
        "force_new": bool(reseed or subreddits),
        "ingest_attempts": 0,
        "best_score": 0.0,
        "answer": "",
        "gen_messages": [],
        "fresh_indexed": 0,
        "blocked": False,
        "intent": "",
        "intent_reason": "",
    })
    return initial


def run_graph_cancellable(initial: dict, cancel_event=None, callbacks=None) -> RAGState:
    """Same as invoke(), but checks `cancel_event` after each node so a
    stop request halts the pipeline at the next node boundary instead of
    running the remaining stages. An in-flight node call itself is not
    interrupted — its result is simply discarded. `callbacks` (e.g. usage
    accounting) propagate to every nested model call."""
    merged = dict(initial)
    config = {"callbacks": callbacks} if callbacks else None
    for update in graph.stream(initial, config=config):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        for _node, node_state in update.items():
            if isinstance(node_state, dict):
                merged.update(node_state)
    return RAGState(**merged)


if __name__ == "__main__":

    graph.get_graph().draw_mermaid_png()

    print(graph.get_graph().draw_mermaid())
