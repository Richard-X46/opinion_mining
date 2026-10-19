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
)
from src.ingestion.ingest import ingest
import logging

logging.basicConfig(level=logging.INFO)

def node_ingest(state: RAGState) -> RAGState:
    from src.graph.store import collection
    logging.info(f"Ingesting for query: {state.query}")
    before = collection.count()
    posts = ingest(state.rewritten_query, session_id=state.session_id)
    fresh = collection.count() - before
    logging.info(f"Fresh comments indexed this turn: {fresh}")
    corpus_queries = state.corpus_queries + [state.rewritten_query]
    titles = [p.get("title", "") for p in posts or []]
    corpus_threads = state.corpus_threads + [t for t in titles if t]
    return {**state.model_dump(), "corpus_queries": corpus_queries, "corpus_threads": corpus_threads, "fresh_indexed": fresh}


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
builder.add_edge("retrieve", "generate_prep")
builder.add_edge("generate_prep", END)

graph = builder.compile()


def run_graph(query: str, state: RAGState | None = None, session_id: str = "") -> RAGState:
    return RAGState(**graph.invoke(build_initial(query, state, session_id)))


class Cancelled(Exception):
    """Raised when a stop request aborts a run between graph nodes."""


def build_initial(query: str, state: RAGState | None = None, session_id: str = "") -> dict:
    initial = state.model_dump() if state else {}
    # Carry over conversation memory (messages, corpus_queries) but reset
    # everything computed fresh each turn — otherwise a new query (or a
    # refreshed page with a new session) inherits stale docs/sources.
    initial.update({
        "query": query,
        "session_id": session_id or initial.get("session_id", ""),
        "rewritten_query": "",
        "docs": [],
        "sources": [],
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