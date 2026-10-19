"""Tests for the graph's routing decisions.

Routing is plain Python, so these tests do not run LangGraph or call models.
"""

from src.graph.edges import route_after_retrieve, route_intent
from src.graph.state import RAGState


def make_state(**overrides):
    """A follow-up state that is already answering from a good corpus."""
    values = {
        "query": "what is a good starting strategy?",
        "intent": "followup",
        "session_subs": ["TheFarmerWasReplaced"],
        "docs": ["one relevant thread"],
        "best_score": 0.60,
        "ingest_attempts": 0,
    }
    values.update(overrides)
    return RAGState(**values)


def test_followup_retrieves_from_the_current_topic():
    assert route_intent(make_state()) == "retrieve"


def test_new_topic_ingests_fresh_reddit_data():
    assert route_intent(make_state(intent="new")) == "ingest"


def test_weak_followup_searches_the_same_anchor():
    state = make_state(docs=[], best_score=0.0)
    assert route_after_retrieve(state) == "ingest"


def test_strong_followup_goes_to_answer_generation():
    assert route_after_retrieve(make_state()) == "generate_prep"


def test_followup_retries_only_once():
    state = make_state(docs=[], ingest_attempts=1)
    assert route_after_retrieve(state) == "generate_prep"


def test_recency_followup_refreshes_even_with_a_good_corpus():
    state = make_state(query="what is the latest update?", best_score=0.90)
    assert route_after_retrieve(state) == "ingest"
