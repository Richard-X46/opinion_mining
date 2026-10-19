"""Small, readable tests for subreddit entity/keyword extraction.

These are pure functions: no Reddit account, network, or LLM calls.
"""

from src.ingestion.praw_fetcher import (
    _normalize_name,
    entity_candidates,
    keyword_queries,
)


def test_normalize_name_ignores_case_and_spaces():
    assert _normalize_name("The Farmer Was Replaced") == "thefarmerwasreplaced"


def test_entity_candidates_keeps_the_dash_separated_subject():
    raw = "there is this game - the farmer was replaced, any useful tips?"
    phrases = [phrase.lower() for phrase in entity_candidates([raw])]
    assert "the farmer was replaced" in phrases


def test_keyword_queries_prefers_the_compact_subject_term():
    queries = keyword_queries(
        "what does fertilizer do?",
        "fertilizer effect in The Farmer Was Replaced",
        primary_entity="The Farmer Was Replaced",
    )
    assert queries[0] == "fertilizer"
    assert all("the farmer was replaced" not in query.lower() for query in queries)


def test_keyword_queries_without_a_primary_entity():
    queries = keyword_queries("how do I solve the maze", "maze solution strategy")
    assert queries
    assert any("maze" in query for query in queries)
