"""Tests for the small pieces of chat state used by follow-up routing."""

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import RAGState, _concrete_message


def test_new_state_has_sensible_defaults():
    state = RAGState(query="hello")
    assert state.session_subs == []
    assert state.topic_id == ""
    assert state.ingest_attempts == 0


def test_messages_are_restored_to_their_real_classes():
    state = RAGState(
        query="hello",
        messages=[
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "hello"},
        ],
    )
    assert isinstance(state.messages[0], HumanMessage)
    assert isinstance(state.messages[1], AIMessage)


def test_existing_message_objects_are_kept():
    message = HumanMessage(content="hi")
    assert _concrete_message(message) is message
