from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Annotated
from langchain_core.messages import BaseMessage
from operator import add


def _concrete_message(m):
    """Coerce a stored message back to its concrete role class.

    State round-trips (model_dump / merge / revalidate) flatten messages to
    BaseMessage with only a `type` string left. Providers like Groq dispatch
    on isinstance, so a flattened history crashes the next turn's stream.
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    if type(m) in (HumanMessage, AIMessage, SystemMessage):
        return m
    d = m if isinstance(m, dict) else (m.model_dump() if isinstance(m, BaseMessage) else None)
    if not isinstance(d, dict):
        return None
    content = d.get("content", "")
    msg_type = d.get("type")
    if msg_type == "ai":
        return AIMessage(content=content)
    if msg_type == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class RAGState(BaseModel):
    query: Annotated[str, lambda x, y: y]  
    session_id: str = Field(default="")  # owning chat session; scopes Chroma retrieval
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    rewritten_query: str = Field(default="")
    docs: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)  # comment permalinks aligned to docs
    source_details: List[Dict] = Field(default_factory=list)  # {title, subreddit, url} for display
    subreddits: List[str] = Field(default_factory=list)  # subs used this turn (user-picked or resolved)
    session_subs: List[str] = Field(default_factory=list)  # active topic anchor: follow-ups reuse these
    topic_id: str = Field(default="")  # scopes Chroma docs to the current topic
    topic_label: str = Field(default="")  # human-readable current topic, for intent reasoning
    force_new: bool = Field(default=False)  # user reseeded or pinned subs: never reuse the old anchor
    ingest_attempts: int = Field(default=0)  # per-turn ingest runs; bounds the follow-up re-search loop
    anchor_attempts: int = Field(default=0)  # reseed rounds used looking for the right anchor
    best_score: float = Field(default=0.0)  # top retrieval cosine sim; guards corpus fallback
    answer: str = Field(default="")
    gen_messages: List[BaseMessage] = Field(default_factory=list)  # what the final LLM call actually sends

    blocked: bool = Field(default=False)
    retries: int = Field(default=0)
    corpus_queries: List[str] = Field(default_factory=list)
    corpus_threads: List[str] = Field(default_factory=list)  # ingested thread titles, for intent reasoning
    intent: str = Field(default="")  # "followup" | "new", set by node_intent each turn
    intent_reason: str = Field(default="")
    fresh_indexed: int = Field(default=0)  # comments newly indexed by this turn's ingest

    @field_validator("messages", "gen_messages", mode="before")
    @classmethod
    def _restore_message_roles(cls, v):
        if not isinstance(v, list):
            return v
        return [m for m in (_concrete_message(item) for item in v) if m is not None]
