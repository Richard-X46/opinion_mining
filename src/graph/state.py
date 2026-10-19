from pydantic import BaseModel, Field
from typing import List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from operator import add


class RAGState(BaseModel):
    query: Annotated[str, lambda x, y: y]  
    session_id: str = Field(default="")  # owning chat session; scopes Chroma retrieval
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    rewritten_query: str = Field(default="")
    docs: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)  # comment permalinks aligned to docs
    answer: str = Field(default="")
    gen_messages: List[BaseMessage] = Field(default_factory=list)  # what the final LLM call actually sends

    blocked: bool = Field(default=False)
    retries: int = Field(default=0)
    corpus_queries: List[str] = Field(default_factory=list)
    corpus_threads: List[str] = Field(default_factory=list)  # ingested thread titles, for intent reasoning
    intent: str = Field(default="")  # "followup" | "new", set by node_intent each turn
    intent_reason: str = Field(default="")
    fresh_indexed: int = Field(default=0)  # comments newly indexed by this turn's ingest