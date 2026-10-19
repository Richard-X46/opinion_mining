import warnings
warnings.filterwarnings("ignore", message=".*JsonPlusSerializer.*")

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
import uvicorn
import logfire
import logging
import json
import uuid
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from src.config.notifications import notify_discord
from src.config.settings import GENERATION_MODEL, DISCORD_ID, DISCORD_TOKEN, OUTPUT_SAFETY_ENABLED
from src.graph.graph import (
    build_initial,
    run_graph_cancellable,
    Cancelled,
)
from src.graph.usage import UsageCallbackHandler
from src.graph.state import RAGState
from src.graph.edges import FOLLOWUP_RETRY_FLOOR
from src.graph.nodes import run_output_safety

try:
    logfire.configure(service_name="opinion-mining")
except Exception as e:
    logging.warning(f"logfire disabled: {e}")
# logfire.instrument_pydantic() # llm streaming response raises too many logs
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logfire.info("App started successfully")


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    subreddits: list[str] | None = None  # user-pinned subs; null/empty = auto-resolve
    reseed_anchor: bool = False  # "none of these" — wipe anchor, discover fresh


class StopRequest(BaseModel):
    session_id: str


class FeedbackRequest(BaseModel):
    text: str


app = FastAPI()
try:
    logfire.instrument_fastapi(app)
except Exception as e:
    logging.warning(f"logfire fastapi instrumentation disabled: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, RAGState] = {}
# Cooperative cancellation flags, one per live /chat stream. /stop sets the
# flag; the pipeline checks it between graph nodes and on every token.
cancel_events: dict[str, threading.Event] = {}
templates = Jinja2Templates(directory="src/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/chat")
def chat(request: Request, payload: ChatRequest):
    logging.info(f"Query: {payload.query}")
    logging.info(f"Session ID received: {payload.session_id}")
    logging.info(f"Active sessions: {list(sessions.keys())}")
    if DISCORD_ID and DISCORD_TOKEN:
        notify_discord(f"User Query: '{payload.query}'")

    session_id = payload.session_id or str(uuid.uuid4())
    existing_state = sessions.get(session_id)
    logging.info(f"Existing state found: {existing_state is not None}")
    if existing_state is None:
        # Brand-new session (incl. page refresh): blank slate — drop other
        # sessions' cached docs so retrieval starts clean.
        from src.graph.store import prune_other_sessions
        prune_other_sessions([*sessions.keys(), session_id])

    ev = cancel_events.setdefault(session_id, threading.Event())
    ev.clear()

    def stopped_payload():
        return f"data: {json.dumps({'type': 'stopped', 'session_id': session_id})}\n\n"

    def generate():
        usage_cb = UsageCallbackHandler()
        tracker = usage_cb.tracker
        try:
            # Immediate heartbeat so the UI can show "working" during the long
            # silent ingest (PRAW discovery + thread fetching) before any other
            # event arrives.
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching Reddit for opinions…'})}\n\n"

            # run graph up to generate_prep, aborting at node boundaries on /stop
            try:
                with logfire.span("langgraph-pipeline", query=payload.query):
                    initial = build_initial(payload.query, state=existing_state, session_id=session_id,
                                            subreddits=payload.subreddits or None,
                                            reseed=payload.reseed_anchor)
                    state = run_graph_cancellable(initial, ev, callbacks=[usage_cb])
            except Cancelled:
                logging.info(f"Run cancelled at node boundary (session {session_id})")
                yield stopped_payload()
                return

            if state.blocked:
                sessions[session_id] = state  # keep corpus/history so the id round-trips
                yield f"data: {json.dumps({'type': 'blocked', 'session_id': session_id})}\n\n"
                return

            if (state.intent or "") == "followup" and state.fresh_indexed == 0 and state.best_score < FOLLOWUP_RETRY_FLOOR:
                # Follow-up answered from the session corpus, but nothing in it
                # is actually close to this question — refuse instead of
                # confabulating from junk. "None of these" re-searches fresh.
                sessions[session_id] = state
                yield f"data: {json.dumps({'type': 'no_sources', 'session_id': session_id, 'message': 'I couldn\\u2019t find that in what we\\u2019ve already discussed \\u2014 try rephrasing, or tick \\u2018none of these\\u2019 to search fresh subs.'})}\n\n"
                return

            if not state.docs:
                sessions[session_id] = state
                if (state.anchor_attempts or 0) >= 2:
                    no_sources_msg = ("Sorry — we weren't able to find this on Reddit after another look. "
                                      "Please consider a web search for this one.")
                else:
                    no_sources_msg = "No relevant Reddit discussion found for this query."
                yield f"data: {json.dumps({'type': 'no_sources', 'session_id': session_id, 'message': no_sources_msg})}\n\n"
                return

            # Deterministic honesty guard (code, not prompt): a new question that
            # indexed zero fresh comments gets a canned reply — no LLM call, so the
            # model can never dress up stale/parametric knowledge as an answer.
            # Follow-ups the intent node routed to the session corpus are exempt:
            # state.docs is non-empty here, and generate_prep carries a staleness
            # caveat for them.
            prior_queries = existing_state.corpus_queries if existing_state else []
            is_followup = (state.intent or "") == "followup"
            if not is_followup and state.rewritten_query not in prior_queries and state.fresh_indexed == 0 and state.best_score >= FOLLOWUP_RETRY_FLOOR:
                # Nothing new indexed (discovery whiffed) but the session corpus
                # already answers this well — answer from it with a staleness
                # caveat instead of refusing.
                logging.info(f"No fresh ingest but corpus best={state.best_score:.2f} — answering from session corpus")
                state = state.model_copy(update={"intent": "followup", "intent_reason": "no fresh ingest; session corpus covers the query"})
            if not is_followup and state.rewritten_query not in prior_queries and state.fresh_indexed == 0 and state.best_score < FOLLOWUP_RETRY_FLOOR:
                full_answer = "I couldn't find fresh Reddit discussion on that — try rephrasing, or coming at it from a different angle."
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                for word in full_answer.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
                updated = state.model_copy(update={
                    "answer": full_answer,
                    "messages": state.messages + [
                        HumanMessage(content=payload.query),
                        AIMessage(content=full_answer),
                    ],
                })
                sessions[session_id] = updated
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'corpus_queries': updated.corpus_queries, 'sources': [], 'source_details': [], 'subs': state.subreddits, 'usage': tracker.snapshot()})}\n\n"
                return

            # stream LLM response
            llm = ChatGroq(model=GENERATION_MODEL)
            full_answer = ""

            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            for chunk in llm.stream(state.gen_messages):
                if ev.is_set():
                    logging.info(f"Run cancelled during streaming (session {session_id})")
                    yield stopped_payload()
                    return
                um = getattr(chunk, "usage_metadata", None)
                if um:
                    tracker.record_groq(um.get("input_tokens", 0), um.get("output_tokens", 0))
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # output safety on complete answer
            if OUTPUT_SAFETY_ENABLED:
                is_safe = run_output_safety(full_answer)
            else:
                logging.info("Output safety skipped (OUTPUT_SAFETY_ENABLED=false)")
                is_safe = True
            if not is_safe:
                yield f"data: {json.dumps({'type': 'blocked', 'session_id': session_id})}\n\n"
                return

            # update state with answer and messages
            new_messages = [HumanMessage(content=payload.query), AIMessage(content=full_answer)]
            updated = state.model_copy(update={
                "answer": full_answer,
                "messages": state.messages + new_messages,
            })
            sessions[session_id] = updated

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'corpus_queries': updated.corpus_queries, 'sources': updated.sources, 'source_details': updated.source_details, 'subs': state.subreddits, 'usage': tracker.snapshot()})}\n\n"
        finally:
            # One-shot flag: a finished stream must not cancel the next query.
            cancel_events.pop(session_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/stop")
def stop(payload: StopRequest):
    """Halt the live /chat run for a session at the next node/token boundary."""
    ev = cancel_events.get(payload.session_id)
    if ev is not None:
        ev.set()
    return {"ok": True}


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    """Suggestion box -> notifications."""
    text = (payload.text or "").strip()[:1000]
    if not text:
        return {"ok": False}
    notify_discord(f"Suggestion from Ask Reddit: {text}")
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=5001, reload=True)
