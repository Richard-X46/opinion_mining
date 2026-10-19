from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.settings import (
    INPUT_SAFETY_MODEL,
    REWRITE_MODEL,
    OUTPUT_SAFETY_MODEL,
    INPUT_SAFETY_ENABLED,
    DECISION_BACKEND,
    TOP_K,
    SIM_FLOOR,
    SIM_MARGIN,
)
from src.graph.state import RAGState
from src.graph.store import collection
from src.ingestion.embedder import get_embedder
import logging


def node_input_safety(state: RAGState) -> RAGState:
    if not INPUT_SAFETY_ENABLED:
        logging.info("Input safety skipped (INPUT_SAFETY_ENABLED=false)")
        return state
    if DECISION_BACKEND == "jev":
        from src.graph.jev import noul
        prob = noul(
            {"user_message": state.query},
            "Is this user message harmful, illegal, unsafe, hateful, or policy-violating content that should not be processed?",
        )
        if prob is None:
            logging.info("Input safety (jev) failed — defaulting to safe")
            is_safe = True
        else:
            is_safe = prob < 0.5
            logging.info(f"Input safety (jev) score: {prob:.3f}")
        return {**state.model_dump(), "blocked": not is_safe}
    llm = ChatGroq(model=INPUT_SAFETY_MODEL)
    result = llm.invoke(state.query).content.strip()
    score = None
    try:
        score = float(result)
        is_safe = score < 0.5
        logging.info(f"Input safety score: {score}")
    except ValueError:
        is_safe = True
        logging.info(f"Input safety unparseable ({result!r}), defaulting to safe")
    return {**state.model_dump(), "blocked": not is_safe}


def node_rewrite(state: RAGState) -> RAGState:
    """Rewrite the user question as a short Reddit search query (not an answer).

    The output feeds subreddit resolution and Reddit's thread search, so it
    must stay a single short line — never markdown, bullets, or a full response.
    """
    llm = ChatGroq(model=REWRITE_MODEL)
    history = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.messages[-4:]
    ])
    prompt = f"""Rewrite the user question below as a short standalone web search query for Reddit opinions.

Rules:
- Output ONE line only, 15 words or fewer
 - Topic keywords ONLY — NEVER include filler like reddit, opinions, buzz,
  thread, comments, posts. Those words dilute the search and the embedding.
  Bad: "Reddit opinions on latest buzz about Navier-Stokes equations".
  Good: "Navier-Stokes equations breakthrough"
 - EXCEPTION: keep recency words when the user wants fresh info — latest,
   newest, recent, current. They help match fresh Reddit threads.
  "whats the latest onepiece chapter about" -> "latest One Piece chapter"
- STANDALONE: resolve every pronoun and fragment from the history into the actual topic.
  "it", "this", "that", "the solution", "check X" must name the topic outright,
  so the line makes sense with zero conversation context.

Conversation history:
{history}

Question: {state.query}

Search query:"""
    raw = llm.invoke(prompt).content.strip()
    # Keep only the first line, strip wrapping quotes, hard-cap length so a
    # chatty model can never blow up the Reddit query string.
    rewritten = raw.split("\n")[0].strip().strip("\"'")[:150]
    if not rewritten:
        rewritten = state.query[:150]
        logging.warning("Rewrite empty, falling back to raw query")
    logging.info(f"Rewritten query: {rewritten}")
    return {**state.model_dump(), "rewritten_query": rewritten}


def _intent_state(state: RAGState) -> dict:
    return {
        "active_topic": state.topic_label or state.query,
        "active_subreddits": list(state.session_subs),
        "history": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state.messages[-4:]
        ],
        "ingested_threads": [t[:120] for t in state.corpus_threads[-15:]],
        "question": state.query,
    }


INTENT_INSTRUCTION = (
    "Is this question a continuation of the active topic, or does it switch to a "
    "different subject/entity? A follow-up may ask for new information and still "
    "belong to the same topic. A correction that replaces the subject (\"I meant X\"), "
    "or a question about another game/product/person/team, is a topic switch."
)


def _apply_topic(state: RAGState, intent: str, reason: str) -> RAGState:
    """Keep the topic scope on follow-ups; start a new one only on a real switch."""
    if state.force_new:
        intent = "new"
        reason = "reseed/user-pinned subs"
    if intent == "followup" and not state.session_subs:
        # No active anchor to continue from; discover fresh rather than
        # retrieving from an empty/unknown scope.
        intent = "new"
        reason = f"{reason}; no active anchor"
    topic_id = state.topic_id
    topic_label = state.topic_label
    if intent != "followup" or not topic_id:
        import uuid
        topic_id = uuid.uuid4().hex
        topic_label = state.query[:120]
    logging.info(f"Intent: {intent} — {reason}")
    return {**state.model_dump(), "intent": intent, "intent_reason": reason,
            "topic_id": topic_id, "topic_label": topic_label, "force_new": False}


def node_intent(state: RAGState) -> RAGState:
    """Route the turn by topic continuity, not by whether the corpus can answer.

    Follow-ups keep the active topic and its subreddit anchor. They are
    allowed to need fresh information: the graph will re-search the same
    anchor subs if retrieval is weak. A new topic (including "I meant X")
    starts a new scope and resolves subreddits from scratch.
    """
    if not state.messages and not state.corpus_threads and not state.session_subs:
        return _apply_topic(state, "new", "first turn, empty session")
    if DECISION_BACKEND == "jev":
        from src.graph.jev import choice
        result = choice(
            _intent_state(state),
            INTENT_INSTRUCTION,
            {
                "followup": "Same topic or a clarification/expansion of it, even if new facts are needed.",
                "new": "Different subject/entity, or a correction that replaces the active subject.",
            },
        )
        if result is None:
            intent_out, reason = "new", "intent jev fail-open"
        else:
            label, conf = result
            intent_out = label
            reason = f"jev conf={conf:.2f}"
        return _apply_topic(state, intent_out, reason)
    history = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.messages[-4:]
    ])
    threads = "\n".join(f"- {t[:100]}" for t in state.corpus_threads[-15:]) or "(none yet)"
    prompt = f"""You route topic continuity for a Reddit opinion-mining chatbot.

Active topic: {state.topic_label or "(none yet)"}
Active subreddits: {', '.join(state.session_subs) or "(none yet)"}

Conversation history (most recent last):
{history}

Threads already ingested this session (titles):
{threads}

Current user question: {state.query}

Rules:
- FOLLOWUP if it continues, clarifies, or expands the active topic. A follow-up may ask for fresh opinions/info; that does NOT make it a new topic.
- NEW if it switches to a different subject/entity, or corrects the subject ("I meant X").
- Do NOT call it NEW just because the current threads cannot answer it.
- If there is no active topic, choose NEW.

Reply with a JSON object ONLY: {{"intent": "FOLLOWUP" or "NEW", "reason": "<=15 words"}}"""
    try:
        llm = ChatGroq(model=REWRITE_MODEL)
        raw = (llm.invoke(prompt).content or "").strip()
        import json as _json
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            raise ValueError("no JSON object in reply")
        parsed = _json.loads(m.group(0))
        intent = str(parsed.get("intent", "")).upper()
        if intent not in ("FOLLOWUP", "NEW"):
            raise ValueError(f"unknown intent {intent!r}")
        reason = str(parsed.get("reason", ""))[:120]
        intent_out = intent.lower()
    except Exception as e:
        intent_out, reason = "new", f"intent LLM fail-open: {e}"
    return _apply_topic(state, intent_out, reason)


def node_retrieve(state: RAGState) -> RAGState:
    """Retrieve + rank: cosine similarity gates what enters the prompt and
    what gets cited. Dynamic rule — keep docs within SIM_MARGIN of the best
    match, never below SIM_FLOOR. Source count varies (0-5), not fixed."""
    from src.ingestion.ingest import clean_text  # same normalization as indexed docs
    embedder = get_embedder()
    embedding = embedder.encode([clean_text(state.rewritten_query)]).tolist()
    # Scope retrieval to this chat session AND this topic: the collection is
    # global/persistent, and one session can span several topics. Without the
    # topic filter, an earlier topic's comments leak into a new answer.
    if state.session_id and state.topic_id:
        where = {"$and": [
            {"session_id": state.session_id},
            {"topic_id": state.topic_id},
        ]}
    elif state.session_id:
        where = {"session_id": state.session_id}
    else:
        where = None
    results = collection.query(
        query_embeddings=embedding,
        n_results=TOP_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    ranked = []
    for i, doc in enumerate(docs):
        d = distances[i] if i < len(distances) else 1.0
        sim = max(0.0, min(1.0, 1.0 - d))  # cosine space => distance = 1 - cos_sim
        meta = metadatas[i] if i < len(metadatas) else {}
        ranked.append((sim, doc, meta or {}))
    ranked.sort(key=lambda t: t[0], reverse=True)

    best = ranked[0][0] if ranked else 0.0
    cutoff = max(SIM_FLOOR, best - SIM_MARGIN) if best >= 0.10 else 1.0
    kept = [(doc, meta) for sim, doc, meta in ranked if sim >= cutoff]
    logging.info(
        "Retrieved %d docs, kept %d (session=%s, best %.2f, cutoff %.2f): %s",
        len(docs), len(kept), state.session_id or "global", best, cutoff,
        [f"{s:.2f}" for s, _, _ in ranked[:TOP_K]],
    )

    seen, sources, details = set(), [], []
    for _, meta in kept:
        # One pill per THREAD: comments from the same thread carry different
        # comment permalinks, so deduping on URL shows one thread five times.
        thread_key = meta.get("thread_url") or meta.get("title") or meta.get("url", "")
        link = meta.get("thread_url") or meta.get("url", "")
        if thread_key and link and thread_key not in seen:
            seen.add(thread_key)
            sources.append(link)
            details.append({
                "title": (meta.get("title", "") or "")[:120],
                "subreddit": meta.get("subreddit", "") or "",
                "url": link,
            })
        if len(sources) >= 5:
            break
    return {**state.model_dump(),
            "docs": [doc for doc, _ in kept],
            "sources": sources,
            "source_details": details,
            "best_score": round(best, 4)}


GEN_SYSTEM = """Answer like ChatGPT: natural, conversational, in your own words. Ground every answer in the supplied background first, and fill gaps from your own knowledge where that helps.

Hard rules:
- Give a complete answer with real substance: several sentences up to a couple of short paragraphs, covering the main angles (the appeal, the trade-offs, the caveats) when the question warrants it. Go longer only when genuinely needed. Cut filler, never pad — but don't starve a complex question either.
- Reddit-sourced specifics (what commenters claim, argue, or recommend) must come straight from the background — never invent them, and never present your own knowledge as something Redditors said. Do not describe what fans are doing, feeling, or looking for unless the background says so.
- For generic, stable facts the background doesn't cover (release dates, casting, widely-known context), use your own knowledge to complete the answer. If you're unsure about a fact, say so in a clause rather than guessing it into existence. If the background was thin and the answer leans mostly on your own knowledge, say so in a brief clause.
- Answer the exact question asked. If it asks for the best X, name them — do not describe the subject instead.
- A few bullets ONLY for listing items (places, prices, options).
- NEVER mention commenters, usernames, tables, or the background itself. No "users say", no "Reddit", no source talk at all.
- NEVER quote or reproduce background text. Synthesize it invisibly.
- Conversation history sets the topic: use it to recover the full intent of short follow-up questions, which often only make sense in light of earlier turns. It is context only — never treat earlier assistant messages as background facts about the world.
- Start with the answer. No preamble."""


def node_generate_prep(state: RAGState) -> RAGState:
    """Builds the final chat messages, does NOT call the LLM. Streaming happens in FastAPI.

    Best practice per LangGraph chatbot guides: the model gets real message
    roles — rules as a SystemMessage, conversation as actual message objects,
    evidence + question as the closing HumanMessage — instead of one
    flattened string where it can't tell rules from history from evidence.
    """
    import re as _re
    pieces = []
    budget = 4000
    for doc in state.docs:
        text = _re.sub(r"\bu\/[A-Za-z0-9_-]+", "a user", doc)
        text = text[:600]
        if len("\n\n".join(pieces)) + len(text) > budget:
            break
        pieces.append(f"- {text}")
    context = "\n\n".join(pieces)
    history = list(state.messages[-6:])
    staleness = ""
    if (state.intent or "") == "followup" and not state.fresh_indexed:
        staleness = (
            "\nNote: the background below was ingested earlier this session, not fresh. "
            "If the question asks for the latest, answer from it but say plainly "
            "what it covers and that newer discussion was not found."
        )
    system = GEN_SYSTEM + staleness
    question = (
        "Background opinions (do not mention or reference these):\n"
        f"{context}\n\nQuestion: {state.query}\n\nAnswer:"
    )
    gen_messages = [SystemMessage(content=system), *history, HumanMessage(content=question)]
    return {**state.model_dump(), "gen_messages": gen_messages}


def node_output_safety(state: RAGState) -> RAGState:
    llm = ChatGroq(model=OUTPUT_SAFETY_MODEL)
    prompt = f"""Is the following response safe, appropriate and free from harmful content?
Reply with only SAFE or UNSAFE.

Response: {state.answer}"""
    result = llm.invoke(prompt).content.strip().upper()
    is_safe = result == "SAFE"
    retries = state.retries + (0 if is_safe else 1)
    logging.info(f"Output safety: {result}")
    return {**state.model_dump(), "blocked": not is_safe, "retries": retries}


def run_output_safety(answer: str) -> bool:
    """Standalone output safety check called from FastAPI after streaming."""
    llm = ChatGroq(model=OUTPUT_SAFETY_MODEL)
    prompt = f"""Is the following response safe, appropriate and free from harmful content?
Reply with only SAFE or UNSAFE.

Response: {answer}"""
    result = llm.invoke(prompt).content.strip().upper()
    logging.info(f"Output safety: {result}")
    return result == "SAFE"