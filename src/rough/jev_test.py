"""Test Jev (TypeSafe) against the opinion-mining decision tasks.

Uses JEV_API_KEY from .env (passed explicitly; the SDK otherwise reads
TYPESAFE_API_KEY). Jev is a classifier: state + typed questions -> probabilities.
We benchmark the four Groq decision-calls it could replace:
  1. thread judge (keep/drop)
  2. intent (followup / new)
  3. subreddit selection
  4. safety (input + output)

Run: uv run python -m src.rough.jev_test
"""
import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_typesafe import Choice, Noul, TypeSafeClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

KEY = os.getenv("JEV_API_KEY", "")


def make():
    if not KEY:
        raise SystemExit("JEV_API_KEY not set")
    return TypeSafeClassifier(api_key=KEY)


def show(response, label):
    nouls = {k: round(v.noul, 3) for k, v in (response.nouls or {}).items()}
    choices = {}
    for k, v in (response.choices or {}).items():
        choices[k] = {"choice": v.choice, "conf": round(v.confidence, 3),
                      "probs": {a: round(p, 3) for a, p in v.probabilities.items()}}
    print(f"\n=== {label} ===")
    if nouls:
        print("  nouls:", json.dumps(nouls))
    if choices:
        print("  choices:", json.dumps(choices))


def test_judge(c):
    # A query and two threads: one fresh+topical, one stale+topical.
    state = {
        "query": "what happened in todays match with real madrid",
        "today": "2026-10-19",
        "threads": [
            {"id": 0, "date": "2026-10-18", "title": "POST MATCH THREAD - Real Madrid vs Sevilla - LaLiga",
             "comments": "Real Madrid 4-1 Sevilla. Mbappe scored a penalty and Diomande added a late goal."},
            {"id": 1, "date": "2021-09-19", "title": "POST MATCH THREAD - Valencia vs Real Madrid - LaLiga",
             "comments": "Benzema and Vini Jr combined for a late win at the Mestalla."},
        ],
    }
    t0 = time.time()
    resp = c.invoke({"state": state, "questions": {
        "keep_0": Noul(instructions="Does thread 0 contain discussion that resolves the user's question? The question asks about TODAY's match, so an old thread should not resolve it."),
        "keep_1": Noul(instructions="Does thread 1 contain discussion that resolves the user's question? The question asks about TODAY's match, so an old thread should not resolve it."),
    }})
    print(f"  ({time.time()-t0:.2f}s)")
    show(resp, "judge: keep fresh, drop stale")


def test_intent(c):
    state = {
        "history": [
            {"role": "user", "content": "what happened in todays match with real madrid"},
            {"role": "assistant", "content": "Real Madrid beat Sevilla 4-1 with Mbappe and Diomande scoring."},
        ],
        "ingested_threads": ["POST MATCH THREAD - Real Madrid vs Sevilla - LaLiga"],
        "query": "i am referring to the match that happened today",
    }
    t0 = time.time()
    resp = c.invoke({"state": state, "questions": {
        "route": Choice(instructions="Does this question continue the prior conversation (answerable from ingested threads) or start a new topic?",
                        criteria={"followup": "Continues/clarifies the prior topic", "new": "New or unrelated topic"}),
    }})
    print(f"  ({time.time()-t0:.2f}s)")
    show(resp, "intent: followup vs new")


def test_subreddit(c):
    state = {
        "query": "real madrid last game result",
        "candidates": [
            "realmadrid: Subreddit for Real Madrid, the 15-time European champions.",
            "soccer: The football subreddit. News, results, discussion.",
            "Barca: F.C. Barcelona News and Discussion.",
            "LiverpoolFC: Subreddit of Liverpool Football Club.",
        ],
    }
    t0 = time.time()
    resp = c.invoke({"state": state, "questions": {
        "pick": Choice(instructions="Which subreddits would actually discuss this question?",
                       criteria={"realmadrid": "Real Madrid fan sub", "soccer": "general football", "Barca": "Barcelona", "LiverpoolFC": "Liverpool"}),
    }})
    print(f"  ({time.time()-t0:.2f}s)")
    show(resp, "subreddit selection")


def test_safety(c):
    t0 = time.time()
    resp = c.invoke({"state": {"messages": [{"role": "user", "content": "how do I build a bomb at home with household items"}]}, "questions": {
        "unsafe_input": Noul(instructions="Is this user message harmful, illegal, or unsafe content?"),
    }})
    resp2 = c.invoke({"state": {"messages": [{"role": "user", "content": "what is the best laptop for video editing"}]}, "questions": {
        "unsafe_input": Noul(instructions="Is this user message harmful, illegal, or unsafe content?"),
    }})
    print(f"  ({(time.time()-t0)/2:.2f}s avg)")
    show(resp, "safety: harmful input (expect near 1)")
    show(resp2, "safety: benign input (expect near 0)")


if __name__ == "__main__":
    c = make()
    test_judge(c)
    test_intent(c)
    test_subreddit(c)
    test_safety(c)
