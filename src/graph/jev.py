"""Jev (TypeSafe) decision helpers.

Jev is a classifier: state + typed questions -> probabilities, no text out.
This module is the single choke point for every DECISION in the pipeline
(safety, intent, thread judge) so the backend swap lives in one place.

Every helper fails open — on missing key, network error, or a reply we can't
read, callers get the safe default (None here; each node decides what its
fail-open is). That keeps a Jev outage from taking down the app.
"""
import logging
from functools import lru_cache

from src.config.settings import JEV_API_KEY, JEV_MODEL


@lru_cache(maxsize=1)
def get_classifier():
    """Lazy singleton; keep long-lived for connection pooling."""
    if not JEV_API_KEY:
        raise RuntimeError("JEV_API_KEY not set")
    from langchain_typesafe import TypeSafeClassifier
    return TypeSafeClassifier(api_key=JEV_API_KEY, model=JEV_MODEL)


def _state_of(value) -> object:
    """Normalise whatever the caller passes into something Jev accepts."""
    return value


def noul(state, instruction: str, criteria=None) -> float | None:
    """Binary yes/no probability in [0, 1], or None on failure."""
    try:
        from langchain_typesafe import Noul
        resp = get_classifier().invoke({
            "state": _state_of(state),
            "questions": {"q": Noul(instructions=instruction, criteria=criteria)},
        })
        return resp.nouls["q"].noul
    except Exception as e:
        logging.warning(f"Jev noul failed: {e}")
        return None


def choice(state, instruction: str, criteria: dict) -> tuple[str, float] | None:
    """Single-label choice. Returns (label, confidence) or None on failure."""
    try:
        from langchain_typesafe import Choice
        resp = get_classifier().invoke({
            "state": _state_of(state),
            "questions": {"q": Choice(instructions=instruction, criteria=criteria)},
        })
        a = resp.choices["q"]
        return a.choice, float(a.confidence)
    except Exception as e:
        logging.warning(f"Jev choice failed: {e}")
        return None


def batch_noul(state, questions: dict[str, str]) -> dict[str, float]:
    """One request, many binary questions sharing one state. Returns {id: prob}.

    Missing keys (any per-question failure) are omitted; callers decide their
    own fail-open per question.
    """
    try:
        from langchain_typesafe import Noul
        qs = {k: Noul(instructions=instr) for k, instr in questions.items()}
        resp = get_classifier().invoke({"state": _state_of(state), "questions": qs})
        return {k: resp.nouls[k].noul for k in questions if k in resp.nouls}
    except Exception as e:
        logging.warning(f"Jev batch_noul failed: {e}")
        return {}
