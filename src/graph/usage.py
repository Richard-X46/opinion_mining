"""Per-request token accounting, split by provider.

Groq reports OpenAI-style usage on every non-streamed invoke (picked up via
a LangChain callback) and on the final chunk of a stream (accumulated by the
caller). Jev reports input/output tokens on every ClassifierResponse (picked
up in src/graph/jev.py).

One tracker lives in a ContextVar per /chat request so concurrent requests
never mix numbers. Always fail-safe: accounting must never break a run.
"""
import contextvars
import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageTracker:
    def __init__(self) -> None:
        self.groq_in = 0
        self.groq_out = 0
        self.groq_calls = 0
        self.jev_in = 0
        self.jev_out = 0
        self.jev_calls = 0

    def record_groq(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        try:
            self.groq_in += int(prompt_tokens or 0)
            self.groq_out += int(completion_tokens or 0)
            self.groq_calls += 1
        except (TypeError, ValueError):
            pass

    def record_jev(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        try:
            self.jev_in += int(input_tokens or 0)
            self.jev_out += int(output_tokens or 0)
            self.jev_calls += 1
        except (TypeError, ValueError):
            pass

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            "groq": {"in": self.groq_in, "out": self.groq_out, "calls": self.groq_calls},
            "jev": {"in": self.jev_in, "out": self.jev_out, "calls": self.jev_calls},
        }


_tracker_var: contextvars.ContextVar[Optional[UsageTracker]] = contextvars.ContextVar(
    "usage_tracker", default=None
)


def use_tracker(tracker: UsageTracker):
    """Set the active tracker for this context; returns a token to reset it."""
    return _tracker_var.set(tracker)


def reset_tracker(token) -> None:
    try:
        _tracker_var.reset(token)
    except Exception:
        pass


def active_tracker() -> Optional[UsageTracker]:
    return _tracker_var.get()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _groq_tokens_from_llm_output(llm_output: Any) -> tuple[int, int]:
    """Best-effort parse of OpenAI-style token_usage from ChatGroq output."""
    if not isinstance(llm_output, dict):
        return 0, 0
    usage = llm_output.get("token_usage", llm_output)
    if not isinstance(usage, dict):
        return 0, 0
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return _as_int(prompt), _as_int(completion)


class UsageCallbackHandler(BaseCallbackHandler):
    """Counts Groq tokens on every nested (non-streamed) LLM invoke."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            tracker = active_tracker()
            if tracker is None:
                return
            prompt, completion = _groq_tokens_from_llm_output(
                getattr(response, "llm_output", None)
            )
            if prompt or completion:
                tracker.record_groq(prompt, completion)
        except Exception as e:
            logging.debug(f"Usage handler skipped a run: {e}")
