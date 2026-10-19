"""Per-request token accounting, split by provider.

Groq reports OpenAI-style usage on every non-streamed invoke (picked up via
a LangChain callback) and on the final chunk of a stream (accumulated by the
caller). Jev reports input/output tokens on every ClassifierResponse
(picked up from the chain output via the same callback).

One handler is created per /chat request and passed explicitly through the
LangGraph config, so accounting survives thread-hops between streamed
chunks. All parsing is best-effort: accounting must never break a run.
"""
import logging
from typing import Any, Dict

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
    """Counts Groq + Jev tokens for one request. Passed via LangGraph config
    so it follows the run across threads; owns its tracker outright."""

    def __init__(self) -> None:
        super().__init__()
        self.tracker = UsageTracker()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            prompt, completion = _groq_tokens_from_llm_output(
                getattr(response, "llm_output", None)
            )
            if prompt or completion:
                self.tracker.record_groq(prompt, completion)
        except Exception as e:
            logging.debug(f"Usage handler skipped an LLM run: {e}")

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        try:
            usage = getattr(outputs, "usage", None)
            if usage is not None:
                self.tracker.record_jev(
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                )
        except Exception as e:
            logging.debug(f"Usage handler skipped a chain run: {e}")
