# settings.py - configuration for agentic RAG architecture
import os
from dotenv import load_dotenv

load_dotenv()

# provider configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configuration (Groq free-tier IDs — see docs/state.md §4)
# llama-3.1-8b-instant / llama-3.3-70b-versatile were shut down 2026-08-16.
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "openai/gpt-oss-20b")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")
INPUT_SAFETY_MODEL = os.getenv("INPUT_SAFETY_MODEL", "meta-llama/llama-prompt-guard-2-86m")
OUTPUT_SAFETY_MODEL = os.getenv("OUTPUT_SAFETY_MODEL", "openai/gpt-oss-safeguard-20b")
# Safety switches. Input guard (prompt-guard-86m) is cheap and kept on; the
# output guard (safeguard-20b, 3 RPM) is the rate-limit cliff, defaulted off
# while building. Flip via env, no code change needed.
INPUT_SAFETY_ENABLED = os.getenv("INPUT_SAFETY_ENABLED", "true").lower() == "true"
OUTPUT_SAFETY_ENABLED = os.getenv("OUTPUT_SAFETY_ENABLED", "false").lower() == "true"
# Thread-relevance judge: cheap model, one batched call per ingest.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", REWRITE_MODEL)
JUDGE_ENABLED = os.getenv("JUDGE_ENABLED", "true").lower() == "true"

# TypeSafe Jev — a classifier, not a chat model. Used for every DECISION in
# the pipeline (safety, intent, thread judge) via /v1/systemone. Groq stays
# for everything that produces text (rewrite, final answer).
JEV_API_KEY = os.getenv("JEV_API_KEY", "")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
# "jev" uses Jev for the decision nodes; "groq" keeps the old LLM-judge path.
# Fall back to "groq" automatically if no key is set.
DECISION_BACKEND = os.getenv("DECISION_BACKEND", "jev" if JEV_API_KEY else "groq").lower()
# Jev Noul probability above which a thread is kept by the judge.
# Kept low-ish on purpose: a wrong drop is unrecoverable, a wrong keep is
# just embedding + retrieval noise for the ranker to sort out.
JUDGE_THRESHOLD = float(os.getenv("JUDGE_THRESHOLD", "0.35"))

# Retrieval configuration (free tier is 8K TPM — keep this small)
TOP_K = int(os.getenv("TOP_K", "10"))

# Min cosine similarity (query ↔ comment) for a doc to enter the prompt
# and for its URL to be cited. Dynamic rule in node_retrieve: keep docs
# within SIM_MARGIN of the best match, never below SIM_FLOOR.
SIM_FLOOR = float(os.getenv("SIM_FLOOR", "0.15"))
SIM_MARGIN = float(os.getenv("SIM_MARGIN", "0.20"))

# Discord webhook configuration
DISCORD_ID = os.getenv("DISCORD_ID")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_WEBHOOK_URL = f"https://discord.com/api/webhooks/{DISCORD_ID}/{DISCORD_TOKEN}"

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "opinions")

# reddit
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")