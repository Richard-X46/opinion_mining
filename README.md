## Reddit Opinion Mining

A conversational RAG chatbot that mines Reddit for real opinions. Ask a question and it searches Reddit in real time, indexes comments in ChromaDB, and answers with citations.

- PRAW-native subreddit discovery and thread fetching (deterministic, no web search)
- Thread relevance filtering with TypeSafe Jev classifiers
- Embedding-based retrieval and ranking over a per-session corpus
- Streaming answers from Groq LLMs over SSE, with cancellable runs

**Stack:** FastAPI · LangGraph · LangChain · Groq · TypeSafe Jev · ChromaDB · PRAW
**Deploy:** Docker + Caddy on Oracle Cloud, GitHub Actions → GHCR

### Status
Experimental prototype — answers reflect indexed Reddit opinions, not facts.

🔗 Demo: [Ask Reddit](https://richard-pears.dev/opinion-mining/)

## License

[MIT License](LICENSE)
