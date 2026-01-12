# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app 

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/

EXPOSE 5001

CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5001"]