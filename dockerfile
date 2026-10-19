FROM python:3.13-slim

RUN pip install uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY pyproject.toml uv.lock ./

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5001/').status==200 else 1)"

CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5001"]