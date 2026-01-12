# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (system-wide, no venv needed in Docker)
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY src/ src/

# Expose port
EXPOSE 5001

# Run the application
# We use the CLI command directly, which is best practice for Docker
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5001"]