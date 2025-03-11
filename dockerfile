# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install UV and dependencies
RUN pip install --no-cache-dir uv

# Install dependencies from pyproject.toml
RUN uv pip install --system -e .
#UV pip compile pyproject 
# OR try this alternative if the above doesn't work
# RUN uv pip install flask flask-limiter flask-wtf flask-talisman pandas psycopg2-binary praw python-dotenv

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app 
ENV PYTHONUNBUFFERED=1

# logging
ENV LOG_LEVEL=INFO


# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "application/app.py"]