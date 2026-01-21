FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/

# Create data directory for uploads
RUN mkdir -p data/uploads

# Railway sets the PORT environment variable
# Default to 8001 if PORT is not set
ENV PORT=8001

# Expose the port
EXPOSE $PORT

# Run the application using Railway's PORT
CMD uv run uvicorn src.app.api:app --host 0.0.0.0 --port $PORT
