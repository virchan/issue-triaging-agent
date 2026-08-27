# Single image, two entrypoints - the Cloud Run Job (scripts/run_daily_job.py)
# and the Cloud Run service (app.py, via uvicorn) both run from this image,
# with the actual command chosen per Cloud Run resource at deploy time.
# Defaults to the service here since a CMD is required either way.

FROM python:3.12-slim

# uv's own distributed image ships the binary standalone - copying it in is
# faster and more reproducible than installing uv via pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, isolated from application code, so this (slow) layer
# only rebuilds when pyproject.toml/uv.lock actually change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app.py ./
COPY src/ ./src/
# Whole directory, not individual files: listing scripts by name here
# once caused a real production failure - scripts/backfill_issue_embeddings.py
# was never copied in, so its Cloud Run Job failed with
# "No module named scripts.backfill_issue_embeddings" the first time it
# ran. Copying the directory wholesale means a new production script
# never needs a Dockerfile change to actually ship - the handful of
# eval/local-only scripts that come along for the ride are harmless,
# unused unless explicitly invoked.
COPY scripts/ ./scripts/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Cloud Run injects PORT (default 8080); uvicorn reads it explicitly since
# it doesn't expand env vars in a plain CMD array.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
