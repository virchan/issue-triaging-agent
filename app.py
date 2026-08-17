"""FastAPI service for issue-triaging-agent.

Deployed as a separate Cloud Run service alongside the Cloud Run Job that
runs the daily schedule (see design-plan.md and open-questions.md item 7 /
LOG.md entry 7 for why this exists at all, rather than a bare batch job).

Run locally with:
    uv run uvicorn app:app --reload
"""

from __future__ import annotations

import logging
import os

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from src.corrections import CaptureResult
from src.daily_job import run_daily_cycle
from src.db import JudgmentAuditEntry, connect, get_judgment_audit_trail
from src.gemini_client import (
    GeminiConfigurationError,
    GeminiJudge,
    GeminiResponseError,
    GeminiUnavailableError,
)
from src.github_client import GitHubClient, GitHubClientError
from src.logging_config import configure_logging
from src.pipeline import PipelineResult

load_dotenv()
configure_logging()

LOGGER = logging.getLogger(__name__)

SOURCE_OWNER = "scikit-learn"
SOURCE_REPO = "scikit-learn"
SHADOW_OWNER = "virchan"
SHADOW_REPO = "issue-triaging-agent-digests"
TRIAGE_LABEL = "Needs Triage"
GEMINI_MODEL = "gemini-3.5-flash"

app = FastAPI(title="issue-triaging-agent")


class HealthResponse(BaseModel):
    status: str


class TriggerResponse(BaseModel):
    pipeline: PipelineResult
    digest_id: int
    issue_count: int
    published_issue_number: int | None
    published_issue_url: str | None
    reviews: list[CaptureResult]


@app.get("/health")
def health() -> HealthResponse:
    """Liveness/readiness check - actually probes the database, not just
    "the process is up", since a DB-unreachable service isn't healthy."""

    try:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as error:
        LOGGER.exception("Health check failed: database unreachable")
        raise HTTPException(status_code=503, detail="Database unavailable.") from error

    return HealthResponse(status="ok")


@app.get("/judgments")
def judgments(limit: int = Query(default=50, gt=0, le=500)) -> list[JudgmentAuditEntry]:
    """Read audit trail: recent judgments, digest status, and any correction."""

    with connect() as connection:
        return get_judgment_audit_trail(connection, limit=limit)


@app.post("/trigger")
def trigger(x_trigger_token: str | None = Header(default=None)) -> TriggerResponse:
    """Manually run the full state machine: fetch -> judge -> digest ->
    publish, plus correction capture for any previously-published digest
    not yet marked reviewed (a no-op for any whose GitHub issue isn't
    closed yet). The poll window (see LOG.md entry 53) is computed inside
    run_daily_cycle, same as the scheduled job - this endpoint runs the
    identical logic on demand, not a separate "today" concept.

    Requires a shared-secret header (TRIGGER_SECRET) - this endpoint
    spends real Gemini/GitHub API calls and posts a real issue, so it
    can't be left open.
    """

    expected_token = os.environ.get("TRIGGER_SECRET")
    if not expected_token or x_trigger_token != expected_token:
        raise HTTPException(status_code=401, detail="Missing or invalid trigger token.")

    try:
        with (
            GitHubClient(token=os.environ.get("GITHUB_TOKEN")) as github_client,
            GitHubClient(token=os.environ["SHADOW_REPO_TOKEN"]) as shadow_client,
            connect() as connection,
        ):
            gemini_judge = GeminiJudge(
                model=GEMINI_MODEL, api_key=os.environ["GEMINI_API_KEY"]
            )
            result = run_daily_cycle(
                github_client=github_client,
                shadow_client=shadow_client,
                gemini_judge=gemini_judge,
                connection=connection,
                source_owner=SOURCE_OWNER,
                source_repo=SOURCE_REPO,
                shadow_owner=SHADOW_OWNER,
                shadow_repo=SHADOW_REPO,
                label=TRIAGE_LABEL,
                manually_triggered=True,
            )
    except (
        GitHubClientError,
        GeminiUnavailableError,
        GeminiResponseError,
        GeminiConfigurationError,
        psycopg.Error,
    ) as error:
        LOGGER.exception("Triggered run failed: a dependency did not complete")
        raise HTTPException(
            status_code=502, detail="A dependency failed to complete the run."
        ) from error

    return TriggerResponse(
        pipeline=result.pipeline,
        digest_id=result.digest.digest_id,
        issue_count=result.digest.issue_count,
        published_issue_number=result.published[0] if result.published else None,
        published_issue_url=result.published[1] if result.published else None,
        reviews=result.reviews,
    )
