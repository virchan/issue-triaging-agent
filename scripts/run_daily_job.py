"""The actual daily job entrypoint: run_daily_cycle, invoked directly.

Distinct from app.py's POST /trigger, which runs the same underlying
function but on-demand over HTTP. This script is what a Cloud Run Job
(triggered by Cloud Scheduler) would actually run - no HTTP server
involved.

Run with:
    uv run python -m scripts.run_daily_job
"""

from __future__ import annotations

import datetime as dt
import os

from dotenv import load_dotenv

from src.daily_job import run_daily_cycle
from src.db import connect
from src.gemini_client import GeminiJudge
from src.github_client import GitHubClient

SOURCE_OWNER = "scikit-learn"
SOURCE_REPO = "scikit-learn"
SHADOW_OWNER = "virchan"
SHADOW_REPO = "issue-triaging-agent-digests"
TRIAGE_LABEL = "Needs Triage"
GEMINI_MODEL = "gemini-3.5-flash"


def main() -> None:
    load_dotenv()

    today = dt.datetime.now(dt.UTC).date()

    with (
        GitHubClient() as github_client,
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
            date=today,
        )

    print(f"Pipeline: {result.pipeline}")
    print(
        f"Digest: id={result.digest.digest_id}, issue_count={result.digest.issue_count}"
    )
    print(f"Published: {result.published}")
    print(f"Reviews checked: {len(result.reviews)}")
    for review in result.reviews:
        print(f"  {review}")


if __name__ == "__main__":
    main()
