"""One-time (safe to re-run) backfill of duplicate-candidate embeddings.

LOG.md entry 75: the operator's real adjudicated duplicate examples came
from scikit-learn's broader issue history, not just issues this agent
has triaged - so the duplicate-candidate pool needs backfilling with a
real slice of that history, not only issues judged going forward.
Scoped to the last 2 years (the operator's own choice - a bounded,
realistic window, not scikit-learn's entire multi-year history).

Chunked into 7-day windows, not one big query: GitHub's Search API caps
results at 1000 per query, and scikit-learn's real issue volume over
even a month could plausibly approach that. A short delay between
chunks keeps this comfortably under the Search API's own (stricter than
the regular REST API's) rate limit.

Idempotent and resumable: already-embedded issues are skipped (checked
per issue, not per chunk), and each chunk commits independently, so a
crash partway through loses at most one chunk's progress, not the whole
run.

Run with:
    uv run python -m scripts.backfill_issue_embeddings
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from collections.abc import Iterator

from dotenv import load_dotenv

from src.bot_filter import partition_bot_issues
from src.db import (
    connect,
    get_issue_embedding,
    save_issue_embedding,
    save_issue_snapshots,
)
from src.duplicate_detection import issue_text
from src.embeddings import EMBEDDING_MODEL, IssueEmbedder
from src.gemini_client import GeminiUnavailableError
from src.github_client import GitHubClient

LOGGER = logging.getLogger(__name__)

OWNER = "scikit-learn"
REPO = "scikit-learn"
BACKFILL_WINDOW = dt.timedelta(days=365 * 2)
CHUNK_SIZE = dt.timedelta(days=7)
DELAY_BETWEEN_CHUNKS_SECONDS = 2.0


def _chunk_windows(
    start: dt.datetime, end: dt.datetime, size: dt.timedelta
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    current = start
    while current < end:
        chunk_end = min(current + size, end)
        yield current, chunk_end
        current = chunk_end


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    github_client = GitHubClient(token=os.environ.get("GITHUB_TOKEN"))
    embedder = IssueEmbedder(api_key=os.environ["GEMINI_API_KEY"])

    window_end = dt.datetime.now(dt.UTC)
    window_start = window_end - BACKFILL_WINDOW

    total_fetched = 0
    total_embedded = 0
    total_skipped = 0
    total_failed = 0

    with connect() as connection:
        for chunk_start, chunk_end in _chunk_windows(
            window_start, window_end, CHUNK_SIZE
        ):
            issues = github_client.fetch_issues_created_between(
                OWNER, REPO, chunk_start, chunk_end
            )
            non_bot, bot = partition_bot_issues(issues)
            total_fetched += len(issues)

            ids_by_number = save_issue_snapshots(connection, OWNER, REPO, non_bot, bot)

            for issue in non_bot:
                issue_id = ids_by_number[issue.number]
                if get_issue_embedding(connection, issue_id) is not None:
                    total_skipped += 1
                    continue

                try:
                    embedding = embedder.embed(issue_text(issue.title, issue.body))
                    save_issue_embedding(
                        connection, issue_id, EMBEDDING_MODEL, embedding
                    )
                    total_embedded += 1
                except GeminiUnavailableError as error:
                    LOGGER.warning(
                        f"Embedding failed for issue #{issue.number}: {error}"
                    )
                    total_failed += 1

            connection.commit()
            LOGGER.info(
                f"Chunk {chunk_start.date()} to {chunk_end.date()}: "
                f"fetched={len(issues)}, embedded so far={total_embedded}"
            )
            time.sleep(DELAY_BETWEEN_CHUNKS_SECONDS)

    print(
        f"Done. Fetched {total_fetched} issues, embedded {total_embedded} new, "
        f"skipped {total_skipped} already-embedded, {total_failed} failed."
    )


if __name__ == "__main__":
    main()
