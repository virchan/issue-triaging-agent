"""Backfill of duplicate-candidate embeddings - full sweep on first run,
incremental on every run after.

The operator's real adjudicated duplicate examples came from
scikit-learn's broader issue history, not just issues this agent has
triaged - so the duplicate-candidate pool needs backfilling with a real
slice of that history, not only issues judged going forward.
Scoped to the last 2 years on first run (the operator's own choice - a
bounded, realistic window, not scikit-learn's entire multi-year
history). Runs weekly via a scheduled Cloud Run Job -
get_backfill_state/set_backfill_state track how far a prior run got, so
a weekly re-run only fetches what's new since then (~7 days' worth, a
few seconds of work) rather than re-sweeping all 2 years every time.
OVERLAP_BUFFER re-checks a little of the prior run's own window too - a
deliberate, cheap safety margin against edge cases near the boundary,
not a sign the boundary itself is unreliable; already-embedded issues
are skipped regardless; re-checking them costs a query, not a fresh
embedding.

Chunked into 7-day windows, not one big query: GitHub's Search API caps
results at 1000 per query, and scikit-learn's real issue volume over
even a month could plausibly approach that (confirmed empirically -
1,159 real issues in the full 2-year window). A short
delay between chunks keeps this comfortably under the Search API's own
(stricter than the regular REST API's) rate limit. Chunking still
matters even once runs are incremental: a missed scheduled run could
leave a gap wider than one week.

DELAY_BETWEEN_EMBEDS_SECONDS paces individual embedding calls too, not
just chunks - the first real run lost 751 of 1,159 issues to 429 Too
Many Requests, because a chunk with 15-20 issues fired that many
embed() calls back-to-back with nothing spacing them out.
IssueEmbedder.embed() itself now also retries on a 429 specifically, so
this is a second, complementary layer - reduces how often a 429 happens
at all, rather than only recovering after it does.

Idempotent and resumable within a run: already-embedded issues are
skipped (checked per issue, not per chunk), and each chunk commits
independently, so a crash partway through loses at most one chunk's
progress, not the whole run - though since set_backfill_state is only
called once, at the very end, a crash partway through also means the
next run starts from the same point again (safe, just some repeated
work - never a gap or a skipped issue).

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
    get_backfill_state,
    get_issue_embedding,
    save_issue_embedding,
    save_issue_snapshots,
    set_backfill_state,
)
from src.duplicate_detection import issue_text
from src.embeddings import EMBEDDING_MODEL, IssueEmbedder
from src.gemini_client import GeminiUnavailableError
from src.github_client import GitHubClient
from src.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)

OWNER = "scikit-learn"
REPO = "scikit-learn"
BACKFILL_WINDOW = dt.timedelta(days=365 * 2)
OVERLAP_BUFFER = dt.timedelta(days=1)
CHUNK_SIZE = dt.timedelta(days=7)
DELAY_BETWEEN_CHUNKS_SECONDS = 2.0
DELAY_BETWEEN_EMBEDS_SECONDS = 1.0


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
    configure_logging()

    github_client = GitHubClient(token=os.environ.get("GITHUB_TOKEN"))
    embedder = IssueEmbedder(api_key=os.environ["GEMINI_API_KEY"])

    window_end = dt.datetime.now(dt.UTC)

    total_fetched = 0
    total_embedded = 0
    total_skipped = 0
    total_failed = 0

    with connect() as connection:
        last_window_end = get_backfill_state(connection)
        if last_window_end is None:
            window_start = window_end - BACKFILL_WINDOW
            LOGGER.info("No prior backfill state - running the full 2-year sweep.")
        else:
            window_start = last_window_end - OVERLAP_BUFFER
            LOGGER.info(
                f"Prior backfill reached {last_window_end.isoformat()} - "
                f"running incrementally from {window_start.isoformat()}."
            )

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
                time.sleep(DELAY_BETWEEN_EMBEDS_SECONDS)

            connection.commit()
            LOGGER.info(
                f"Chunk {chunk_start.date()} to {chunk_end.date()}: "
                f"fetched={len(issues)}, embedded so far={total_embedded}"
            )
            time.sleep(DELAY_BETWEEN_CHUNKS_SECONDS)

        # Recorded only after every chunk above has completed - see the
        # module docstring for why a crash partway through deliberately
        # doesn't advance this.
        set_backfill_state(connection, window_end)

    # A structured event, not just the print() below - the
    # backfill-trigger workflow parses these as real JSON fields (Cloud
    # Logging jsonPayload), not by regexing free text out of stdout.
    LOGGER.info(
        "Backfill run completed",
        extra={
            "event": "backfill_run",
            "fetched": total_fetched,
            "embedded": total_embedded,
            "skipped": total_skipped,
            "failed": total_failed,
        },
    )
    print(
        f"Done. Fetched {total_fetched} issues, embedded {total_embedded} new, "
        f"skipped {total_skipped} already-embedded, {total_failed} failed."
    )


if __name__ == "__main__":
    main()
