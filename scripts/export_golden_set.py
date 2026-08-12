"""Export the golden evaluation set from real reviewed triage history.

Overwrites eval/golden_set.json with the complete current state - the
database is the source of truth, this file is a versioned, CI-usable
snapshot of it (see LOG.md for why: CI runs credential-free, with no
database access, so the regression suite in Step 24 needs a static file
rather than a live query).

Run with:
    uv run python -m scripts.export_golden_set
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from src.db import GoldenExample, connect, get_all_reviewed_judgments

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_set.json"


def _serialize(example: GoldenExample) -> dict:
    return {
        "github_number": example.github_number,
        "issue_title": example.issue_title,
        "issue_body": example.issue_body,
        "judgment": example.judgment.model_dump(),
        "correction_text": example.correction_text,
        "digest_date": example.digest_date.isoformat(),
    }


def main() -> None:
    load_dotenv()

    with connect() as connection:
        examples = get_all_reviewed_judgments(connection)

    OUTPUT_PATH.write_text(
        json.dumps([_serialize(example) for example in examples], indent=2) + "\n"
    )

    print(f"Exported {len(examples)} golden examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
