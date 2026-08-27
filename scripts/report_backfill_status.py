"""Post a backfill-trigger workflow status report as an issue in
issue-triaging-agent itself (the code repo, not the digests repo) -
written directly in response to a real backfill run losing 65% of its
work to rate limiting with no visible signal.

Run from .github/workflows/backfill-trigger.yml, with the real
fetched/embedded/skipped/failed counts pulled from Cloud Logging by the
workflow itself and passed in as arguments. Uses GitHub Actions' own
automatic, repo-scoped GITHUB_TOKEN - no new secret.

Run with:
    uv run python -m scripts.report_backfill_status \
        --success true --fetched 1159 --embedded 257 --skipped 0 --failed 751 \
        --run-url https://github.com/virchan/issue-triaging-agent/actions/runs/123
"""

from __future__ import annotations

import argparse
import os

from src.ci_status import format_backfill_status_report
from src.github_client import GitHubClient

OWNER = "virchan"
REPO = "issue-triaging-agent"
LABEL = "CI: status report"


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success", required=True)
    parser.add_argument("--fetched", type=int, required=True)
    parser.add_argument("--embedded", type=int, required=True)
    parser.add_argument("--skipped", type=int, required=True)
    parser.add_argument("--failed", type=int, required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()

    title, body = format_backfill_status_report(
        success=_bool(args.success),
        fetched=args.fetched,
        embedded=args.embedded,
        skipped=args.skipped,
        failed=args.failed,
        run_url=args.run_url,
    )

    client = GitHubClient(token=os.environ["GITHUB_TOKEN"])
    _, html_url = client.create_issue(OWNER, REPO, title, body, labels=[LABEL])
    print(f"Posted status report: {html_url}")


if __name__ == "__main__":
    main()
