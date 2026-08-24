"""Post a Deploy workflow status report as an issue in issue-triaging-agent
itself (the code repo, not the digests repo) - LOG.md entry 80.

Run from .github/workflows/deploy.yml with real values as arguments.
Uses GitHub Actions' own automatic, repo-scoped GITHUB_TOKEN - no new
secret, since the issue is created in the same repo the workflow runs in.

Run with:
    uv run python -m scripts.report_deploy_status \
        --success true --commit-sha abc1234 \
        --image-tag us-east1-docker.pkg.dev/.../app:latest \
        --revision issue-triaging-agent-00020-xyz \
        --health-check-ok true \
        --run-url https://github.com/virchan/issue-triaging-agent/actions/runs/123
"""

from __future__ import annotations

import argparse
import os

from src.ci_status import format_deploy_status_report
from src.github_client import GitHubClient

OWNER = "virchan"
REPO = "issue-triaging-agent"
LABEL = "CI: status report"


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--health-check-ok", required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()

    title, body = format_deploy_status_report(
        success=_bool(args.success),
        commit_sha=args.commit_sha,
        image_tag=args.image_tag,
        revision=args.revision or None,
        health_check_ok=_bool(args.health_check_ok),
        run_url=args.run_url,
    )

    client = GitHubClient(token=os.environ["GITHUB_TOKEN"])
    _, html_url = client.create_issue(OWNER, REPO, title, body, labels=[LABEL])
    print(f"Posted status report: {html_url}")


if __name__ == "__main__":
    main()
