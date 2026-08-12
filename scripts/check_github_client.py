"""Manual live check for src.github_client against the real GitHub API.

Run with:
    uv run python -m scripts.check_github_client
"""

from __future__ import annotations

import datetime as dt
import os

from dotenv import load_dotenv

from src.github_client import GitHubClient


def main() -> None:
    load_dotenv()

    token = os.environ.get("GITHUB_TOKEN")
    today = dt.datetime.now(dt.UTC).date()

    with GitHubClient(token=token) as client:
        issues = client.fetch_issues_created_on(
            owner="scikit-learn",
            repo="scikit-learn",
            date=today,
        )

    print(f"Authenticated: {token is not None}")
    print(f"Issues created on {today.isoformat()}: {len(issues)}")
    for issue in issues:
        print(f"  #{issue.number} by {issue.author_login} - {issue.title}")


if __name__ == "__main__":
    main()
