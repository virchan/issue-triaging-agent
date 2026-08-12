from __future__ import annotations

import datetime as dt
from typing import Any, Self

import httpx
from pydantic import BaseModel

GITHUB_API_BASE = "https://api.github.com"


class GitHubClientError(RuntimeError):
    """Raised when the GitHub API cannot complete a request."""


class GitHubIssue(BaseModel):
    """A minimal, typed view of a GitHub issue relevant to triage."""

    number: int
    title: str
    body: str | None
    author_login: str
    created_at: dt.datetime
    html_url: str


def _parse_issue(item: dict[str, Any]) -> GitHubIssue:
    return GitHubIssue(
        number=item["number"],
        title=item["title"],
        body=item.get("body"),
        author_login=item["user"]["login"],
        created_at=item["created_at"],
        html_url=item["html_url"],
    )


class GitHubClient:
    """Small read-only wrapper around the GitHub Search API."""

    def __init__(self, token: str | None = None, timeout: float = 30.0) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "issue-triaging-agent",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._client = httpx.Client(
            base_url=GITHUB_API_BASE,
            headers=headers,
            timeout=timeout,
        )

    def fetch_issues_created_on(
        self,
        owner: str,
        repo: str,
        date: dt.date,
    ) -> list[GitHubIssue]:
        """Fetch issues (not pull requests) created on the given UTC date."""

        query = f"repo:{owner}/{repo} is:issue created:{date.isoformat()}"

        issues: list[GitHubIssue] = []
        page = 1
        while True:
            try:
                response = self._client.get(
                    "/search/issues",
                    params={"q": query, "per_page": 100, "page": page},
                )
                response.raise_for_status()
            except httpx.HTTPError as error:
                raise GitHubClientError(
                    "GitHub could not complete the issue search."
                ) from error

            items = response.json().get("items", [])
            issues.extend(_parse_issue(item) for item in items)

            if len(items) < 100:
                break
            page += 1

        return issues

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
