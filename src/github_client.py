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


class GitHubComment(BaseModel):
    """A minimal, typed view of a GitHub issue comment."""

    id: int
    body: str
    created_at: dt.datetime


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
    """Small wrapper around the GitHub REST/Search API.

    Read methods (fetch_issues_created_between, fetch_labels) are used against
    scikit-learn with no token or GITHUB_TOKEN. create_issue is used
    against the shadow repo with SHADOW_REPO_TOKEN - a separate instance,
    constructed with a different token, per the deliberate read/write
    credential split.
    """

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

    def fetch_issues_created_between(
        self,
        owner: str,
        repo: str,
        window_start: dt.datetime,
        window_end: dt.datetime,
        label: str | None = None,
    ) -> list[GitHubIssue]:
        """Fetch issues (not pull requests) created in [window_start, window_end).

        A relative time-range query, not a fixed calendar-day match: a
        fixed-day query requires deciding "which day, in which timezone,"
        which is exactly the decision that broke on a real scheduled run.
        A range query has no such decision to get wrong. If label is
        given, only issues carrying that exact label are returned (e.g.
        "Needs Triage") - quoted in the query since labels can contain
        spaces.
        """

        query = (
            f"repo:{owner}/{repo} is:issue "
            f"created:{window_start.isoformat()}..{window_end.isoformat()}"
        )
        if label:
            query += f' label:"{label}"'

        issues: list[GitHubIssue] = []
        page = 1
        while True:
            try:
                response = self._client.get(
                    "/search/issues",
                    params={"q": query, "per_page": 100, "page": page},
                )
                response.raise_for_status()
                items = response.json().get("items", [])
                issues.extend(_parse_issue(item) for item in items)
            except httpx.HTTPError as error:
                raise GitHubClientError(
                    "GitHub could not complete the issue search."
                ) from error
            except (KeyError, TypeError, ValueError) as error:
                raise GitHubClientError(
                    "GitHub returned an unexpected issue search response."
                ) from error

            if len(items) < 100:
                break
            page += 1

        return issues

    def fetch_open_issues_with_label(
        self,
        owner: str,
        repo: str,
        label: str,
        per_page: int = 100,
    ) -> list[GitHubIssue]:
        """Fetch currently-open issues carrying `label`, newest-created first.

        Used for backlog catch-up (see src.pipeline.fetch_and_judge_backlog)
        - when a poll finds nothing new, this looks for open issues
        carrying `label`. Newest-first, not oldest-first, per the
        operator's explicit choice. Single page (default 100): enough to
        find a small handful of candidates, not built to page through an
        entire large backlog.
        """

        query = f'repo:{owner}/{repo} is:issue is:open label:"{label}"'

        try:
            response = self._client.get(
                "/search/issues",
                params={
                    "q": query,
                    "sort": "created",
                    "order": "desc",
                    "per_page": per_page,
                },
            )
            response.raise_for_status()
            items = response.json().get("items", [])
            return [_parse_issue(item) for item in items]
        except httpx.HTTPError as error:
            raise GitHubClientError(
                "GitHub could not complete the issue search."
            ) from error
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError(
                "GitHub returned an unexpected issue search response."
            ) from error

    def fetch_labels(self, owner: str, repo: str) -> list[str]:
        """Fetch the repository's real, currently valid label names."""

        labels: list[str] = []
        page = 1
        while True:
            try:
                response = self._client.get(
                    f"/repos/{owner}/{repo}/labels",
                    params={"per_page": 100, "page": page},
                )
                response.raise_for_status()
                items = response.json()
                labels.extend(item["name"] for item in items)
            except httpx.HTTPError as error:
                raise GitHubClientError(
                    "GitHub could not complete the label listing."
                ) from error
            except (KeyError, TypeError, ValueError) as error:
                raise GitHubClientError(
                    "GitHub returned an unexpected label listing response."
                ) from error

            if len(items) < 100:
                break
            page += 1

        return labels

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> tuple[int, str]:
        """Create an issue, returning (issue_number, html_url).

        `labels` must already exist in the repo - this does not create
        them (they're created by hand).
        """

        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels

        try:
            response = self._client.post(
                f"/repos/{owner}/{repo}/issues",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["number"], data["html_url"]
        except httpx.HTTPError as error:
            raise GitHubClientError("GitHub could not create the issue.") from error
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError(
                "GitHub returned an unexpected issue-creation response."
            ) from error

    def fetch_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """Fetch a single issue's full content by number.

        Unlike get_issue_state (state only), this returns the same
        GitHubIssue shape as the search/window-based fetch methods -
        needed wherever a specific, already-known issue number has to be
        looked up directly rather than discovered by a query (e.g. the
        duplicate-detection eval script, scripts/eval_duplicate_heuristic.py).
        """

        try:
            response = self._client.get(f"/repos/{owner}/{repo}/issues/{issue_number}")
            response.raise_for_status()
            return _parse_issue(response.json())
        except httpx.HTTPError as error:
            raise GitHubClientError("GitHub could not fetch the issue.") from error
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError(
                "GitHub returned an unexpected issue response."
            ) from error

    def get_issue_state(self, owner: str, repo: str, issue_number: int) -> str:
        """Return 'open' or 'closed' for the given issue."""

        try:
            response = self._client.get(f"/repos/{owner}/{repo}/issues/{issue_number}")
            response.raise_for_status()
            return response.json()["state"]
        except httpx.HTTPError as error:
            raise GitHubClientError("GitHub could not fetch the issue.") from error
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError(
                "GitHub returned an unexpected issue response."
            ) from error

    def fetch_issue_comments(
        self, owner: str, repo: str, issue_number: int
    ) -> list[GitHubComment]:
        """Fetch all comments on an issue."""

        comments: list[GitHubComment] = []
        page = 1
        while True:
            try:
                response = self._client.get(
                    f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
                    params={"per_page": 100, "page": page},
                )
                response.raise_for_status()
                items = response.json()
                comments.extend(
                    GitHubComment(
                        id=item["id"], body=item["body"], created_at=item["created_at"]
                    )
                    for item in items
                )
            except httpx.HTTPError as error:
                raise GitHubClientError(
                    "GitHub could not fetch the issue comments."
                ) from error
            except (KeyError, TypeError, ValueError) as error:
                raise GitHubClientError(
                    "GitHub returned an unexpected issue comments response."
                ) from error

            if len(items) < 100:
                break
            page += 1

        return comments

    def create_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> int:
        """Post a comment on an issue, returning the comment id."""

        try:
            response = self._client.post(
                f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
                json={"body": body},
            )
            response.raise_for_status()
            return response.json()["id"]
        except httpx.HTTPError as error:
            raise GitHubClientError("GitHub could not post the comment.") from error
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError(
                "GitHub returned an unexpected comment response."
            ) from error

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
