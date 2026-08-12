from __future__ import annotations

import datetime as dt

import pytest

from src.bot_filter import is_bot_issue, partition_bot_issues
from src.github_client import GitHubIssue


def _issue(number: int, author_login: str) -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=f"Issue {number}",
        body=None,
        author_login=author_login,
        created_at=dt.datetime(2026, 8, 4, tzinfo=dt.UTC),
        html_url=f"https://github.com/scikit-learn/scikit-learn/issues/{number}",
    )


@pytest.mark.parametrize(
    "author_login",
    [
        "scikit-learn-bot",
        "Scikit-Learn-Bot",
        "dependabot[bot]",
        "github-actions[bot]",
    ],
)
def test_is_bot_issue_true_for_known_bot_logins(author_login: str) -> None:
    assert is_bot_issue(_issue(1, author_login)) is True


@pytest.mark.parametrize(
    "author_login",
    ["lorentzenchr", "cakedev0", "scikit-learn-botfan"],
)
def test_is_bot_issue_false_for_human_logins(author_login: str) -> None:
    assert is_bot_issue(_issue(1, author_login)) is False


def test_partition_bot_issues_splits_and_preserves_order() -> None:
    issues = [
        _issue(1, "lorentzenchr"),
        _issue(2, "scikit-learn-bot"),
        _issue(3, "cakedev0"),
        _issue(4, "dependabot[bot]"),
    ]

    non_bot, bot = partition_bot_issues(issues)

    assert [issue.number for issue in non_bot] == [1, 3]
    assert [issue.number for issue in bot] == [2, 4]


def test_partition_bot_issues_handles_empty_list() -> None:
    assert partition_bot_issues([]) == ([], [])
