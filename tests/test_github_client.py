from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pytest

from src.github_client import GitHubClient, GitHubClientError


def _raw_issue(number: int, login: str = "lorentzenchr") -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "body": "Some description",
        "user": {"login": login},
        "created_at": "2026-08-04T18:02:54Z",
        "html_url": f"https://github.com/scikit-learn/scikit-learn/issues/{number}",
    }


def _mock_response(mocker: Any, items: list[dict[str, Any]]) -> Any:
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"items": items}
    return response


def test_fetch_issues_created_on_parses_response(mocker: Any) -> None:
    get = mocker.patch.object(
        httpx.Client,
        "get",
        return_value=_mock_response(mocker, [_raw_issue(34649)]),
    )

    client = GitHubClient()
    issues = client.fetch_issues_created_on(
        "scikit-learn", "scikit-learn", dt.date(2026, 8, 4)
    )

    assert len(issues) == 1
    issue = issues[0]
    assert issue.number == 34649
    assert issue.title == "Issue 34649"
    assert issue.author_login == "lorentzenchr"
    assert issue.created_at == dt.datetime(2026, 8, 4, 18, 2, 54, tzinfo=dt.UTC)
    assert issue.html_url.endswith("/34649")
    get.assert_called_once()


def test_fetch_issues_created_on_handles_null_body(mocker: Any) -> None:
    raw = _raw_issue(1)
    raw["body"] = None
    mocker.patch.object(httpx.Client, "get", return_value=_mock_response(mocker, [raw]))

    client = GitHubClient()
    issues = client.fetch_issues_created_on(
        "scikit-learn", "scikit-learn", dt.date(2026, 8, 4)
    )

    assert issues[0].body is None


def test_fetch_issues_created_on_builds_expected_query(mocker: Any) -> None:
    get = mocker.patch.object(
        httpx.Client, "get", return_value=_mock_response(mocker, [])
    )

    client = GitHubClient()
    client.fetch_issues_created_on("scikit-learn", "scikit-learn", dt.date(2026, 8, 4))

    _, kwargs = get.call_args
    assert (
        kwargs["params"]["q"]
        == "repo:scikit-learn/scikit-learn is:issue created:2026-08-04"
    )


def test_fetch_issues_created_on_paginates(mocker: Any) -> None:
    page_1 = _mock_response(mocker, [_raw_issue(i) for i in range(100)])
    page_2 = _mock_response(mocker, [_raw_issue(100)])
    get = mocker.patch.object(httpx.Client, "get", side_effect=[page_1, page_2])

    client = GitHubClient()
    issues = client.fetch_issues_created_on(
        "scikit-learn", "scikit-learn", dt.date(2026, 8, 4)
    )

    assert len(issues) == 101
    assert get.call_count == 2


def test_fetch_issues_created_on_wraps_http_errors(mocker: Any) -> None:
    response = mocker.Mock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=mocker.Mock(), response=mocker.Mock()
    )
    mocker.patch.object(httpx.Client, "get", return_value=response)

    client = GitHubClient()
    with pytest.raises(GitHubClientError):
        client.fetch_issues_created_on(
            "scikit-learn", "scikit-learn", dt.date(2026, 8, 4)
        )


def test_client_adds_authorization_header_when_token_given() -> None:
    client = GitHubClient(token="secret-token")
    assert client._client.headers["Authorization"] == "Bearer secret-token"


def test_client_omits_authorization_header_when_no_token() -> None:
    client = GitHubClient()
    assert "Authorization" not in client._client.headers
