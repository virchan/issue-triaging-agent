from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.db import connect, save_issue_snapshot, save_issue_snapshots
from src.github_client import GitHubIssue


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(
        number=34649,
        title="Add option to drop infrequent categories",
        body="Description",
        author_login="lorentzenchr",
        created_at=dt.datetime(2026, 8, 4, 18, 2, 54, tzinfo=dt.UTC),
        html_url="https://github.com/scikit-learn/scikit-learn/issues/34649",
    )


def _mock_connection(mocker: Any) -> Any:
    connection = mocker.MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    return connection, cursor


def test_save_issue_snapshot_returns_new_id_on_insert(
    mocker: Any, issue: GitHubIssue
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (7,)

    result_id = save_issue_snapshot(
        connection, "scikit-learn", "scikit-learn", issue, is_bot=False
    )

    assert result_id == 7
    # Only the INSERT ran; no fallback SELECT needed since it returned a row.
    assert cursor.execute.call_count == 1
    sql = cursor.execute.call_args.args[0]
    assert "INSERT INTO issues" in sql
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql


def test_save_issue_snapshot_falls_back_to_select_on_conflict(
    mocker: Any, issue: GitHubIssue
) -> None:
    connection, cursor = _mock_connection(mocker)
    # First fetchone (after INSERT) is None -> conflict; second (after
    # SELECT) returns the existing row's id.
    cursor.fetchone.side_effect = [None, (3,)]

    result_id = save_issue_snapshot(
        connection, "scikit-learn", "scikit-learn", issue, is_bot=False
    )

    assert result_id == 3
    assert cursor.execute.call_count == 2
    select_sql = cursor.execute.call_args_list[1].args[0]
    assert "SELECT id FROM issues" in select_sql


def test_save_issue_snapshot_passes_is_bot_flag(
    mocker: Any, issue: GitHubIssue
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (1,)

    save_issue_snapshot(connection, "scikit-learn", "scikit-learn", issue, is_bot=True)

    params = cursor.execute.call_args.args[1]
    assert params[-1] is True  # is_bot is the last bound parameter


def test_save_issue_snapshots_tags_both_partitions_and_commits(
    mocker: Any, issue: GitHubIssue
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.side_effect = [(1,), (2,)]

    bot_issue = issue.model_copy(
        update={"number": 34717, "author_login": "scikit-learn-bot"}
    )

    ids = save_issue_snapshots(
        connection,
        "scikit-learn",
        "scikit-learn",
        non_bot_issues=[issue],
        bot_issues=[bot_issue],
    )

    assert ids == [1, 2]
    first_call_params, second_call_params = (
        call.args[1] for call in cursor.execute.call_args_list
    )
    assert first_call_params[-1] is False
    assert second_call_params[-1] is True
    connection.commit.assert_called_once()


def test_connect_reads_database_url_from_environment(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/test")
    mock_connect = mocker.patch("src.db.psycopg.connect")

    connect()

    mock_connect.assert_called_once_with("postgresql://example/test")
