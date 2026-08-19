from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.db import (
    connect,
    get_all_reviewed_judgments,
    get_judged_issues_by_numbers,
    get_recent_reviewed_judgments,
    get_unreviewed_digests,
    save_correction,
    save_issue_snapshot,
    save_issue_snapshots,
)
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

    ids_by_number = save_issue_snapshots(
        connection,
        "scikit-learn",
        "scikit-learn",
        non_bot_issues=[issue],
        bot_issues=[bot_issue],
    )

    assert ids_by_number == {issue.number: 1, bot_issue.number: 2}
    first_call_params, second_call_params = (
        call.args[1] for call in cursor.execute.call_args_list
    )
    assert first_call_params[-1] is False
    assert second_call_params[-1] is True
    connection.commit.assert_called_once()


def test_save_correction_returns_new_id_on_insert(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (9,)

    result_id = save_correction(
        connection, 501, 1, "text", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )

    assert result_id == 9
    assert cursor.execute.call_count == 1
    sql = cursor.execute.call_args.args[0]
    assert "ON CONFLICT (github_comment_id, judgment_id)" in sql


def test_save_correction_falls_back_to_select_on_conflict(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.side_effect = [None, (4,)]

    result_id = save_correction(
        connection, 501, 1, "text", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )

    assert result_id == 4
    assert cursor.execute.call_count == 2
    select_sql, select_params = cursor.execute.call_args_list[1].args
    assert "github_comment_id = %s AND judgment_id = %s" in select_sql
    assert select_params == (1, 501)


def test_save_correction_allows_the_same_comment_for_a_different_judgment(
    mocker: Any,
) -> None:
    """Regression test: one GitHub comment correcting several issues must
    be able to save a correction per judgment, not just the first -
    uniqueness is (comment, judgment), not comment alone."""

    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (1,)

    save_correction(
        connection, 501, 1, "text a", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )
    save_correction(
        connection, 502, 1, "text b", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )

    assert cursor.execute.call_count == 2
    first_params = cursor.execute.call_args_list[0].args[1]
    second_params = cursor.execute.call_args_list[1].args[1]
    assert first_params[0] == 501
    assert second_params[0] == 502
    assert first_params[1] == second_params[1] == 1


def test_get_recent_reviewed_judgments_maps_rows_and_passes_limit(
    mocker: Any,
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = [
        (
            "Title A",
            "Body A",
            "Bug",
            False,
            "summary A",
            "high",
            "rationale A",
            0.9,
            "Fix the label",
        ),
        ("Title B", None, None, False, "summary B", "low", "rationale B", 0.5, None),
    ]

    results = get_recent_reviewed_judgments(connection, limit=5)

    assert len(results) == 2
    assert results[0].issue_title == "Title A"
    assert results[0].correction_text == "Fix the label"
    assert results[0].judgment.suggested_label == "Bug"
    assert results[1].correction_text is None
    assert results[1].judgment.suggested_label is None

    sql, params = cursor.execute.call_args.args
    assert "state = 'reviewed'" in sql
    assert params == (5,)


def test_get_unreviewed_digests_maps_rows(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = [
        (
            3,
            "virchan",
            "issue-triaging-agent-digests",
            6,
            dt.datetime(2026, 8, 15, 0, 0, 0, tzinfo=dt.UTC),
        ),
        (
            4,
            "virchan",
            "issue-triaging-agent-digests",
            7,
            dt.datetime(2026, 8, 16, 0, 0, 0, tzinfo=dt.UTC),
        ),
    ]

    results = get_unreviewed_digests(connection)

    assert len(results) == 2
    assert results[0].digest_id == 3
    assert results[0].shadow_issue_number == 6
    assert results[0].window_end == dt.datetime(2026, 8, 15, 0, 0, 0, tzinfo=dt.UTC)
    assert results[1].digest_id == 4
    sql = cursor.execute.call_args.args[0]
    assert "state = 'published'" in sql


def test_get_all_reviewed_judgments_maps_rows_unbounded(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = [
        (
            34648,
            "Title A",
            "Body A",
            "Documentation",
            False,
            "summary",
            "low",
            "rationale",
            0.95,
            "#34648 should be labelled as array API.",
            dt.date(2026, 8, 4),
        ),
    ]

    results = get_all_reviewed_judgments(connection)

    assert len(results) == 1
    assert results[0].github_number == 34648
    assert results[0].digest_date == dt.date(2026, 8, 4)
    assert results[0].correction_text == "#34648 should be labelled as array API."
    sql = cursor.execute.call_args.args[0]
    assert "LIMIT" not in sql


def test_get_judged_issues_by_numbers_returns_empty_without_querying(
    mocker: Any,
) -> None:
    connection, cursor = _mock_connection(mocker)

    result = get_judged_issues_by_numbers(
        connection, "scikit-learn", "scikit-learn", []
    )

    assert result == []
    cursor.execute.assert_not_called()


def test_get_judged_issues_by_numbers_maps_rows(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = [
        (
            1,
            501,
            34648,
            "Title A",
            "https://github.com/scikit-learn/scikit-learn/issues/34648",
            "scikit-learn",
            "scikit-learn",
            "Documentation",
            False,
            "summary",
            "low",
            "rationale",
            0.95,
        ),
    ]

    result = get_judged_issues_by_numbers(
        connection, "scikit-learn", "scikit-learn", [34648]
    )

    assert len(result) == 1
    assert result[0].github_number == 34648
    assert result[0].judgment.suggested_label == "Documentation"
    sql, params = cursor.execute.call_args.args
    assert "i.github_number = ANY(%s)" in sql
    assert params == ("scikit-learn", "scikit-learn", [34648])


def test_connect_reads_database_url_from_environment(
    mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/test")
    mock_connect = mocker.patch("src.db.psycopg.connect")

    connect()

    mock_connect.assert_called_once_with("postgresql://example/test")
