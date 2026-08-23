from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.db import (
    connect,
    get_all_issue_embeddings,
    get_all_reviewed_judgments,
    get_authoritative_correction_digest,
    get_backfill_state,
    get_correction_field_counts,
    get_issue_embedding,
    get_judged_issues_by_numbers,
    get_judgment_id_for_issue_number,
    get_recent_reviewed_judgments,
    get_rejudge_context,
    get_unreviewed_digests,
    mark_corrections_superseded,
    prune_old_issue_embeddings,
    save_correction,
    save_issue_embedding,
    save_issue_snapshot,
    save_issue_snapshots,
    set_backfill_state,
    set_correction_changed_fields,
    set_possible_duplicate,
    update_judgment,
)
from src.github_client import GitHubIssue
from src.judgment import IssueJudgment


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


def test_get_issue_embedding_returns_the_stored_vector(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = ([0.1, 0.2, 0.3],)

    assert get_issue_embedding(connection, 501) == [0.1, 0.2, 0.3]


def test_get_issue_embedding_returns_none_when_absent(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = None

    assert get_issue_embedding(connection, 501) is None


def test_save_issue_embedding_upserts_on_issue_id(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)

    save_issue_embedding(connection, 501, "gemini-embedding-001", [0.1, 0.2])

    sql, params = cursor.execute.call_args.args
    assert "ON CONFLICT (issue_id) DO UPDATE" in sql
    assert params == (501, "gemini-embedding-001", [0.1, 0.2])


def test_get_all_issue_embeddings_maps_rows(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = [
        (1, 34649, [0.1, 0.2]),
        (2, 34650, [0.3, 0.4]),
    ]

    result = get_all_issue_embeddings(connection)

    assert result == [(1, 34649, [0.1, 0.2]), (2, 34650, [0.3, 0.4])]


def test_set_possible_duplicate_updates_the_judgment_row(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)

    set_possible_duplicate(connection, 501, 34649, 0.87)

    sql, params = cursor.execute.call_args.args
    assert "UPDATE judgments" in sql
    assert "possible_duplicate_number" in sql
    assert params == (34649, 0.87, 501)


def test_set_possible_duplicate_can_clear_to_none(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)

    set_possible_duplicate(connection, 501, None, None)

    _, params = cursor.execute.call_args.args
    assert params == (None, None, 501)


def test_prune_old_issue_embeddings_returns_the_deleted_count(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.rowcount = 3
    cutoff = dt.datetime(2024, 8, 23, tzinfo=dt.UTC)

    deleted = prune_old_issue_embeddings(connection, cutoff)

    assert deleted == 3
    sql, params = cursor.execute.call_args.args
    assert "DELETE FROM issue_embeddings" in sql
    assert params == (cutoff,)


def test_get_backfill_state_returns_the_last_window_end(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    window_end = dt.datetime(2026, 8, 17, tzinfo=dt.UTC)
    cursor.fetchone.return_value = (window_end,)

    assert get_backfill_state(connection) == window_end


def test_get_backfill_state_returns_none_when_never_run(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = None

    assert get_backfill_state(connection) is None


def test_set_backfill_state_upserts_the_single_row(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    window_end = dt.datetime(2026, 8, 24, tzinfo=dt.UTC)

    set_backfill_state(connection, window_end)

    sql, params = cursor.execute.call_args.args
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert params == (window_end,)
    connection.commit.assert_called_once()


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
        connection, 501, 20, 1, "text", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )

    assert result_id == 9
    assert cursor.execute.call_count == 1
    sql = cursor.execute.call_args.args[0]
    assert "ON CONFLICT (github_comment_id, judgment_id)" in sql
    params = cursor.execute.call_args.args[1]
    assert params == (501, 20, 1, "text", dt.datetime(2026, 8, 4, tzinfo=dt.UTC), False)


def test_save_correction_falls_back_to_select_on_conflict(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.side_effect = [None, (4,)]

    result_id = save_correction(
        connection, 501, 20, 1, "text", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )

    assert result_id == 4
    assert cursor.execute.call_count == 2
    select_sql, select_params = cursor.execute.call_args_list[1].args
    assert "github_comment_id = %s AND judgment_id = %s" in select_sql
    assert select_params == (1, 501)


def test_save_correction_passes_superseded_flag(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (1,)

    save_correction(
        connection,
        501,
        20,
        1,
        "text",
        dt.datetime(2026, 8, 4, tzinfo=dt.UTC),
        superseded=True,
    )

    params = cursor.execute.call_args.args[1]
    assert params[-1] is True


def test_save_correction_allows_the_same_comment_for_a_different_judgment(
    mocker: Any,
) -> None:
    """Regression test: one GitHub comment correcting several issues must
    be able to save a correction per judgment, not just the first -
    uniqueness is (comment, judgment), not comment alone."""

    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (1,)

    save_correction(
        connection, 501, 20, 1, "text a", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )
    save_correction(
        connection, 502, 20, 1, "text b", dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    )

    assert cursor.execute.call_count == 2
    first_params = cursor.execute.call_args_list[0].args[1]
    second_params = cursor.execute.call_args_list[1].args[1]
    assert first_params[0] == 501
    assert second_params[0] == 502
    assert first_params[2] == second_params[2] == 1


def test_get_judgment_id_for_issue_number_not_scoped_to_a_digest(
    mocker: Any,
) -> None:
    """Regression test: must find a judgment by github_number alone, not
    require it to match a specific digest_id - a judgment's digest_id is
    reassigned whenever backlog catch-up re-surfaces it into a newer
    digest, so an older digest's own link can't be trusted."""

    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (501,)

    result = get_judgment_id_for_issue_number(connection, 34649)

    assert result == 501
    sql, params = cursor.execute.call_args.args
    assert "digest_id" not in sql
    assert params == (34649,)


def test_get_judgment_id_for_issue_number_returns_none_when_not_found(
    mocker: Any,
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = None

    assert get_judgment_id_for_issue_number(connection, 99999) is None


def test_mark_corrections_superseded_updates_only_non_superseded_rows(
    mocker: Any,
) -> None:
    connection, cursor = _mock_connection(mocker)

    mark_corrections_superseded(connection, 501)

    sql, params = cursor.execute.call_args.args
    assert "superseded = TRUE" in sql
    assert "superseded = FALSE" in sql
    assert params == (501,)
    # Deliberately does not commit - see the function's own docstring:
    # it's one step of a single logical event with save_correction and
    # update_judgment, which must land together.
    connection.commit.assert_not_called()


def test_get_authoritative_correction_digest_returns_most_recent(
    mocker: Any,
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (16, dt.datetime(2026, 8, 19, tzinfo=dt.UTC))

    result = get_authoritative_correction_digest(connection, 501)

    assert result == (16, dt.datetime(2026, 8, 19, tzinfo=dt.UTC))
    sql = cursor.execute.call_args.args[0]
    assert "superseded = FALSE" in sql
    assert "ORDER BY d.window_end DESC" in sql


def test_get_authoritative_correction_digest_returns_none_when_no_correction(
    mocker: Any,
) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = None

    assert get_authoritative_correction_digest(connection, 501) is None


def test_get_rejudge_context_maps_row(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = (
        "Issue title",
        "Issue body",
        "Bug",
        False,
        "summary",
        "medium",
        "rationale",
        0.8,
    )

    result = get_rejudge_context(connection, 501)

    assert result is not None
    assert result.title == "Issue title"
    assert result.body == "Issue body"
    assert result.judgment == IssueJudgment(
        suggested_label="Bug",
        is_spam=False,
        summary="summary",
        priority="medium",
        rationale="rationale",
        confidence=0.8,
    )


def test_get_rejudge_context_returns_none_when_not_found(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchone.return_value = None

    assert get_rejudge_context(connection, 501) is None


def test_update_judgment_overwrites_in_place(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    judgment = IssueJudgment(
        suggested_label="module:decomposition",
        is_spam=False,
        summary="revised summary",
        priority="medium",
        rationale="revised rationale",
        confidence=0.9,
    )

    update_judgment(connection, 501, judgment)

    sql, params = cursor.execute.call_args.args
    assert "UPDATE judgments" in sql
    assert params == (
        "module:decomposition",
        False,
        "revised summary",
        "medium",
        "revised rationale",
        0.9,
        501,
    )
    # Deliberately does not commit - see the function's own docstring.
    connection.commit.assert_not_called()


def test_set_correction_changed_fields_updates_by_id(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)

    set_correction_changed_fields(connection, 42, ["suggested_label", "priority"])

    sql, params = cursor.execute.call_args.args
    assert "UPDATE corrections" in sql
    assert "changed_fields" in sql
    assert params == (["suggested_label", "priority"], 42)
    # Deliberately does not commit - same logical event as update_judgment.
    connection.commit.assert_not_called()


def test_get_correction_field_counts_maps_rows(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = [("suggested_label", 3), ("priority", 1)]

    result = get_correction_field_counts(connection)

    assert result == {"suggested_label": 3, "priority": 1}
    sql, params = cursor.execute.call_args.args
    assert "changed_fields IS NOT NULL" in sql
    assert "unnest(changed_fields)" in sql
    assert params == []


def test_get_correction_field_counts_filters_by_since(mocker: Any) -> None:
    connection, cursor = _mock_connection(mocker)
    cursor.fetchall.return_value = []
    since = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)

    get_correction_field_counts(connection, since=since)

    sql, params = cursor.execute.call_args.args
    assert "captured_at >= %s" in sql
    assert params == [since]


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
    assert "c.superseded = FALSE" in sql
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
            "Body A",
            "https://github.com/scikit-learn/scikit-learn/issues/34648",
            "scikit-learn",
            "scikit-learn",
            "Documentation",
            False,
            "summary",
            "low",
            "rationale",
            0.95,
            None,
            None,
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
