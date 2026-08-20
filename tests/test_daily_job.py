from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.corrections import CaptureResult
from src.daily_job import WINDOW_DURATION, run_daily_cycle
from src.db import UnreviewedDigest
from src.digest import DigestContent
from src.pipeline import BACKLOG_CAP, PipelineResult

_EMPTY = PipelineResult(fetched=0, bot_excluded=0, judged=0, already_judged=0)


def _unreviewed(
    digest_id: int,
    shadow_issue_number: int,
    window_end: dt.datetime = dt.datetime(2026, 8, 16, 12, 0, 0, tzinfo=dt.UTC),
) -> UnreviewedDigest:
    return UnreviewedDigest(
        digest_id=digest_id,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=shadow_issue_number,
        window_end=window_end,
    )


@pytest.fixture
def clients_and_connection(mocker: Any) -> tuple[Any, Any, Any, Any]:
    return mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()


def test_run_daily_cycle_runs_forward_pipeline_and_checks_unreviewed_digests(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    pipeline_result = PipelineResult(
        fetched=2, bot_excluded=0, judged=2, already_judged=0
    )
    mocker.patch("src.daily_job.fetch_and_judge", return_value=pipeline_result)

    digest = DigestContent(digest_id=5, title="t", body="b", issue_count=2)
    mocker.patch("src.daily_job.build_digest", return_value=digest)
    mocker.patch(
        "src.daily_job.publish_digest",
        return_value=(10, "https://example.com/10"),
    )

    unreviewed = [_unreviewed(3, 6), _unreviewed(4, 7)]
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=unreviewed)
    github_client.fetch_labels.return_value = ["Bug"]
    mocker.patch("src.daily_job.get_recent_reviewed_judgments", return_value=[])

    capture_results = [
        CaptureResult(issue_still_open=True, already_reviewed=False),
        CaptureResult(issue_still_open=False, already_reviewed=False, captured=1),
    ]
    capture_mock = mocker.patch(
        "src.daily_job.capture_corrections", side_effect=capture_results
    )

    result = run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    assert result.pipeline is pipeline_result
    assert result.digest is digest
    assert result.published == (10, "https://example.com/10")
    assert result.reviews == capture_results

    assert capture_mock.call_count == 2
    first_call = capture_mock.call_args_list[0]
    assert first_call.kwargs["digest_id"] == 3
    assert first_call.kwargs["shadow_issue_number"] == 6
    assert first_call.kwargs["digest_window_end"] == unreviewed[0].window_end
    assert first_call.kwargs["known_labels"] == ["Bug"]
    second_call = capture_mock.call_args_list[1]
    assert second_call.kwargs["digest_id"] == 4
    assert second_call.kwargs["shadow_issue_number"] == 7


def test_run_daily_cycle_with_no_unreviewed_digests(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch("src.daily_job.fetch_and_judge", return_value=_EMPTY)
    mocker.patch(
        "src.daily_job.fetch_and_judge_backlog",
        return_value=(_EMPTY, []),
    )
    digest = DigestContent(digest_id=1, title="t", body="b", issue_count=0)
    mocker.patch("src.daily_job.build_digest", return_value=digest)
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])
    capture_mock = mocker.patch("src.daily_job.capture_corrections")

    result = run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    assert result.published is None
    assert result.reviews == []
    capture_mock.assert_not_called()


def test_run_daily_cycle_uses_a_fixed_lookback_window(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    """Regression test for the watermark-chain redesign: with no WIP
    digest, window_start is WINDOW_DURATION back from "now", not
    derived from any previous digest."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    fetch_and_judge_mock = mocker.patch(
        "src.daily_job.fetch_and_judge", return_value=_EMPTY
    )
    mocker.patch(
        "src.daily_job.fetch_and_judge_backlog",
        return_value=(_EMPTY, []),
    )
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=0),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    fetch_kwargs = fetch_and_judge_mock.call_args.kwargs
    build_kwargs = build_digest_mock.call_args.kwargs
    assert fetch_kwargs["window_end"] - fetch_kwargs["window_start"] == WINDOW_DURATION
    assert fetch_kwargs["cap"] is None
    assert build_kwargs["window_start"] == fetch_kwargs["window_start"]
    assert build_kwargs["window_end"] == fetch_kwargs["window_end"]
    assert build_kwargs["label"] == "Needs Triage"
    assert build_kwargs["wip_digest_issue_number"] is None


# --- Backlog catch-up orchestration ---


def test_run_daily_cycle_does_not_trigger_backlog_when_new_issues_found(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=1, bot_excluded=0, judged=1, already_judged=0
        ),
    )
    backlog_mock = mocker.patch("src.daily_job.fetch_and_judge_backlog")
    mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=1),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    result = run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    backlog_mock.assert_not_called()
    assert result.backlog is None


def test_run_daily_cycle_triggers_backlog_when_nothing_new_found(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch("src.daily_job.fetch_and_judge", return_value=_EMPTY)
    backlog_result = PipelineResult(
        fetched=2, bot_excluded=0, judged=2, already_judged=0
    )
    backlog_mock = mocker.patch(
        "src.daily_job.fetch_and_judge_backlog",
        return_value=(backlog_result, [42, 43]),
    )
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=2),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    result = run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    backlog_mock.assert_called_once_with(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        label="Needs Triage",
    )
    assert result.backlog is backlog_result
    build_kwargs = build_digest_mock.call_args.kwargs
    assert build_kwargs["backlog_issue_numbers"] == [42, 43]


def test_run_daily_cycle_does_not_trigger_backlog_without_a_label(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch("src.daily_job.fetch_and_judge", return_value=_EMPTY)
    backlog_mock = mocker.patch("src.daily_job.fetch_and_judge_backlog")
    mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=0),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    result = run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label=None,
    )

    backlog_mock.assert_not_called()
    assert result.backlog is None


# --- Digest labels ---


def test_run_daily_cycle_applies_agent_triggered_label_by_default(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=1, bot_excluded=0, judged=1, already_judged=0
        ),
    )
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=1),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    assert build_digest_mock.call_args.kwargs["labels"] == [
        "daily digest",
        "triggered-by:agent",
        "testing",
    ]


def test_run_daily_cycle_adds_manually_triggered_label_when_set(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=1, bot_excluded=0, judged=1, already_judged=0
        ),
    )
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=1),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
        manually_triggered=True,
    )

    assert build_digest_mock.call_args.kwargs["labels"] == [
        "daily digest",
        "manually-triggered",
        "testing",
    ]


def test_run_daily_cycle_omits_testing_label_when_toggled_off(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    """STILL_TESTING is a one-line toggle for when the operator considers
    the system production-ready - confirm the label actually responds
    to it rather than being hardcoded into the list."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch("src.daily_job.STILL_TESTING", False)
    mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=1, bot_excluded=0, judged=1, already_judged=0
        ),
    )
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=1),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    assert build_digest_mock.call_args.kwargs["labels"] == [
        "daily digest",
        "triggered-by:agent",
    ]


# --- WIP-digest handling ---


def test_run_daily_cycle_uses_most_recent_wip_window_end_as_start(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    older_wip = _unreviewed(
        3, 11, window_end=dt.datetime(2026, 8, 15, 0, 0, 0, tzinfo=dt.UTC)
    )
    newer_wip = _unreviewed(
        4, 12, window_end=dt.datetime(2026, 8, 16, 0, 1, 0, tzinfo=dt.UTC)
    )
    mocker.patch(
        "src.daily_job.get_unreviewed_digests", return_value=[older_wip, newer_wip]
    )
    github_client.fetch_labels.return_value = ["Bug"]
    mocker.patch("src.daily_job.get_recent_reviewed_judgments", return_value=[])
    fetch_and_judge_mock = mocker.patch(
        "src.daily_job.fetch_and_judge", return_value=_EMPTY
    )
    backlog_mock = mocker.patch("src.daily_job.fetch_and_judge_backlog")
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=5, title="t", body="b", issue_count=0),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)
    mocker.patch(
        "src.daily_job.capture_corrections",
        side_effect=[
            CaptureResult(issue_still_open=True, already_reviewed=False),
            CaptureResult(issue_still_open=True, already_reviewed=False),
        ],
    )

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    fetch_kwargs = fetch_and_judge_mock.call_args.kwargs
    assert fetch_kwargs["window_start"] == newer_wip.window_end
    assert fetch_kwargs["cap"] == BACKLOG_CAP
    backlog_mock.assert_not_called()
    build_kwargs = build_digest_mock.call_args.kwargs
    assert build_kwargs["wip_digest_issue_number"] == 12


def test_run_daily_cycle_excludes_a_wip_digest_closed_before_this_run(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    """Regression test for a real bug found on the first live day under
    this design: the operator closed digest #13 before this run started,
    but the reminder still said "Still working on #13?" - because the old
    code determined the WIP digest from get_unreviewed_digests'
    pre-backward-pass snapshot, not from whether capture_corrections'
    live GitHub check found it actually still open. Backward pass must
    run first, and WIP detection must use
    its result, not the stale pre-pass list."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    closed_digest = _unreviewed(3, 13)
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[closed_digest])
    github_client.fetch_labels.return_value = ["Bug"]
    mocker.patch("src.daily_job.get_recent_reviewed_judgments", return_value=[])
    capture_mock = mocker.patch(
        "src.daily_job.capture_corrections",
        return_value=CaptureResult(
            issue_still_open=False, already_reviewed=False, captured=1
        ),
    )
    fetch_and_judge_mock = mocker.patch(
        "src.daily_job.fetch_and_judge", return_value=_EMPTY
    )
    backlog_mock = mocker.patch(
        "src.daily_job.fetch_and_judge_backlog", return_value=(_EMPTY, [])
    )
    build_digest_mock = mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=5, title="t", body="b", issue_count=0),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)

    result = run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    capture_mock.assert_called_once_with(
        connection,
        shadow_client,
        gemini_judge,
        digest_id=3,
        digest_window_end=closed_digest.window_end,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=13,
        known_labels=["Bug"],
        recent_examples=[],
    )
    fetch_kwargs = fetch_and_judge_mock.call_args.kwargs
    assert fetch_kwargs["window_end"] - fetch_kwargs["window_start"] == WINDOW_DURATION
    assert fetch_kwargs["cap"] is None
    backlog_mock.assert_called_once()
    assert build_digest_mock.call_args.kwargs["wip_digest_issue_number"] is None
    assert result.reviews[0].captured == 1


def test_run_daily_cycle_does_not_skip_backlog_when_no_wip_exists(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    """Sanity check that the WIP branch above doesn't leak into the
    normal no-WIP path: backlog still runs when the window is empty and
    there's no unreviewed digest sitting open."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=[])
    mocker.patch("src.daily_job.fetch_and_judge", return_value=_EMPTY)
    backlog_mock = mocker.patch(
        "src.daily_job.fetch_and_judge_backlog", return_value=(_EMPTY, [])
    )
    mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=0),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    backlog_mock.assert_called_once()


def test_run_daily_cycle_reuses_unreviewed_digests_for_the_backward_pass(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    """get_unreviewed_digests should only be called once per cycle - the
    same list drives both WIP detection and correction capture, rather
    than querying twice."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    get_unreviewed_mock = mocker.patch(
        "src.daily_job.get_unreviewed_digests", return_value=[]
    )
    mocker.patch("src.daily_job.fetch_and_judge", return_value=_EMPTY)
    mocker.patch("src.daily_job.fetch_and_judge_backlog", return_value=(_EMPTY, []))
    mocker.patch(
        "src.daily_job.build_digest",
        return_value=DigestContent(digest_id=1, title="t", body="b", issue_count=0),
    )
    mocker.patch("src.daily_job.publish_digest", return_value=None)

    run_daily_cycle(
        github_client=github_client,
        shadow_client=shadow_client,
        gemini_judge=gemini_judge,
        connection=connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        label="Needs Triage",
    )

    get_unreviewed_mock.assert_called_once_with(connection)
