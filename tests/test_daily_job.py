from __future__ import annotations

from typing import Any

import pytest

from src.corrections import CaptureResult
from src.daily_job import WINDOW_DURATION, run_daily_cycle
from src.db import UnreviewedDigest
from src.digest import DigestContent
from src.pipeline import PipelineResult

_EMPTY = PipelineResult(fetched=0, bot_excluded=0, judged=0, already_judged=0)


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

    unreviewed = [
        UnreviewedDigest(
            digest_id=3,
            shadow_owner="virchan",
            shadow_repo="issue-triaging-agent-digests",
            shadow_issue_number=6,
        ),
        UnreviewedDigest(
            digest_id=4,
            shadow_owner="virchan",
            shadow_repo="issue-triaging-agent-digests",
            shadow_issue_number=7,
        ),
    ]
    mocker.patch("src.daily_job.get_unreviewed_digests", return_value=unreviewed)

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
    second_call = capture_mock.call_args_list[1]
    assert second_call.kwargs["digest_id"] == 4
    assert second_call.kwargs["shadow_issue_number"] == 7


def test_run_daily_cycle_with_no_unreviewed_digests(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=0, bot_excluded=0, judged=0, already_judged=0
        ),
    )
    mocker.patch(
        "src.daily_job.fetch_and_judge_backlog",
        return_value=(
            PipelineResult(fetched=0, bot_excluded=0, judged=0, already_judged=0),
            [],
        ),
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
    """Regression test for the watermark-chain redesign (see LOG.md entry
    56): window_start is always WINDOW_DURATION back from "now", not
    derived from any previous digest - the operator explicitly chose
    this over the entry-53 watermark-chain design."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    fetch_and_judge_mock = mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=0, bot_excluded=0, judged=0, already_judged=0
        ),
    )
    mocker.patch(
        "src.daily_job.fetch_and_judge_backlog",
        return_value=(
            PipelineResult(fetched=0, bot_excluded=0, judged=0, already_judged=0),
            [],
        ),
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
    assert build_kwargs["window_start"] == fetch_kwargs["window_start"]
    assert build_kwargs["window_end"] == fetch_kwargs["window_end"]
    assert build_kwargs["label"] == "Needs Triage"


# --- Backlog catch-up orchestration (Phase 8 idea A - see LOG.md/daily-log.md) ---


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
