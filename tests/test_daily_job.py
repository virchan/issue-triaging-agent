from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.corrections import CaptureResult
from src.daily_job import BOOTSTRAP_WINDOW, run_daily_cycle
from src.db import UnreviewedDigest
from src.digest import DigestContent
from src.pipeline import PipelineResult


@pytest.fixture
def clients_and_connection(mocker: Any) -> tuple[Any, Any, Any, Any]:
    return mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()


def test_run_daily_cycle_runs_forward_pipeline_and_checks_unreviewed_digests(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch(
        "src.daily_job.get_latest_digest_window_end",
        return_value=dt.datetime(2026, 8, 11, 20, 0, 0, tzinfo=dt.UTC),
    )

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

    mocker.patch("src.daily_job.get_latest_digest_window_end", return_value=None)
    mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=0, bot_excluded=0, judged=0, already_judged=0
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


def test_run_daily_cycle_uses_previous_digest_window_end_as_start(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    """Regression test for the real 2026-08-13/14 incident (see LOG.md
    entry 53): window_start must chain from the previous digest's
    window_end, not be derived from "today" in any timezone."""

    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    previous_window_end = dt.datetime(2026, 8, 13, 23, 47, 31, tzinfo=dt.UTC)
    mocker.patch(
        "src.daily_job.get_latest_digest_window_end", return_value=previous_window_end
    )
    fetch_and_judge_mock = mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=0, bot_excluded=0, judged=0, already_judged=0
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
    assert fetch_kwargs["window_start"] == previous_window_end
    assert build_kwargs["window_start"] == previous_window_end
    assert fetch_kwargs["window_end"] == build_kwargs["window_end"]
    assert build_kwargs["label"] == "Needs Triage"


def test_run_daily_cycle_bootstraps_window_when_no_previous_digest(
    mocker: Any, clients_and_connection: tuple[Any, Any, Any, Any]
) -> None:
    github_client, shadow_client, gemini_judge, connection = clients_and_connection

    mocker.patch("src.daily_job.get_latest_digest_window_end", return_value=None)
    fetch_and_judge_mock = mocker.patch(
        "src.daily_job.fetch_and_judge",
        return_value=PipelineResult(
            fetched=0, bot_excluded=0, judged=0, already_judged=0
        ),
    )
    mocker.patch(
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
    assert fetch_kwargs["window_end"] - fetch_kwargs["window_start"] == BOOTSTRAP_WINDOW
