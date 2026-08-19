from __future__ import annotations

from typing import Any

import pytest

import scripts.run_daily_job as run_daily_job_module
from scripts.run_daily_job import main


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("SHADOW_REPO_TOKEN", "fake-token")


def test_main_logs_structured_failure_and_reraises(mocker: Any, caplog: Any) -> None:
    mocker.patch.object(
        run_daily_job_module, "GitHubClient", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module, "GeminiJudge", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module, "connect", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module,
        "run_daily_cycle",
        side_effect=RuntimeError("scikit-learn's API is unreachable"),
    )

    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="unreachable"):
        main()

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "daily_cycle_failed"
    ]
    assert len(records) == 1


def test_main_does_not_log_failure_event_on_success(mocker: Any, caplog: Any) -> None:
    mocker.patch.object(
        run_daily_job_module, "GitHubClient", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module, "GeminiJudge", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module, "connect", return_value=mocker.MagicMock()
    )

    cycle_result = mocker.MagicMock()
    cycle_result.pipeline = "pipeline-summary"
    cycle_result.digest.digest_id = 1
    cycle_result.digest.issue_count = 0
    cycle_result.published = None
    cycle_result.reviews = []
    mocker.patch.object(
        run_daily_job_module, "run_daily_cycle", return_value=cycle_result
    )

    with caplog.at_level("INFO"):
        main()

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "daily_cycle_failed"
    ]
    assert records == []


def test_main_does_not_pass_a_date_kwarg(mocker: Any) -> None:
    """The poll window is computed inside run_daily_cycle, not by the
    caller. main() must not resurrect a "today" concept of its own (that
    caused a real incident: a schedule firing at 00:00 UTC/17:00 PDT
    means a UTC-based "today" is 0 seconds old at run time)."""

    mocker.patch.object(
        run_daily_job_module, "GitHubClient", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module, "GeminiJudge", return_value=mocker.MagicMock()
    )
    mocker.patch.object(
        run_daily_job_module, "connect", return_value=mocker.MagicMock()
    )

    cycle_result = mocker.MagicMock()
    cycle_result.pipeline = "pipeline-summary"
    cycle_result.digest.digest_id = 1
    cycle_result.digest.issue_count = 0
    cycle_result.published = None
    cycle_result.reviews = []
    run_daily_cycle = mocker.patch.object(
        run_daily_job_module, "run_daily_cycle", return_value=cycle_result
    )

    main()

    assert "date" not in run_daily_cycle.call_args.kwargs
