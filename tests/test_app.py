from __future__ import annotations

import datetime as dt
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import app
from src.daily_job import DailyCycleResult
from src.db import JudgmentAuditEntry
from src.digest import DigestContent
from src.github_client import GitHubClientError
from src.pipeline import PipelineResult


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_ok_when_db_reachable(mocker: Any, client: TestClient) -> None:
    connection = mocker.MagicMock()
    mocker.patch("app.connect", return_value=connection)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_503_when_db_unreachable(mocker: Any, client: TestClient) -> None:
    mocker.patch(
        "app.connect",
        side_effect=RuntimeError(
            "connection to server at ep-summer-hill-axt2hw18-pooler.c-4."
            "us-east-2.aws.neon.tech failed"
        ),
    )

    response = client.get("/health")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == "Database unavailable."
    assert "neon.tech" not in detail


def test_judgments_returns_audit_trail(mocker: Any, client: TestClient) -> None:
    entries = [
        JudgmentAuditEntry(
            github_number=34649,
            title="Some issue",
            suggested_labels=["Bug"],
            is_spam=False,
            priority="medium",
            confidence=0.9,
            digest_date=dt.date(2026, 8, 4),
            digest_state="reviewed",
            correction_text=None,
        )
    ]
    mocker.patch("app.connect", return_value=mocker.MagicMock())
    get_trail = mocker.patch("app.get_judgment_audit_trail", return_value=entries)

    response = client.get("/judgments?limit=5")

    assert response.status_code == 200
    assert response.json()[0]["github_number"] == 34649
    assert get_trail.call_args.kwargs["limit"] == 5


def test_judgments_rejects_limit_above_500(mocker: Any, client: TestClient) -> None:
    mocker.patch("app.connect", return_value=mocker.MagicMock())
    get_trail = mocker.patch("app.get_judgment_audit_trail", return_value=[])

    response = client.get("/judgments?limit=501")

    assert response.status_code == 422
    get_trail.assert_not_called()


def test_judgments_rejects_non_positive_limit(mocker: Any, client: TestClient) -> None:
    mocker.patch("app.connect", return_value=mocker.MagicMock())
    get_trail = mocker.patch("app.get_judgment_audit_trail", return_value=[])

    response = client.get("/judgments?limit=0")

    assert response.status_code == 422
    get_trail.assert_not_called()


def test_trigger_requires_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIGGER_SECRET", "the-real-secret")

    response = client.post("/trigger")

    assert response.status_code == 401


def test_trigger_rejects_wrong_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIGGER_SECRET", "the-real-secret")

    response = client.post("/trigger", headers={"X-Trigger-Token": "wrong"})

    assert response.status_code == 401


def test_trigger_runs_pipeline_with_correct_token(
    mocker: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIGGER_SECRET", "the-real-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("SHADOW_REPO_TOKEN", "fake-token")

    mocker.patch("app.connect", return_value=mocker.MagicMock())
    mocker.patch("app.GitHubClient", return_value=mocker.MagicMock())
    mocker.patch("app.GeminiJudge", return_value=mocker.MagicMock())

    cycle_result = DailyCycleResult(
        pipeline=PipelineResult(fetched=2, bot_excluded=1, judged=1, already_judged=0),
        digest=DigestContent(digest_id=7, title="t", body="b", issue_count=1),
        published=(9, "https://example.com/9"),
        reviews=[],
    )
    mocker.patch("app.run_daily_cycle", return_value=cycle_result)

    response = client.post("/trigger", headers={"X-Trigger-Token": "the-real-secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["digest_id"] == 7
    assert body["published_issue_number"] == 9
    assert body["pipeline"]["fetched"] == 2
    assert body["reviews"] == []


def test_trigger_does_not_pass_a_date_kwarg(
    mocker: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The poll window is computed inside run_daily_cycle, not by the
    caller. /trigger must not resurrect a "today" concept of its own."""

    monkeypatch.setenv("TRIGGER_SECRET", "the-real-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("SHADOW_REPO_TOKEN", "fake-token")

    mocker.patch("app.connect", return_value=mocker.MagicMock())
    mocker.patch("app.GitHubClient", return_value=mocker.MagicMock())
    mocker.patch("app.GeminiJudge", return_value=mocker.MagicMock())

    cycle_result = DailyCycleResult(
        pipeline=PipelineResult(fetched=0, bot_excluded=0, judged=0, already_judged=0),
        digest=DigestContent(digest_id=1, title="t", body="b", issue_count=0),
        published=None,
        reviews=[],
    )
    run_daily_cycle = mocker.patch("app.run_daily_cycle", return_value=cycle_result)

    client.post("/trigger", headers={"X-Trigger-Token": "the-real-secret"})

    assert "date" not in run_daily_cycle.call_args.kwargs


def test_trigger_returns_502_on_github_failure(
    mocker: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIGGER_SECRET", "the-real-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("SHADOW_REPO_TOKEN", "fake-token")

    mocker.patch("app.connect", return_value=mocker.MagicMock())
    mocker.patch("app.GitHubClient", return_value=mocker.MagicMock())
    mocker.patch("app.GeminiJudge", return_value=mocker.MagicMock())
    mocker.patch(
        "app.run_daily_cycle",
        side_effect=GitHubClientError("scikit-learn's API is unreachable"),
    )

    response = client.post("/trigger", headers={"X-Trigger-Token": "the-real-secret"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail == "A dependency failed to complete the run."
    assert "unreachable" not in detail


def test_trigger_returns_502_on_db_failure(
    mocker: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIGGER_SECRET", "the-real-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("SHADOW_REPO_TOKEN", "fake-token")

    mocker.patch("app.connect", return_value=mocker.MagicMock())
    mocker.patch("app.GitHubClient", return_value=mocker.MagicMock())
    mocker.patch("app.GeminiJudge", return_value=mocker.MagicMock())
    mocker.patch(
        "app.run_daily_cycle",
        side_effect=psycopg.OperationalError(
            "connection to server at ep-summer-hill-axt2hw18-pooler.c-4."
            "us-east-2.aws.neon.tech failed"
        ),
    )

    response = client.post("/trigger", headers={"X-Trigger-Token": "the-real-secret"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail == "A dependency failed to complete the run."
    assert "neon.tech" not in detail
