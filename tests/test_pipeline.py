from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.db import ReviewedJudgment
from src.gemini_client import GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubIssue
from src.judgment import IssueJudgment
from src.pipeline import fetch_and_judge, fetch_and_judge_backlog

WINDOW_START = dt.datetime(2026, 8, 3, 20, 0, 0, tzinfo=dt.UTC)
WINDOW_END = dt.datetime(2026, 8, 4, 20, 0, 0, tzinfo=dt.UTC)


def _issue(number: int, author_login: str = "reporter") -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=f"Issue {number}",
        body="body",
        author_login=author_login,
        created_at=dt.datetime(2026, 8, 4, tzinfo=dt.UTC),
        html_url=f"https://github.com/scikit-learn/scikit-learn/issues/{number}",
    )


def _judgment() -> IssueJudgment:
    return IssueJudgment(
        suggested_label="Bug",
        is_spam=False,
        summary="summary",
        priority="medium",
        rationale="rationale",
        confidence=0.8,
    )


@pytest.fixture
def github_client(mocker: Any) -> Any:
    client = mocker.Mock()
    client.fetch_labels.return_value = ["Bug", "Documentation"]
    return client


@pytest.fixture
def gemini_judge(mocker: Any) -> Any:
    judge = mocker.Mock()
    judge.judge.return_value = _judgment()
    return judge


@pytest.fixture
def connection(mocker: Any) -> Any:
    return mocker.Mock()


@pytest.fixture(autouse=True)
def no_recent_examples(mocker: Any) -> Any:
    """Default: no reviewed history yet. Individual tests override this."""

    return mocker.patch("src.pipeline.get_recent_reviewed_judgments", return_value=[])


def test_fetch_and_judge_logs_poll_run_summary(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any, caplog: Any
) -> None:
    github_client.fetch_issues_created_between.return_value = [_issue(1, "human")]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={1: 101})
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    mocker.patch("src.pipeline.save_judgment", return_value=201)

    with caplog.at_level("INFO"):
        fetch_and_judge(
            github_client=github_client,
            gemini_judge=gemini_judge,
            connection=connection,
            owner="scikit-learn",
            repo="scikit-learn",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )

    records = [r for r in caplog.records if getattr(r, "event", None) == "poll_run"]
    assert len(records) == 1
    record = records[0]
    assert record.owner == "scikit-learn"
    assert record.repo == "scikit-learn"
    assert record.window_start == WINDOW_START.isoformat()
    assert record.window_end == WINDOW_END.isoformat()
    assert record.fetched == 1
    assert record.judged == 1
    assert record.failure_count == 0
    assert record.duration_seconds >= 0


def test_fetch_and_judge_happy_path(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    non_bot = _issue(1, "human")
    bot = _issue(2, "scikit-learn-bot")
    github_client.fetch_issues_created_between.return_value = [non_bot, bot]

    save_snapshots = mocker.patch(
        "src.pipeline.save_issue_snapshots", return_value={1: 101, 2: 102}
    )
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    save_judgment = mocker.patch("src.pipeline.save_judgment", return_value=201)

    result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    assert result.fetched == 2
    assert result.bot_excluded == 1
    assert result.judged == 1
    assert result.already_judged == 0
    assert result.failures == []

    save_snapshots.assert_called_once_with(
        connection, "scikit-learn", "scikit-learn", [non_bot], [bot]
    )
    gemini_judge.judge.assert_called_once_with(
        title="Issue 1",
        body="body",
        known_labels=["Bug", "Documentation"],
        recent_examples=[],
    )
    save_judgment.assert_called_once_with(connection, 101, _judgment())
    github_client.fetch_labels.assert_called_once_with("scikit-learn", "scikit-learn")
    github_client.fetch_issues_created_between.assert_called_once_with(
        "scikit-learn", "scikit-learn", WINDOW_START, WINDOW_END, label=None
    )


def test_fetch_and_judge_passes_label_through_to_the_fetch(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_issues_created_between.return_value = []
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={})

    fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        label="Needs Triage",
    )

    github_client.fetch_issues_created_between.assert_called_once_with(
        "scikit-learn", "scikit-learn", WINDOW_START, WINDOW_END, label="Needs Triage"
    )


def test_fetch_and_judge_passes_recent_examples_to_every_judge_call(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_issues_created_between.return_value = [_issue(1), _issue(2)]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={1: 101, 2: 102})
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    mocker.patch("src.pipeline.save_judgment")

    examples = [
        ReviewedJudgment(
            issue_title="Prior issue",
            issue_body="body",
            judgment=_judgment(),
            correction_text=None,
        )
    ]
    get_examples = mocker.patch(
        "src.pipeline.get_recent_reviewed_judgments", return_value=examples
    )

    fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    # Fetched once for the whole run, not once per issue.
    get_examples.assert_called_once_with(connection)
    assert gemini_judge.judge.call_count == 2
    for call in gemini_judge.judge.call_args_list:
        assert call.kwargs["recent_examples"] == examples


def test_fetch_and_judge_skips_already_judged(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    issue = _issue(1)
    github_client.fetch_issues_created_between.return_value = [issue]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={1: 101})
    mocker.patch("src.pipeline.has_judgment", return_value=True)
    save_judgment = mocker.patch("src.pipeline.save_judgment")

    result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    assert result.judged == 0
    assert result.already_judged == 1
    gemini_judge.judge.assert_not_called()
    save_judgment.assert_not_called()


def test_fetch_and_judge_continues_after_single_failure(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_issues_created_between.return_value = [_issue(1), _issue(2)]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={1: 101, 2: 102})
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    save_judgment = mocker.patch("src.pipeline.save_judgment")

    gemini_judge.judge.side_effect = [
        GeminiResponseError("bad response"),
        _judgment(),
    ]

    result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    assert result.judged == 1
    assert result.failures == [(1, "bad response")]
    save_judgment.assert_called_once_with(connection, 102, _judgment())


def test_fetch_and_judge_handles_unavailable_error(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_issues_created_between.return_value = [_issue(1)]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={1: 101})
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    save_judgment = mocker.patch("src.pipeline.save_judgment")

    gemini_judge.judge.side_effect = GeminiUnavailableError("down")

    result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    assert result.judged == 0
    assert result.failures == [(1, "down")]
    save_judgment.assert_not_called()


# --- fetch_and_judge_backlog ---


def test_fetch_and_judge_backlog_fetches_newest_first_and_judges_candidates(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_open_issues_with_label.return_value = [
        _issue(10, "human"),
        _issue(11, "human"),
    ]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={10: 110, 11: 111})
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    save_judgment = mocker.patch("src.pipeline.save_judgment")

    result, judged_numbers = fetch_and_judge_backlog(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        label="Needs Triage",
    )

    assert result.judged == 2
    assert judged_numbers == [10, 11]
    assert save_judgment.call_count == 2
    github_client.fetch_open_issues_with_label.assert_called_once_with(
        "scikit-learn", "scikit-learn", "Needs Triage"
    )


def test_fetch_and_judge_backlog_reuses_already_judged_candidates(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    """Regression test: a candidate already judged in a prior run is
    still included (reused, not excluded) - it must not silently vanish
    from the digest just because it was judged once."""

    github_client.fetch_open_issues_with_label.return_value = [
        _issue(10, "human"),
        _issue(11, "human"),
    ]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={10: 110, 11: 111})
    mocker.patch(
        "src.pipeline.has_judgment", side_effect=lambda conn, issue_id: issue_id == 110
    )
    mocker.patch("src.pipeline.save_judgment")

    _, judged_numbers = fetch_and_judge_backlog(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        label="Needs Triage",
    )

    assert set(judged_numbers) == {10, 11}
    assert gemini_judge.judge.call_count == 1


def test_fetch_and_judge_backlog_respects_the_cap(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_open_issues_with_label.return_value = [
        _issue(n, "human") for n in range(10, 20)
    ]
    mocker.patch(
        "src.pipeline.save_issue_snapshots",
        return_value={n: 100 + n for n in range(10, 20)},
    )
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    mocker.patch("src.pipeline.save_judgment")

    _, judged_numbers = fetch_and_judge_backlog(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        label="Needs Triage",
        cap=3,
    )

    assert len(judged_numbers) == 3
    assert judged_numbers == [10, 11, 12]


def test_fetch_and_judge_backlog_excludes_bots(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    non_bot = _issue(10, "human")
    bot = _issue(11, "scikit-learn-bot")
    github_client.fetch_open_issues_with_label.return_value = [non_bot, bot]
    save_snapshots = mocker.patch(
        "src.pipeline.save_issue_snapshots", return_value={10: 110, 11: 111}
    )
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    mocker.patch("src.pipeline.save_judgment")

    result, judged_numbers = fetch_and_judge_backlog(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        label="Needs Triage",
    )

    assert result.bot_excluded == 1
    assert judged_numbers == [10]
    save_snapshots.assert_called_once_with(
        connection, "scikit-learn", "scikit-learn", [non_bot], [bot]
    )


def test_fetch_and_judge_backlog_returns_only_successful_judgments(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_open_issues_with_label.return_value = [
        _issue(10, "human"),
        _issue(11, "human"),
    ]
    mocker.patch("src.pipeline.save_issue_snapshots", return_value={10: 110, 11: 111})
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    mocker.patch("src.pipeline.save_judgment")
    gemini_judge.judge.side_effect = [GeminiResponseError("bad"), _judgment()]

    result, judged_numbers = fetch_and_judge_backlog(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        label="Needs Triage",
    )

    assert judged_numbers == [11]
    assert result.failures == [(10, "bad")]


def test_fetch_and_judge_respects_an_optional_cap(
    mocker: Any, github_client: Any, gemini_judge: Any, connection: Any
) -> None:
    github_client.fetch_issues_created_between.return_value = [
        _issue(n, "human") for n in range(1, 6)
    ]
    mocker.patch(
        "src.pipeline.save_issue_snapshots",
        return_value={n: 100 + n for n in range(1, 6)},
    )
    mocker.patch("src.pipeline.has_judgment", return_value=False)
    mocker.patch("src.pipeline.save_judgment")

    result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner="scikit-learn",
        repo="scikit-learn",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        cap=2,
    )

    assert result.judged == 2
    assert gemini_judge.judge.call_count == 2
