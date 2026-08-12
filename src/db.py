from __future__ import annotations

import os
from typing import Any

import psycopg

from src.github_client import GitHubIssue


def connect() -> psycopg.Connection[Any]:
    """Open a connection using DATABASE_URL from the environment."""

    return psycopg.connect(os.environ["DATABASE_URL"])


def save_issue_snapshot(
    connection: psycopg.Connection[Any],
    repo_owner: str,
    repo_name: str,
    issue: GitHubIssue,
    is_bot: bool,
) -> int:
    """Insert an issue snapshot, or return the existing row's id if already stored.

    Idempotent on (repo_owner, repo_name, github_number): re-fetching the
    same issue on a later run (e.g. after a retry) does not create a
    duplicate row or clobber a snapshot a judgment may already reference.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO issues (
                repo_owner, repo_name, github_number, title, body,
                author_login, github_created_at, html_url, is_bot
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo_owner, repo_name, github_number) DO NOTHING
            RETURNING id
            """,
            (
                repo_owner,
                repo_name,
                issue.number,
                issue.title,
                issue.body,
                issue.author_login,
                issue.created_at,
                issue.html_url,
                is_bot,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]

        cursor.execute(
            """
            SELECT id FROM issues
            WHERE repo_owner = %s AND repo_name = %s AND github_number = %s
            """,
            (repo_owner, repo_name, issue.number),
        )
        existing = cursor.fetchone()
        assert existing is not None
        return existing[0]


def save_issue_snapshots(
    connection: psycopg.Connection[Any],
    repo_owner: str,
    repo_name: str,
    non_bot_issues: list[GitHubIssue],
    bot_issues: list[GitHubIssue],
) -> list[int]:
    """Store both partitions from bot_filter.partition_bot_issues.

    Bot issues are stored too (tagged is_bot=True), not discarded, so
    there's an audit trail of what the filter excluded and why.
    """

    ids = [
        save_issue_snapshot(connection, repo_owner, repo_name, issue, is_bot=False)
        for issue in non_bot_issues
    ]
    ids += [
        save_issue_snapshot(connection, repo_owner, repo_name, issue, is_bot=True)
        for issue in bot_issues
    ]
    connection.commit()
    return ids
