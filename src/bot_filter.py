from __future__ import annotations

from src.github_client import GitHubIssue

# Known automation accounts on the target repo that don't present as
# GitHub-flagged bot accounts: scikit-learn-bot's `user.type` is "User",
# not "Bot", so it can't be caught by type alone.
KNOWN_BOT_LOGINS = frozenset({"scikit-learn-bot"})


def is_bot_issue(issue: GitHubIssue) -> bool:
    """Whether an issue was authored by a known bot/automation account.

    Covers two patterns: an explicit denylist of accounts (like
    scikit-learn-bot) that operate as automation but report as a normal
    "User" type, and the standard GitHub App bot login convention
    (a "[bot]" suffix, e.g. "dependabot[bot]").
    """

    login = issue.author_login.casefold()
    return login in KNOWN_BOT_LOGINS or login.endswith("[bot]")


def partition_bot_issues(
    issues: list[GitHubIssue],
) -> tuple[list[GitHubIssue], list[GitHubIssue]]:
    """Split issues into (non_bot, bot) issues, preserving input order."""

    non_bot: list[GitHubIssue] = []
    bot: list[GitHubIssue] = []
    for issue in issues:
        (bot if is_bot_issue(issue) else non_bot).append(issue)
    return non_bot, bot
