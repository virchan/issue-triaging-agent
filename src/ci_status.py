"""Status reports for the Deploy and backfill-trigger GitHub Actions
workflows - posted as a new issue in issue-triaging-agent itself (the
code repo, not the digests repo), so a workflow's real outcome can be
confirmed by fetching an issue rather than only checking the deployed
artifacts afterward (LOG.md entry 80).

Rendered via this project's existing Jinja templates (src.rendering),
not built as inline strings in the workflow YAML - the same reasoning
as entry 69's digest/acknowledgment templates.
"""

from __future__ import annotations

from src.rendering import render_template


def format_deploy_status_report(
    *,
    success: bool,
    commit_sha: str,
    image_tag: str,
    revision: str | None,
    health_check_ok: bool,
    run_url: str,
) -> tuple[str, str]:
    """Returns (title, body) for a Deploy workflow status-report issue."""

    title = f"Deploy {'succeeded' if success else 'failed'} - {commit_sha}"
    body = render_template(
        "deploy-status-report.md.jinja",
        success=success,
        commit_sha=commit_sha,
        image_tag=image_tag,
        revision=revision,
        health_check_ok=health_check_ok,
        run_url=run_url,
    )
    return title, body


def format_backfill_status_report(
    *,
    success: bool,
    fetched: int,
    embedded: int,
    skipped: int,
    failed: int,
    run_url: str,
) -> tuple[str, str]:
    """Returns (title, body) for a backfill-trigger workflow status-report issue."""

    title = f"Backfill {'succeeded' if success else 'failed'} - fetched {fetched}, embedded {embedded}, failed {failed}"
    body = render_template(
        "backfill-status-report.md.jinja",
        success=success,
        fetched=fetched,
        embedded=embedded,
        skipped=skipped,
        failed=failed,
        run_url=run_url,
    )
    return title, body
