from __future__ import annotations

from src.ci_status import format_backfill_status_report, format_deploy_status_report


def test_format_deploy_status_report_success() -> None:
    title, body = format_deploy_status_report(
        success=True,
        commit_sha="abc1234",
        image_tag="us-east1-docker.pkg.dev/x/y/app:latest",
        revision="issue-triaging-agent-00020-xyz",
        health_check_ok=True,
        run_url="https://github.com/virchan/issue-triaging-agent/actions/runs/1",
    )

    assert "succeeded" in title
    assert "abc1234" in title
    assert "✅" in body
    assert "❌" not in body
    assert "`abc1234`" in body
    assert "issue-triaging-agent-00020-xyz" in body
    assert "200 OK" in body
    assert "https://github.com/virchan/issue-triaging-agent/actions/runs/1" in body


def test_format_deploy_status_report_failure() -> None:
    title, body = format_deploy_status_report(
        success=False,
        commit_sha="abc1234",
        image_tag="us-east1-docker.pkg.dev/x/y/app:latest",
        revision=None,
        health_check_ok=False,
        run_url="https://github.com/virchan/issue-triaging-agent/actions/runs/2",
    )

    assert "failed" in title
    assert "❌" in body
    assert "✅" not in body
    assert "(not reached)" in body
    assert "failed or not reached" in body


def test_format_backfill_status_report_success_no_failures() -> None:
    title, body = format_backfill_status_report(
        success=True,
        fetched=10,
        embedded=10,
        skipped=0,
        failed=0,
        run_url="https://github.com/virchan/issue-triaging-agent/actions/runs/3",
    )

    assert "succeeded" in title
    assert "✅" in body
    assert "**Fetched:** 10" in body
    assert "**Failed:** 0" in body
    # No failures - the extra warning callout shouldn't appear.
    assert "check Cloud Logging" not in body


def test_format_backfill_status_report_flags_real_failures() -> None:
    """Regression case for the real entry-80 incident: 751 of 1159
    fetched issues failed silently until checked by hand - the report
    must make a non-zero failure count impossible to miss."""

    title, body = format_backfill_status_report(
        success=True,
        fetched=1159,
        embedded=257,
        skipped=0,
        failed=751,
        run_url="https://github.com/virchan/issue-triaging-agent/actions/runs/4",
    )

    assert "failed 751" in title
    assert "**Failed:** 751" in body
    assert "751 issue(s) failed to embed" in body
    assert "check Cloud Logging" in body
