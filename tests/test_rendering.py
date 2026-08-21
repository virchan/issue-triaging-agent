from __future__ import annotations

import jinja2
import pytest

from src.rendering import render_template


def test_render_template_trims_to_a_single_trailing_newline() -> None:
    """digest.md.jinja's own block tags leave behind incidental blank
    space (blank lines from macros/conditionals that didn't fire) -
    callers shouldn't have to know how much. Every render ends in exactly
    one trailing newline, matching the convention format_digest_body
    already relied on before this became a template."""

    body = render_template(
        "digest.md.jinja",
        date="2026-08-21",
        scope="non-bot issue(s)",
        wip_digest_issue_number=None,
        has_issues=False,
        has_backlog=False,
        issue_count=0,
        backlog_count=0,
        issue_groups=[],
        backlog_groups=[],
    )

    assert body == "No newly created non-bot issue(s) were found for 2026-08-21.\n"


def test_render_template_raises_for_an_unknown_template() -> None:
    with pytest.raises(jinja2.TemplateNotFound):
        render_template("does-not-exist.md.jinja")
