from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

# Shared by digest.py and corrections.py - one Environment, not one per
# call site, so template files are only ever parsed/cached once.
#
# autoescape is deliberately off: this renders Markdown for a GitHub
# issue body, not HTML, and issue titles/summaries/rationale can contain
# characters (backticks, angle brackets in code samples, etc.) that HTML
# escaping would mangle. This matches the unescaped behavior of the
# hand-built strings these templates replaced - not a new gap introduced
# here. Issue titles/bodies are untrusted (real GitHub issue content we
# don't control), but the digest is rendered into the operator's own
# private shadow repo, not somewhere a script tag could execute against
# a wider audience - the risk this would otherwise mitigate doesn't apply
# here the way it would for a public-facing page.
_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # see docstring above
)


def render_template(name: str, **context: Any) -> str:
    """Render templates/{name} with context, trimmed to a single
    trailing newline - callers don't need to worry about exactly how
    much incidental blank space the template's own block tags leave
    behind."""

    return _ENV.get_template(name).render(**context).strip() + "\n"
