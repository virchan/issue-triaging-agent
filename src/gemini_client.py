from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from src.db import ReviewedJudgment
from src.judgment import IssueJudgment
from src.rendering import render_template

LOGGER = logging.getLogger(__name__)


class GeminiConfigurationError(ValueError):
    """Raised when Gemini client configuration is invalid."""


class GeminiUnavailableError(RuntimeError):
    """Raised when the Gemini service cannot complete a request."""


class GeminiResponseError(RuntimeError):
    """Raised when Gemini returns an invalid or ungrounded structured response."""


# Kept as separate template files (src/templates/*-instructions.md.jinja),
# not inline string constants - editing the wording of a rule (the kind
# of change entry 90 made and reverted) is then a plain-text edit with no
# Python diff noise, and the prompt reads the way it will actually be
# sent, not wrapped in a triple-quoted Python string. Loaded once here via
# the same render_template mechanism as every other rendered document in
# this project (src/rendering.py), not a second, parallel loading
# mechanism - these templates happen to have no {{ }} substitution today,
# but Jinja renders static text unchanged, and reusing one mechanism
# keeps "how do we load a text asset" answered one way, not two.
# .strip() matches the original inline constants' own `.strip()` exactly,
# so the text actually sent to Gemini is byte-for-byte unchanged from
# before this refactor - verified via test_gemini_client.py.
_JUDGE_INSTRUCTIONS = render_template("judge-instructions.md.jinja").strip()
_REJUDGE_INSTRUCTIONS = render_template("rejudge-instructions.md.jinja").strip()


def _format_examples(examples: list[ReviewedJudgment]) -> str:
    if not examples:
        return ""

    lines = ["Recent reviewed judgments (for context on your past accuracy):", ""]
    for index, example in enumerate(examples, start=1):
        judgment = example.judgment
        lines.append(f'{index}. Issue: "{example.issue_title}"')
        lines.append(
            f"   Your judgment: label={judgment.suggested_label or '(none)'}, "
            f"is_spam={judgment.is_spam}, priority={judgment.priority}"
        )
        if example.correction_text:
            lines.append(f"   Outcome: corrected: {example.correction_text!r}")
        else:
            lines.append("   Outcome: confirmed correct, no correction needed")
        lines.append("")

    return "\n".join(lines).strip()


class GeminiJudge:
    """Produce a structured, label-grounded IssueJudgment for one issue."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        model = model.strip()
        if not model:
            raise GeminiConfigurationError("A Gemini model name is required.")

        if client is None:
            if api_key is None or not api_key.strip():
                raise GeminiConfigurationError("A Gemini API key is required.")
            client = genai.Client(api_key=api_key.strip())

        self._model = model
        self._client = client

    def judge(
        self,
        *,
        title: str,
        body: str | None,
        known_labels: list[str],
        recent_examples: list[ReviewedJudgment] | None = None,
    ) -> IssueJudgment:
        """Produce a structured judgment for one issue.

        recent_examples (see src.db.get_recent_reviewed_judgments) are
        injected as few-shot context - real past corrections and
        confirmations, not model fine-tuning.

        Raises GeminiResponseError if the model suggests a label outside
        known_labels - this is verified deterministically, not trusted
        from the model's own claim that it followed the instructions.
        """

        title = title.strip()
        if not title:
            raise ValueError("An issue title is required.")

        prompt = self._build_prompt(
            title=title,
            body=body,
            known_labels=known_labels,
            recent_examples=recent_examples or [],
        )

        return self._request_judgment(prompt=prompt, known_labels=known_labels)

    def judge_with_correction(
        self,
        *,
        title: str,
        body: str | None,
        previous_judgment: IssueJudgment,
        correction_text: str,
        known_labels: list[str],
        recent_examples: list[ReviewedJudgment] | None = None,
    ) -> IssueJudgment:
        """Produce a revised judgment for an issue, informed by a human
        correction - distinct from judge()'s few-shot mechanism, which
        only ever influences *other, future* judgments. This re-judges
        the same issue directly, given the previous judgment and what
        was wrong with it.

        Raises the same errors as judge() for the same reasons.
        """

        title = title.strip()
        if not title:
            raise ValueError("An issue title is required.")

        correction_text = correction_text.strip()
        if not correction_text:
            raise ValueError("A correction is required.")

        prompt = self._build_correction_prompt(
            title=title,
            body=body,
            previous_judgment=previous_judgment,
            correction_text=correction_text,
            known_labels=known_labels,
            recent_examples=recent_examples or [],
        )

        return self._request_judgment(prompt=prompt, known_labels=known_labels)

    def _request_judgment(
        self, *, prompt: str, known_labels: list[str]
    ) -> IssueJudgment:
        """Shared call/parse/validate logic for judge() and judge_with_correction()."""

        start = time.monotonic()
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=IssueJudgment,
                    temperature=0,
                ),
            )
        except Exception as error:
            LOGGER.exception(
                f"Gemini judgment request failed for model {self._model}",
                extra={
                    "event": "judgment_latency",
                    "model": self._model,
                    "latency_seconds": time.monotonic() - start,
                    "outcome": "error",
                },
            )
            raise GeminiUnavailableError(
                "The judgment service could not complete the request."
            ) from error

        LOGGER.info(
            f"Gemini judgment request completed for model {self._model}",
            extra={
                "event": "judgment_latency",
                "model": self._model,
                "latency_seconds": time.monotonic() - start,
                "outcome": "ok",
            },
        )

        response_text = getattr(response, "text", None)
        if not response_text:
            raise GeminiResponseError(
                "The judgment service returned an empty response."
            )

        try:
            judgment = IssueJudgment.model_validate_json(response_text)
        except ValidationError as error:
            raise GeminiResponseError(
                "The judgment service returned an invalid response."
            ) from error

        if (
            judgment.suggested_label is not None
            and judgment.suggested_label not in known_labels
        ):
            raise GeminiResponseError(
                "The judgment service suggested an unknown label: "
                f"{judgment.suggested_label!r}"
            )

        return judgment

    @staticmethod
    def _build_prompt(
        *,
        title: str,
        body: str | None,
        known_labels: list[str],
        recent_examples: list[ReviewedJudgment],
    ) -> str:
        labels_json = json.dumps(known_labels, ensure_ascii=False, indent=2)
        examples_section = _format_examples(recent_examples)

        return f"""
{_JUDGE_INSTRUCTIONS}

{examples_section}

Repository labels:
{labels_json}

Issue title:
<issue_title>
{title}
</issue_title>

Issue body:
<issue_body>
{body or "(no body provided)"}
</issue_body>
""".strip()

    @staticmethod
    def _build_correction_prompt(
        *,
        title: str,
        body: str | None,
        previous_judgment: IssueJudgment,
        correction_text: str,
        known_labels: list[str],
        recent_examples: list[ReviewedJudgment],
    ) -> str:
        labels_json = json.dumps(known_labels, ensure_ascii=False, indent=2)
        examples_section = _format_examples(recent_examples)
        previous_json = previous_judgment.model_dump_json(indent=2)

        return f"""
{_REJUDGE_INSTRUCTIONS}

{examples_section}

Repository labels:
{labels_json}

Issue title:
<issue_title>
{title}
</issue_title>

Issue body:
<issue_body>
{body or "(no body provided)"}
</issue_body>

Previous judgment:
<previous_judgment>
{previous_json}
</previous_judgment>

Human correction:
<correction>
{correction_text}
</correction>
""".strip()
