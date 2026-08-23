"""Semantic-similarity embeddings for duplicate-issue detection.

Escalation from src/duplicate_detection.py's deterministic heuristic,
which entry 74's real evaluation ruled out - character-level text
similarity couldn't separate real adjudicated duplicate pairs from
superficially similar but unrelated ones. gemini-embedding-001 with
task_type="SEMANTIC_SIMILARITY" is Google's own documented use case for
duplicate detection (confirmed via ai.google.dev, not assumed), and free
of charge in the free tier - see LOG.md entry 74.
"""

from __future__ import annotations

import math
from typing import Any

from google import genai
from google.genai import types

from src.gemini_client import GeminiConfigurationError, GeminiUnavailableError

EMBEDDING_MODEL = "gemini-embedding-001"


class IssueEmbedder:
    """Produce a semantic-similarity embedding vector for issue text."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if api_key is None or not api_key.strip():
                raise GeminiConfigurationError("A Gemini API key is required.")
            client = genai.Client(api_key=api_key.strip())

        self._client = client

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for text - a list of floats,
        deterministic per call (not sampled), same input always yields
        the same vector.

        Raises GeminiUnavailableError on any provider-side failure - the
        same translation pattern GeminiJudge.judge() uses, so callers
        never see a raw SDK exception from either code path.
        """

        text = text.strip()
        if not text:
            raise ValueError("Text to embed cannot be empty.")

        try:
            response = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            return list(response.embeddings[0].values)
        except Exception as error:
            raise GeminiUnavailableError(
                "The embedding service could not complete the request."
            ) from error


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity in [-1, 1] - 1.0 for identical direction, 0.0
    for orthogonal, negative for opposite direction. Pure Python, no new
    dependency - vector sizes here (low thousands of floats at most)
    don't warrant numpy."""

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
