from __future__ import annotations

import math
from typing import Any
from unittest.mock import Mock

import pytest
from google.genai import errors

from src.embeddings import (
    EMBEDDING_MODEL,
    MAX_RATE_LIMIT_RETRIES,
    IssueEmbedder,
    cosine_similarity,
)
from src.gemini_client import GeminiConfigurationError, GeminiUnavailableError


def _rate_limit_error() -> errors.APIError:
    return errors.APIError(429, {"error": {"message": "rate limited"}})


def _server_error() -> errors.APIError:
    return errors.APIError(500, {"error": {"message": "boom"}})


@pytest.fixture
def client(mocker: Any) -> Any:
    return mocker.Mock()


@pytest.fixture
def embedder(client: Any) -> IssueEmbedder:
    return IssueEmbedder(client=client)


def mocker_embedding_response(values: list[float]) -> Any:
    response = Mock()
    response.embeddings = [Mock(values=values)]
    return response


def test_requires_an_api_key_or_client() -> None:
    with pytest.raises(GeminiConfigurationError):
        IssueEmbedder(api_key="   ")


def test_embed_returns_the_vector(client: Any, embedder: IssueEmbedder) -> None:
    client.models.embed_content.return_value = mocker_embedding_response(
        [0.1, 0.2, 0.3]
    )

    vector = embedder.embed("PCA raises a ValueError on float32 input")

    assert vector == [0.1, 0.2, 0.3]


def test_embed_uses_the_semantic_similarity_task_type(
    client: Any, embedder: IssueEmbedder
) -> None:
    client.models.embed_content.return_value = mocker_embedding_response([0.1])

    embedder.embed("some issue text")

    _, kwargs = client.models.embed_content.call_args
    assert kwargs["model"] == EMBEDDING_MODEL
    assert kwargs["config"].task_type == "SEMANTIC_SIMILARITY"


def test_embed_rejects_blank_text(embedder: IssueEmbedder) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        embedder.embed("   ")


def test_embed_translates_provider_failure(
    client: Any, embedder: IssueEmbedder
) -> None:
    client.models.embed_content.side_effect = RuntimeError("boom")

    with pytest.raises(GeminiUnavailableError):
        embedder.embed("some issue text")


def test_embed_retries_on_rate_limit_then_succeeds(
    mocker: Any, client: Any, embedder: IssueEmbedder
) -> None:
    """A real backfill run once lost 751 of 1,159
    issues to 429s with no retry at all. A transient rate limit that
    clears within a couple of tries must not be treated as a permanent
    failure."""

    sleep = mocker.patch("src.embeddings.time.sleep")
    client.models.embed_content.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        mocker_embedding_response([0.1, 0.2]),
    ]

    vector = embedder.embed("some issue text")

    assert vector == [0.1, 0.2]
    assert client.models.embed_content.call_count == 3
    assert sleep.call_count == 2
    # Exponential: second wait longer than the first.
    assert sleep.call_args_list[1].args[0] > sleep.call_args_list[0].args[0]


def test_embed_gives_up_after_exhausting_rate_limit_retries(
    mocker: Any, client: Any, embedder: IssueEmbedder
) -> None:
    mocker.patch("src.embeddings.time.sleep")
    client.models.embed_content.side_effect = _rate_limit_error()

    with pytest.raises(GeminiUnavailableError):
        embedder.embed("some issue text")

    # The initial attempt, plus every retry - never more, never fewer.
    assert client.models.embed_content.call_count == MAX_RATE_LIMIT_RETRIES + 1


def test_embed_does_not_retry_a_non_rate_limit_api_error(
    mocker: Any, client: Any, embedder: IssueEmbedder
) -> None:
    sleep = mocker.patch("src.embeddings.time.sleep")
    client.models.embed_content.side_effect = _server_error()

    with pytest.raises(GeminiUnavailableError):
        embedder.embed("some issue text")

    client.models.embed_content.assert_called_once()
    sleep.assert_not_called()


def test_cosine_similarity_identical_vectors_is_one() -> None:
    vec = [0.5, 0.5, 0.5]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_is_symmetric() -> None:
    a, b = [0.1, 0.9, 0.4], [0.7, 0.2, 0.6]
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


def test_cosine_similarity_handles_a_zero_vector() -> None:
    """A zero vector has no direction - cosine similarity is undefined
    mathematically (division by zero norm); returning 0.0 rather than
    raising keeps this usable as a plain similarity score everywhere
    else, at the cost of not distinguishing "undefined" from "orthogonal"."""

    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_matches_manual_calculation() -> None:
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    dot = 1 * 4 + 2 * 5 + 3 * 6
    norm_a = math.sqrt(1 + 4 + 9)
    norm_b = math.sqrt(16 + 25 + 36)
    assert cosine_similarity(a, b) == pytest.approx(dot / (norm_a * norm_b))
