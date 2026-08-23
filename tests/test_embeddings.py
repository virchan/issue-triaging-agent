from __future__ import annotations

import math
from typing import Any
from unittest.mock import Mock

import pytest

from src.embeddings import EMBEDDING_MODEL, IssueEmbedder, cosine_similarity
from src.gemini_client import GeminiConfigurationError, GeminiUnavailableError


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
