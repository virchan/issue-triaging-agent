"""Duplicate-issue detection.

similarity_score is the original deterministic heuristic (LOG.md entry
73) - real evaluation against adjudicated pairs (entry 74) ruled it out,
kept here since scripts/eval_duplicate_heuristic.py still references it
as the comparison baseline, not because it's used in production.

find_possible_duplicate is the mechanism actually used (entry 76):
gemini-embedding-001 similarity, real-evaluated to have no threshold
that cleanly separates duplicate from non-duplicate pairs - so this
ranks and surfaces the single most similar candidate as a suggestion,
never a duplicate/not-duplicate classification.

find_similar_reviewed_examples (entry 96) reuses the same embeddings
and the same loose floor for a different purpose: retrieval-augmented
few-shot context in judge()'s prompt, not a duplicate suggestion.
"""

from __future__ import annotations

import difflib
from dataclasses import replace

from src.db import ReviewedJudgment
from src.embeddings import cosine_similarity

# Below this, a "most similar match" isn't worth surfacing at all - a
# loose sanity floor (guards against an empty/irrelevant candidate pool,
# e.g. early in the backfill), not a tuned classification boundary. Real
# evaluation (entry 76) found even genuinely unrelated real pairs scored
# 0.818-0.918, and a real duplicate scored as low as 0.799 - there's no
# threshold in that range that would improve on just always surfacing
# the top match, so this is set well below all of it deliberately.
MINIMUM_SIMILARITY = 0.5


def issue_text(title: str, body: str | None) -> str:
    """The text this module actually compares - title and body combined,
    since a duplicate pair can differ mainly in one or the other (a
    reworded title over near-identical body content, or vice versa)."""

    return f"{title}\n\n{body or ''}"


def similarity_score(text_a: str, text_b: str) -> float:
    """A similarity ratio in [0, 1] - 1.0 for identical text, 0.0 for
    completely disjoint text. Symmetric by construction: SequenceMatcher's
    own .ratio() is a heuristic, not a true distance metric, and isn't
    actually symmetric in general (score(a, b) can differ slightly from
    score(b, a), confirmed empirically while writing this) - averaging
    both directions guarantees symmetry rather than leaving it to chance.
    """

    a_to_b = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    b_to_a = difflib.SequenceMatcher(None, text_b, text_a).ratio()
    return (a_to_b + b_to_a) / 2


def find_possible_duplicate(
    target_embedding: list[float],
    candidates: list[tuple[int, list[float]]],
) -> tuple[int, float] | None:
    """The single most similar candidate to target_embedding, as
    (github_number, similarity) - or None if candidates is empty or
    nothing clears MINIMUM_SIMILARITY.

    candidates is (github_number, embedding) pairs - the caller is
    responsible for excluding the target issue's own embedding (it
    won't be in the pool yet at the point this is called - see
    src.pipeline._judge_and_persist). A ranked suggestion, not a
    classification: this always returns the best match found above the
    loose sanity floor, never attempts a duplicate/not-duplicate
    verdict - real evaluation (LOG.md entry 76) found no threshold that
    would make that verdict reliable.
    """

    best: tuple[int, float] | None = None
    for github_number, embedding in candidates:
        score = cosine_similarity(target_embedding, embedding)
        if best is None or score > best[1]:
            best = (github_number, score)

    if best is None or best[1] < MINIMUM_SIMILARITY:
        return None
    return best


# How many semantically similar past reviewed judgments to surface as
# retrieval-augmented few-shot context in judge()'s prompt (LOG.md entry
# 96) - kept small: the same real evaluation that ruled out a clean
# duplicate/non-duplicate threshold (entries 74-76) means embedding
# similarity is a noisy signal here too, not something to lean on for a
# large context dump.
SIMILAR_EXAMPLES_LIMIT = 3


def find_similar_reviewed_examples(
    target_embedding: list[float],
    candidates: list[tuple[int, list[float], ReviewedJudgment]],
    limit: int = SIMILAR_EXAMPLES_LIMIT,
) -> list[ReviewedJudgment]:
    """Rank candidates by cosine similarity to target_embedding, above
    MINIMUM_SIMILARITY, returning up to `limit` ReviewedJudgment objects
    (each with .similarity set) - most similar first.

    candidates is (github_number, embedding, ReviewedJudgment) triples -
    the caller is responsible for excluding the target issue's own
    embedding first, same convention as find_possible_duplicate. Reuses
    the same loose floor deliberately: this is retrieval-augmented
    context for the model to weigh, not a precision-guaranteed lookup -
    an occasionally irrelevant example can slip through here, the same
    real caveat as an occasional wrong duplicate suggestion (LOG.md
    entry 85).
    """

    scored = [
        (cosine_similarity(target_embedding, embedding), reviewed)
        for _, embedding, reviewed in candidates
    ]
    above_floor = [pair for pair in scored if pair[0] >= MINIMUM_SIMILARITY]
    above_floor.sort(key=lambda pair: pair[0], reverse=True)

    return [
        replace(reviewed, similarity=score) for score, reviewed in above_floor[:limit]
    ]
