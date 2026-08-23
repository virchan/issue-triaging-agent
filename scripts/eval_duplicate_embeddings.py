"""Check embeddings-based duplicate detection against the same real,
adjudicated scikit-learn issue pairs used in scripts/eval_duplicate_heuristic.py.

The deterministic heuristic was ruled out by that script's real result
(LOG.md entry 74) - this is the escalation path, checked the same
rigorous way: does cosine similarity on gemini-embedding-001 embeddings
actually separate the operator's real duplicate pairs from the real
non-duplicate (false-positive-check) pairs?

Requires GEMINI_API_KEY (real API calls, one embedding per distinct
issue - 20 calls for the 10 pairs below, free tier).

Run with:
    uv run python -m scripts.eval_duplicate_embeddings
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.duplicate_detection import issue_text
from src.embeddings import IssueEmbedder, cosine_similarity
from src.github_client import GitHubClient

OWNER = "scikit-learn"
REPO = "scikit-learn"

# Same real pairs as scripts/eval_duplicate_heuristic.py - transcribed
# from duplicated-issues-example.md.
PAIRS: list[tuple[int, int, bool]] = [
    (31593, 32150, True),
    (32961, 33002, True),
    (33219, 33355, True),
    (31719, 31725, True),
    (31799, 31806, True),
    (33737, 34046, False),
    (33094, 34431, False),
    (28711, 30308, False),
    (34312, 34351, False),
    (34618, 34672, False),
]


def main() -> None:
    load_dotenv()

    github = GitHubClient()
    embedder = IssueEmbedder(api_key=os.environ["GEMINI_API_KEY"])

    issue_cache: dict[int, str] = {}
    embedding_cache: dict[int, list[float]] = {}

    def embedding_for(number: int) -> list[float]:
        if number not in embedding_cache:
            if number not in issue_cache:
                issue = github.fetch_issue(OWNER, REPO, number)
                issue_cache[number] = issue_text(issue.title, issue.body)
            embedding_cache[number] = embedder.embed(issue_cache[number])
        return embedding_cache[number]

    results: list[tuple[int, int, bool, float]] = []
    for a, b, is_duplicate in PAIRS:
        score = cosine_similarity(embedding_for(a), embedding_for(b))
        results.append((a, b, is_duplicate, score))

    results.sort(key=lambda row: row[3], reverse=True)

    print(f"{'issue A':>8}  {'issue B':>8}  {'expected':>10}  {'score':>7}")
    for a, b, is_duplicate, score in results:
        expected = "duplicate" if is_duplicate else "unrelated"
        print(f"{a:>8}  {b:>8}  {expected:>10}  {score:>7.3f}")

    duplicate_scores = [row[3] for row in results if row[2]]
    unrelated_scores = [row[3] for row in results if not row[2]]
    print()
    print(
        f"Duplicate pairs   - min {min(duplicate_scores):.3f}, max {max(duplicate_scores):.3f}"
    )
    print(
        f"Unrelated pairs   - min {min(unrelated_scores):.3f}, max {max(unrelated_scores):.3f}"
    )
    if min(duplicate_scores) > max(unrelated_scores):
        print(
            f"Clean separation - any threshold in ({max(unrelated_scores):.3f}, {min(duplicate_scores):.3f}) works."
        )
    else:
        print("No clean separation - the two groups' score ranges overlap.")


if __name__ == "__main__":
    main()
