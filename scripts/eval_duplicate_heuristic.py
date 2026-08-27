"""Check the deterministic duplicate-detection heuristic against real,
adjudicated scikit-learn issue pairs.

The pairs below are transcribed from the operator's own local,
personally-adjudicated notes (kept local, not part of this repo) -
real cases spanning both true duplicates and superficially similar but
unrelated issues (the false-positive check). This script answers, with
real data rather than speculation: does similarity_score actually
separate the two groups, and where would a threshold go?

Run with:
    uv run python -m scripts.eval_duplicate_heuristic
"""

from __future__ import annotations

from src.duplicate_detection import issue_text, similarity_score
from src.github_client import GitHubClient

OWNER = "scikit-learn"
REPO = "scikit-learn"

# (issue_a, issue_b, is_duplicate) - transcribed from the operator's own local notes.
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
    client = GitHubClient()
    cache: dict[int, str] = {}

    def text_for(number: int) -> str:
        if number not in cache:
            issue = client.fetch_issue(OWNER, REPO, number)
            cache[number] = issue_text(issue.title, issue.body)
        return cache[number]

    results: list[tuple[int, int, bool, float]] = []
    for a, b, is_duplicate in PAIRS:
        score = similarity_score(text_for(a), text_for(b))
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
