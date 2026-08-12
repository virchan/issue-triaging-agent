"""Manual live check for src.gemini_client against the real Gemini API.

Run with:
    uv run python -m scripts.check_gemini_client
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.gemini_client import GeminiJudge
from src.github_client import GitHubClient

MODEL = "gemini-3.5-flash"


def main() -> None:
    load_dotenv()

    with GitHubClient() as client:
        labels = client.fetch_labels("scikit-learn", "scikit-learn")

    judge = GeminiJudge(model=MODEL, api_key=os.environ["GEMINI_API_KEY"])
    judgment = judge.judge(
        title="OneHotEncoder crashes with sparse input and unknown categories",
        body=(
            "When fitting OneHotEncoder on sparse input with "
            "handle_unknown='ignore' and an unseen category at transform "
            "time, a ValueError is raised instead of the documented "
            "silent-ignore behavior."
        ),
        known_labels=labels,
    )

    print(f"Model: {MODEL}")
    print(f"Fetched {len(labels)} real labels")
    print(judgment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
