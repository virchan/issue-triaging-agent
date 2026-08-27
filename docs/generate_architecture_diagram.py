"""Generate docs/architecture.png - a rendered version of the README's
Architecture section, replacing the ASCII-art diagram with something
that actually looks like a diagram.

matplotlib is a docs-only dependency (see pyproject.toml's `docs` group)
- never imported by the running application, only by this one-off
generation script. Re-run whenever the real pipeline shape changes:

    uv run --group docs python docs/generate_architecture_diagram.py
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUTPUT_PATH = Path(__file__).parent / "architecture.png"

# Okabe-Ito colorblind-safe categorical palette (Okabe & Ito, 2008) -
# blue/vermillion/bluish-green are chosen specifically because they stay
# distinguishable under protanopia, deuteranopia, and tritanopia
# simulations, unlike a naive blue/orange/green pick. Fills are light
# tints of each edge color, not a separate arbitrary color.
GCP_BLUE = "#e6f0f7"
GCP_BLUE_EDGE = "#0072b2"
STEP_FILL = "#ffffff"
STEP_EDGE = "#333333"
RAG_FILL = "#fbeae0"
RAG_EDGE = "#d55e00"
HUMAN_FILL = "#e0f5ef"
HUMAN_EDGE = "#009e73"
TEXT_COLOR = "#202124"
ARROW_COLOR = "#5f6368"

STEP_X = 3.6
STEP_WIDTH = 3.8
STEP_HEIGHT = 1.0
STEP_GAP = 1.4  # vertical distance between consecutive step centers


def draw_box(
    ax,
    center,
    width,
    height,
    text,
    *,
    fill=STEP_FILL,
    edge=STEP_EDGE,
    fontsize=10,
    fontweight="normal",
):
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        facecolor=fill,
        edgecolor=edge,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=TEXT_COLOR,
        zorder=4,
        linespacing=1.4,
    )


def draw_arrow(ax, start, end, *, linestyle="solid", color=ARROW_COLOR, mutation=14):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=1.4,
        linestyle=linestyle,
        color=color,
        zorder=2,
    )
    ax.add_patch(arrow)


def main() -> None:
    # Step centers, evenly spaced, top to bottom - the whole layout is
    # derived from this list so nothing can silently overlap the way a
    # hand-placed extra box did in an earlier version of this script.
    steps = [
        ("fetch\n(GitHub API, read-only)", 11.0),
        ("embed\n(Gemini embeddings)", 11.0 - STEP_GAP),
        ("judge\n(Gemini, structured output)", 11.0 - 2 * STEP_GAP),
        ("persist\n(Postgres / Neon)", 11.0 - 3 * STEP_GAP),
        ("digest + publish\n(GitHub issue, shadow repo)", 11.0 - 4 * STEP_GAP),
    ]
    fetch_y = steps[0][1]
    embed_y = steps[1][1]
    judge_y = steps[2][1]
    digest_y = steps[-1][1]

    scheduler_center = (5, fetch_y + STEP_GAP + 0.9)
    job_top = fetch_y + 0.5 + 0.55
    job_bottom = digest_y - 0.5 - 0.3
    review_center = (STEP_X, job_bottom - 0.9)
    next_run_center = (STEP_X, review_center[1] - STEP_GAP)
    caption_y = next_run_center[1] - 0.75

    fig, ax = plt.subplots(figsize=(8, 9.7))
    ax.set_xlim(-0.3, 10)
    ax.set_ylim(caption_y - 0.5, scheduler_center[1] + 0.9)
    ax.axis("off")

    # --- Cloud Scheduler ---
    draw_box(
        ax,
        scheduler_center,
        4.4,
        0.9,
        "Cloud Scheduler\n(daily, cron)",
        fill=GCP_BLUE,
        edge=GCP_BLUE_EDGE,
        fontweight="bold",
    )

    # --- Cloud Run Job container ---
    ax.add_patch(
        Rectangle(
            (0.6, job_bottom),
            8.8,
            job_top - job_bottom,
            linewidth=1.6,
            linestyle="dashed",
            edgecolor=GCP_BLUE_EDGE,
            facecolor=GCP_BLUE,
            alpha=0.2,
            zorder=1,
        )
    )
    ax.text(
        1.0,
        job_top - 0.4,
        "Cloud Run Job",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=GCP_BLUE_EDGE,
        zorder=2,
    )

    for text, y in steps:
        draw_box(ax, (STEP_X, y), STEP_WIDTH, STEP_HEIGHT, text)
    for (_, y1), (_, y2) in pairwise(steps):
        draw_arrow(ax, (STEP_X, y1 - 0.5), (STEP_X, y2 + 0.5))

    # Scheduler -> first step
    draw_arrow(
        ax, (scheduler_center[0], scheduler_center[1] - 0.45), (STEP_X, fetch_y + 0.5)
    )

    # --- RAG side branch off "embed", feeding into "judge" ---
    rag_center = (7.6, embed_y)
    draw_box(
        ax,
        rag_center,
        3.2,
        1.1,
        "retrieve similar past\njudgments (RAG)",
        fill=RAG_FILL,
        edge=RAG_EDGE,
        fontsize=9,
    )
    draw_arrow(
        ax,
        (STEP_X + STEP_WIDTH / 2, embed_y),
        (rag_center[0] - 1.6, rag_center[1] + 0.15),
        color=RAG_EDGE,
    )
    draw_arrow(
        ax,
        (rag_center[0] - 1.6, rag_center[1] - 0.25),
        (STEP_X + STEP_WIDTH / 2, judge_y),
        color=RAG_EDGE,
    )

    # --- Human review (outside the Cloud Run Job) ---
    draw_arrow(ax, (STEP_X, job_bottom), (STEP_X, review_center[1] + 0.5))
    draw_box(
        ax,
        review_center,
        5.6,
        1.0,
        "Human review + correction\n(comment on the digest issue)",
        fill=HUMAN_FILL,
        edge=HUMAN_EDGE,
    )

    # --- Next run captures the correction ---
    draw_arrow(ax, (STEP_X, review_center[1] - 0.5), (STEP_X, next_run_center[1] + 0.5))
    draw_box(
        ax,
        next_run_center,
        6.6,
        1.0,
        "Next run:\ncapture correction → re-judge → acknowledge",
        fill=STEP_FILL,
        edge=STEP_EDGE,
    )
    ax.text(
        STEP_X,
        caption_y,
        "(runs on the next scheduled Cloud Run Job execution)",
        ha="center",
        va="center",
        fontsize=8,
        color=ARROW_COLOR,
        style="italic",
    )

    fig.suptitle(
        "issue-triaging-agent — daily pipeline", fontsize=13, fontweight="bold", y=0.985
    )
    fig.savefig(
        OUTPUT_PATH, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white"
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
