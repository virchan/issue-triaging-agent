# Evaluation rubric

What "correct" means for this project, and why it isn't a simple exact-match check. Implemented in `src/eval.py`; this document is the prose version of the same logic, kept in sync with it.

## What gets compared

Only the **structured decision fields** of a judgment: `suggested_label`, `is_spam`, `priority`. Free-text fields (`summary`, `rationale`) are never compared — differently worded but equally correct phrasing shouldn't count as a failure, and comparing free text meaningfully would need its own (weaker, fuzzier) evaluation method. This is a deliberate scope boundary, not an oversight: the goal is a real, honest signal on the decisions that matter operationally, not a plausible-looking-text checker (see the original project brief: "Avoid creating an evaluation suite that merely checks whether the LLM produced plausible-looking text").

## The two outcomes

Every golden example is one of two kinds — **confirmed** (the operator reviewed the digest and made no correction for this issue) or **corrected** (the operator explicitly said something was wrong, in free text, which triggered a fresh judgment call informed by that correction). Both kinds are scored by the identical rule when judgment is re-run against the same issue:

| New judgment | Outcome |
|---|---|
| Matches the golden example | **PASS** |
| Differs from the golden example | **REGRESSION** |

## Why confirmed and corrected examples aren't scored differently

An earlier version of this rubric treated a corrected example specially: reproducing the golden example's label was scored as a REGRESSION (on the theory that the golden label was "the original mistake" being reproduced), and differing from it was scored as a third outcome, NEEDS_REVIEW (on the theory that there was no structured target to confirm the new judgment against).

That theory doesn't hold against how this system actually persists a correction. Applying a correction always calls `judge_with_correction` — a fresh judgment call informed by the correction's free text — and then overwrites the stored judgment's `suggested_label`/`is_spam`/`priority` in place with that revised result (see `update_judgment` in `src/db.py`). No column anywhere preserves the pre-correction value; only *which* fields changed is recorded, not their old values. So by the time a corrected example is exported into the golden set, its stored fields are already the human-confirmed final answer — exactly the same kind of ground truth a plain confirmed example has, not "the original mistake." There is no separate, less-certain target to give partial credit against, so there's no honest basis for a third outcome: a match is a real pass, and a difference is a real regression, the same as for a confirmed example.

## What counts as a CI-failing regression

REGRESSION fails CI. PASS does not.
