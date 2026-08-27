# Evaluation rubric

What "correct" means for this project, and why it isn't a simple exact-match check. Implemented in `src/eval.py`; this document is the prose version of the same logic, kept in sync with it.

## What gets compared

Only the **structured decision fields** of a judgment: `suggested_label`, `is_spam`, `priority`. Free-text fields (`summary`, `rationale`) are never compared — differently worded but equally correct phrasing shouldn't count as a failure, and comparing free text meaningfully would need its own (weaker, fuzzier) evaluation method. This is a deliberate scope boundary, not an oversight: the goal is a real, honest signal on the decisions that matter operationally, not a plausible-looking-text checker (see the original project brief: "Avoid creating an evaluation suite that merely checks whether the LLM produced plausible-looking text").

## The three outcomes

Every golden example is one of two kinds — **confirmed** (the operator reviewed the digest and made no correction for this issue) or **corrected** (the operator explicitly said something was wrong, in free text). Re-running judgment against the same issue produces one of three outcomes:

| Golden example | New judgment | Outcome |
|---|---|---|
| Confirmed | Matches the original | **PASS** |
| Confirmed | Differs from the original | **REGRESSION** |
| Corrected | Reproduces the exact original (wrong) judgment | **REGRESSION** |
| Corrected | Differs from the original | **NEEDS_REVIEW** |

## Why "corrected + differs" isn't an automatic pass

A correction is a free-text comment (e.g. `"#34648 should be labelled as \"array API\"."`), not a structured expected answer. If the new judgment stops reproducing the known-wrong original, that's genuine evidence something changed for the better — but there's no automated way to confirm the new judgment now satisfies what the operator actually meant, since we never parsed the correction into a structured target. Calling this a pass would be exactly the kind of unearned confidence this project's evaluation is designed to avoid — correctness claims need to be honest, not optimistic.

**NEEDS_REVIEW** is a deliberate partial-credit outcome: it says "this got better, but a human still needs to look" rather than silently claiming success or bluntly claiming failure.

## What counts as a CI-failing regression

Both REGRESSION cases fail CI. NEEDS_REVIEW does not fail CI, but is reported — it's a real signal, just not an automatable pass/fail one.

## Known limitation

Since corrections aren't parsed into structured targets, a genuinely correct fix that happens to coincidentally reproduce the exact original judgment (rare, but possible if the correction was actually about something outside the three compared fields - e.g. the summary text) would be misclassified as REGRESSION. This hasn't happened in the real golden set yet (the one real correction, `#34648`, was specifically about `suggested_label`), but it's a real edge case worth knowing about, not a hidden one.
