# Failure-mode matrix

The dimensions this project's evaluation should cover, checked against what the real golden set (`eval/golden_set.json`, 7 examples as of 2026-08-12) actually contains — not a hypothetical list. This matrix is built *before* declaring a vertical slice done, so scope gaps are caught here, not discovered later by an external reviewer.

At the time this matrix was written, "no clear duplicate" was dropped as a dimension, since duplicate-candidate detection hadn't been built yet — evaluating a capability that doesn't exist would be dishonest, not thorough. (Duplicate detection has since been implemented; this matrix is due for a refresh to reflect that - see the golden set's own growth for the current state.)

## Coverage as of 2026-08-12 (7 real examples)

| Dimension | Covered? | Detail |
|---|---|---|
| Issue type: bug report | Yes | `#34647`, `#34734` |
| Issue type: feature/enhancement request | Yes | `#34649` |
| Issue type: documentation | Yes | `#34648` |
| Issue type: testing/CI infrastructure | Yes | `#34730`, `#34738` |
| Issue type: performance/parallelism | Yes | `#34736` |
| Issue type: question/discussion-only | **No** | No real example yet |
| Ambiguous or underspecified issue | **No** | All 7 real reports are well-specified; no genuinely ambiguous case yet |
| Spam / clearly off-topic (`is_spam=True`) | **No** | Zero real examples with `is_spam=True` - the spam path is completely unverified against real data |
| No clear label applies (`suggested_label=null`) | **No** | Every real example so far got a confident label |
| `priority=high` | **No** | Priority is skewed: 6/7 medium, 1/7 low, 0/7 high |
| Corrected (known past mistake) | Yes, thin | Exactly 1 real example (`#34648`) - enough to prove the mechanism works, not enough to be confident in it |
| Bot-authored issue reaching judgment | N/A, correctly | Bots are filtered before judgment (Step 6); this is deterministic and already covered by `tests/test_bot_filter.py`, not something the golden set needs to re-verify |

## What this means

Five real gaps, stated plainly rather than papered over: no question/discussion-only issue, no ambiguous issue, no spam example, no null-label example, no high-priority example. None of these are fabricatable honestly — this project's own philosophy of preferring real operation over synthetic curation rules out inventing synthetic examples to fill them in.

These gaps are expected to close as real operation continues (Phase 7) and the golden set grows via `scripts/export_golden_set.py`. Until they do, evaluation results should be read as "verified for the issue shapes seen so far," not "verified in general" - an honest scope statement, not a hedge.
