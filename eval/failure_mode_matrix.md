# Failure-mode matrix

The dimensions this project's evaluation should cover, checked against what the real golden set (`eval/golden_set.json`, 19 examples as of 2026-08-28) actually contains — not a hypothetical list. This matrix is built *before* declaring a vertical slice done, so scope gaps are caught here, not discovered later by an external reviewer.

## Coverage as of 2026-08-28 (19 real examples)

| Dimension | Covered? | Detail |
|---|---|---|
| Issue type: bug report | Yes | `#34744`, `#34734`, `#34436`, `#34768`, `#34822` |
| Issue type: feature/enhancement request | Yes | `#34628`, `#34660`, `#33963` |
| Issue type: documentation | Yes | `#34641`, `#34669`, `#34807`, `#34820`, `#34825` |
| Issue type: testing/CI infrastructure | Yes | `#34730`, `#34738`, `#34715` |
| Issue type: performance/parallelism | Yes | `#34736`, `#34618` |
| Issue type: question/discussion-only (RFC) | Yes | `#34691` - closed since the last matrix, which had no real example for this |
| Ambiguous or underspecified issue | **No** | All 19 real reports are well-specified; no genuinely ambiguous case yet |
| Spam / clearly off-topic (`is_spam=True`) | **No** | Zero real examples with `is_spam=True` - the spam path is completely unverified against real data |
| No clear label applies (`suggested_labels=[]`) | **No** | Every real example so far got at least one confident label |
| Multi-label judgment (more than one suggested label) | **No** | Zero real examples yet - the schema only started supporting more than one label today; expected to close as real corrections accumulate |
| `priority=high` | **No** | Priority is skewed: 12/19 medium, 7/19 low, 0/19 high |
| Corrected (known past mistake) | Yes | 7 real examples (`#34715`, `#34436`, `#34618`, `#34768`, `#34807`, `#34820`, `#34825`) - up from 1 real example in the last matrix, and now genuinely exercises `similar_examples`/`recent_examples` few-shot retrieval against real embeddings, not just the bare judge path |
| Bot-authored issue reaching judgment | N/A, correctly | Bots are filtered before judgment (Step 6); this is deterministic and already covered by `tests/test_bot_filter.py`, not something the golden set needs to re-verify |

## What this means

Five real gaps remain, stated plainly rather than papered over: no ambiguous issue, no spam example, no empty-label example, no multi-label example, no high-priority example. The question/discussion-only gap flagged in the last version of this matrix has since closed organically, through real operation, not by curating an example to fill it in - consistent with this project's own philosophy of preferring real operation over synthetic curation, which also rules out fabricating the five gaps that remain.

These gaps are expected to close as real operation continues and the golden set grows via `scripts/export_golden_set.py`. Until they do, evaluation results should be read as "verified for the issue shapes seen so far," not "verified in general" - an honest scope statement, not a hedge.
