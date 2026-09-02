# Runbook

Documented operational procedures for known failure modes of `issue-triaging-agent`. Each entry records what would be observed, how it can be confirmed, how it is fixed, and how the fix is verified. Entries are written in a generalized, undated form for the procedure itself - the date on each entry marks when the failure mode was first encountered and documented, not an expiry. Entries are ordered newest first.

* <details><summary>2026-08-31 - 2026-09-02: An additive correction replaced a label instead of adding to it</summary>

  ### Symptom

  A correction phrased additively - "should also be labelled X," "should include X," "X in addition to Y" - appears to have no effect, or worse, silently replaces the issue's existing label with the new one instead of adding to it.

  ### Why this happened

  `suggested_label` held exactly one string. A correction is always applied by re-judging the issue with the correction as context, then overwriting the stored judgment with the result - there was no way to represent "keep the existing label and add this one," so an additive correction got arbitrarily collapsed into a full replacement. Read against real history: most real corrections turn out to be additive rather than replacements, so this wasn't an edge case - it was the more common shape, and it's the concrete, evidenced reason label feedback sometimes didn't seem to "stick."

  ### Fix

  `suggested_label` became `suggested_labels`, a list. The correction-handling prompt now explicitly distinguishes additive phrasing (keep existing labels, add the new one) from explicit replacement phrasing ("X is incorrect," "should not include X"), and the acknowledgment reports a deterministic added/removed diff computed from the label sets before and after, rather than a single "updated to" line - so it's clear at a glance whether a correction was additive or a genuine replacement.

  ### Verification

  `uv run pytest -q` passes with tests covering the diff rendering and the subset-match golden-eval rule this required. Confirmed live and organically, not just by manual reproduction: the first real additive correction after this shipped (`#34860`, "also needs an `API` label") produced `→ API added (now: module:preprocessing, API)` in the real acknowledgment, and the stored judgment shows both labels - the failure mode described above did not recur.

  ### Status

  Resolved. This exact failure mode is not expected to recur, since the single-label constraint that caused it no longer exists in the schema. If a correction still appears to be silently dropped or replaced, that's a different failure mode - see the correction-comment entry below.

  ### Related

  Distinct from the correction-comment-attribution entry below, which is about identifying *which issue* a correction refers to - this entry is about what happens to the label *once* the right issue is correctly identified.

  </details>

* <details><summary>2026-08-25: An already-reviewed issue reappears in a later digest as if new</summary>

  ### Symptom

  A digest re-displays an issue in full, identically to how it appeared in an earlier digest, even though the run that produced it found nothing genuinely new (a fetch count of zero). This is most visible after a digest has stayed open, unclosed, for several days.

  ### Why this happens

  While a digest stays open, the next run's lookback window is intentionally widened - it starts from that still-open digest's own window end, not from a fixed one-day lookback, so that reopening it later still surfaces everything genuinely new since it was first opened. That widening is correct and intentional. The bug is a separate, downstream step: selecting which already-judged issues count as "new enough to show" used that same wide window purely as a time filter, with no memory of whether a given issue had already been shown in an earlier digest. An issue created a few days ago, still inside the widened window, could therefore be pulled back in and re-rendered as if freshly reviewed, even on a day when nothing new was actually found.

  ### Diagnosis

  The real fetch window used by the run in question is read directly from its structured log output, rather than assumed from the digest's date alone - a window spanning more than roughly a day, alongside a fetched count of zero, is the signature of this happening. The specific issue being re-shown is then checked against the digest it originally appeared in, to confirm it is a genuine repeat rather than a legitimate backlog re-surface (backlog catch-up intentionally and correctly re-shows an issue on every run for as long as it stays open and unlabeled on GitHub - that is expected, unrelated behavior, not this bug).

  ### Fix

  An already-judged issue is only ever selected as "new" once - the first time it is linked to any digest, whichever digest that turns out to be. Backlog catch-up is a separate query path and is unaffected by this restriction, since it is expected to keep re-surfacing a still-open, still-unlabeled issue by design.

  ### Verification

  `uv run pytest -q` passes with a test confirming an already-linked judgment is excluded from a later, wider window even though its creation timestamp falls inside that window, while a freshly-judged issue in the same window is still included. In production, a digest published on a day with a widened window and zero genuinely new issues is confirmed to read a plain "no newly created issue(s)... were found," with no re-rendering of a previously shown issue.

  ### Related

  Distinct from the scheduler double-fire entry below, which produces a duplicate digest issue for an unrelated reason (a duplicated trigger, not a stale window).

  </details>

* <details><summary>2026-08-24: Stale GCP credentials after a long-running `--wait` in `backfill-trigger.yml`</summary>

  ### Symptom

  `backfill-trigger.yml`'s `Extract the real result` or `Report status` step fails - for example, `report_backfill_status.py: error: argument --fetched: invalid int value: ''`. No "CI: status report" issue is posted for that run, even though the underlying Cloud Run Job execution may have succeeded.

  ### Why this happens

  `gcloud run jobs execute --wait` blocks until the Cloud Run Job execution reaches a terminal state, which can take well over an hour if the job hits its own timeout-and-retry (see the task-timeout entry below). GCP access tokens minted via Workload Identity Federation (`google-github-actions/auth@v2`) default to a one-hour lifetime. If the token expires while `--wait` is still blocking, the first `gcloud` command run after it returns (`executions list`) fails authentication. Under GitHub Actions' default `bash -e` shell, the entire step is aborted at that point - none of its `$GITHUB_OUTPUT` lines are reached, so every downstream output (`fetched`, `embedded`, `skipped`, `failed`) is left blank rather than set to `0`, which is why the report script's `int()` parsing is what visibly fails, even though the real problem occurs upstream of it.

  As of commit `a032673`, the workflow re-authenticates (`google-github-actions/auth@v2`) immediately after `--wait` returns and before extraction, so this failure mode is not expected to recur. If it does, that should be treated as evidence the re-authentication step itself failed or was skipped, not a repeat of the original bug.

  ### Diagnosis

  The workflow's own tokens aren't independently inspectable after the fact, and the repository's available GitHub tokens can't read this repository's own Actions data, so this is diagnosed from the workflow run's own log in the Actions tab rather than an external query. The following are checked:

  - How long the `Run backfill job and wait for completion` step actually ran (its start/end timestamps, compared in the Actions UI).
  - Whether the re-authentication step immediately after it ran and succeeded.
  - The exact error on `Extract the real result`, if the re-authentication step is present and did not help.

  Independently of the workflow, the real embedding result can always be obtained directly, without relying on the report:

  ```bash
  gcloud logging read 'resource.type="cloud_run_job"
    AND resource.labels.job_name="issue-triaging-agent-backfill"
    AND jsonPayload.event="backfill_run"' --limit=1 --format=json
  ```

  ### Fix

  If the re-authentication step is missing (for example, if this entry is being read against an older commit), a second `google-github-actions/auth@v2` step is added between `--wait` and `Extract the real result`, marked `if: always()` so that it also runs when the job execution itself failed.

  If the re-authentication step is present and this still recurs, the extraction step itself has likely become the long-running part (unlikely, since it consists of a few quick `gcloud` reads) - what changed there should be investigated, rather than assuming the same one-hour-token cause is repeating.

  ### Verification

  Confirmation that a "CI: status report" issue was actually posted for the run, and that its numbers match the real `backfill_run` log event pulled directly from Cloud Logging.

  ### Related

  Reachable only when `--wait` runs long, which is itself caused by the task-timeout entry below (or, in principle, any unusually slow run) - that should be fixed first if it is the root cause; this entry concerns the reporting layer failing on top of it.

  </details>

* <details><summary>2026-08-24: Backfill job terminated by Cloud Run's default task timeout</summary>

  ### Symptom

  A backfill execution disappears mid-run without a `backfill_run` log event, and the following is shown in Cloud Logging: `Terminating task because it has reached the maximum timeout of 3600 seconds.` The execution's `retriedCount` is incremented and the task is restarted by Cloud Run; if that retry also times out and the job's `maxRetries` is exhausted, the execution is marked FAILED and the Cloud Monitoring "a Cloud Run Job execution failed" alert fires for real. (The alert's timestamp should be checked against any recently-known incident before it is assumed to be a new failure rather than a delayed notification for something already fixed.)

  ### Why this happens

  The Cloud Run Job's task timeout defaults to 3600 seconds (one hour). A full two-year sweep - triggered whenever `backfill_state` is empty, either on a genuine first run or after a deliberate reset - fetches and embeds roughly 1,000+ issues. Most of the sweep completes quickly, since already-embedded issues are skipped cheaply, but the chunks nearest the current date are the most exposed to rate-limit retries (see the rate-limiting entry below); sustained 429 storms there can push the total runtime past the one-hour limit.

  Because `set_backfill_state` is written only once, at the end of a successful run - a deliberate choice, so that a partial run cannot advance the watermark and silently skip a real gap - a timeout-triggered retry restarts the entire sweep from scratch rather than resuming from where it was terminated.

  ### Diagnosis

  ```bash
  # Confirmation that this is a timeout, not a different crash
  gcloud logging read 'resource.type="cloud_run_job"
    AND resource.labels.job_name="issue-triaging-agent-backfill"
    AND severity>=ERROR' --format="value(timestamp,textPayload)"

  # The execution's attempt count and current state
  gcloud run jobs executions describe EXECUTION_NAME --region=us-east1 \
    --format="value(status.completionTime,status.retriedCount,status.conditions)"

  # The job's configured timeout and retry budget
  gcloud run jobs describe issue-triaging-agent-backfill --region=us-east1 \
    --format="value(spec.template.spec.template.spec.timeoutSeconds,spec.template.spec.template.spec.maxRetries)"
  ```

  ### Fix

  No lever exists mid-flight: `gcloud run jobs update --task-timeout` takes effect only on the next execution, not one already running, so an already-running job's timeout cannot be extended.

  If an execution still has a retry remaining, it is often left to run - most of a re-swept run consists of cheap skip-only work, so it may complete within the same 3600-second window the first attempt nearly survived.

  If its retries are exhausted, or the recurrence is to be prevented, the timeout is raised before the next trigger:

  ```bash
  gcloud run jobs update issue-triaging-agent-backfill --region=us-east1 \
    --task-timeout=7200
  ```

  ### Verification

  A run is triggered (or the next scheduled one is awaited), and `gcloud run jobs executions describe` is used to confirm that `status.completionTime` is set with no `Retry`/timeout condition, and that the structured `backfill_run` log event was actually written.

  ### Related

  See the rate-limiting entry below for why the final chunks are the likely source of the slowdown, and the credential-expiry entry above for the downstream failure this can trigger in `backfill-trigger.yml` once a run takes over an hour.

  </details>

* <details><summary>2026-08-24: Two digest issues appear for the same day</summary>

  ### Symptom

  Two digest issues are published for what is effectively the same day - the second one arriving shortly after the first, reporting that no newly created issues were found, and carrying a reminder pointing back at the digest that was just published.

  ### Why this happens

  Cloud Scheduler does not guarantee exactly-once delivery for scheduled HTTP targets - an occasional duplicate dispatch of the same scheduled trigger is a documented platform characteristic, not a configuration error on this project's side. Two independent, both-successful Cloud Run Job executions a short time apart, each traced back to a distinct attempt in Cloud Scheduler's own execution log with its own scheduled time (not one attempt retried after a perceived failure), is the signature of this happening.

  ### Diagnosis

  ```bash
  # Real executions of the daily job around the suspected time
  gcloud run jobs executions list --job=issue-triaging-agent-daily \
    --region=us-east1 --limit=5 --sort-by="~createTime"

  # Cloud Scheduler's own dispatch log - looking for two distinct scheduled times
  gcloud logging read 'resource.type="cloud_scheduler_job"' \
    --format="value(timestamp,jsonPayload.status,jsonPayload.scheduledTime)" --limit=10
  ```

  A manual trigger (a GitHub Actions run or a direct `gcloud` call) is ruled out first, since it would look superficially similar; confirming two independent, Scheduler-originated attempts, each cleanly succeeding, is what distinguishes this from a manual duplicate or a retry following a real failure.

  ### Fix

  The daily job includes a same-day duplicate guard: if a WIP digest already exists for the same calendar day and this run's own window found nothing new, publishing a second digest is skipped entirely rather than posting a redundant reminder. A duplicate dispatch is therefore expected to be harmless and silent going forward; the underlying double-fire itself cannot be prevented from this project's side, since it is Cloud Scheduler's own delivery guarantee, not a bug in its configuration.

  ### Verification

  Confirmation that only one digest issue exists for the affected day, and that Cloud Run Job execution logs still show two executions if a genuine double-fire occurred - the guard is expected to suppress the second digest's publication, not the second execution itself.

  ### Related

  Distinct from the stale-window entry above, which produces an incorrect digest for an unrelated reason (a widened lookback window re-showing an already-seen issue, not a duplicated trigger).

  </details>

* <details><summary>2026-08-23: Rate-limited embedding calls during backfill</summary>

  ### Symptom

  The backfill job (`issue-triaging-agent-backfill`, run via `scripts/backfill_issue_embeddings.py`) completes, but its structured `backfill_run` log event (or the GitHub issue posted by `scripts/report_backfill_status.py`, labeled `CI: status report`) shows a non-zero `failed` count. Repeated `HTTP/1.1 429 Too Many Requests` responses from `generativelanguage.googleapis.com/.../gemini-embedding-001:batchEmbedContents` are visible in Cloud Logging, interleaved with `WARNING Embedding failed for issue #<n>` lines.

  ### Why this happens

  `gemini-embedding-001`'s free-tier quota is enforced per minute, not just per request. A backfill chunk can contain 15-20 issues; if embedding calls are not paced, the quota is saturated by back-to-back requests. Two mitigations are already in place (`src/embeddings.py` and `scripts/backfill_issue_embeddings.py`):

  - Each `IssueEmbedder.embed()` call is retried specifically on a 429 response, up to `MAX_RATE_LIMIT_RETRIES = 5` times, with exponential backoff starting at `INITIAL_BACKOFF_SECONDS = 2.0` (2s, 4s, 8s, 16s, 32s).
  - Individual embed calls are paced by the backfill script (`DELAY_BETWEEN_EMBEDS_SECONDS = 1.0`), separately from the existing `DELAY_BETWEEN_CHUNKS_SECONDS = 2.0` pacing between chunks.

  A non-zero `failed` count indicates a 429 storm outlasted the full retry budget (worst case, around 62 seconds of backoff) for that issue. This is expected to happen occasionally and is not, by itself, a sign the mitigations have failed.

  ### Diagnosis

  The real counts and the shape of the failures are confirmed directly, rather than inferred from an alert alone:

  ```bash
  # The execution is located
  gcloud run jobs executions list --job=issue-triaging-agent-backfill \
    --region=us-east1 --limit=5 --sort-by="~createTime"

  # The real final counts are pulled
  gcloud logging read 'resource.type="cloud_run_job"
    AND resource.labels.job_name="issue-triaging-agent-backfill"
    AND jsonPayload.event="backfill_run"' --limit=1 --format=json

  # The issues that failed permanently are listed
  gcloud logging read 'resource.type="cloud_run_job"
    AND resource.labels.job_name="issue-triaging-agent-backfill"
    AND jsonPayload.message:"Embedding failed"' --format="value(timestamp,jsonPayload.message)"
  ```

  ### Fix

  A handful of permanent failures (single digits to low tens, out of roughly 1,000+ issues) is within the expected residual rate and requires no action - the next scheduled weekly run's `OVERLAP_BUFFER` window will not retry these specifically (they fall outside its one-day overlap unless recent), and they are not load-bearing: they only mean those specific issues are absent from the duplicate-candidate pool.

  If the failure count is large again (tens of percent, as in the original 751-of-1,159 incident), the pacing and retry constants themselves should be reconsidered - whether the free-tier quota has changed, or whether `DELAY_BETWEEN_EMBEDS_SECONDS` needs to be increased, should be checked. An immediate re-run is not recommended in this case, since it would likely run under the same quota pressure and repeat the same failure rate.

  A targeted retry of already-failed issues isn't directly supported, but forcing one is straightforward: failed issues are simply absent from `issue_embeddings`, so resetting `backfill_state` (`DELETE FROM backfill_state WHERE id = 1`) forces the next run to re-sweep the full two-year window and pick them up. Already-embedded issues are skipped cheaply, so this is safe, if slower than a targeted retry would be.

  ### Verification

  `scripts/backfill_issue_embeddings.py` is re-run (or the `Trigger Backfill` GitHub Actions workflow is triggered), and the same `backfill_run` log query above is used to confirm that `failed` has dropped and that the specific issue numbers from the prior failed list now have rows in `issue_embeddings`.

  ### Related

  A backfill run that becomes stuck in sustained rate-limit retries for long enough can also run long enough to hit Cloud Run's default one-hour task timeout, and separately, long enough to outlast a Workload Identity Federation access token's default one-hour lifetime in `backfill-trigger.yml`. Both are documented as separate entries above, not variations of this one.

  </details>

* <details><summary>2026-08-18/19 - 2026-08-28: A correction comment doesn't produce the expected effect</summary>

  ### Symptom

  A correction comment posted when closing a digest issue doesn't result in the expected label change (or any recorded correction at all) on the next day's run. The acknowledgment reply either omits the intended correction, or explicitly notes that one or more references "didn't match any issue this agent has judged" - even though the comment clearly names a real, judged issue.

  ### Why this happens

  Which judgment a correction line is about is determined entirely by pattern-matching an issue reference out of the comment's own text - there's no separate structured field for "which issue is this about." Several genuinely different real phrasings have each defeated this matching, at different times, in different ways:

  - A reference format the parser's patterns didn't cover at all (for example, the digest's own `owner/repo/number` format, encountered before the parser was updated to expect it) - the whole comment was reported as unattributable and dropped.
  - A markdown link's own target URL (for example, `[...](https://redirect.github.com/owner/repo/issues/N)`) happening to match the same `owner/repo/number` shape as a real reference, and winning because it appeared first in the line - misattributing the whole line to a nonexistent "issue" formed from the URL's path segments.
  - A short alias-plus-separator combination (for example, `scikit-learn#NNN`) falling into a gap between two existing patterns, neither of which covered that exact combination.
  - Two valid references in the same line - one naming the issue the correction is actually about, one naming a different issue mentioned only for comparison - with whichever one appeared first in the sentence winning regardless of which one actually has a real judgment behind it.

  Each of these has been fixed as it was found, but the underlying risk is structural, not fully closed off: real corrections have more phrasing variety than any fixed set of patterns can be certain to fully anticipate, so a new phrasing defeating matching in some new way remains possible.

  ### Diagnosis

  The exact comment text is run directly through the real parsing function, rather than guessed at:

  ```bash
  uv run python -c "
  from src.corrections import extract_corrections_by_issue
  print(extract_corrections_by_issue('''<the exact real comment body>'''))
  "
  ```

  An empty or incomplete result confirms a matching gap; which reference(s) were and weren't picked up narrows down which known gap shape (or a new one) applies. Separately, the acknowledgment reply already posted by the next day's run (or a direct read of the `corrections` table for that judgment) confirms what was actually recorded in production, independent of what should have been recorded.

  ### Fix

  The specific matching gap identified in diagnosis is closed with a targeted addition to the reference patterns or resolution logic in `src/corrections.py`, verified first by reproducing the exact real comment line as a test case before any pattern is broadened speculatively. If the affected digest is already closed and marked reviewed by the time the gap is found, the correction is not recoverable automatically - correction capture is a no-op for an already-reviewed digest by design - so the intended label change is applied by hand for that specific case, and the fix only prevents recurrence going forward.

  ### Verification

  `uv run pytest -q` passes with a new test reproducing the exact real comment line; the parser is also re-run directly against that same real text (as in Diagnosis) to confirm the fix outside the test suite as well. The next real digest closed with a similarly-shaped comment is watched to confirm its correction is recorded as expected.

  ### Related

  Distinct from a reference that is correctly recognized but points at an issue this system genuinely never judged - that case is reported honestly by the acknowledgment reply as unmatched, which is expected, working behavior, not a bug.

  </details>
