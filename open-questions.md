# Open Questions

Working file for decisions not yet settled in `design-plan.md` (frozen) or `LOG.md` (chronological record of what happened). Each entry records the question, the options actually considered, and — once consensus is reached — a pointer to the `LOG.md` entry where the decision and its rationale were logged.

Entries are not deleted once resolved; the deliberation stays visible even after the outcome is known.

---

## 1. Final project name

**Status:** resolved

**Why it matters:** `triage-agent` was a working title. The final name shows up in the repo, any bot identity, and public docs, and needs to avoid implying scikit-learn affiliation (see design-plan.md §3, Non-affiliation).

**Operator's comment:** issue-triaging-agent

**Operator's confirmation:** Yes.

**Assistant's response:** Finalized. Working directory renamed from `triage-agent` to `issue-triaging-agent`; `LOG.md` and this file updated. `design-plan.md` intentionally still reads the old placeholder text — it's frozen, not being edited to match.

**Options considered:** `issue-triaging-agent` (operator-proposed). **Chosen.**

**Resolution:** `issue-triaging-agent`. See `LOG.md` entry 5.

---

## 2. LLM provider

**Status:** resolved

**Why it matters:** Drives cost, structured-output/tool-calling reliability, and latency. design-plan.md §5 already settled that it can be a standard hosted API rather than a self-hosted open-weight model — this question is which hosted provider/model.

**Operator's comment:** Free-tier Gemini

**Assistant's response:** Matches the recommendation — Google AI Studio's free-tier Gemini API, same mechanism already proven in `us-census-chat-agent`. "ChatGPT free version" (mentioned in the original comment) turned out not to be a real option, since OpenAI's free consumer chat product has no callable API.

**Options considered:**
- Google AI Studio (Gemini API, free tier) — already used successfully in `us-census-chat-agent`. **Chosen.**
- OpenAI API — not free for sustained backend use; ruled out.

**Resolution:** Google AI Studio free-tier Gemini API. See `LOG.md` entry 1.

---

## 3. Judgment schema (label taxonomy, duplicate-candidate format)

**Status:** resolved (exact Pydantic field names deferred to Phase 2 implementation)

**Why it matters:** This is the structured-output contract the whole pipeline depends on — what fields a "judgment" contains (labels, duplicate candidates, summary, confidence, etc.) and how each is validated.

**Operator's comment:** This is how I imagine my UX. The agent will look for issues created today. Then, it will open a new issue in a different Github repo to summarise those issues, rank their importance, and make recommendation to them, such as:
- this issue concerns linear model. (so I would determine if I should add the linear model label to the original scikit-learn issue.)
- this issue is a spam.
- this issue is created by the scikit-learn CI bot. So we don't need to include this here.

The agent doesn't interact with the issue/PR directly: it makes recommendation in the issue thread I mentioned earlier, and I will be the judge. I can comment on the issue, like "this issue is not about linear model, it is about SVC", to provide feedback. After I'm done with providing feedback for today's work, and closed the issue, the agent will use my comments in the issue thread to tune itself.

**Operator's response (on "tune itself"):** Go with (a) — feed recent corrections back into the prompt as examples, not literal fine-tuning.

**Assistant's response:** Both clarifications resolved this. Bot-authored issues (e.g., scikit-learn's CI bot) are excluded via a deterministic pre-filter before anything reaches the LLM or the digest — not a judgment the model makes. Corrections become prompt-context/examples for future runs, not fine-tuning data (fine-tuning was considered and rejected as disproportionate to the correction volume). This is a meaningfully better shape than the per-issue, admin-page-driven flow originally sketched in design-plan.md §5 — the shadow-repo issue thread itself becomes the review surface. It also directly resolves item 5 below.

**Options considered:** the operator's daily-digest design, with bot-issue pre-filtering and few-shot correction feedback. **Chosen.**

**Resolution:** Daily-digest design with deterministic bot-issue filtering and few-shot correction feedback. See `LOG.md` entry 2.

---

## 4. Golden evaluation set construction

**Status:** resolved (exact seeding mechanics deferred to Phase 4 implementation)

**Why it matters:** design-plan.md §8 requires a golden set of real historical scikit-learn issues with known outcomes. Open question is how it's built: fully hand-labeled by the author, partially derived from real historical resolutions (e.g., issues actually closed as duplicate), or a mix — and how large it needs to be to cover the failure-mode matrix from §8.

**Operator's comment:** I would say I can triage 2-3 issues per day on average, and a maximum of 5 issues on a day. I hope this helps....

**Operator's comment (follow-up):** We can use the issues I helped triage to create the seed. But I see this as an implementation detail, and can discuss more when we start coding.

**Assistant's response:** Agreed — sourcing the golden set from the operator's own real daily triage activity (rather than a separate bulk historical import) is the more authentic choice and consistent with the project's evidence philosophy. Exact construction mechanics (timing, format, how much to backfill) are deferred to Phase 4, as the operator requested.

**Options considered:**
- Pure organic growth from the operator's own daily digest reviews and corrections. **Chosen** (mechanics deferred to implementation).
- Bulk historical import via the GitHub API — not pursued as a separate initiative.

**Resolution:** Golden set sourced from the operator's own triage activity; construction mechanics deferred to Phase 4. See `LOG.md` entry 4.

---

## 5. Approval-gate design, final confirmation

**Status:** resolved

**Why it matters:** design-plan.md §5 records a *default* design (judgments sit `pending` until the author approves) but flags it as "pending final confirmation." Needs an explicit yes/no, or a revised design, before Phase 3 implementation.

**Operator's comment:** the publish-then-review-and-correct model from item 3 is the final design for this

**Assistant's response:** Confirmed and logged. This explicitly supersedes design-plan.md §5's pre-publish pending-approval default — the digest publishes automatically once judged, and the operator's review/correction happens afterward via public comments, with closing the issue marking review as complete.

**Options considered:** design-plan.md §5's pre-publish pending-approval default (superseded); item 3's publish-then-review model. **Chosen.**

**Resolution:** Publish-then-review-and-correct, per item 3's design. See `LOG.md` entry 3.

---

## 6. Cloud deployment target

**Status:** resolved

**Why it matters:** design-plan.md §6 lists cloud deployment as category A (required) but marks it "TBD." Affects Phase 5 and the observability/ops design in Phase 6.

**Operator's comment (round 1):** I want a free and quick solution.

**Operator's comment (round 2):** More familiar with AWS, ECR, EKS, S3. Could we do this free?

**Assistant's response:** Full EKS was ruled out on cost alone, not preference — the control plane bills a flat ~$0.10/hour (~$73/month) with no free tier, regardless of workload size. ECR and S3 fit fine on their own, but AWS's only genuinely *permanent* free-tier compute option is Lambda; ECR/S3 are free for 12 months on a new account and bill (cheaply) afterward.

**Operator's comment (round 3):** I want free-tier everything. Is it possible?

**Assistant's response:** Compared AWS (Lambda permanent-free, ECR/S3 12-month-then-billed), GCP (Cloud Run jobs, Cloud Scheduler, Artifact Registry, and Cloud Storage are all under a permanent Always Free allowance, not a trial), and Azure (Functions has a comparable permanent free grant, but confidence was too low on ACR/Blob Storage permanence to recommend without verification). GCP was the only option where every needed piece satisfies "free, indefinitely" without a time limit. No major cloud's managed Postgres has a permanent free tier, so Neon/Supabase stays the choice regardless of cloud.

**Operator's confirmation:** Yes, use GCP.

**Options considered:**
- Full EKS/Kubernetes — ruled out (no free tier for the control plane; also redundant with existing evidence).
- AWS Lambda + EventBridge Scheduler + ECR + S3 + Neon/Supabase Postgres — free for compute, only free-for-a-year (then cheap) for registry/storage.
- Azure Functions + ACR + Blob Storage + Neon/Supabase Postgres — plausible but not recommended without verifying current ACR/Blob Storage free-tier terms.
- GCP Cloud Run (jobs) + Cloud Scheduler + Artifact Registry + Cloud Storage + Neon/Supabase Postgres — **chosen**; only option with every piece under a permanent free allowance.

**Resolution:** GCP — Cloud Run (jobs) + Cloud Scheduler + Artifact Registry + Cloud Storage (if needed) + Neon/Supabase Postgres. See `LOG.md` entry 6.

**Postgres provider follow-up (Step 27):** Neon vs. Supabase was left open here. Resolved as **Neon** — Supabase's free tier pauses the whole project after 1 week of inactivity, requiring a manual dashboard action to resume; Neon scales to zero after 5 minutes idle but auto-resumes in ~1 second on the next connection. For a database only ever touched by an unattended, cron-triggered daily job, Neon's auto-resume avoids a real risk: a single missed run tipping Supabase past a week idle and silently breaking every subsequent scheduled run until manually noticed. See `LOG.md` entry 41.

---

## 7. Does the system need an HTTP surface (FastAPI service), or is it a pure batch job?

**Status:** resolved

**Why it matters:** design-plan.md's architecture (§5) names "FastAPI backend" in the pipeline, and its hiring-gap map (§6) lists "Backend service ownership | FastAPI service" as category **A** (required by the product problem). But the daily-digest design (item 3) plus the GCP Cloud Run *Jobs* decision (item 6) describe something that could run as a bare scheduled batch script with no HTTP server at all — the actual product problem (poll once/day, judge, publish a digest) doesn't strictly require anything to be listening on a port.

If that's all it is, "FastAPI service" was arguably mis-classified as A in the frozen plan — it would really be category **C** (added to close the "backend service / REST API" hiring gap named in the original brief), which isn't disqualifying but should be named honestly rather than left as an unexamined A. design-plan.md itself doesn't get corrected either way (frozen); this is about what actually gets built.

**Assistant's response:** I'd lean toward building a small, genuinely useful FastAPI service alongside the batch job, rather than either dropping FastAPI entirely or keeping it as a "backend" that does nothing an HTTP server would justify. Concretely: a low/zero-traffic Cloud Run *service* (separate from the Cloud Run *Job* that Cloud Scheduler triggers daily) exposing things backed by real data — e.g. `GET /health`, `GET /judgments` (read the Postgres audit trail), `POST /trigger` (manually kick a run, useful for testing and for recovering from a missed scheduled run). That keeps FastAPI honestly earning its place (real reads against real state, real operational utility) rather than being decorative, and it still fits inside GCP's Always Free tier at this traffic level.

**Operator's confirmation:** Agreed with the suggestion.

**Options considered:**
- Pure batch job, no HTTP surface at all — leaves "FastAPI backend" and "REST API" in the hiring-gap map unearned.
- Minimal FastAPI service (health/read/manual-trigger) alongside the Cloud Run Job — **chosen**, keeps the claim honest without inflating scope.

**Resolution:** Minimal FastAPI service (`/health`, `/judgments`, `/trigger`) deployed as a separate Cloud Run service alongside the Cloud Run Job. See `LOG.md` entry 7.
