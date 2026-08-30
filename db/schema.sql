-- Schema for issue-triaging-agent.
-- Applied manually for now (see scripts/init_db.py); a migration tool can
-- replace this if the schema outgrows a single file.

CREATE TABLE IF NOT EXISTS issues (
    id                  SERIAL PRIMARY KEY,
    repo_owner          TEXT NOT NULL,
    repo_name           TEXT NOT NULL,
    github_number       INTEGER NOT NULL,
    title               TEXT NOT NULL,
    body                TEXT,
    author_login        TEXT NOT NULL,
    github_created_at   TIMESTAMPTZ NOT NULL,
    html_url            TEXT NOT NULL,
    is_bot              BOOLEAN NOT NULL DEFAULT FALSE,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo_owner, repo_name, github_number)
);

-- One row per polled time window's digest, published to the operator-owned
-- shadow repo. window_start is a fixed lookback from "now" (or, if a
-- previous digest is still unreviewed, that digest's window_end) - not
-- a calendar day: a fixed calendar-day design broke on the very first
-- real scheduled run (17:00 PDT lands exactly on 00:00 UTC).
CREATE TABLE IF NOT EXISTS digests (
    id                      SERIAL PRIMARY KEY,
    window_start            TIMESTAMPTZ NOT NULL,
    window_end              TIMESTAMPTZ NOT NULL,
    shadow_repo_owner       TEXT NOT NULL,
    shadow_repo_name        TEXT NOT NULL,
    shadow_issue_number     INTEGER,
    state                   TEXT NOT NULL DEFAULT 'pending'
                            CHECK (state IN ('pending', 'published', 'reviewed')),
    published_at            TIMESTAMPTZ,
    closed_at                TIMESTAMPTZ
);

-- Matches src/judgment.py's IssueJudgment, plus two columns for the
-- duplicate-candidate feature (dropped from the original MVP, revisited
-- once real evidence supported it). possible_duplicate_*
-- is a ranked suggestion computed once at judgment time from
-- issue_embeddings below, not a classification - real evaluation
-- against adjudicated pairs found no threshold that cleanly separates
-- duplicate from non-duplicate, so this is deliberately "most similar
-- match found," always surfaced above a loose sanity floor, not a
-- duplicate/not-duplicate verdict.
CREATE TABLE IF NOT EXISTS judgments (
    id                              SERIAL PRIMARY KEY,
    issue_id                        INTEGER NOT NULL UNIQUE REFERENCES issues (id),
    digest_id                       INTEGER REFERENCES digests (id),
    -- A native array, not a join table: nothing here ever queries "every
    -- judgment with label X" independently of fetching the full judgment,
    -- so a join table would add real complexity (extra JOIN, ordering,
    -- delete-then-insert on update) for no actual benefit. Empty array
    -- means "no clear label," replacing the old NULL sentinel. A single
    -- TEXT column had no way to represent a genuinely multi-label
    -- judgment, which real corrections often call for (e.g. "should also
    -- be labelled X" - additive, not a replacement).
    suggested_labels                TEXT[] NOT NULL DEFAULT '{}',
    is_spam                         BOOLEAN NOT NULL DEFAULT FALSE,
    summary                         TEXT NOT NULL,
    priority                        TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    rationale                       TEXT NOT NULL,
    confidence                      REAL NOT NULL,
    possible_duplicate_number       INTEGER,
    possible_duplicate_similarity   REAL,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One embedding per issue - both issues this agent has judged and
-- issues fetched purely as duplicate-candidate backfill (see
-- scripts/backfill_issue_embeddings.py) share this table; not every
-- issue_id here has a corresponding judgments row. model is recorded so
-- a future embedding-model change (the prior text-embedding-004 was
-- deprecated in Jan 2026) can be detected rather than silently mixing
-- incompatible vectors. Pruned on a rolling 2-year window (see
-- prune_old_issue_embeddings) to keep storage bounded - not kept
-- forever, unlike issues/judgments/corrections themselves.
CREATE TABLE IF NOT EXISTS issue_embeddings (
    id              SERIAL PRIMARY KEY,
    issue_id        INTEGER NOT NULL UNIQUE REFERENCES issues (id) ON DELETE CASCADE,
    model           TEXT NOT NULL,
    embedding       DOUBLE PRECISION[] NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Single-row table tracking scripts/backfill_issue_embeddings.py's own
-- progress, separate from anything the daily judgment pipeline touches
-- (which also writes to issue_embeddings continuously - MAX(created_at)
-- there would just reflect "whenever an issue was last judged," not how
-- far the backfill itself has swept). Absent means "never run" (the
-- full BACKFILL_WINDOW lookback applies); present means "only fetch
-- since last_window_end" - what makes weekly re-runs incremental
-- instead of re-sweeping the entire history every time.
CREATE TABLE IF NOT EXISTS backfill_state (
    id                  INTEGER PRIMARY KEY DEFAULT 1,
    last_window_end     TIMESTAMPTZ NOT NULL,
    CHECK (id = 1)
);

-- The operator's corrections, parsed from their comments on a digest
-- issue - publish-then-review-and-correct. These feed back into future
-- prompts as few-shot context and grow the golden evaluation set.
-- One GitHub comment can correct several
-- issues at once (one bullet per issue), so the uniqueness is on the
-- (comment, judgment) pair, not the comment alone - a comment can
-- legitimately produce more than one correction row.
--
-- digest_id records which thread a correction came from - needed because
-- the same real issue can be re-surfaced into a newer digest (backlog
-- catch-up) while an older digest referencing it is still open, so more
-- than one thread can carry a correction for the same judgment over
-- time. superseded distinguishes which one is authoritative: when a
-- correction arrives on a thread that isn't the most recently created
-- one referencing that issue, it's recorded but marked superseded - not
-- used to re-judge, not surfaced as few-shot context - rather than
-- silently dropped.
CREATE TABLE IF NOT EXISTS corrections (
    id                  SERIAL PRIMARY KEY,
    judgment_id         INTEGER NOT NULL REFERENCES judgments (id),
    digest_id           INTEGER NOT NULL REFERENCES digests (id),
    github_comment_id   BIGINT NOT NULL,
    comment_body        TEXT NOT NULL,
    github_created_at   TIMESTAMPTZ NOT NULL,
    captured_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded          BOOLEAN NOT NULL DEFAULT FALSE,
    -- NULL means no live re-judge happened for this correction (capped,
    -- failed, or superseded) - "we don't know what would have changed".
    -- An empty array means a re-judge did happen but none of the tracked
    -- fields (suggested_labels, is_spam, priority) differed - "we know,
    -- and nothing changed". Set after the fact (see update_judgment) -
    -- absent at insert time since the outcome isn't known yet.
    changed_fields      TEXT[],
    UNIQUE (github_comment_id, judgment_id)
);
