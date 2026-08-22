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

-- Matches src/judgment.py's IssueJudgment. Note what's deliberately
-- absent: no duplicate-candidate fields (dropped from the MVP).
CREATE TABLE IF NOT EXISTS judgments (
    id                  SERIAL PRIMARY KEY,
    issue_id            INTEGER NOT NULL UNIQUE REFERENCES issues (id),
    digest_id           INTEGER REFERENCES digests (id),
    suggested_label     TEXT,
    is_spam             BOOLEAN NOT NULL DEFAULT FALSE,
    summary             TEXT NOT NULL,
    priority            TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    rationale           TEXT NOT NULL,
    confidence          REAL NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The operator's corrections, parsed from their comments on a digest
-- issue - publish-then-review-and-correct. These feed back into future
-- prompts as few-shot context and grow the golden evaluation set
-- (open-questions.md items 3-4). One GitHub comment can correct several
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
    -- fields (suggested_label, is_spam, priority) differed - "we know,
    -- and nothing changed". Set after the fact (see update_judgment) -
    -- absent at insert time since the outcome isn't known yet.
    changed_fields      TEXT[],
    UNIQUE (github_comment_id, judgment_id)
);
