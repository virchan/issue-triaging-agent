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

-- One row per calendar day's digest, published to the operator-owned
-- shadow repo (see design-plan.md §3 and LOG.md entries 2-3).
CREATE TABLE IF NOT EXISTS digests (
    id                      SERIAL PRIMARY KEY,
    digest_date             DATE NOT NULL UNIQUE,
    shadow_repo_owner       TEXT NOT NULL,
    shadow_repo_name        TEXT NOT NULL,
    shadow_issue_number     INTEGER,
    state                   TEXT NOT NULL DEFAULT 'pending'
                            CHECK (state IN ('pending', 'published', 'reviewed')),
    published_at            TIMESTAMPTZ,
    closed_at                TIMESTAMPTZ
);

-- Draft shape: the exact fields here (label taxonomy, confidence, etc.)
-- are expected to be revisited at Step 11, when the judgment schema gets
-- finalized as a Pydantic model. This is a reasonable starting point, not
-- a locked contract.
CREATE TABLE IF NOT EXISTS judgments (
    id                  SERIAL PRIMARY KEY,
    issue_id            INTEGER NOT NULL UNIQUE REFERENCES issues (id),
    digest_id           INTEGER REFERENCES digests (id),
    suggested_label     TEXT,
    is_spam             BOOLEAN NOT NULL DEFAULT FALSE,
    rationale           TEXT,
    confidence          REAL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The operator's corrections, parsed from their comments on a digest
-- issue (see LOG.md entry 3 — publish-then-review-and-correct). These
-- feed back into future prompts as few-shot context and grow the golden
-- evaluation set (open-questions.md items 3-4).
CREATE TABLE IF NOT EXISTS corrections (
    id                  SERIAL PRIMARY KEY,
    judgment_id         INTEGER NOT NULL REFERENCES judgments (id),
    github_comment_id   BIGINT NOT NULL UNIQUE,
    comment_body        TEXT NOT NULL,
    github_created_at   TIMESTAMPTZ NOT NULL,
    captured_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
