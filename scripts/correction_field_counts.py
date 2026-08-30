"""Print how many corrections have touched each tracked judgment field.

Phase 8 evidence: which parts of a judgment (suggested_labels, is_spam,
priority) real corrections tend to be about, so far. Only reflects
corrections captured after the changed_fields column existed - see
get_correction_field_counts's docstring for why older rows aren't
backfilled.

Run with:
    uv run python -m scripts.correction_field_counts [--since-days N]
"""

from __future__ import annotations

import argparse
import datetime as dt

from dotenv import load_dotenv

from src.db import connect, get_correction_field_counts


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="Only count corrections captured in the last N days (default: all-time).",
    )
    args = parser.parse_args()

    since = None
    if args.since_days is not None:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(days=args.since_days)

    with connect() as connection:
        counts = get_correction_field_counts(connection, since=since)

    if not counts:
        print("No re-judged corrections yet.")
        return

    total = sum(counts.values())
    for field, count in counts.items():
        print(f"{field}: {count} ({count / total:.0%})")


if __name__ == "__main__":
    main()
