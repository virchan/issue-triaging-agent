"""Apply db/schema.sql against DATABASE_URL.

Run with:
    uv run python -m scripts.init_db
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.db import connect

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def main() -> None:
    load_dotenv()

    schema_sql = SCHEMA_PATH.read_text()

    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(schema_sql)
        connection.commit()

    print(f"Applied {SCHEMA_PATH.name} successfully.")


if __name__ == "__main__":
    main()
