"""Fast, GPU-free tests for the core library (schema parsing + execution grader).

Run directly (`python tests/test_core.py`) or with pytest. No model required, so
this is safe to run in CI on every push.
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlforge.exec_eval import execution_match, run_sql, schema_from_sqlite
from sqlforge.prompts import build_correction_messages, schema_identifiers


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE "singer" ("Singer_ID" int, "Name" text, "Age" int)')
    conn.executemany("INSERT INTO singer VALUES (?,?,?)",
                     [(1, "Ana", 30), (2, "Ben", 25), (3, "Cy", 40)])
    conn.commit()
    conn.close()


def test_schema_identifiers():
    ddl = ('CREATE TABLE "singer" (\n"Singer_ID" int,\n"Name" text,\n'
           'PRIMARY KEY ("Singer_ID")\n);')
    assert schema_identifiers(ddl) == {"singer": ["Singer_ID", "Name"]}


def test_correction_prompt_lists_real_names():
    ddl = 'CREATE TABLE "concert" (\n"concert_ID" int,\n"Theme" text\n);'
    msgs = build_correction_messages(ddl, "q", "SELECT bad FROM nope", "no such table: nope")
    user = msgs[-1]["content"]
    assert "concert(concert_ID, Theme)" in user
    assert "do not invent" in user


def test_execution_match_and_run():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "t.sqlite")
        _make_db(db)

        # schema round-trips out of the DB
        assert "singer" in schema_from_sqlite(db).lower()

        # a correct query matches the gold
        ok, err = execution_match(db, "SELECT count(*) FROM singer",
                                  "SELECT count(*) FROM singer")
        assert ok is True and err is None

        # different row order still matches when the gold has no ORDER BY
        ok, _ = execution_match(db, "SELECT Name FROM singer ORDER BY Age DESC",
                                "SELECT Name FROM singer")
        assert ok is True  # gold has no ORDER BY -> comparison is order-insensitive

        # a crashing query reports the error, not a false match
        ok, err = execution_match(db, "SELECT nope FROM singer", "SELECT count(*) FROM singer")
        assert ok is False and "nope" in err

        # run_sql surfaces errors directly
        _, err = run_sql(db, "SELECT * FROM missing_table")
        assert err and "no such table" in err


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
