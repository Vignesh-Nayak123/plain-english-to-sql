"""
Execution accuracy — the real metric.

A predicted query is correct if, run against the actual SQLite database, it
returns the SAME result set as the gold query. This is far more meaningful than
string matching: two very different-looking queries can be equally correct, and
two near-identical strings can differ by one wrong column.

Comparison is order-insensitive (a multiset of rows) unless the gold query
contains ORDER BY, in which case row order must match too.
"""
import sqlite3
import threading
from pathlib import Path


def db_file(db_root: str | Path, db_id: str) -> Path:
    """Path to a Spider database: <db_root>/<db_id>/<db_id>.sqlite"""
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def schema_from_sqlite(path: str | Path) -> str:
    """Reconstruct the CREATE TABLE schema string from a SQLite file.

    Matches the format the model was trained on (the raw DDL from sqlite_master),
    so the demo server can feed the model a real DB's schema with no extra work.
    """
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def run_sql(path: str | Path, sql: str, timeout: float = 15.0):
    """Execute SQL; return (rows, error). A watchdog interrupts runaway queries."""
    conn = sqlite3.connect(str(path))
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore") if isinstance(b, bytes) else b
    watchdog = threading.Timer(timeout, conn.interrupt)
    watchdog.start()
    try:
        cur = conn.execute(sql)
        return cur.fetchall(), None
    except Exception as exc:  # noqa: BLE001 - any SQL/exec error => prediction is wrong
        return None, str(exc)
    finally:
        watchdog.cancel()
        conn.close()


def _norm(rows, order_sensitive: bool):
    """Stringify cells so types compare cleanly; sort unless order matters."""
    table = [tuple(str(c) for c in row) for row in rows]
    return table if order_sensitive else sorted(table)


def execution_match(path: str | Path, pred_sql: str, gold_sql: str):
    """
    Return (is_correct, error).
      is_correct = True/False, or None if the GOLD query itself fails (skip example).
    """
    gold_rows, gerr = run_sql(path, gold_sql)
    if gerr is not None:
        return None, f"gold failed: {gerr}"

    pred_rows, perr = run_sql(path, pred_sql)
    if perr is not None:
        return False, perr

    order_sensitive = "order by" in gold_sql.lower()
    return _norm(pred_rows, order_sensitive) == _norm(gold_rows, order_sensitive), None
