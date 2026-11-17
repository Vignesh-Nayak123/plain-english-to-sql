"""
Phase 1c — build schema-aware training data.

Combines:
  - the question/SQL pairs            (data/spider_pairs, from download_data.py)
  - the per-database schemas          (tables.json, from download_databases.py)

into chat-formatted JSONL where each example is:
  system : instruction to write SQLite SQL
  user   : CREATE TABLE schema + the question
  assistant : the gold SQL query

The model must SEE the schema to write correct SQL, so we render each database
as real CREATE TABLE DDL (with primary/foreign keys) — the format LLMs are most
fluent in from pre-training.

Run from the project root:
    python scripts/prepare_data.py
"""
import json
import sys
from pathlib import Path

from datasets import load_from_disk

# shared prompt format (single source of truth with inference/eval)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlforge.prompts import build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES_JSON = DATA / "spider_raw" / "spider_data" / "tables.json"
PAIRS = DATA / "spider_pairs"
OUT = DATA / "processed"

# Spider column types -> SQL-ish types for the rendered DDL
TYPE_MAP = {
    "text": "TEXT",
    "number": "INTEGER",
    "time": "DATETIME",
    "boolean": "BOOLEAN",
    "others": "TEXT",
}

def render_schema(db: dict) -> str:
    """Render one Spider db entry as CREATE TABLE statements."""
    cols = db["column_names_original"]          # [[table_idx, "ColName"], ...]
    types = db["column_types"]                  # aligned to cols
    tables = db["table_names_original"]
    pk = set(db["primary_keys"])                # column indices
    fks = db["foreign_keys"]                    # [[src_col_idx, ref_col_idx], ...]

    statements = []
    for ti, tname in enumerate(tables):
        defs = []
        # column definitions for this table
        for ci, (col_tbl, col_name) in enumerate(cols):
            if col_tbl != ti:
                continue
            sql_type = TYPE_MAP.get(types[ci], "TEXT")
            defs.append(f'  "{col_name}" {sql_type}')

        # composite-safe primary key constraint
        pk_cols = [cols[ci][1] for ci in pk if 0 <= ci < len(cols) and cols[ci][0] == ti]
        if pk_cols:
            joined = ", ".join(f'"{c}"' for c in pk_cols)
            defs.append(f"  PRIMARY KEY ({joined})")

        # foreign keys originating from this table
        for src, ref in fks:
            if cols[src][0] != ti:
                continue
            src_col = cols[src][1]
            ref_table = tables[cols[ref][0]]
            ref_col = cols[ref][1]
            defs.append(
                f'  FOREIGN KEY ("{src_col}") REFERENCES "{ref_table}"("{ref_col}")'
            )

        statements.append(f'CREATE TABLE "{tname}" (\n' + ",\n".join(defs) + "\n);")

    return "\n\n".join(statements)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) schema lookup: db_id -> rendered DDL
    raw = json.loads(TABLES_JSON.read_text(encoding="utf-8"))
    schemas = {db["db_id"]: render_schema(db) for db in raw}
    print(f"-> Rendered schemas for {len(schemas)} databases")

    # 2) join pairs with schemas, write JSONL
    ds = load_from_disk(str(PAIRS))
    for split, fname in [("train", "train.jsonl"), ("validation", "validation.jsonl")]:
        rows, skipped, char_lens = 0, 0, []
        with (OUT / fname).open("w", encoding="utf-8") as f:
            for ex in ds[split]:
                schema = schemas.get(ex["db_id"])
                if schema is None:
                    skipped += 1
                    continue
                rec = {
                    "db_id": ex["db_id"],
                    "question": ex["question"],
                    "query": ex["query"],
                    "schema": schema,
                    "messages": build_messages(schema, ex["question"], ex["query"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1
                char_lens.append(len(schema) + len(ex["question"]))
        avg = sum(char_lens) // max(len(char_lens), 1)
        mx = max(char_lens) if char_lens else 0
        print(f"   {split:10s}: wrote {rows} rows  (skipped {skipped}) "
              f"| schema+question chars avg={avg} max={mx}")

    print(f"\nDone -> {OUT}\nNext: baseline eval (Phase 2), then training (Phase 3).")


if __name__ == "__main__":
    main()
