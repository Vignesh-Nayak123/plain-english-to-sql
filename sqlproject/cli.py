"""sqlproject CLI — generate (and optionally run) SQL from a natural-language question.

Examples:
    sqlproject -q "How many users signed up after 2020?" --db app.sqlite --run
    sqlproject -q "list all singers" --schema 'CREATE TABLE singer ("Name" text);'
"""
import argparse
import os
import sys

DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER = "Abdullahkousa2/sqlproject-qwen2.5-coder-1.5b"


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="sqlproject",
        description="Turn a natural-language question into SQL with the fine-tuned sqlproject model.")
    ap.add_argument("-q", "--question", required=True, help="the question to answer in SQL")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="path to a SQLite database (schema auto-extracted)")
    src.add_argument("--schema", help="raw CREATE TABLE text, or a path to a .sql file")
    ap.add_argument("--base", default=DEFAULT_BASE, help="base model (HF id or local path)")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="LoRA adapter (HF id or local path); pass 'none' for the plain base model")
    ap.add_argument("--no-4bit", action="store_true",
                    help="load in bf16/fp32 instead of 4-bit (use this on CPU)")
    ap.add_argument("--run", action="store_true", help="also execute the query (requires --db)")
    args = ap.parse_args()

    from .inference import generate_sql, generate_sql_with_retry, load_model

    if args.db:
        from .exec_eval import schema_from_sqlite
        schema = schema_from_sqlite(args.db)
    else:
        schema = open(args.schema, encoding="utf-8").read() if os.path.isfile(args.schema) else args.schema

    adapter = None if (args.adapter or "").lower() == "none" else args.adapter
    model, tok = load_model(args.base, adapter_path=adapter, four_bit=not args.no_4bit)

    if args.db and args.run:
        from .exec_eval import run_sql
        sql, _ = generate_sql_with_retry(
            model, tok, schema, args.question, validate=lambda s: run_sql(args.db, s)[1])
        print(sql)
        rows, err = run_sql(args.db, sql)
        if err:
            print(f"-- execution error: {err}", file=sys.stderr)
            sys.exit(1)
        for row in rows[:50]:
            print(row)
    else:
        print(generate_sql(model, tok, schema, args.question))


if __name__ == "__main__":
    main()
