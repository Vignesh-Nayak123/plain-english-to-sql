"""
Phase 2/4 — execution-accuracy evaluation.

Generates SQL for each example in a processed split, runs it against the real
SQLite database, and compares result sets to the gold query.

  Baseline (no fine-tuning):
      python scripts/evaluate.py --limit 200

  Fine-tuned model (after training):
      python scripts/evaluate.py --adapter outputs/qwen2.5-coder-1.5b-sql

Results (per-example correctness + summary) are written to data/eval/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# model + tokenizer are local; never reach for the (flaky) HF CDN at eval time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlforge.exec_eval import db_file, execution_match, run_sql  # noqa: E402
from sqlforge.inference import (  # noqa: E402
    DEFAULT_BASE, generate_sql, generate_sql_with_retry, load_model)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DB_ROOT = ROOT / "data" / "spider_raw" / "spider_data" / "database"
EVAL_OUT = ROOT / "data" / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-4bit", action="store_true",
                    help="load in bf16 instead of 4-bit (1.5B fits in 8GB for inference)")
    ap.add_argument("--wandb", action="store_true",
                    help="log execution accuracy to Weights & Biases")
    ap.add_argument("--self-correct", action="store_true",
                    help="retry queries that crash, feeding the DB error back to the model")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="max self-correction attempts per query (default 2)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (PROC / f"{args.split}.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tag = "finetuned" if args.adapter else "baseline"
    if args.self_correct:
        tag += "-selfcorrect"
    print(f"== {tag} eval | model={args.base_model} | adapter={args.adapter} | "
          f"{len(rows)} examples"
          + (f" | self-correct (max {args.max_retries} retries)" if args.self_correct else "")
          + " ==")

    model, tok = load_model(args.base_model, adapter_path=args.adapter,
                            four_bit=not args.no_4bit)

    correct = total = errors = skipped = recovered = 0
    records = []
    t0 = time.time()
    for i, ex in enumerate(rows, 1):
        path = db_file(DB_ROOT, ex["db_id"])
        if args.self_correct:
            pred, attempts = generate_sql_with_retry(
                model, tok, ex["schema"], ex["question"],
                validate=lambda sql: run_sql(path, sql)[1],
                max_retries=args.max_retries, max_new_tokens=args.max_new_tokens)
        else:
            pred = generate_sql(model, tok, ex["schema"], ex["question"],
                                max_new_tokens=args.max_new_tokens)
            attempts = 1
        ok, err = execution_match(path, pred, ex["query"])

        if ok is None:                 # gold query itself failed — don't count
            skipped += 1
        else:
            total += 1
            correct += int(ok)
            if err:
                errors += 1
            elif attempts > 1:         # ran clean only after a self-correction retry
                recovered += 1
        records.append({"db_id": ex["db_id"], "question": ex["question"],
                        "gold": ex["query"], "pred": pred, "correct": ok,
                        "error": err, "attempts": attempts})

        if i % 25 == 0 or i == len(rows):
            acc = 100 * correct / total if total else 0.0
            print(f"  [{i}/{len(rows)}] exec-acc={acc:.1f}%  "
                  f"(correct={correct}/{total}, pred-errors={errors}, skipped={skipped}"
                  + (f", recovered={recovered}" if args.self_correct else "") + ")")

    acc = 100 * correct / total if total else 0.0
    dt = time.time() - t0
    print(f"\n== {tag}: execution accuracy = {acc:.2f}%  "
          f"({correct}/{total}) | {dt:.0f}s | {dt/max(len(rows),1):.1f}s/example ==")

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    out = EVAL_OUT / f"{tag}_{args.split}.json"
    out.write_text(json.dumps(
        {"tag": tag, "split": args.split, "base_model": args.base_model,
         "adapter": args.adapter, "self_correct": args.self_correct,
         "execution_accuracy": acc,
         "correct": correct, "total": total, "pred_errors": errors,
         "gold_skipped": skipped, "recovered": recovered, "records": records},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   wrote {out}")

    if args.wandb:
        import wandb  # logs to the same project the training run used
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "sqlforge-text2sql"),
            name=f"eval-{tag}", job_type="eval",
            config={"tag": tag, "base_model": args.base_model, "adapter": args.adapter,
                    "split": args.split, "limit": args.limit,
                    "max_new_tokens": args.max_new_tokens,
                    "self_correct": args.self_correct,
                    "max_retries": args.max_retries if args.self_correct else 0},
        )
        wandb.log({"execution_accuracy": acc, "correct": correct, "total": total,
                   "pred_errors": errors, "gold_skipped": skipped,
                   "recovered": recovered})
        # promote to summary so it shows as a sortable column in the runs table
        run.summary["execution_accuracy"] = acc
        run.summary["pred_errors"] = errors
        run.finish()
        print(f"   logged to W&B as run 'eval-{tag}'")


if __name__ == "__main__":
    main()
