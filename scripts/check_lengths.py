"""
Phase 1d — measure the REAL token-length distribution of the training examples,
using the actual Qwen tokenizer + chat template.

This tells us what to set max_seq_length to, and how many examples would be lost
at each cutoff. Downloads only the tokenizer (a few MB), no model weights, no GPU.

Run from the project root:
    python scripts/check_lengths.py
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
THRESHOLDS = [1024, 1280, 1536, 2048]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = int(round((p / 100) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    for fname in ("train.jsonl", "validation.jsonl"):
        path = OUT / fname
        lengths = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                msgs = json.loads(line)["messages"]
                ids = tok.apply_chat_template(msgs, tokenize=True)
                lengths.append(len(ids))
        lengths.sort()
        n = len(lengths)
        print(f"\n=== {fname}  (n={n}) ===")
        print(f"  min={lengths[0]}  p50={percentile(lengths,50)}  "
              f"p90={percentile(lengths,90)}  p95={percentile(lengths,95)}  "
              f"p99={percentile(lengths,99)}  max={lengths[-1]}")
        for t in THRESHOLDS:
            over = sum(1 for x in lengths if x > t)
            print(f"  > {t:5d} tokens: {over:5d} examples ({100*over/n:.1f}%)")


if __name__ == "__main__":
    main()
