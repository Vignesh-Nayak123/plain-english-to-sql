"""
Phase 1 — download the Spider text-to-SQL dataset.

Pulls two things:
  1. The question/SQL pairs (train + validation) via the `datasets` library.
  2. The raw Spider files (tables.json + database/*.sqlite) via huggingface_hub —
     needed later for schema-aware prompts and execution-accuracy evaluation.

Run from the project root:
    python scripts/download_data.py
"""
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)

    print("-> Downloading question/SQL pairs (xlangai/spider) ...")
    ds = load_dataset("xlangai/spider")
    ds.save_to_disk(str(DATA / "spider_pairs"))
    print(f"   train={len(ds['train'])}  validation={len(ds['validation'])}")
    sample = {k: ds["train"][0][k] for k in ("db_id", "question", "query")}
    print("   sample:", sample)

    print("-> Downloading raw Spider files (schemas + sqlite DBs) ...")
    snapshot_download(
        repo_id="xlangai/spider",
        repo_type="dataset",
        local_dir=str(DATA / "spider_raw"),
        allow_patterns=["*.json", "database/**", "*.zip"],
    )
    print("   done -> data/spider_raw")
    print("\nNext: inspect the data, then run scripts/prepare_data.py")


if __name__ == "__main__":
    main()
