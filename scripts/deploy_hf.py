"""
One-shot deploy to a Hugging Face Space (Docker SDK).

Assembles the 5 example databases into demo_dbs/, then uploads the serving
files (app/, sqlforge/, demo_dbs/, Dockerfile, requirements-serve.txt) plus
README_HF.md (as the Space's README.md) using the huggingface_hub API.

Usage (Anaconda Prompt, Nenv active):
    python scripts/deploy_hf.py --token hf_xxxxxxxx
    # or set HF_TOKEN in the environment and omit --token

Notes:
  - Uses upload_folder with the token passed directly (no `huggingface-cli login`,
    no git-LFS — both are flaky on Windows).
  - The Space must already exist (SDK: Docker). Create at https://huggingface.co/new-space
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_SRC = ROOT / "data" / "spider_raw" / "spider_data" / "database"
DEMO = ROOT / "demo_dbs"
EXAMPLE_DBS = ["concert_singer", "pets_1", "world_1", "car_1",
               "student_transcripts_tracking"]


def assemble_demo_dbs() -> None:
    DEMO.mkdir(exist_ok=True)
    for db in EXAMPLE_DBS:
        src = DB_SRC / db / f"{db}.sqlite"
        if not src.exists():
            sys.exit(f"missing example DB: {src} — run scripts/download_databases.py first")
        dst = DEMO / db
        dst.mkdir(exist_ok=True)
        shutil.copy2(src, dst / f"{db}.sqlite")
    print(f"-> assembled demo_dbs/ ({len(EXAMPLE_DBS)} databases)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Abdullahkousa2/sqlforge",
                    help="HF Space id (user/space-name)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token (or set HF_TOKEN env var)")
    args = ap.parse_args()
    if not args.token:
        sys.exit("provide a write token via --token or the HF_TOKEN env var")

    assemble_demo_dbs()

    from huggingface_hub import HfApi
    api = HfApi()

    # serving files only — ignore everything heavy/dev
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.repo,
        repo_type="space",
        token=args.token,
        allow_patterns=["app/**", "sqlforge/**", "demo_dbs/**",
                        "Dockerfile", "requirements-serve.txt"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
    )
    # README with HF frontmatter -> the Space's README.md
    api.upload_file(
        path_or_fileobj=str(ROOT / "README_HF.md"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="space",
        token=args.token,
    )
    print(f"-> deployed to https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
