"""
Phase 1b — download the Spider SQLite databases + tables.json (schemas).

The xlangai/spider HF dataset only ships the question/SQL pairs. The actual
SQLite databases (needed for execution-accuracy eval) and tables.json (needed to
build schema-aware prompts) come from the official Spider archive on Google Drive:
    https://yale-lily.github.io/spider

Run from the project root:
    python scripts/download_databases.py

If the Google Drive download ever fails with a quota error, download the zip
manually from the link above and drop it at  data/spider_data.zip , then re-run.
"""
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "spider_raw"

# Official Spider archive (spider_data.zip, ~1.4 GB)
GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "spider_data.zip"

    if not zip_path.exists():
        print("-> Downloading Spider archive (~1.4 GB) from Google Drive ...")
        gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False)
    else:
        print(f"-> Archive already present: {zip_path}")

    print("-> Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)

    # Locate the extracted tables.json + database files to confirm success
    tables = list(RAW.rglob("tables.json"))
    dbs = list(RAW.rglob("*.sqlite"))
    print(f"   tables.json found : {len(tables)} -> {tables[0] if tables else 'MISSING'}")
    print(f"   sqlite databases  : {len(dbs)}")
    if dbs:
        print(f"   example db        : {dbs[0]}")
    print("\nNext: run scripts/prepare_data.py")


if __name__ == "__main__":
    main()
