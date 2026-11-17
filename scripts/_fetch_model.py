"""
Resilient model fetch for flaky networks.

HF serves Qwen weights via its Xet CDN, which times out intermittently on some
connections. hf_hub_download resumes from the partial .incomplete file, so we
just retry until the 3GB safetensors lands. One-off helper; not part of the lib.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FILES = ["config.json", "generation_config.json", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

for fname in FILES:
    for attempt in range(1, 501):
        try:
            p = hf_hub_download(REPO, fname)
            print(f"[OK] {fname} -> {p}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[retry {attempt}] {fname}: {type(exc).__name__}: "
                  f"{str(exc)[:120]} — resuming in 5s", flush=True)
            time.sleep(5)
    else:
        print(f"[FAIL] gave up on {fname}", flush=True)
        sys.exit(1)

print("ALL FILES PRESENT", flush=True)
