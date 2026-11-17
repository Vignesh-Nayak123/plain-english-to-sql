"""
Phase 3 — QLoRA fine-tuning of Qwen2.5-Coder-1.5B on Spider (text-to-SQL).

4-bit (nf4) base + a small LoRA adapter trained on the chat-formatted pairs in
data/processed/train.jsonl. Tuned to fit a single RTX 3070 (8GB).

  Quick smoke test (a few steps, no W&B):
      python scripts/train.py --limit 64 --max-steps 5 --no-wandb

  Full run:
      python scripts/train.py

Only the LoRA adapter is saved (a few MB) to configs.training.output_dir.
"""
# --- Windows UTF-8 guard -----------------------------------------------------
# TRL reads bundled .jinja chat templates with the platform default codec; on
# Windows that's cp1252 and the files contain non-cp1252 bytes -> import crashes.
# Re-exec the interpreter in UTF-8 mode so open() defaults to utf-8 everywhere.
import os
import sys
if sys.flags.utf8_mode == 0:
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])
# -----------------------------------------------------------------------------

# base model is a local dir; don't reach for the (flaky) HF CDN during training.

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DEFAULT_CONFIG = ROOT / "configs" / "train_config.yaml"

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dataset(tok, max_len: int, limit: int | None):
    """Load chat-formatted rows; DROP (not truncate) anything over max_len tokens.

    The gold SQL is the last turn, so truncation would silently delete the label
    we want the model to learn. Dropping the ~3% of long examples is correct.
    """
    rows = [json.loads(l) for l in (PROC / "train.jsonl").open(encoding="utf-8")]
    if limit:
        rows = rows[:limit]

    kept, dropped = [], 0
    for r in rows:
        n = len(tok.apply_chat_template(r["messages"], tokenize=True,
                                        add_generation_prompt=False))
        if n > max_len:
            dropped += 1
            continue
        kept.append({"messages": r["messages"]})

    print(f"-> dataset: kept {len(kept)} / {len(rows)} examples "
          f"(dropped {dropped} over {max_len} tokens)")
    return Dataset.from_list(kept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N train examples (smoke test)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (smoke test); overrides epochs")
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable Weights & Biases logging")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in output_dir")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    m, lora, tr = cfg["model"], cfg["lora"], cfg["training"]
    out_dir = str(ROOT / tr["output_dir"])

    # resolve a local model dir against the project root (so CWD doesn't matter);
    # a HF repo id like "Qwen/..." is left untouched.
    base_model = m["base_model"]
    if (ROOT / base_model).is_dir():
        base_model = str(ROOT / base_model)

    use_wandb = False
    if not args.no_wandb and tr.get("report_to") == "wandb":
        try:
            import wandb
            # picks up creds from `wandb login` (netrc) OR a WANDB_API_KEY env var;
            # purely local, no network/prompt.
            use_wandb = bool(os.environ.get("WANDB_API_KEY")) or bool(wandb.Api().api_key)
        except Exception:
            use_wandb = False
        if use_wandb:
            os.environ.setdefault("WANDB_PROJECT", "sqlforge-text2sql")
        else:
            print("!! W&B not logged in — run `wandb login <KEY>` once to enable live "
                  "charts (or pass --no-wandb to silence). Logging to console only.")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # dataset (filtered)
    train_ds = build_dataset(tok, tr["max_seq_length"], args.limit)

    # 4-bit base model
    bnb = BitsAndBytesConfig(
        load_in_4bit=m["load_in_4bit"],
        bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_DTYPE[m["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=m["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb,
        device_map="auto", dtype=torch.float16,
    )
    model.config.use_cache = False                      # required w/ grad checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=tr["gradient_checkpointing"])

    lora_cfg = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    )

    sft = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=tr["num_train_epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_steps=15,
        max_length=tr["max_seq_length"],               # TRL 1.5 renamed max_seq_length -> max_length
        packing=False,
        gradient_checkpointing=tr["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False, fp16=True,
        optim=tr["optim"],
        logging_steps=tr["logging_steps"],
        save_strategy=tr["save_strategy"], save_steps=50, save_total_limit=2,
        report_to="wandb" if use_wandb else "none",
        seed=tr["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=lora_cfg,
    )

    for _p in trainer.model.parameters():
        if _p.requires_grad:
            _p.data = _p.data.float()
    trainer.train(resume_from_checkpoint=args.resume)

    # save just the adapter + tokenizer
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\nDone -> adapter saved to {out_dir}\n"
          f"Next: python scripts/evaluate.py --adapter {tr['output_dir']}")


if __name__ == "__main__":
    main()
