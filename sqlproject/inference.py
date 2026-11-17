"""
Model loading + SQL generation. Shared by the eval harness and the demo server.
"""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompts import build_correction_messages, build_inference_messages

DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def load_model(base_model: str = DEFAULT_BASE, adapter_path: str | None = None,
               four_bit: bool = True):
    """Load the base model (4-bit by default) and optionally attach a LoRA adapter."""
    tok = AutoTokenizer.from_pretrained(base_model)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def clean_sql(text: str) -> str:
    """Strip markdown fences / prose; keep a single SQL statement."""
    text = text.strip()
    m = _SQL_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # cut at first statement terminator, drop trailing chatter
    if ";" in text:
        text = text.split(";")[0]
    return " ".join(text.split()).strip()


@torch.no_grad()
def _generate_from_messages(model, tok, msgs: list[dict], max_new_tokens: int = 256) -> str:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,            # greedy — deterministic, reproducible eval
        pad_token_id=tok.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return clean_sql(tok.decode(gen, skip_special_tokens=True))


def generate_sql(model, tok, schema: str, question: str, max_new_tokens: int = 256) -> str:
    return _generate_from_messages(
        model, tok, build_inference_messages(schema, question), max_new_tokens)


def generate_sql_with_retry(model, tok, schema: str, question: str, validate,
                            max_retries: int = 2, max_new_tokens: int = 256,
                            trace: list | None = None):
    """Generate SQL, then self-correct if it crashes against the database.

    `validate(sql) -> error_str_or_None` runs the query and returns the DB error
    (or None if it executes). On error we feed the failed query + error back to
    the model and regenerate, up to `max_retries` times. Returns
    (final_sql, attempts) where attempts == 1 means no correction was needed.

    If `trace` is provided, each attempt is appended as
    {"attempt": n, "sql": ..., "error": ...} — used by the demo UI to show the
    agent fixing itself.

    Note: this only fixes queries that *crash*. A query that runs but returns the
    wrong rows is left alone — there's no error signal to correct against.
    """
    sql = generate_sql(model, tok, schema, question, max_new_tokens)
    err = validate(sql)
    attempts = 1
    if trace is not None:
        trace.append({"attempt": attempts, "sql": sql, "error": err})
    while err is not None and attempts <= max_retries:
        msgs = build_correction_messages(schema, question, sql, err)
        fixed = _generate_from_messages(model, tok, msgs, max_new_tokens)
        attempts += 1
        new_err = validate(fixed)
        # accept the retry if it now runs, or as the new working draft to refine
        sql, err = fixed, new_err
        if trace is not None:
            trace.append({"attempt": attempts, "sql": sql, "error": err})
        if err is None:
            break
    return sql, attempts
