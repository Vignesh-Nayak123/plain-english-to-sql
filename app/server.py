"""
SQLForge — demo server.

A FastAPI app that loads the fine-tuned text-to-SQL model and turns a natural
-language question + database schema into a SQL query. When a real SQLite
database is available it also *runs* the query, shows the results, and uses
self-correction (feed the DB error back to the model and retry) so you can watch
the agent fix its own mistakes.

Run locally:
    uvicorn app.server:app --reload --port 8000
Then open http://localhost:8000
"""
import os
import sqlite3
import threading
import time
from pathlib import Path

# the model + tokenizer are local; don't reach for the (flaky) HF CDN at serve time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sqlforge.exec_eval import run_sql, schema_from_sqlite
from sqlforge.inference import generate_sql, generate_sql_with_retry, load_model

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

# --- config (all overridable by env, so HF Spaces / CI can swap paths) ----------
BASE_MODEL = os.environ.get("SQLFORGE_BASE", "models/qwen2.5-coder-1.5b")
if (ROOT / BASE_MODEL).is_dir():
    BASE_MODEL = str(ROOT / BASE_MODEL)
ADAPTER = os.environ.get("SQLFORGE_ADAPTER", str(ROOT / "outputs" / "qwen2.5-coder-1.5b-sql"))
FOUR_BIT = os.environ.get("SQLFORGE_4BIT", "1") == "1"
DB_DIR = Path(os.environ.get("SQLFORGE_DB_DIR",
                             str(ROOT / "data" / "spider_raw" / "spider_data" / "database")))
MAX_ROWS = 100    # cap result rows sent to the browser
GEN_TOKENS = 192  # SQL is short; cap new tokens for a snappier demo
MAX_RETRIES = 1   # one self-correction retry (worst case ~2 generations, not 3)

# curated example databases — every question below is verified to run cleanly
# against the real DB. The last car question deliberately triggers self-correction
# (a JOIN it fixes itself) to showcase the agentic recovery succeeding.
EXAMPLES = [
    {"db_id": "concert_singer", "label": "Concerts & Singers",
     "questions": ["How many singers do we have?",
                   "What is the average, minimum, and maximum age of all singers?",
                   "Show the name and country of all singers ordered by age from oldest to youngest."]},
    {"db_id": "pets_1", "label": "Students & Pets",
     "questions": ["How many pets are there?",
                   "What is the average weight of all pets?",
                   "How many students are there?"]},
    {"db_id": "world_1", "label": "World (countries)",
     "questions": ["What are the names of all countries that became independent after 1950?",
                   "How many countries have a republic as their form of government?",
                   "What is the average life expectancy of countries in Africa?"]},
    {"db_id": "car_1", "label": "Cars",
     "questions": ["How many continents are there?",
                   "What is the maximum horsepower of any car?",
                   "How many countries does each continent have? List continent id, name and count."]},
    {"db_id": "student_transcripts_tracking", "label": "Student Transcripts",
     "questions": ["How many courses in total are listed?",
                   "How many students are there?",
                   "List the first and last name of every student."]},
]

# --- model state (loaded in the background so the server starts instantly) ------
STATE = {"status": "loading", "model": ADAPTER, "device": None, "error": None}
_MODEL = {"model": None, "tok": None}
_LOCK = threading.Lock()  # one generation at a time (single GPU)


def _load():
    try:
        model, tok = load_model(BASE_MODEL, adapter_path=ADAPTER, four_bit=FOUR_BIT)
        _MODEL["model"], _MODEL["tok"] = model, tok
        STATE["device"] = str(getattr(model, "device", "cuda"))
        STATE["status"] = "online"
        print(f"[sqlforge] model online ({STATE['device']}, 4bit={FOUR_BIT})")
    except Exception as exc:  # noqa: BLE001
        STATE["status"] = "error"
        STATE["error"] = str(exc)
        print(f"[sqlforge] model failed to load: {exc}")


app = FastAPI(title="SQLForge", description="Fine-tuned text-to-SQL demo")


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load, daemon=True).start()


def _db_path(db_id: str) -> Path | None:
    """Resolve a known example db_id to its .sqlite file (no path traversal)."""
    if not db_id or "/" in db_id or "\\" in db_id or ".." in db_id:
        return None
    p = DB_DIR / db_id / f"{db_id}.sqlite"
    return p if p.exists() else None


# --- API ------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    question: str
    schema_text: str | None = None   # raw CREATE TABLE text (custom mode)
    db_id: str | None = None         # example DB id (executes + self-corrects)
    self_correct: bool = True


@app.get("/api/health")
def health():
    return STATE


@app.get("/api/examples")
def examples():
    return [e for e in EXAMPLES if _db_path(e["db_id"])]


@app.get("/api/schema")
def schema(db_id: str):
    p = _db_path(db_id)
    if not p:
        raise HTTPException(404, f"unknown database '{db_id}'")
    return {"db_id": db_id, "schema": schema_from_sqlite(p)}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if STATE["status"] != "online":
        raise HTTPException(503, f"model not ready ({STATE['status']})")
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    db_path = _db_path(req.db_id) if req.db_id else None
    schema_text = req.schema_text
    if db_path and not schema_text:
        schema_text = schema_from_sqlite(db_path)
    if not schema_text:
        raise HTTPException(400, "provide a schema or pick an example database")

    model, tok = _MODEL["model"], _MODEL["tok"]
    trace: list = []
    t0 = time.time()
    with _LOCK:
        if db_path and req.self_correct:
            sql, attempts = generate_sql_with_retry(
                model, tok, schema_text, req.question,
                validate=lambda s: run_sql(db_path, s)[1],
                max_retries=MAX_RETRIES, max_new_tokens=GEN_TOKENS, trace=trace)
        else:
            sql = generate_sql(model, tok, schema_text, req.question,
                               max_new_tokens=GEN_TOKENS)
            attempts = 1
    elapsed = round(time.time() - t0, 2)

    resp = {"sql": sql, "attempts": attempts, "trace": trace,
            "self_corrected": attempts > 1, "elapsed_s": elapsed,
            "executed": False, "columns": None, "rows": None,
            "row_count": None, "error": None}

    # if we have the real DB, run the final query and return the result preview
    if db_path:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(sql)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(MAX_ROWS)
            resp.update(executed=True, columns=cols,
                        rows=[list(r) for r in rows], row_count=len(rows))
        except Exception as exc:  # noqa: BLE001
            resp.update(executed=True, error=str(exc))
        finally:
            conn.close()
    return resp


# --- static frontend (mounted last so /api/* wins) ------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC)), name="static")
