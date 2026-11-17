"""
Single source of truth for the prompt format.

Both data prep (training) and inference (eval + serving) import from here, so the
model is always evaluated with the exact format it was trained on — no skew.
"""
import re

SYSTEM_PROMPT = (
    "You are an expert data analyst. Given a SQLite database schema and a "
    "question, write a single valid SQLite SQL query that answers it. "
    "Respond with only the SQL query and nothing else."
)


def build_user(schema: str, question: str) -> str:
    return f"Database schema:\n{schema}\n\nQuestion: {question}"


def build_messages(schema: str, question: str, query: str) -> list[dict]:
    """Full training example (system + user + assistant)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user(schema, question)},
        {"role": "assistant", "content": query},
    ]


def build_inference_messages(schema: str, question: str) -> list[dict]:
    """Prompt for generation (no assistant turn — model completes it)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user(schema, question)},
    ]


_CREATE_RE = re.compile(r'CREATE\s+TABLE\s+["`\[]?(\w+)', re.IGNORECASE)
_COL_RE = re.compile(r'^["`\[]?(\w+)["`\]]?\s+\w')
_SKIP = {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK", "KEY"}


def schema_identifiers(schema: str) -> dict[str, list[str]]:
    """Parse a CREATE TABLE schema string into {table: [columns]}.

    The model's crashes are almost always invented/pluralised identifiers, so we
    hand it the *exact* names back in the correction prompt. Line-based parse
    matches the one-column-per-line DDL both the dataset and SQLite produce.
    """
    tables: dict[str, list[str]] = {}
    current = None
    for raw in schema.splitlines():
        line = raw.strip()
        m = _CREATE_RE.match(line)
        if m:
            current = m.group(1)
            tables[current] = []
            continue
        if current is None:
            continue
        if line.startswith(")"):
            current = None
            continue
        toks = line.split()
        if toks and toks[0].strip('("`[]').upper() in _SKIP:
            continue
        cm = _COL_RE.match(line)
        if cm:
            tables[current].append(cm.group(1))
    return tables


def _schema_facts(schema: str) -> str:
    ids = schema_identifiers(schema)
    if not ids:
        return ""
    lines = [f"- {t}({', '.join(cols)})" for t, cols in ids.items() if cols]
    return "The database has ONLY these tables and columns:\n" + "\n".join(lines)


def build_correction_messages(schema: str, question: str, bad_sql: str,
                              error: str) -> list[dict]:
    """Self-correction prompt: the model's previous query failed to execute.

    Keeps the original system/task framing (so it stays in-distribution with
    training) and appends the failed attempt + the SQLite error, PLUS the exact
    valid table/column names parsed from the schema — so the model can't keep
    inventing or pluralising identifiers. Also nudges it away from needless JOINs
    (the 1.5B's main failure mode).
    """
    facts = _schema_facts(schema)
    user = (
        f"{build_user(schema, question)}\n\n"
        f"Your previous query failed with a database error.\n"
        f"Previous query: {bad_sql}\n"
        f"Error: {error}\n\n"
        f"{facts}\n\n"
        f"Use ONLY the exact table and column names listed above — do not invent, "
        f"pluralise, or rename them. Prefer the simplest query and avoid "
        f"unnecessary JOINs. Then write a corrected SQLite query. "
        f"Respond with only the SQL query."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
