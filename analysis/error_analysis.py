#!/usr/bin/env python3
"""Error analysis for text-to-SQL evaluation output.

Reads the JSON written by scripts/evaluate.py (a dict with a "records" list; each
record has db_id, question, gold, pred, correct, error, attempts) and writes a
Markdown report.

Everything here is derived from the SQL *text* and the SQLite error strings, so it
needs no database access. The difficulty buckets and structural tags are heuristics,
NOT Spider's official hardness labels or the official evaluation script.

Usage:
    python analysis/error_analysis.py eval/finetuned_validation.json \
        --baseline eval/baseline_validation.json --out analysis/error_report.md
"""
import argparse
import collections
import json
import re


def norm(sql):
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def feats(sql):
    s = norm(sql)
    return {
        "joins": len(re.findall(r"\bjoin\b", s)),
        "group": " group by " in s,
        "order": " order by " in s,
        "limit": " limit " in s,
        "distinct": "distinct" in s,
        "sub": bool(re.search(r"\(\s*select", s)),
        "setop": bool(re.search(r"\b(intersect|union|except)\b", s)),
        "having": " having " in s,
    }


def bucket(gold):
    f = feats(gold)
    score = f["joins"] + 2 * f["sub"] + 2 * f["setop"] + f["group"] + f["having"] + f["order"]
    if f["setop"] or f["sub"] or f["joins"] >= 3 or score >= 5:
        return "hard/extra"
    if f["joins"] >= 1 or f["group"] or f["order"] or score >= 2:
        return "medium"
    return "easy"


def error_kind(err):
    e = err.lower()
    if "no such column" in e:
        return "no such column"
    if "no such table" in e:
        return "no such table"
    if "ambiguous" in e:
        return "ambiguous column"
    if "syntax" in e:
        return "syntax error"
    if "misuse of aggregate" in e:
        return "misuse of aggregate"
    if "timeout" in e or "interrupt" in e:
        return "timeout"
    return "other execution error"


def pct(c, t):
    return f"{100 * c / t:.1f}%" if t else "n/a"


def analyse(path):
    d = json.load(open(path))
    return d, d["records"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--baseline", help="baseline eval JSON, for a before/after comparison")
    ap.add_argument("--out", default="error_report.md")
    args = ap.parse_args()

    d, R = analyse(args.eval_json)
    n = len(R)
    ok = sum(r["correct"] for r in R)
    wrong = [r for r in R if not r["correct"]]
    out = []
    w = out.append

    w(f"# Error analysis: {d.get('tag', 'run')} ({d.get('split', '')})\n")
    w(f"Execution accuracy: **{ok}/{n} = {pct(ok, n)}**  |  failures: {len(wrong)}\n")
    w("Method notes: failure categories come from SQLite error strings and from comparing "
      "gold vs predicted SQL text. Difficulty buckets are a heuristic on the gold SQL, "
      "not Spider's official hardness. Counts from structural tags can overlap.\n")

    # 1. failure type
    kinds = collections.Counter()
    for r in wrong:
        kinds[error_kind(r["error"]) if r["error"] else "valid SQL, wrong result"] += 1
    w("## 1. Why predictions fail\n")
    w("| Failure type | Count | Share of failures |\n|---|---:|---:|")
    for k, v in kinds.most_common():
        w(f"| {k} | {v} | {pct(v, len(wrong))} |")
    n_exec = sum(1 for r in wrong if r["error"])
    w(f"\n{n_exec} of {len(wrong)} failures ({pct(n_exec, len(wrong))}) do not even execute. "
      "These are the cases an execution-feedback (self-correction) loop could in principle fix; "
      f"the ceiling would be +{100 * n_exec / n:.1f} points, and a real gain is expected to be smaller.\n")

    # 2. difficulty
    B = collections.defaultdict(lambda: [0, 0])
    for r in R:
        b = bucket(r["gold"])
        B[b][1] += 1
        B[b][0] += r["correct"]
    w("## 2. Accuracy by approximate difficulty\n")
    w("| Bucket | Correct / total | Accuracy |\n|---|---:|---:|")
    for b in ("easy", "medium", "hard/extra"):
        c, t = B[b]
        w(f"| {b} | {c}/{t} | {pct(c, t)} |")

    # 3. SQL feature in gold
    w("\n## 3. Accuracy when the gold query uses a feature\n")
    w("| Gold query has | Correct / total | Accuracy |\n|---|---:|---:|")
    for name in ("join", "group by", "order by", "subquery", "set operation", "having", "distinct"):
        c = t = 0
        for r in R:
            f = feats(r["gold"])
            v = {"join": f["joins"] >= 1, "group by": f["group"], "order by": f["order"],
                 "subquery": f["sub"], "set operation": f["setop"], "having": f["having"],
                 "distinct": f["distinct"]}[name]
            if v:
                t += 1
                c += r["correct"]
        w(f"| {name} | {c}/{t} | {pct(c, t)} |")

    # 4. structural differences on wrong-result cases
    wr = [r for r in wrong if not r["error"]]
    tags = collections.Counter()
    for r in wr:
        g, p = feats(r["gold"]), feats(r["pred"])
        cur = []
        if p["joins"] > g["joins"]:
            cur.append("extra join")
        elif p["joins"] < g["joins"]:
            cur.append("missing join")
        for k in ("group", "order", "limit", "distinct", "sub", "setop", "having"):
            if g[k] and not p[k]:
                cur.append(f"missing {k}")
            if p[k] and not g[k]:
                cur.append(f"extra {k}")
        if not cur:
            cur.append("same structure (column/value/condition choice)")
        tags.update(cur)
    w(f"\n## 4. Valid-but-wrong queries ({len(wr)}): structural difference from gold\n")
    w("| Difference | Count |\n|---|---:|")
    for k, v in tags.most_common():
        w(f"| {k} | {v} |")

    # 5. databases
    D = collections.defaultdict(lambda: [0, 0])
    for r in R:
        D[r["db_id"]][1] += 1
        D[r["db_id"]][0] += r["correct"]
    w("\n## 5. Weakest databases (20+ questions)\n")
    w("| Database | Correct / total | Accuracy |\n|---|---:|---:|")
    rows = [(k, c, t) for k, (c, t) in D.items() if t >= 20]
    for k, c, t in sorted(rows, key=lambda x: x[1] / x[2])[:8]:
        w(f"| {k} | {c}/{t} | {pct(c, t)} |")

    # 6. examples
    w("\n## 6. Sample failures\n")
    shown = collections.Counter()
    for r in wrong:
        key = error_kind(r["error"]) if r["error"] else "wrong result"
        if shown[key] >= 2:
            continue
        shown[key] += 1
        w(f"**{key}** ({r['db_id']}): {r['question']}\n")
        w(f"- gold: `{r['gold']}`\n- pred: `{r['pred']}`")
        if r["error"]:
            w(f"- error: `{r['error'][:120]}`")
        w("")

    # 7. baseline comparison
    if args.baseline:
        bd, BR = analyse(args.baseline)
        bmap = {(r["db_id"], r["question"], r["gold"]): r["correct"] for r in BR}
        fixed = broke = same_ok = same_bad = 0
        for r in R:
            k = (r["db_id"], r["question"], r["gold"])
            if k not in bmap:
                continue
            b, a = bmap[k], r["correct"]
            fixed += (not b) and a
            broke += b and (not a)
            same_ok += b and a
            same_bad += (not b) and (not a)
        w("\n## 7. Baseline vs fine-tuned, per question\n")
        w(f"Baseline: {bd['correct']}/{bd['total']} = {pct(bd['correct'], bd['total'])}\n")
        w("| Outcome | Count |\n|---|---:|")
        w(f"| wrong before, right after (fixed) | {fixed} |")
        w(f"| right before, wrong after (regressed) | {broke} |")
        w(f"| right both | {same_ok} |")
        w(f"| wrong both | {same_bad} |")
        w(f"\nNet change: {fixed - broke:+d} questions.")

    open(args.out, "w").write("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
