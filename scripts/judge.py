"""Evaluación de la corrección de las respuestas (LLM-as-judge, a ciegas).

    uv run python scripts/judge.py prepare   # crea out/judge/items.jsonl (sin el nombre del modelo)
    uv run python scripts/judge.py score     # junta out/judge/judgments.jsonl y saca los resultados

El juez solo debe leer items.jsonl y escribir judgments.jsonl:
    {"item_id": "J001", "verdict": "correct|partial|incorrect|no_answer", "reason": "..."}
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "judge"
RESULTS = ROOT / "out" / "results" / "gen_test.jsonl"
DATASET = ROOT / "data" / "evals" / "test.jsonl"
VERDICTS = ("correct", "partial", "incorrect", "no_answer")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spec_rows(table, obj, paths):
    rows = []
    for path in paths:
        match = table[(table.object == obj) & (table.path == path)]
        if match.empty:
            match = table[table.path == path].head(1)
        for r in match.itertuples():
            rows.append({"object": r.object, "path": r.path, "type": r.type, "nullable": bool(r.nullable),
                         "expandable": bool(r.expandable), "enum": r.enum.split("|") if r.enum else [],
                         "description": r.description})
    return rows


def prepare():
    table = pd.read_parquet(ROOT / "out" / "table.parquet")
    questions = {q["id"]: q for q in read_jsonl(DATASET)}
    items, mapping = [], {}
    records = [r for r in read_jsonl(RESULTS) if r["id"] in questions]
    random.Random(2026).shuffle(records)
    for n, r in enumerate(records, start=1):
        q = questions[r["id"]]
        item_id = f"J{n:03d}"
        mapping[item_id] = {"model": r["model"], "qid": r["id"], "category": q["category"]}
        items.append({
            "item_id": item_id,
            "question": q["query"],
            "object": q["object"],
            "answerable": q["answerable"],
            "expected_fields": spec_rows(table, q["object"], q["relevant_paths"]),
            "answer": r["answer"],
            "model_said_found": r["found"],
            "cited_fields": spec_rows(table, q["object"], r["field_paths"]),
            "cited_paths_not_in_spec": r["invented_paths"],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    (OUT / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} respuestas a juzgar -> {OUT / 'items.jsonl'}")


def ci(values, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def enum_coverage(table):
    enum = {(r.object, r.path): r.enum.split("|") for r in table.itertuples() if "|" in r.enum}
    questions = {q["id"]: q for q in read_jsonl(DATASET)}
    out = defaultdict(list)
    for r in read_jsonl(RESULTS):
        q = questions.get(r["id"])
        if not q or q["category"] != "test_enum":
            continue
        values = enum[(q["object"], q["relevant_paths"][0])]
        text = r["answer"].lower()
        found = [v for v in values if re.search(rf"(?<![a-z0-9_]){re.escape(v.lower())}(?![a-z0-9_])", text)]
        out[r["model"]].append(len(found) / len(values))
    return out


def score():
    mapping = json.loads((OUT / "mapping.json").read_text(encoding="utf-8"))
    judgments = {j["item_id"]: j for j in read_jsonl(OUT / "judgments.jsonl")}
    missing = sorted(set(mapping) - set(judgments))
    bad = [k for k, j in judgments.items() if j.get("verdict") not in VERDICTS]
    if missing or bad:
        raise SystemExit(f"Faltan {len(missing)} juicios o hay {len(bad)} veredictos no válidos: {(missing + bad)[:5]}")
    by_model = defaultdict(lambda: defaultdict(list))
    for item_id, m in mapping.items():
        by_model[m["model"]]["answerable" if m["category"] != "test_unanswerable" else "unanswerable"].append(
            (judgments[item_id]["verdict"], m["category"]))
    lines = ["| Modelo | n | Correcta | Parcial | Incorrecta | No responde | Correcta o parcial |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for model, groups in by_model.items():
        verdicts = [v for v, _ in groups["answerable"]]
        rates = {v: verdicts.count(v) / len(verdicts) for v in VERDICTS}
        correct = [float(v == "correct") for v in verdicts]
        ok = [float(v in ("correct", "partial")) for v in verdicts]
        (lo, hi), (lo2, hi2) = ci(correct), ci(ok)
        unans = [v for v, _ in groups["unanswerable"]]
        summary[model] = {"rates": rates, "correct_ci95": [lo, hi], "correct_or_partial_ci95": [lo2, hi2],
                          "unanswerable_correct": unans.count("correct"), "unanswerable": len(unans),
                          "by_category": {c: sum(v == "correct" for v, cc in groups["answerable"] if cc == c)
                                          / sum(cc == c for _, cc in groups["answerable"])
                                          for c in sorted({c for _, c in groups["answerable"]})}}
        lines.append(f"| {model} | {len(verdicts)} | {rates['correct']:.3f} [{lo:.2f}, {hi:.2f}] | {rates['partial']:.3f} | "
                     f"{rates['incorrect']:.3f} | {rates['no_answer']:.3f} | {sum(ok) / len(ok):.3f} [{lo2:.2f}, {hi2:.2f}] |")
    lines += ["", "| Modelo | Sin respuesta bien tratadas | Correctas por categoría | Cobertura de valores enum |",
              "|---|---:|---|---:|"]
    coverage = enum_coverage(pd.read_parquet(ROOT / "out" / "table.parquet"))
    for model, s in summary.items():
        cats = ", ".join(f"{c.replace('test_', '')} {v:.2f}" for c, v in s["by_category"].items())
        cov = coverage.get(model, [])
        s["enum_coverage"] = sum(cov) / len(cov) if cov else None
        lines.append(f"| {model} | {s['unanswerable_correct']} de {s['unanswerable']} | {cats} | "
                     f"{s['enum_coverage']:.2f} ({sum(c == 1 for c in cov)} de {len(cov)} completas) |")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    report = "\n".join(lines)
    (OUT / "summary.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
