"""Genera un conjunto de control a partir de la tabla, para medir UNA vez al final.

No se usa para decidir nada durante las iteraciones. Uso:
    RAG_OUT=out/experiments/E1_sin_copias uv run python scripts/make_control_set.py
"""
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RAG_OUT", ROOT / "out"))
TARGET = ROOT / "data" / "evals" / "control.jsonl"
PER_TYPE = 40


def words(text):
    return text.replace("_", " ").replace(".", " ")


def clause(description):
    first = re.split(r"(?<=[.!?])\s", description)[0].rstrip(".")
    first = first[0].lower() + first[1:]
    return " ".join(first.split()[:20])


def main():
    if TARGET.exists():
        raise SystemExit(f"{TARGET.name} ya existe. No se regenera: es el conjunto de control.")
    rows = pd.read_parquet(OUT / "table.parquet")
    rows = rows[(~rows.has_children) & (rows.depth <= 2) & (rows.description.str.split().str.len() >= 5)]
    rows = rows[~rows.name.isin(["object", "id", "livemode", "metadata"])]
    rng = random.Random(2026)
    picked = rows.sample(n=min(len(rows), PER_TYPE * 3), random_state=2026).to_dict("records")
    rng.shuffle(picked)

    questions = []
    for i, r in enumerate(picked):
        obj = words(r["object"])
        kind = ["desc", "name", "ident"][i % 3]
        if kind == "desc":
            query = f"On a {obj} object, which field holds {clause(r['description'])}?"
        elif kind == "name":
            leaf = words(r["name"])
            parent = words(r["path"].rsplit(".", 1)[0]) if "." in r["path"] else ""
            query = f"What is the {leaf}{' in the ' + parent if parent else ''} of a {obj}?"
        else:
            query = f"What does `{r['path']}` mean on a {obj}?"
        questions.append({
            "id": f"heldout-{i + 1:03d}", "query": query, "object": r["object"],
            "category": f"heldout_{kind}", "difficulty": "medium", "answerable": True,
            "relevant_paths": [r["path"]],
        })
    TARGET.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    print(f"{len(questions)} preguntas de control -> {TARGET}")


if __name__ == "__main__":
    main()
