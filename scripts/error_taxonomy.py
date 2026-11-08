"""Clasifica cada pregunta fallada del test en una fase (búsqueda o generación) y un tipo de error.

    uv run python scripts/error_taxonomy.py [--model google/gemma-4-e4b]   # escribe out/analysis/error_taxonomy_<modelo>.md

Usa los resultados ya calculados: out/results/gen_test.jsonl y los veredictos del juez (out/judge/).
Las reglas son deterministas; solo los tipos de generación dependen del veredicto del juez.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag.retrieve import mentioned_objects

OUT = ROOT / "out" / "analysis"
FIELD_LINE = re.compile(r"^\s*- `([^`]+)` \(", re.MULTILINE)

RETRIEVAL = {
    "multi_object": "Varios objetos en la pregunta",
    "wrong_object": "Objeto equivocado",
    "twin_field": "Campo gemelo en otra caja",
    "grab_bag": "Cajón de sastre (grupo de primer nivel)",
    "wrong_box": "Caja equivocada del objeto correcto",
}
GENERATION = {
    "invented": "Campo inventado",
    "wrong_field": "Elige un campo equivocado del contexto",
    "partial": "Incompleta o con un matiz falso",
    "false_abstention": "Falsa abstención",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def covers(paths, expected):
    return any(p == e or p.startswith(e + ".") for p in paths for e in expected)


def gold_rank(question, obj, expected, chunks):
    import sentence_transformers

    model = sentence_transformers.SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    own = [c for c in chunks if c["object"] == obj]
    vectors = model.encode([c["text"] for c in own], normalize_embeddings=True)
    q = model.encode([question], normalize_embeddings=True)[0]
    ranked = [own[i] for i in (vectors @ q).argsort()[::-1]]
    for rank, c in enumerate(ranked, start=1):
        if covers(FIELD_LINE.findall(c["text"]), expected):
            return rank, len(own), c
    return None, len(own), None


def classify_retrieval(record, question, chunks, objects):
    obj, expected = question["object"], question["relevant_paths"]
    by_id = {c["id"]: c for c in chunks}
    top = [by_id[r["id"]] for r in record["retrieved"]]
    rank, total, gold = gold_rank(question["query"], obj, expected, chunks)
    detail = {"gold_rank_in_object": rank, "object_chunks": total,
              "gold_parent": gold["parent"] if gold else None, "top_objects": [c["object"] for c in top]}
    mentioned = mentioned_objects(question["query"], objects)
    leaves = {e.rsplit(".", 1)[-1] for e in expected}
    twins = [p for c in top if c["object"] == obj for p in FIELD_LINE.findall(c["text"])
             if p.rsplit(".", 1)[-1] in leaves and not covers([p], expected)]
    if len(mentioned) > 1 and obj in mentioned and top[0]["object"] != obj:
        kind = "multi_object"
    elif all(c["object"] != obj for c in top):
        kind = "wrong_object"
    elif twins:
        kind, detail["twins"] = "twin_field", sorted(set(twins))
    elif gold is not None and not gold["parent"]:
        kind = "grab_bag"
    else:
        kind = "wrong_box"
    return kind, detail


def classify_generation(record, verdict):
    if record["hallucinated"]:
        return "invented"
    if verdict == "no_answer" or not record["found"]:
        return "false_abstention"
    return "partial" if verdict == "partial" else "wrong_field"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-e4b")
    args = ap.parse_args()

    questions = {q["id"]: q for q in read_jsonl(ROOT / "data" / "evals" / "test.jsonl")}
    records = {r["id"]: r for r in read_jsonl(ROOT / "out" / "results" / "gen_test.jsonl") if r["model"] == args.model}
    mapping = json.loads((ROOT / "out" / "judge" / "mapping.json").read_text(encoding="utf-8"))
    judgments = {j["item_id"]: j for j in read_jsonl(ROOT / "out" / "judge" / "judgments.jsonl")}
    verdicts = {m["qid"]: judgments[k] for k, m in mapping.items() if m["model"] == args.model}
    chunks = read_jsonl(ROOT / "out" / "chunks.jsonl")
    objects = {c["object"] for c in chunks}

    rows = []
    for qid, q in questions.items():
        r, v = records[qid], verdicts[qid]
        if not q["answerable"] or v["verdict"] == "correct":
            continue
        if r["retrieval_hit"]:
            phase, kind, detail = "generación", classify_generation(r, v["verdict"]), {}
        else:
            phase, (kind, detail) = "búsqueda", classify_retrieval(r, q, chunks, objects)
        rows.append({"id": qid, "query": q["query"], "object": q["object"], "category": q["category"],
                     "phase": phase, "type": kind, "label": {**RETRIEVAL, **GENERATION}[kind],
                     "verdict": v["verdict"], "judge_reason": v["reason"], "answer": r["answer"],
                     "cited": r["field_paths"], "expected": q["relevant_paths"], **detail})

    answerable = sum(q["answerable"] for q in questions.values())
    counts = Counter((r["phase"], r["label"]) for r in rows)
    lines = [f"Modelo: {args.model} · {len(rows)} fallos de {answerable} preguntas con respuesta", "",
             "| Fase | Tipo de error | Fallos |", "|---|---|---:|"]
    for phase in ("búsqueda", "generación"):
        for label in (RETRIEVAL if phase == "búsqueda" else GENERATION).values():
            if counts[(phase, label)]:
                lines.append(f"| {phase} | {label} | {counts[(phase, label)]} |")
    lines += ["", "| Pregunta | Objeto | Fase | Tipo | Puesto del bueno en su objeto |", "|---|---|---|---|---|"]
    for r in rows:
        rank = f"{r['gold_rank_in_object']} de {r['object_chunks']}" if r.get("gold_rank_in_object") else "—"
        lines.append(f"| {r['id']}: {r['query'][:70]} | {r['object']} | {r['phase']} | {r['label']} | {rank} |")
    OUT.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model.split("/")[-1]).strip("_")
    (OUT / f"error_taxonomy_{slug}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = "\n".join(lines)
    (OUT / f"error_taxonomy_{slug}.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
