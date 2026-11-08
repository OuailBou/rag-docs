import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .common import OUT, load_config
from .eval_retrieval import NON_RETRIEVAL, bootstrap_ci, evaluate_question, expected_object, mean
from .retrieve import add_retrieval_args, describe, retriever_from_args


def load_items(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def scored_items(items):
    return [item for item in items if item["category"] not in NON_RETRIEVAL]


def path_matches(expected, paths):
    return sorted(e for e in expected if any(p == e or p.startswith(e + ".") for p in paths))


def evaluate_item(item, chunks, result, check, top_k):
    paths = check.get("field_paths", [])
    record = {
        "id": item["id"],
        "query": item["query"],
        "object": expected_object(item),
        "category": item["category"],
        "difficulty": item["difficulty"],
        "answerable": bool(item["answerable"]),
        "model": result.get("model"),
        "found": bool(result.get("found")),
        "grounded": bool(check.get("grounded")),
        "hallucinated": bool(check.get("invented_paths")),
        "unsupported": bool(check.get("unsupported_paths")),
        "parse_error": bool(result.get("parse_error")),
        "latency_ms": result.get("latency_ms", 0),
        "answer": result.get("answer", ""),
        "field_paths": paths,
        "raw_field_paths": result.get("field_paths", []),
        "cited_chunks": result.get("cited_chunks", []),
        "invented_paths": check.get("invented_paths", []),
        "unsupported_paths": check.get("unsupported_paths", []),
        "invalid_citations": check.get("invalid_citations", []),
        "retrieved": [{"id": c["id"], "object": c.get("object"), "parent": c.get("parent")} for c in chunks],
    }
    if not item["answerable"]:
        record["abstained"] = not record["found"]
        return record
    expected = set(item["relevant_paths"])
    matches = path_matches(expected, paths) if record["found"] else []
    record.update({
        "expected_paths": sorted(expected),
        "matched_paths": matches,
        "retrieval_hit": evaluate_question(item, chunks, (top_k,))[f"recall_at_{top_k}"] > 0,
        "path_hit": bool(matches),
        "path_recall": len(matches) / len(expected) if expected else 0.0,
    })
    return record


def evaluate_items(items, retriever, answer_fn, verify_fn, known, top_k, cfg=None, done=None, on_result=None):
    done = done or set()
    records = []
    for item in scored_items(items):
        if item["id"] in done:
            continue
        chunks = retriever.retrieve(item["query"], top_k)
        result = answer_fn(item["query"], chunks, cfg=cfg)
        record = evaluate_item(item, chunks, result, verify_fn(result, chunks, known), top_k)
        records.append(record)
        if on_result:
            on_result(record)
    return records


def rate(records, key):
    values = [float(bool(r[key])) for r in records]
    return {"rate": mean(values), "ci95": bootstrap_ci(values), "count": int(sum(values)), "questions": len(values)}


def summarize(records):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    retrieved = [r for r in answerable if r["retrieval_hit"]]
    found = [r for r in records if r["found"]]
    latencies = [r["latency_ms"] for r in records]
    misses = {
        "retrieval_miss": [not r["retrieval_hit"] for r in answerable],
        "generation_miss": [r["retrieval_hit"] and not r["path_hit"] for r in answerable],
    }
    summary = {
        "questions": len(records),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "retrieval_hit_rate": rate(answerable, "retrieval_hit"),
        "path_accuracy": rate(answerable, "path_hit"),
        "path_accuracy_when_retrieved": rate(retrieved, "path_hit"),
        "path_recall": mean([r["path_recall"] for r in answerable]),
        "errors": {name: {"count": sum(v), "rate": mean([float(x) for x in v])} for name, v in misses.items()},
        "false_abstention_rate": mean([float(not r["found"]) for r in answerable]),
        "abstention_rate": rate(unanswerable, "abstained"),
        "hallucination_rate": rate(records, "hallucinated"),
        "unsupported_rate": rate(records, "unsupported"),
        "grounded_rate": rate(found, "grounded"),
        "parse_error_rate": rate(records, "parse_error"),
        "latency_ms": {"mean": mean(latencies), "median": statistics.median(latencies) if latencies else 0.0},
        "by_category": {},
    }
    groups = defaultdict(list)
    for r in answerable:
        groups[r["category"]].append(r)
    for category, items in sorted(groups.items()):
        hits = [r for r in items if r["retrieval_hit"]]
        summary["by_category"][category] = {
            "questions": len(items),
            "path_accuracy": mean([float(r["path_hit"]) for r in items]),
            "path_accuracy_when_retrieved": mean([float(r["path_hit"]) for r in hits]),
            "retrieval_hit_rate": mean([float(r["retrieval_hit"]) for r in items]),
        }
    return summary


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def same_run(record, run):
    return all(record.get(k) == v for k, v in run.items())


def appender(path, run):
    path.parent.mkdir(parents=True, exist_ok=True)
    broken = path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n")
    handle = open(path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115 (se cierra con "with" en run_dataset)
    if broken:
        handle.write("\n")

    def write(record):
        handle.write(json.dumps({**record, **run}, ensure_ascii=False) + "\n")
        handle.flush()

    return handle, write


def run_dataset(dataset, items, retriever, generate, known, cfg, run, output_dir):
    stem = Path(dataset).stem
    output = output_dir / f"gen_{stem}.jsonl"
    done = {r["id"] for r in read_records(output) if same_run(r, run)}
    handle, write = appender(output, run)
    with handle:
        new = evaluate_items(items, retriever, generate.answer, generate.verify, known, run["top_k"], cfg, done,
                             write)
    ids = {item["id"] for item in scored_items(items)}
    latest = {r["id"]: r for r in read_records(output) if same_run(r, run) and r["id"] in ids}
    records = [latest[i] for i in sorted(latest)]
    summary = {"dataset": str(dataset), **run, "new_questions": len(new), "skipped_questions": len(done & ids),
               "summary": summarize(records)}
    summary_path = output_dir / f"gen_{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, output, summary_path)
    return summary


def print_summary(report, output, summary_path):
    s = report["summary"]
    print(f"Modelo: {report['model']} | Recuperación: {report['retrieval']} (top_k={report['top_k']}) | "
          f"{Path(report['dataset']).name}")
    print(f"Preguntas: {s['questions']} (con respuesta: {s['answerable_questions']}, "
          f"sin respuesta: {s['unanswerable_questions']}; nuevas: {report['new_questions']}, "
          f"reanudadas: {report['skipped_questions']})")
    for label, key in (("Acierto de campo", "path_accuracy"),
                       ("Acierto de campo con recuperación correcta", "path_accuracy_when_retrieved"),
                       ("Abstención en preguntas sin respuesta", "abstention_rate"),
                       ("Alucinación (campos inventados)", "hallucination_rate"),
                       ("Campos no presentes en el contexto", "unsupported_rate"),
                       ("Respuestas fundamentadas", "grounded_rate"),
                       ("Errores de formato", "parse_error_rate")):
        lo, hi = s[key]["ci95"]
        print(f"{label}: {s[key]['rate']:.3f} [{lo:.3f}, {hi:.3f}] (n={s[key]['questions']})")
    e = s["errors"]
    print(f"Fallos de recuperación: {e['retrieval_miss']['count']} | "
          f"fallos de generación: {e['generation_miss']['count']}")
    print(f"Latencia media: {s['latency_ms']['mean']:.0f} ms | mediana: {s['latency_ms']['median']:.0f} ms")
    print(f"Resultados: {output}\nResumen: {summary_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evalúa las respuestas del modelo con un conjunto de preguntas.")
    parser.add_argument("--dataset", action="append", default=None, help="Se puede repetir")
    parser.add_argument("--limit", type=int, default=None)
    add_retrieval_args(parser, cfg)
    parser.add_argument("--output-dir", default=str(OUT / "results"))
    args = parser.parse_args()

    from . import generate

    datasets = args.dataset or ["data/evals/dev_charge.jsonl"]
    run = {"model": cfg["llm"]["model"], "retrieval": describe(args), "top_k": args.top_k}
    retriever = retriever_from_args(args)
    known = generate.known_paths()
    try:
        for dataset in datasets:
            items = scored_items(load_items(dataset))[:args.limit]
            run_dataset(dataset, items, retriever, generate, known, cfg, run, Path(args.output_dir))
    except RuntimeError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
