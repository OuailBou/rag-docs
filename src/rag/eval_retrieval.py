import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from .common import OUT, field_paths, load_config
from .retrieve import add_retrieval_args, describe, retriever_from_args

CUTOFFS = (1, 3, 5)
NON_RETRIEVAL = {"metadata"}
def expected_object(item):
    return item["object"]


def chunk_paths(result, obj):
    if result.get("object", obj) != obj:
        return set()
    return field_paths(result["text"])


def matched(expected, paths):
    return sorted(e for e in expected if any(p == e or p.startswith(e + ".") for p in paths))


def evaluate_question(item, results, cutoffs=CUTOFFS):
    obj = expected_object(item)
    evaluation = {
        "id": item["id"],
        "query": item["query"],
        "object": obj,
        "category": item["category"],
        "difficulty": item["difficulty"],
        "answerable": bool(item["answerable"]),
        "retrieval": item["category"] not in NON_RETRIEVAL,
        "retrieved": [{"id": r["id"], "object": r.get("object"), "parent": r.get("parent")} for r in results],
    }
    if not item["answerable"]:
        return evaluation

    expected = set(item["relevant_paths"])
    matches = [matched(expected, chunk_paths(r, obj)) for r in results]
    evaluation["expected_paths"] = sorted(expected)
    evaluation["matched_paths"] = {str(rank): m for rank, m in enumerate(matches, start=1)}
    first = next((rank for rank, m in enumerate(matches, start=1) if m), None)

    for cutoff in cutoffs:
        found = set().union(*matches[:cutoff]) if matches[:cutoff] else set()
        relevant_ranks = [rank for rank, m in enumerate(matches[:cutoff], start=1) if m]
        evaluation[f"recall_at_{cutoff}"] = len(found) / len(expected) if expected else 0.0
        ideal = min(len(expected), cutoff)
        dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
        evaluation[f"ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0
    evaluation["mrr"] = 1 / first if first else 0.0
    return evaluation


def mean(values):
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values, n=5000, seed=0):
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(values) for _ in values]) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def summarize(evaluations, cutoffs=CUTOFFS):
    scored = [e for e in evaluations if e["answerable"] and e["retrieval"]]
    metrics = [f"recall_at_{c}" for c in cutoffs] + [f"ndcg_at_{c}" for c in cutoffs] + ["mrr"]
    summary = {
        "questions": len(evaluations),
        "scored_questions": len(scored),
        "unanswerable_questions": sum(not e["answerable"] for e in evaluations),
        "metadata_questions": sum(not e["retrieval"] for e in evaluations),
        "overall": {m: mean([e[m] for e in scored]) for m in metrics},
        "ci95_recall_at_5": bootstrap_ci([e["recall_at_5"] for e in scored]),
        "by_category": {},
    }
    groups = defaultdict(list)
    for e in scored:
        groups[e["category"]].append(e)
    for category, items in sorted(groups.items()):
        summary["by_category"][category] = {
            "questions": len(items),
            **{f"recall_at_{c}": mean([e[f"recall_at_{c}"] for e in items]) for c in cutoffs},
            "mrr": mean([e["mrr"] for e in items]),
        }
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evalúa la búsqueda con un conjunto de preguntas.")
    parser.add_argument("--dataset", default="data/evals/dev_charge.jsonl")
    add_retrieval_args(parser, load_config(), top_k=False)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    items = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = retriever_from_args(args)
    evaluations = [evaluate_question(item, retriever.retrieve(item["query"], top_k=max(CUTOFFS))) for item in items]
    method = describe(args)
    report = {"method": method, "dataset": args.dataset, "summary": summarize(evaluations), "questions": evaluations}
    output = Path(args.output or OUT / "results" / f"{Path(args.dataset).stem}_{method}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"Método: {method} | {Path(args.dataset).name}")
    print(f"Preguntas evaluadas: {s['scored_questions']} (sin respuesta: {s['unanswerable_questions']}, "
          f"metadata: {s['metadata_questions']})")
    for metric, value in s["overall"].items():
        print(f"{metric}: {value:.3f}")
    lo, hi = s["ci95_recall_at_5"]
    print(f"recall_at_5 intervalo 95%: [{lo:.3f}, {hi:.3f}]")
    print(f"Resultados: {output}")


if __name__ == "__main__":
    main()
