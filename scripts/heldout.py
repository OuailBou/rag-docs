"""Evalúa la búsqueda en los conjuntos reservados (control, test, test2) y escribe sus tablas.

Uso:
    uv run python scripts/heldout.py              # los tres conjuntos
    uv run python scripts/heldout.py test2        # solo ese

Las ejecuciones y comparaciones están en la sección "heldout" de configs/experiments.yaml.
Si el índice de una variante no existe, se construye antes con scripts/experiment.py.
"""
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import EXP_OUT, ROOT, ci, run, run_variant


def variant_env(name):
    out = EXP_OUT / name
    env = {**os.environ, "RAG_OUT": str(out), "RAG_CONFIG": str(out / "config.yaml"),
           "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
    if sys.platform == "darwin":
        env.update({"OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    return env


def evaluate(dataset, run_id, plan, base_cfg):
    name, method = run_id.split(":")
    if not (EXP_OUT / name / "rag.faiss").exists():
        run_variant(name, plan["variants"][name], base_cfg)
    base_method, *flags = method.split("+")
    run("rag.eval_retrieval", variant_env(name), "--dataset", f"data/evals/{dataset}.jsonl",
        "--method", base_method, "--object-filter" if "objfilter" in flags else "--no-object-filter",
        "--rerank" if "rerank" in flags else "--no-rerank")
    report = json.loads((EXP_OUT / name / "results" / f"{dataset}_{method}.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in report["questions"] if q["answerable"] and q["retrieval"]}


def mean(values):
    return sum(values) / len(values)


def report_for(dataset, spec, plan, base_cfg):
    results = {r: evaluate(dataset, r, plan, base_cfg) for r in spec["runs"]}
    lines = [f"## {dataset}", "", "| Ejecución | n | R@1 | R@5 | R@5 IC95% | MRR |", "|---|---:|---:|---:|---|---:|"]
    for run_id, qs in results.items():
        r5 = [q["recall_at_5"] for q in qs.values()]
        lo, hi = ci(r5)
        lines.append(f"| {run_id} | {len(qs)} | {mean([q['recall_at_1'] for q in qs.values()]):.3f} | "
                     f"{mean(r5):.3f} | [{lo:.2f}, {hi:.2f}] | {mean([q['mrr'] for q in qs.values()]):.3f} |")
    lines += ["", "| Comparación | ΔR@5 | IC95% | ΔMRR | IC95% |", "|---|---:|---|---:|---|"]
    for a, b in spec.get("compare", []):
        ids = sorted(set(results[a]) & set(results[b]))
        cells = []
        for metric in ("recall_at_5", "mrr"):
            d = [results[a][i][metric] - results[b][i][metric] for i in ids]
            lo, hi = ci(d)
            cells.append(f"{mean(d):+.3f} | [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"| {a} − {b} | {' | '.join(cells)} |")
    final = results[spec["final"]]
    lines += ["", f"| {spec['final']} por categoría | n | R@5 | MRR |", "|---|---:|---:|---:|"]
    for cat in sorted({q["category"] for q in final.values()}):
        qs = [q for q in final.values() if q["category"] == cat]
        lines.append(f"| {cat} | {len(qs)} | {mean([q['recall_at_5'] for q in qs]):.3f} | "
                     f"{mean([q['mrr'] for q in qs]):.3f} |")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plan = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    for dataset in sys.argv[1:] or list(plan["heldout"]):
        report = report_for(dataset, plan["heldout"][dataset], plan, base_cfg)
        (EXP_OUT / f"heldout_{dataset}.md").write_text(report + "\n", encoding="utf-8")
        print(report + "\n")


if __name__ == "__main__":
    main()
