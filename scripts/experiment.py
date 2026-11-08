"""Ejecuta las variantes de configs/experiments.yaml y compara sus resultados.

Uso:
    uv run python scripts/experiment.py            # todas las variantes
    uv run python scripts/experiment.py E1 E2      # solo esas
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_OUT = ROOT / "out" / "experiments"
DATASETS = ["data/evals/dev_charge.jsonl", "data/evals/dev_objects.jsonl"]


def merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def run(module, env, *args):
    proc = subprocess.run([sys.executable, "-W", "ignore", "-m", module, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=False, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"Fallo en {module}:\n{proc.stderr[-2000:]}")
    return proc.stdout


def index_settings(cfg):
    """Solo lo que cambia el índice: si nada de esto cambia, no se vuelve a indexar."""
    return {"data": cfg["data"], "extract": cfg["extract"], "chunker": cfg["chunker"],
            "embedding_model": cfg["embedding"]["model"]}


def run_variant(name, spec, base_cfg):
    out = EXP_OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = merge(base_cfg, spec.get("config"))
    env = {**os.environ, "RAG_OUT": str(out), "RAG_CONFIG": str(out / "config.yaml"),
           "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
    if sys.platform == "darwin":  # faiss y torch chocan en macOS sin esto
        env.update({"OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    previous = out / "config.yaml"
    old = yaml.safe_load(previous.read_text(encoding="utf-8")) if previous.exists() else None
    built = bool(old) and "data" in old and (out / "rag.faiss").exists() and index_settings(old) == index_settings(cfg)
    previous.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    if not built:
        print(f"[{name}] extract -> chunk -> index")
        run("rag.extract", env)
        print("   " + run("rag.chunk", env).strip().replace("\n", "\n   "))
        run("rag.index", env)
    for method in spec.get("methods", ["semantic"]):
        base_method, *flags = method.split("+")
        flag_args = [("--object-filter" if "objfilter" in flags else "--no-object-filter"),
                     ("--rerank" if "rerank" in flags else "--no-rerank")]
        for ds in DATASETS:
            args = ["--dataset", ds, "--method", base_method] + flag_args
            run("rag.eval_retrieval", env, *args)
    return out


def scores(name, method):
    vals = {}
    for ds in DATASETS:
        f = EXP_OUT / name / "results" / f"{Path(ds).stem}_{method}.json"
        for q in json.loads(f.read_text(encoding="utf-8"))["questions"]:
            if q["answerable"] and q["retrieval"]:
                vals[q["id"]] = (q["recall_at_5"], q["mrr"], q["recall_at_1"])
    return vals


def ci(values, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plan = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    wanted = sys.argv[1:] or list(plan["variants"])
    for name in wanted:
        run_variant(name, plan["variants"][name], base_cfg)

    lines = ["| Variante | Método | n | R@1 | R@5 | R@5 IC95% | MRR |", "|---|---|---:|---:|---:|---|---:|"]
    table = {}
    for name in plan["variants"]:
        if not (EXP_OUT / name / "results").exists():
            continue
        for method in plan["variants"][name].get("methods", ["semantic"]):
            s = scores(name, method)
            table[(name, method)] = s
            r5 = [v[0] for v in s.values()]
            lo, hi = ci(r5)
            lines.append(f"| {name} | {method} | {len(s)} | {sum(v[2] for v in s.values()) / len(s):.3f} | "
                         f"{sum(r5) / len(r5):.3f} | [{lo:.2f}, {hi:.2f}] | {sum(v[1] for v in s.values()) / len(s):.3f} |")
    lines += ["", "| Comparación | ΔR@5 | IC95% | ΔMRR | IC95% |", "|---|---:|---|---:|---|"]
    for a, b in plan.get("compare", []):
        sa, sb = table.get(tuple(a.split(":"))), table.get(tuple(b.split(":")))
        if not sa or not sb:
            continue
        ids = sorted(set(sa) & set(sb))
        cells = []
        for k in (0, 1):
            d = [sa[i][k] - sb[i][k] for i in ids]
            lo, hi = ci(d)
            cells.append(f"{sum(d) / len(d):+.3f} | [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"| {a} − {b} | {' | '.join(cells)} |")
    report = "\n".join(lines)
    (EXP_OUT / "summary.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
