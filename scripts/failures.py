"""Lista las preguntas falladas de una variante y qué se recuperó en su lugar.

Uso:
    uv run python scripts/failures.py E1_sin_copias semantic
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    variant, method = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "semantic"
    results = ROOT / "out" / "experiments" / variant / "results"
    for f in sorted(results.glob(f"*_{method}.json")):
        for q in json.loads(f.read_text(encoding="utf-8"))["questions"]:
            if not (q["answerable"] and q["retrieval"]) or q["recall_at_5"] == 1.0:
                continue
            print(f"\n[{q['id']}] R@5={q['recall_at_5']:.2f}  {q['query']}")
            print(f"   esperado: {q['object']} -> {', '.join(q['expected_paths'][:4])}")
            for rank, r in enumerate(q["retrieved"], start=1):
                hit = "OK" if q["matched_paths"].get(str(rank)) else "  "
                print(f"   {hit} {rank}. {r['object']:18} {r['parent'] or '(primer nivel)'}")


if __name__ == "__main__":
    main()
