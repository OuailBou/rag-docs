import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from .common import OUT, load_config


class SemanticRetriever:
    def __init__(
        self,
        index_path=OUT / "rag.faiss",
        metadata_path=OUT / "rag.metadata.jsonl",
        model_name=None,
        query_prefix=None,
    ):
        cfg = load_config()
        model_name = model_name or cfg["embedding"]["model"]
        self.query_prefix = cfg["embedding"]["query_prefix"] if query_prefix is None else query_prefix
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)
        self.metadata = load_metadata(metadata_path)
        self.small_to_big = bool(self.metadata) and "chunk_id" in self.metadata[0]
        if self.small_to_big:
            self.chunks = {c["id"]: c for c in load_metadata(Path(metadata_path).with_name("rag.chunks.jsonl"))}

    def retrieve(self, query, top_k=5):
        query_vector = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        depth = min(self.index.ntotal, max(top_k * 40, 200) if self.small_to_big else top_k)
        scores, positions = self.index.search(query_vector, depth)
        hits = [(float(s), self.metadata[p]) for s, p in zip(scores[0], positions[0]) if p >= 0]
        if not self.small_to_big:
            return [{**row, "score": score} for score, row in hits]
        results, seen = [], set()
        for score, field in hits:
            if field["chunk_id"] not in seen:
                seen.add(field["chunk_id"])
                results.append({**self.chunks[field["chunk_id"]], "score": score, "matched_field": field["parent"]})
            if len(results) == top_k:
                break
        return results


def load_metadata(metadata_path):
    return [json.loads(line) for line in Path(metadata_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def mentioned_objects(query, objects):
    words = " " + " ".join(re.findall(r"[a-z0-9]+", query.lower())) + " "
    found = set()
    for obj in objects:
        phrase = re.escape(obj.replace("_", " ").replace(".", " "))
        if re.search(rf" {phrase}(s|es)? ", words):
            found.add(obj)
    return found


class ObjectFilterRetriever:
    def __init__(self, base, objects, candidates=50):
        self.base = base
        self.objects = sorted(objects)
        self.candidates = candidates

    def retrieve(self, query, top_k=5):
        rows = self.base.retrieve(query, max(self.candidates, top_k))
        found = mentioned_objects(query, self.objects)
        if found:
            rows = [r for r in rows if r["object"] in found] + [r for r in rows if r["object"] not in found]
        return rows[:top_k]


class RerankRetriever:
    def __init__(self, base, model_name=None, candidates=20):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.candidates = candidates
        self.model = CrossEncoder(model_name or load_config()["retrieval"]["reranker_model"])

    def retrieve(self, query, top_k=5):
        rows = self.base.retrieve(query, max(self.candidates, top_k))
        scores = self.model.predict([(query, r["text"]) for r in rows]) if rows else []
        ranked = sorted(zip(scores, rows), key=lambda pair: -pair[0])
        return [{**row, "score": float(score)} for score, row in ranked[:top_k]]


def make_match_query(query):
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("La consulta debe contener al menos una palabra.")
    return " OR ".join(f'"{term}"' for term in terms)


class LexicalRetriever:
    def __init__(self, lexical_index_path=OUT / "rag.sqlite3"):
        self.lexical_index_path = Path(lexical_index_path)

    def retrieve(self, query, top_k=5):
        connection = sqlite3.connect(self.lexical_index_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT chunks.id, chunks.object, chunks.parent, chunks.text, chunks_fts.rank
                FROM chunks_fts
                JOIN chunks ON chunks.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY chunks_fts.rank
                LIMIT ?
                """,
                (make_match_query(query), top_k),
            ).fetchall()
        finally:
            connection.close()
        return [{**dict(row), "score": float(-row["rank"])} for row in rows]


class HybridRetriever:
    def __init__(self, semantic, lexical, rrf_k=60):
        self.semantic = semantic
        self.lexical = lexical
        self.rrf_k = rrf_k

    def retrieve(self, query, top_k=5):
        candidate_k = max(top_k * 4, 20)
        semantic_rows = self.semantic.retrieve(query, candidate_k)
        lexical_rows = self.lexical.retrieve(query, candidate_k)
        by_id = {row["id"]: row for row in semantic_rows + lexical_rows}
        scores = {}
        for rank, row in enumerate(semantic_rows, start=1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1 / (self.rrf_k + rank)
        for rank, row in enumerate(lexical_rows, start=1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1 / (self.rrf_k + rank)
        results = []
        for chunk_id, score in scores.items():
            result = dict(by_id[chunk_id])
            result["score"] = score
            result["retrieval_method"] = "hybrid"
            results.append(result)
        return sorted(results, key=lambda row: row["score"], reverse=True)[:top_k]


def retrieve(
    query,
    top_k=5,
    index_path=OUT / "rag.faiss",
    metadata_path=OUT / "rag.metadata.jsonl",
    lexical_index_path=OUT / "rag.sqlite3",
    model_name=None,
    method="semantic",
    object_filter=False,
    rerank=False,
):
    return build_retriever(
        method, index_path, metadata_path, lexical_index_path, model_name, object_filter, rerank
    ).retrieve(query, top_k)


def build_retriever(
    method="semantic",
    index_path=OUT / "rag.faiss",
    metadata_path=OUT / "rag.metadata.jsonl",
    lexical_index_path=OUT / "rag.sqlite3",
    model_name=None,
    object_filter=False,
    rerank=False,
):
    semantic = SemanticRetriever(index_path, metadata_path, model_name) if method != "lexical" else None
    lexical = LexicalRetriever(lexical_index_path) if method != "semantic" else None
    if method == "semantic":
        retriever = semantic
    elif method == "lexical":
        retriever = lexical
    elif method == "hybrid":
        retriever = HybridRetriever(semantic, lexical)
    else:
        raise ValueError(f"Método desconocido: {method}")
    if rerank:
        retriever = RerankRetriever(retriever)
    if object_filter:
        retriever = ObjectFilterRetriever(retriever, {m["object"] for m in load_metadata(metadata_path)})
    return retriever


METHODS = ("semantic", "lexical", "hybrid")


def add_retrieval_args(parser, cfg, top_k=True):
    """Opciones de búsqueda comunes a todos los comandos; los valores por defecto salen de la configuración."""
    defaults = cfg["retrieval"]
    parser.add_argument("--method", choices=METHODS, default=defaults["method"])
    parser.add_argument("--object-filter", action=argparse.BooleanOptionalAction, default=defaults["object_filter"],
                        help="Prioriza los objetos que nombra la pregunta")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=defaults["rerank"],
                        help="Reordena los candidatos con un cross-encoder")
    if top_k:
        parser.add_argument("--top-k", type=int, default=defaults["top_k"], help="Fragmentos que se recuperan")
    parser.add_argument("--index", default=str(OUT / "rag.faiss"))
    parser.add_argument("--metadata", default=str(OUT / "rag.metadata.jsonl"))
    parser.add_argument("--lexical-index", default=str(OUT / "rag.sqlite3"))


def retriever_from_args(args):
    return build_retriever(args.method, args.index, args.metadata, args.lexical_index,
                           object_filter=args.object_filter, rerank=args.rerank)


def describe(args):
    return args.method + ("+objfilter" if args.object_filter else "") + ("+rerank" if args.rerank else "")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Busca en el índice local.")
    parser.add_argument("query", nargs="+", help="Pregunta o términos de búsqueda")
    add_retrieval_args(parser, load_config())
    args = parser.parse_args()
    if not Path(args.index).exists():
        parser.error(f"No existe el índice: {args.index}. Ejecuta primero: uv run python -m rag.index")
    results = retriever_from_args(args).retrieve(" ".join(args.query), max(1, args.top_k))
    if not results:
        print("Sin resultados.")
        return
    for result in results:
        print(f"[{result['id']}] {result['object']} / {result['parent']}  score={result['score']:.4f}")
        print(result["text"])
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
