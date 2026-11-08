import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .common import OUT, load_config


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_index(
    chunks_path=OUT / "chunks.jsonl",
    index_path=OUT / "rag.faiss",
    metadata_path=OUT / "rag.metadata.jsonl",
    lexical_index_path=OUT / "rag.sqlite3",
    model_name=None,
    fields_path=None,
):
    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    lexical_index_path = Path(lexical_index_path)
    model_name = model_name or load_config()["embedding"]["model"]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    searched = rows
    if fields_path:
        searched = [json.loads(line) for line in Path(fields_path).read_text(encoding="utf-8").splitlines()]
        write_rows(metadata_path.with_name("rag.chunks.jsonl"), rows)
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [row["text"] for row in searched],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    write_rows(metadata_path, searched)
    lexical_index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(lexical_index_path)
    try:
        connection.executescript(
            """
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS chunks_fts;
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                object TEXT NOT NULL,
                parent TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id'
            );
            """
        )
        connection.executemany(
            "INSERT INTO chunks (id, object, parent, text) VALUES (?, ?, ?, ?)",
            [(row["id"], row["object"], row["parent"], row["text"]) for row in rows],
        )
        connection.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks")
        connection.commit()
    finally:
        connection.close()
    return len(rows), vectors.shape[1]


def main():
    parser = argparse.ArgumentParser(description="Build the local FAISS embedding index from generated chunks.")
    parser.add_argument("--chunks", default=str(OUT / "chunks.jsonl"))
    parser.add_argument("--index", default=str(OUT / "rag.faiss"))
    parser.add_argument("--metadata", default=str(OUT / "rag.metadata.jsonl"))
    parser.add_argument("--lexical-index", default=str(OUT / "rag.sqlite3"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    fields = OUT / "fields.jsonl" if load_config()["chunker"]["small_to_big"] else None
    count, dimensions = build_index(args.chunks, args.index, args.metadata, args.lexical_index, args.model, fields)
    searched = "fragmentos de campo (small-to-big)" if fields else "chunks"
    print(f"{count} chunks; se buscan {searched} -> {args.index} ({dimensions} dimensiones) + {args.lexical_index}")


if __name__ == "__main__":
    main()
