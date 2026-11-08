import json

from rag.index import build_index
from rag.retrieve import retrieve


def test_build_and_retrieve(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "rag.faiss"
    metadata_path = tmp_path / "rag.metadata.jsonl"
    lexical_index_path = tmp_path / "rag.sqlite3"
    chunks_path.write_text(
        json.dumps({"id": 0, "object": "charge", "parent": "", "text": "A charge can be refunded."}) + "\n"
        + json.dumps({"id": 1, "object": "customer", "parent": "", "text": "A customer has an email address."}) + "\n",
        encoding="utf-8",
    )

    assert build_index(chunks_path, index_path, metadata_path, lexical_index_path)[0] == 2
    results = retrieve(
        "refunded charge",
        index_path=index_path,
        metadata_path=metadata_path,
        lexical_index_path=lexical_index_path,
    )

    assert results[0]["id"] == 0
    assert results[0]["object"] == "charge"


def test_small_to_big_searches_fields_and_returns_their_chunk(tmp_path):
    from rag.chunk import field_chunks
    from rag.retrieve import SemanticRetriever

    chunks = [
        {"id": 0, "object": "invoice", "parent": "",
         "text": "- `customer_phone` (string)\n  Phone.\n\n- `due_date` (integer)\n  The date on which payment is due."},
        {"id": 1, "object": "invoice", "parent": "",
         "text": "- `status` (enum)\n  The status of the invoice.\n\n- `total` (integer)\n  Total after discounts."},
    ]
    rows = [{"object": "invoice", "title": "Invoice", "path": p, "depth": 0, "type": t, "nullable": False,
             "expandable": False, "enum": "", "description": d}
            for p, t, d in [("customer_phone", "string", "Phone."), ("due_date", "integer", "The date on which payment is due."),
                            ("status", "enum", "The status of the invoice."), ("total", "integer", "Total after discounts.")]]
    fields = field_chunks(rows, chunks)
    assert [f["chunk_id"] for f in fields] == [0, 0, 1, 1]

    chunks_path, fields_path = tmp_path / "chunks.jsonl", tmp_path / "fields.jsonl"
    chunks_path.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    fields_path.write_text("".join(json.dumps(f) + "\n" for f in fields), encoding="utf-8")
    build_index(chunks_path, tmp_path / "rag.faiss", tmp_path / "rag.metadata.jsonl", tmp_path / "rag.sqlite3",
                fields_path=fields_path)

    results = SemanticRetriever(tmp_path / "rag.faiss", tmp_path / "rag.metadata.jsonl").retrieve("When is the invoice due?", 2)

    assert [r["id"] for r in results] == [0, 1]
    assert results[0]["matched_field"] == "due_date"
    assert "customer_phone" in results[0]["text"]
