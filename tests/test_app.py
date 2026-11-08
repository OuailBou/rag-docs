import sys
import types

import rag
from rag import app

CHUNKS = [
    {"id": 3, "object": "charge", "parent": "", "text": "- `currency` (string)\n  Three-letter ISO code.", "score": 0.91},
    {"id": 7, "object": "charge", "parent": "billing_details", "text": "- `billing_details.address.city`", "score": 0.5},
]
RESULT = {
    "found": True, "answer": "A charge uses `currency` [1].", "field_paths": ["currency"],
    "cited_chunks": [1], "latency_ms": 1234, "model": "qwen2.5-coder:7b",
}


class FakeRetriever:
    def retrieve(self, query, top_k):
        return CHUNKS[:top_k]


def install(monkeypatch, answer):
    fake = types.SimpleNamespace(
        answer=answer,
        known_paths=lambda rows=None: {"charge": {"currency"}},
        verify=lambda result, chunks, known: {"field_paths": result["field_paths"], "invented_paths": [],
                                              "unsupported_paths": [], "invalid_citations": [], "grounded": True},
    )
    monkeypatch.setitem(sys.modules, "rag.generate", fake)
    monkeypatch.setattr(rag, "generate", fake, raising=False)
    monkeypatch.setattr(app, "_cache", {"base": FakeRetriever(), "filtered": FakeRetriever()})


def test_format_answer_grounded():
    text = app.format_answer(RESULT, {"grounded": True})
    assert "A charge uses `currency` [1]." in text
    assert "fundamentada" in text and "no fundamentada" not in text
    assert "`currency`" in text and "1234 ms" in text and "qwen2.5-coder:7b" in text


def test_format_answer_not_found_and_hallucinations():
    verification = {"grounded": False, "invented_paths": ["foo.bar"], "unsupported_paths": ["amount"],
                    "invalid_citations": [9]}
    text = app.format_answer({**RESULT, "found": False, "field_paths": []}, verification)
    assert "No encontrado" in text
    assert "campos inventados: `foo.bar`" in text
    assert "campos no respaldados por los fragmentos: `amount`" in text
    assert "[9]" in text and "**Campos citados:** ninguno" in text


def test_format_sources_marks_cited():
    text = app.format_sources(CHUNKS, [1])
    assert "[1] charge / (nivel superior) (0.910) — citado" in text
    assert "[2] charge / billing_details (0.500)" in text
    assert text.count("citado") == 1
    assert "<details open>" in text and "billing_details.address.city" in text
    assert "Ningún fragmento" in app.format_sources([])


def test_ask_success(monkeypatch):
    install(monkeypatch, lambda question, chunks, cfg=None, client=None: RESULT)
    answer, sources = app.ask("What currency does a charge use?", True, 2)
    assert "fundamentada" in answer and "— citado" in sources and "[2]" in sources


def test_ask_ollama_down_still_shows_sources(monkeypatch):
    def down(question, chunks, cfg=None, client=None):
        raise RuntimeError("No se puede conectar con Ollama.")

    install(monkeypatch, down)
    answer, sources = app.ask("What currency does a charge use?", False, 2)
    assert "No se puede conectar con Ollama." in answer
    assert "[1] charge" in sources and "citado" not in sources


def test_ask_empty_question():
    assert app.ask("   ")[1] == ""


def test_format_answer_parse_error_and_normalized_paths():
    text = app.format_answer({**RESULT, "found": False, "parse_error": True}, {"grounded": False, "field_paths": []})
    assert "JSON válido" in text
    text = app.format_answer({**RESULT, "field_paths": ["charge.currency"]}, {"grounded": True, "field_paths": ["currency"]})
    assert "**Campos citados:** `currency`" in text
