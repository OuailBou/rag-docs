import json
import sys
import types

import pytest

import rag
from rag import eval_generation
from rag.eval_generation import evaluate_items, read_records, run_dataset, summarize

CHUNKS = [
    {"id": 1, "object": "charge", "parent": "", "text": "- `amount` (integer)"},
    {"id": 2, "object": "charge", "parent": "outcome", "text": "- `outcome.type` (string)\n- `outcome.reason` (string)"},
]


def item(qid, paths, category="direct_field", answerable=True):
    return {"id": qid, "query": f"question {qid}", "object": "charge", "category": category, "difficulty": "easy",
            "answerable": answerable, "relevant_paths": paths}


class FakeRetriever:
    def __init__(self, chunks=CHUNKS):
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append(query)
        return self.chunks[:top_k]


def fake_answer(answers):
    def answer(question, chunks, cfg=None, client=None):
        found, paths = answers[question]
        return {"found": found, "answer": "x", "field_paths": paths, "cited_chunks": [1], "latency_ms": 100,
                "model": "fake"}

    return answer


def fake_verify(result, chunks, known):
    paths = result["field_paths"]
    invented = [p for p in paths if p.startswith("bogus")]
    return {"field_paths": paths, "invented_paths": invented, "unsupported_paths": [], "invalid_citations": [],
            "grounded": not invented}


def run(items, answers, retriever=None, **kwargs):
    return evaluate_items(items, retriever or FakeRetriever(), fake_answer(answers), fake_verify, {}, 5, **kwargs)


def test_path_hit_rules():
    items = [item("exact", ["outcome.type"]), item("child", ["outcome"]), item("ancestor", ["outcome.type"])]
    answers = {"question exact": (True, ["outcome.type"]), "question child": (True, ["outcome.reason"]),
               "question ancestor": (True, ["outcome"])}

    records = {r["id"]: r for r in run(items, answers)}

    assert records["exact"]["path_hit"]
    assert records["child"]["path_hit"]
    assert not records["ancestor"]["path_hit"]


def test_retrieval_and_generation_misses():
    items = [item("ok", ["amount"]), item("gen", ["amount"]), item("ret", ["customer"])]
    answers = {"question ok": (True, ["amount"]), "question gen": (True, ["currency"]),
               "question ret": (True, ["customer"])}

    records = {r["id"]: r for r in run(items, answers)}
    summary = summarize(list(records.values()))

    assert records["ok"]["retrieval_hit"] and records["gen"]["retrieval_hit"]
    assert not records["ret"]["retrieval_hit"] and records["ret"]["path_hit"]
    assert summary["errors"]["retrieval_miss"]["count"] == 1
    assert summary["errors"]["generation_miss"]["count"] == 1
    assert summary["path_accuracy"]["rate"] == pytest.approx(2 / 3)
    assert summary["path_accuracy_when_retrieved"]["rate"] == 0.5
    assert summary["path_accuracy_when_retrieved"]["questions"] == 2


def test_unanswerable_and_metadata():
    items = [item("u1", [], "unanswerable", False), item("u2", [], "unanswerable", False),
             item("meta", ["amount"], "metadata"), item("a", ["amount"])]
    answers = {"question u1": (False, []), "question u2": (True, ["bogus_field"]), "question a": (True, ["amount"])}
    retriever = FakeRetriever()

    records = {r["id"]: r for r in run(items, answers, retriever)}
    summary = summarize(list(records.values()))

    assert "meta" not in records and "question meta" not in retriever.queries
    assert records["u1"]["abstained"] and not records["u2"]["abstained"]
    assert records["u2"]["hallucinated"] and "path_hit" not in records["u2"]
    assert summary["questions"] == 3
    assert summary["unanswerable_questions"] == 2
    assert summary["abstention_rate"]["rate"] == 0.5
    assert summary["hallucination_rate"]["rate"] == pytest.approx(1 / 3)
    assert summary["grounded_rate"]["rate"] == 0.5
    assert summary["path_accuracy"]["rate"] == 1.0
    assert summary["latency_ms"] == {"mean": 100, "median": 100}
    assert summary["by_category"] == {"direct_field": {"questions": 1, "path_accuracy": 1.0,
                                                       "path_accuracy_when_retrieved": 1.0,
                                                       "retrieval_hit_rate": 1.0}}


def test_done_ids_are_skipped_and_results_streamed():
    items = [item("a", ["amount"]), item("b", ["amount"])]
    answers = {"question a": (True, ["amount"]), "question b": (True, ["amount"])}
    streamed = []

    records = run(items, answers, done={"a"}, on_result=streamed.append)

    assert [r["id"] for r in records] == ["b"]
    assert streamed == records


class FakeGenerate:
    def __init__(self, answers):
        self.calls = 0
        self.answers = answers
        self.verify = fake_verify

    def answer(self, question, chunks, cfg=None, client=None):
        self.calls += 1
        return fake_answer(self.answers)(question, chunks, cfg)


def test_run_dataset_resumes_per_model(tmp_path):
    items = [item("a", ["amount"]), item("b", ["outcome.type"]), item("u", [], "unanswerable", False)]
    answers = {"question a": (True, ["amount"]), "question b": (True, ["outcome"]), "question u": (False, [])}
    run_info = {"model": "m1", "retrieval": "semantic", "top_k": 5}
    previous = {**run(items[:1], answers)[0], **run_info}
    output = tmp_path / "gen_tiny.jsonl"
    output.write_text(json.dumps(previous) + "\n" + json.dumps({**previous, "id": "b", "model": "other"})
                      + '\n{"id": "broken', encoding="utf-8")
    generate = FakeGenerate(answers)

    report = run_dataset(tmp_path / "tiny.jsonl", items, FakeRetriever(), generate, {}, None, run_info, tmp_path)

    assert generate.calls == 2
    assert report["new_questions"] == 2 and report["skipped_questions"] == 1
    assert report["summary"]["questions"] == 3
    assert report["summary"]["path_accuracy"]["rate"] == 0.5
    assert [r["id"] for r in read_records(output)] == ["a", "b", "b", "u"]
    saved = json.loads((tmp_path / "gen_tiny_summary.json").read_text(encoding="utf-8"))
    assert saved["summary"]["abstention_rate"]["rate"] == 1.0


def test_read_records_ignores_truncated_line(tmp_path):
    path = tmp_path / "gen.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b', encoding="utf-8")

    assert read_records(path) == [{"id": "a"}]
    assert read_records(tmp_path / "missing.jsonl") == []


def test_main_exits_cleanly_when_ollama_is_unreachable(tmp_path, monkeypatch):
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(json.dumps(item("a", ["amount"])) + "\n", encoding="utf-8")

    def unreachable(*args, **kwargs):
        raise RuntimeError("No se puede conectar con Ollama")

    module = types.SimpleNamespace(answer=unreachable, verify=fake_verify, known_paths=lambda rows=None: {})
    monkeypatch.setitem(sys.modules, "rag.generate", module)
    monkeypatch.setattr(rag, "generate", module, raising=False)
    monkeypatch.setattr(eval_generation, "retriever_from_args", lambda args: FakeRetriever())
    monkeypatch.setattr(sys, "argv", ["eval_generation", "--dataset", str(dataset), "--output-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as exit_info:
        eval_generation.main()

    assert exit_info.value.code == "No se puede conectar con Ollama"


def test_abstained_answer_is_not_a_hit_even_with_the_right_path():
    from rag.eval_generation import evaluate_item

    item = {"id": "q9", "query": "What is the amount?", "object": "charge", "category": "direct_field", "difficulty": "easy",
            "answerable": True, "relevant_paths": ["amount"]}
    chunks = [{"id": 1, "object": "charge", "parent": "", "text": "- `amount` (integer)"}]
    result = {"found": False, "answer": "not found", "field_paths": ["amount"], "cited_chunks": [1]}
    check = {"field_paths": ["amount"], "invented_paths": [], "unsupported_paths": [], "grounded": False}

    record = evaluate_item(item, chunks, result, check, 5)

    assert record["retrieval_hit"] is True
    assert record["path_hit"] is False
