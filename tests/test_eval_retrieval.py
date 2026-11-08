from rag.eval_retrieval import evaluate_question, summarize
from rag.retrieve import mentioned_objects

ITEM = {
    "id": "q1",
    "query": "What is the amount?",
    "object": "charge",
    "category": "direct_field",
    "difficulty": "easy",
    "answerable": True,
    "relevant_paths": ["amount"],
}


def test_evaluate_question_and_summary():
    results = [
        {"id": 4, "text": "- `currency` (string)"},
        {"id": 9, "text": "- `amount` (integer)"},
    ]

    evaluation = evaluate_question(ITEM, results)

    assert evaluation["recall_at_1"] == 0.0
    assert evaluation["recall_at_3"] == 1.0
    assert evaluation["mrr"] == 0.5
    assert summarize([evaluation])["overall"]["recall_at_3"] == 1.0


def test_other_objects_keep_their_rank_and_do_not_count():
    results = [
        {"id": 1, "object": "customer", "text": "- `amount` (integer)"},
        {"id": 2, "object": "charge", "text": "- `amount` (integer)"},
    ]

    evaluation = evaluate_question(ITEM, results)

    assert evaluation["recall_at_1"] == 0.0
    assert evaluation["mrr"] == 0.5


def test_paths_mentioned_in_descriptions_do_not_count():
    results = [{"id": 1, "text": "- `currency` (string)\n  Uses the same unit as `amount`."}]

    assert evaluate_question(ITEM, results)["recall_at_5"] == 0.0


def test_metadata_questions_are_excluded_from_retrieval_metrics():
    metadata_item = {**ITEM, "id": "q2", "category": "metadata"}
    results = [{"id": 9, "text": "- `amount` (integer)"}]

    summary = summarize([evaluate_question(ITEM, results), evaluate_question(metadata_item, [])])

    assert summary["scored_questions"] == 1
    assert summary["metadata_questions"] == 1
    assert summary["overall"]["recall_at_5"] == 1.0


def test_mentioned_objects():
    objects = ["charge", "payment_intent", "checkout.session", "refund"]

    assert mentioned_objects("Why can a payment intent be canceled?", objects) == {"payment_intent"}
    assert mentioned_objects("List checkout sessions for refunds", objects) == {"checkout.session", "refund"}
    assert mentioned_objects("Was the charge refunded?", objects) == {"charge"}
