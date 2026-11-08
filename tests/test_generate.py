import json

import pytest

from rag.generate import ANSWER_SCHEMA, answer, build_messages, call_llm, known_paths, parse_answer, verify

CFG = {"llm": {"model": "fake-model", "temperature": 0, "seed": 0, "num_ctx": 8192, "top_k": 5}}
CHUNKS = [
    {
        "id": 0,
        "object": "charge",
        "parent": "payment_method_details.ach_debit",
        "text": "Charge → payment method details → ach debit\nPath: payment_method_details.ach_debit\n\n"
        "- `payment_method_details.ach_debit` (object)\n\n"
        "  - `payment_method_details.ach_debit.country` (string, nullable)\n    Two-letter ISO code.",
    },
    {"id": 1, "object": "customer", "parent": "", "text": "Customer\nPath: (top level)\n\n- `email` (string, nullable)\n  Email."},
]
ROWS = [
    {"object": "charge", "path": "amount"},
    {"object": "charge", "path": "payment_method_details.ach_debit"},
    {"object": "charge", "path": "payment_method_details.ach_debit.country"},
    {"object": "customer", "path": "email"},
    {"object": "checkout.session", "path": "url"},
]
KNOWN = known_paths(ROWS)


class FakeClient:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, model, messages, format, options):
        self.calls.append({"model": model, "messages": messages, "format": format, "options": options})
        if self.error:
            raise self.error
        return {"message": {"content": self.content}}


def result(paths, cited=(1,), found=True):
    return {"found": found, "answer": "x", "field_paths": list(paths), "cited_chunks": list(cited)}


def test_build_messages_numbers_and_labels_chunks():
    messages = build_messages("Which country?", CHUNKS)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "[1] (object: charge)\nCharge → payment method details" in user
    assert "[2] (object: customer)\nCustomer" in user
    assert "[3]" not in user
    assert user.rstrip().endswith("Question: Which country?")


def test_parse_answer_valid_json_normalizes_types():
    raw = json.dumps({"found": "true", "answer": " Yes. ", "field_paths": "amount", "cited_chunks": ["1", 2, "[3]"]})
    parsed = parse_answer(raw)
    assert parsed["found"] is True
    assert parsed["answer"] == "Yes."
    assert parsed["field_paths"] == ["amount"]
    assert parsed["cited_chunks"] == [1, 2, 3]
    assert parsed["parse_error"] is False


@pytest.mark.parametrize("raw", ["not json", '{"found": true, "answer": "x"}', "[1, 2]", ""])
def test_parse_answer_invalid_json(raw):
    parsed = parse_answer(raw)
    assert parsed == {"found": False, "answer": raw, "field_paths": [], "cited_chunks": [], "parse_error": True}


def test_verify_grounded_answer():
    check = verify(result(["payment_method_details.ach_debit.country"]), CHUNKS, KNOWN)
    assert check["grounded"] is True
    assert check["invented_paths"] == check["unsupported_paths"] == check["invalid_citations"] == []


def test_verify_invented_path():
    check = verify(result(["payment_method_details.ach_debit.routing"]), CHUNKS, KNOWN)
    assert check["invented_paths"] == ["payment_method_details.ach_debit.routing"]
    assert check["grounded"] is False


def test_verify_unsupported_path():
    check = verify(result(["amount"]), CHUNKS, KNOWN)
    assert check["unsupported_paths"] == ["amount"]
    assert check["invented_paths"] == []
    assert check["grounded"] is False


def test_verify_normalizes_object_prefix_and_backticks():
    check = verify(result([" `charge.payment_method_details.ach_debit.country` ", "customer.email", "checkout.session.url"]), CHUNKS, KNOWN)
    assert check["field_paths"] == ["payment_method_details.ach_debit.country", "email", "url"]
    assert check["unsupported_paths"] == ["url"]


def test_verify_descendant_supports_box():
    chunks = [{"object": "charge", "text": "- `payment_method_details.ach_debit.country` (string)"}]
    check = verify(result(["payment_method_details.ach_debit"]), chunks, KNOWN)
    assert check["grounded"] is True
    assert verify(result(["payment_method_details.ach"]), chunks, KNOWN)["invented_paths"] == ["payment_method_details.ach"]


def test_verify_invalid_citation():
    check = verify(result(["email"], cited=[0, 2, 3]), CHUNKS, KNOWN)
    assert check["invalid_citations"] == [0, 3]
    assert check["grounded"] is False


def test_verify_not_found_still_checks_paths():
    check = verify(result(["made.up"], cited=[9], found=False), CHUNKS, KNOWN)
    assert check["grounded"] is False
    assert check["invented_paths"] == ["made.up"]
    assert check["invalid_citations"] == [9]


def test_answer_with_fake_client():
    content = json.dumps({"found": True, "answer": "It is the country.", "field_paths": ["payment_method_details.ach_debit.country"], "cited_chunks": [1]})
    client = FakeClient(content)
    out = answer("Which country?", CHUNKS, CFG, client)
    assert out["found"] is True and out["cited_chunks"] == [1]
    assert out["model"] == "fake-model"
    assert isinstance(out["latency_ms"], int) and out["latency_ms"] >= 0
    call = client.calls[0]
    assert call["model"] == "fake-model"
    assert call["format"] == ANSWER_SCHEMA
    assert call["options"] == {"temperature": 0, "seed": 0, "num_ctx": 8192}


def test_call_llm_connection_error_is_runtime_error():
    with pytest.raises(RuntimeError, match="ollama serve"):
        call_llm([], CFG, FakeClient(error=ConnectionError("down")))


def test_call_llm_missing_model_is_runtime_error():
    error = Exception('model "fake-model" not found, try pulling it first')
    error.status_code = 404
    with pytest.raises(RuntimeError, match="ollama pull fake-model"):
        call_llm([], CFG, FakeClient(error=error))


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        return self.response


def test_openai_compatible_client_sends_schema_and_reads_content():
    from rag.generate import ANSWER_SCHEMA, OpenAICompatibleClient

    client = OpenAICompatibleClient("http://localhost:1234/v1/")
    content = '{"found": true, "answer": "usd", "field_paths": ["currency"], "cited_chunks": [1]}'
    client.http = FakeHttp(FakeResponse(200, {"choices": [{"message": {"content": content}}]}))

    reply = client.chat(model="m", messages=[{"role": "user", "content": "q"}], format=ANSWER_SCHEMA,
                        options={"temperature": 0, "seed": 0, "num_ctx": 8192})

    url, body = client.http.calls[0]
    assert url == "http://localhost:1234/v1/chat/completions"
    assert body["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA
    assert "num_ctx" not in body and body["temperature"] == 0
    assert reply["message"]["content"] == content


def test_openai_backend_errors_mention_lm_studio():
    from rag.generate import OpenAICompatibleClient, call_llm

    client = OpenAICompatibleClient("http://localhost:1234/v1")
    client.http = FakeHttp(FakeResponse(404, {"error": "model not found"}))
    cfg = {"llm": {"backend": "openai", "model": "google/gemma-4-e4b"}}

    with pytest.raises(RuntimeError, match="LM Studio"):
        call_llm([{"role": "user", "content": "q"}], cfg, client)


def test_default_client_uses_backend():
    from rag.generate import OpenAICompatibleClient, default_client

    assert isinstance(default_client({"backend": "openai", "model": "m"}), OpenAICompatibleClient)
