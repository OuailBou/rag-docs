import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from .common import OUT, field_paths, load_config, load_table

SYSTEM_PROMPT = """You answer questions about the Stripe API using ONLY the numbered documentation excerpts provided.
Rules:
- Use only information that appears in the excerpts. Never use outside knowledge.
- If the answer is not in the excerpts, set "found" to false and say that it could not be found in the documentation.
- "field_paths" must list the exact field paths as written in the excerpts (between backticks), without the object name.
- "cited_chunks" must list the numbers of the excerpts that support the answer.
- Keep "answer" to 1-3 sentences.
Reply with a JSON object with keys: found, answer, field_paths, cited_chunks."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "field_paths": {"type": "array", "items": {"type": "string"}},
        "cited_chunks": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "field_paths", "cited_chunks"],
}

KEYS = ("found", "answer", "field_paths", "cited_chunks")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def llm_config(cfg=None):
    cfg = load_config() if cfg is None else cfg
    return cfg.get("llm", cfg)


def format_context(chunks):
    return "\n\n".join(f"[{i}] (object: {c['object']})\n{c['text'].strip()}" for i, c in enumerate(chunks, start=1))


def build_messages(question, chunks):
    user = f"Documentation excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question.strip()}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class ServerError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleClient:
    """Servidor con la API de OpenAI: LM Studio, llama.cpp, vLLM..."""

    def __init__(self, base_url, timeout=300):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def chat(self, model, messages, format=None, options=None):
        import httpx

        options = options or {}
        body = {"model": model, "messages": messages, "stream": False,
                **{k: options[k] for k in ("temperature", "seed") if k in options}}
        if format:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "strict": True, "schema": format}}
        try:
            response = self.http.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.ConnectError as error:
            raise ConnectionError(str(error)) from error
        if response.status_code >= 400:
            raise ServerError(f"{response.status_code}: {response.text[:300]}", response.status_code)
        return {"message": {"content": response.json()["choices"][0]["message"]["content"]}}


def backend_help(llm):
    model = llm["model"]
    if llm.get("backend") == "openai":
        url = llm.get("base_url", "http://localhost:1234/v1")
        hint = (f"Abre LM Studio, carga el modelo '{model}' y arranca el servidor local "
                f"(pestaña Developer, o 'lms server start' y 'lms load {model}').")
        return f"el servidor de {url}", hint
    return ("Ollama", f"Arranca el servidor con 'ollama serve' y descarga el modelo con 'ollama pull {model}'.")


def default_client(llm):
    if llm.get("backend") == "openai":
        return OpenAICompatibleClient(llm.get("base_url", "http://localhost:1234/v1"), llm.get("timeout", 300))
    try:
        import ollama
    except ImportError as error:
        raise RuntimeError("Falta el paquete 'ollama'. Instálalo con: pip install ollama") from error
    return ollama.Client(host=os.environ.get("OLLAMA_HOST"), timeout=llm.get("timeout", 300))


def call_llm(messages, cfg=None, client=None):
    llm = llm_config(cfg)
    model = llm["model"]
    server, hint = backend_help(llm)
    client = client or default_client(llm)
    options = {key: llm[key] for key in ("temperature", "seed", "num_ctx") if key in llm}
    try:
        response = client.chat(model=model, messages=messages, format=ANSWER_SCHEMA, options=options)
    except OSError as error:
        raise RuntimeError(f"No se pudo conectar con {server} ({error}). {hint}") from error
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status == 404 or (status is not None and "not found" in str(error).lower()):
            raise RuntimeError(f"El modelo '{model}' no está disponible en {server}. {hint}") from error
        if "timeout" in type(error).__name__.lower():
            raise RuntimeError(
                f"{server} no respondió a tiempo ({error}). Comprueba que el modelo '{model}' está cargado "
                "y cabe en memoria, o aumenta llm.timeout en config.yaml."
            ) from error
        raise
    return response["message"]["content"]


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def as_ints(values):
    result = []
    for value in as_list(values):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result.append(int(value))
        else:
            result.extend(int(d) for d in re.findall(r"\d+", str(value)))
    return list(dict.fromkeys(result))


def parse_answer(raw):
    failure = {"found": False, "answer": raw, "field_paths": [], "cited_chunks": [], "parse_error": True}
    match = JSON_OBJECT.search(raw or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or any(key not in data for key in KEYS):
        return failure
    return {
        "found": as_bool(data["found"]),
        "answer": str(data["answer"] or "").strip(),
        "field_paths": [str(p).strip() for p in as_list(data["field_paths"]) if str(p).strip()],
        "cited_chunks": as_ints(data["cited_chunks"]),
        "parse_error": False,
    }


def answer(question, chunks, cfg=None, client=None):
    llm = llm_config(cfg)
    start = time.perf_counter()
    raw = call_llm(build_messages(question, chunks), llm, client)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {**parse_answer(raw), "latency_ms": latency_ms, "model": llm["model"]}


def known_paths(rows=None):
    known = {}
    for row in load_table() if rows is None else rows:
        known.setdefault(row["object"], set()).add(row["path"])
    return known


def normalize_path(path, known):
    path = str(path).strip().strip("`").strip()
    for obj, paths in known.items():
        if path.lower().startswith(obj.lower() + ".") and path[len(obj) + 1:] in paths:
            return path[len(obj) + 1:]
    return path


def is_supported(path, chunk_paths):
    return any(p == path or p.startswith(path + ".") for p in chunk_paths)


def verify(result, chunks, known):
    paths = [normalize_path(p, known) for p in result.get("field_paths", [])]
    paths = list(dict.fromkeys(p for p in paths if p))
    all_known = set().union(*known.values()) if known else set()
    chunk_paths = set().union(*(field_paths(c["text"]) for c in chunks)) if chunks else set()
    invented = [p for p in paths if p not in all_known]
    unsupported = [p for p in paths if p in all_known and not is_supported(p, chunk_paths)]
    invalid = [n for n in result.get("cited_chunks", []) if not 1 <= n <= len(chunks)]
    grounded = bool(result.get("found")) and not (invented or unsupported or invalid)
    return {
        "field_paths": paths,
        "invented_paths": invented,
        "unsupported_paths": unsupported,
        "invalid_citations": invalid,
        "grounded": grounded,
    }


def print_result(result, chunks):
    check = result["verification"]
    print(result["answer"] or "(sin respuesta)")
    print()
    print(f"Encontrado: {'sí' if result['found'] else 'no'}")
    print(f"Campos: {', '.join(check['field_paths']) or '-'}")
    print(f"Citas: {', '.join(f'[{n}]' for n in result['cited_chunks']) or '-'}")
    print(f"Fundamentada: {'sí' if check['grounded'] else 'no'}")
    for label, key in (("Inventados", "invented_paths"), ("Sin respaldo", "unsupported_paths"), ("Citas inválidas", "invalid_citations")):
        if check[key]:
            print(f"{label}: {', '.join(map(str, check[key]))}")
    if result.get("parse_error"):
        print("Aviso: la respuesta del modelo no era un JSON válido.")
    print(f"Modelo: {result['model']}  latencia={result['latency_ms']} ms")
    print("\nFragmentos recuperados:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] {chunk['object']} / {chunk['parent'] or '(top level)'}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from .retrieve import add_retrieval_args, retriever_from_args

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Responde preguntas sobre la API de Stripe con el RAG local.")
    parser.add_argument("question", nargs="+", help="Pregunta")
    add_retrieval_args(parser, cfg)
    parser.add_argument("--json", action="store_true", help="Imprime el resultado completo en JSON")
    args = parser.parse_args()
    if not Path(args.metadata).exists():
        parser.error(f"No existe el índice en {OUT}. Ejecuta primero: python -m rag.index")
    question = " ".join(args.question)
    chunks = retriever_from_args(args).retrieve(question, max(1, args.top_k))
    try:
        result = answer(question, chunks, cfg)
    except RuntimeError as error:
        sys.exit(str(error))
    result["verification"] = verify(result, chunks, known_paths())
    result["chunks"] = [{key: c.get(key) for key in ("id", "object", "parent", "score")} for c in chunks]
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result, chunks)


if __name__ == "__main__":
    main()
