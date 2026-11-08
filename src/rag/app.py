import argparse
import html
import os
import threading

from .common import OUT, load_config

EXAMPLES = [
    "What currency does a charge use?",
    "Which address fields are stored in billing details on a charge?",
    "Why can a payment intent be canceled?",
    "Which fields describe the card used in a payment method?",
]
_lock = threading.Lock()
_cache = {}


def code_list(paths):
    return ", ".join(f"`{p}`" for p in paths) if paths else "ninguno"


def format_verification(verification):
    if verification.get("grounded"):
        return "**Verificación:** fundamentada"
    parts = []
    if verification.get("invented_paths"):
        parts.append(f"campos inventados: {code_list(verification['invented_paths'])}")
    if verification.get("unsupported_paths"):
        parts.append(f"campos no respaldados por los fragmentos: {code_list(verification['unsupported_paths'])}")
    if verification.get("invalid_citations"):
        parts.append(f"citas no válidas: {', '.join(f'[{c}]' for c in verification['invalid_citations'])}")
    return "**Verificación:** no fundamentada — " + ("; ".join(parts) or "sin detalles")


def format_answer(result, verification):
    text = (result.get("answer") or "").strip()
    if result.get("parse_error"):
        text = "_No encontrado: la respuesta del modelo no era un JSON válido._"
    elif not result.get("found"):
        text = "_No encontrado en la documentación recuperada._" + (f"\n\n> {text}" if text else "")
    lines = [
        f"### Respuesta\n\n{text}",
        format_verification(verification),
        f"**Campos citados:** {code_list(verification.get('field_paths', result.get('field_paths', [])))}",
        f"_Latencia: {result.get('latency_ms', 0)} ms · Modelo: {result.get('model', '?')}_",
    ]
    return "\n\n".join(lines)


def format_error(message):
    return (
        f"### No se pudo generar la respuesta\n\n{message}\n\n"
        "_Los fragmentos recuperados se muestran igualmente a la derecha._"
    )


def chunk_title(n, chunk, cited):
    parent = chunk.get("parent") or "(nivel superior)"
    mark = " — citado" if n in cited else ""
    return html.escape(f"[{n}] {chunk.get('object', '?')} / {parent} ({chunk.get('score', 0.0):.3f}){mark}")


def format_sources(chunks, cited=()):
    if not chunks:
        return "### Fragmentos recuperados\n\n_Ningún fragmento recuperado._"
    cited = set(cited or ())
    blocks = ["### Fragmentos recuperados"]
    for n, chunk in enumerate(chunks, start=1):
        is_open = " open" if n in cited else ""
        body = chunk.get("text", "").replace("~~~", "~ ~ ~")
        blocks.append(f"<details{is_open}><summary>{chunk_title(n, chunk, cited)}</summary>\n\n~~~text\n{body}\n~~~\n\n</details>")
    return "\n\n".join(blocks)


def get_retriever(object_filter):
    from .retrieve import ObjectFilterRetriever, build_retriever, load_metadata

    with _lock:
        if "base" not in _cache:
            cfg = load_config()["retrieval"]
            index_path = OUT / "rag.faiss"
            if cfg["method"] != "lexical" and not index_path.exists():
                raise RuntimeError(f"No existe el índice {index_path}. Ejecuta primero: python -m rag.index")
            metadata_path = OUT / "rag.metadata.jsonl"
            _cache["base"] = build_retriever(
                cfg["method"], index_path, metadata_path, OUT / "rag.sqlite3",
                rerank=cfg["rerank"],
            )
            objects = {m["object"] for m in load_metadata(metadata_path)}
            _cache["filtered"] = ObjectFilterRetriever(_cache["base"], objects)
    return _cache["filtered" if object_filter else "base"]


def get_known():
    from . import generate

    with _lock:
        if "known" not in _cache:
            _cache["known"] = generate.known_paths()
    return _cache["known"]


def ask(question, object_filter=True, top_k=5):
    question = (question or "").strip()
    if not question:
        return "_Escribe una pregunta._", ""
    try:
        chunks = get_retriever(object_filter).retrieve(question, int(top_k))
    except Exception as error:  # noqa: BLE001 (la interfaz no debe caerse)
        return format_error(f"Error en la recuperación: {error}"), ""
    try:
        from . import generate

        result = generate.answer(question, chunks)
        verification = generate.verify(result, chunks, get_known())
    except RuntimeError as error:
        return format_error(str(error)), format_sources(chunks)
    except Exception as error:  # noqa: BLE001 (la interfaz no debe caerse)
        return format_error(f"Error inesperado: {type(error).__name__}: {error}"), format_sources(chunks)
    return format_answer(result, verification), format_sources(chunks, result.get("cited_chunks", []))


def build_ui():
    import gradio as gr

    cfg = load_config()
    default_filter = cfg["retrieval"]["object_filter"]
    default_k = cfg["retrieval"]["top_k"]
    with gr.Blocks(title="RAG sobre la API de Stripe", analytics_enabled=False) as demo:
        gr.Markdown("# RAG sobre la API de Stripe\nPregunta sobre los objetos de la API; la respuesta cita los fragmentos usados.")
        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(label="Pregunta", placeholder="What currency does a charge use?", lines=2)
                with gr.Row():
                    object_filter = gr.Checkbox(value=default_filter, label="Filtrar por objeto")
                    top_k = gr.Slider(1, 10, value=default_k, step=1, label="Número de fragmentos")
                button = gr.Button("Preguntar", variant="primary")
                gr.Examples(EXAMPLES, inputs=question, label="Ejemplos")
                answer = gr.Markdown()
            with gr.Column(scale=1):
                sources = gr.Markdown()
        inputs, outputs = [question, object_filter, top_k], [answer, sources]
        button.click(ask, inputs, outputs)
        question.submit(ask, inputs, outputs)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Interfaz web local del RAG.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    build_ui().launch(
        server_name=args.host, server_port=args.port, share=False, ssr_mode=False, mcp_server=False,
        footer_links=[], inbrowser=False,
    )


if __name__ == "__main__":
    main()
