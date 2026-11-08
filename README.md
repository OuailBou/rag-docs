# Stripe RAG

Asistente que responde preguntas sobre la API de Stripe con un modelo de lenguaje local. Usa como fuente la especificación OpenAPI oficial, trocea la documentación siguiendo el árbol de campos de cada objeto, y verifica cada campo que cita el modelo contra la spec.

## Resultados

Test final: 70 preguntas escritas sin ver el sistema y evaluadas una sola vez.

| | |
|---|---|
| El fragmento correcto está entre los 5 recuperados (`Recall@5`) | **0.850** [0.76, 0.93] |
| Troceado jerárquico frente al fijo de LangChain | **+0.233** en `Recall@5` |
| Respuestas correctas en contenido (juez a ciegas) | **0.817** |
| Campos inventados por el modelo | **0 de 70** |
| Preguntas sin respuesta reconocidas | **10 de 10** |

Al pasar de 16 a 50 objetos, el rendimiento no baja. Todos los experimentos, con su explicación, están en [`docs/experimentos.md`](docs/experimentos.md).

## Puesta en marcha (Mac, con LM Studio)

```bash
uv sync
uv run python -m rag.extract
uv run python -m rag.chunk
uv run python -m rag.index
lms server start
lms load google/gemma-4-e4b --context-length 8192
uv run python -m rag.app
```

La interfaz queda en `http://127.0.0.1:7860`. En macOS, si el proceso se cierra sin mensaje al buscar (choque entre `faiss` y `torch`), antes de los comandos:

```bash
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
```

Para Windows con Ollama, sigue [`docs/windows.md`](docs/windows.md) (`scripts\windows\setup.ps1`).

## Comandos

| Para | Comando |
|---|---|
| Ver los objetos disponibles | `uv run python -m rag.extract --list` |
| Buscar | `uv run python -m rag.retrieve "What currency does a charge use?"` |
| Preguntar al modelo | `uv run python -m rag.generate "What currency does a charge use?"` |
| Evaluar la búsqueda | `uv run python -m rag.eval_retrieval --dataset data/evals/dev_charge.jsonl` |
| Evaluar el modelo | `uv run python -m rag.eval_generation --dataset data/evals/test.jsonl` |
| Repetir los experimentos | `uv run python scripts/experiment.py` |
| Evaluar control, test y test2 | `uv run python scripts/heldout.py` |
| Ver los fallos de una variante | `uv run python scripts/failures.py E1_sin_copias semantic+objfilter` |
| Juzgar la corrección de las respuestas | `uv run python scripts/judge.py prepare` y `score` |
| Clasificar los fallos por fase | `uv run python scripts/error_taxonomy.py --model google/gemma-4-e4b` |
| Tests | `uv run pytest` |

## Cambiar parámetros

Todos los parámetros están en [`configs/default.yaml`](configs/default.yaml), agrupados por etapa (`data`, `extract`, `chunker`, `embedding`, `retrieval`, `llm`). Tres formas de cambiarlos:

- **Editar `configs/default.yaml`.** Si cambias algo de `data`, `extract`, `chunker` o `embedding`, vuelve a ejecutar `extract`, `chunk` e `index`.
- **Otra configuración entera:** `RAG_CONFIG=configs/ollama.yaml uv run python -m rag.app`.
- **Solo para un comando**, las opciones de búsqueda: `--method semantic|lexical|hybrid`, `--object-filter/--no-object-filter`, `--rerank/--no-rerank` y `--top-k N`.

## Documentación

- [`docs/arquitectura.md`](docs/arquitectura.md): cómo funciona, módulo por módulo, formatos y configuración.
- [`docs/experimentos.md`](docs/experimentos.md): cada experimento, sus resultados y lo que significan.
- [`docs/windows.md`](docs/windows.md): instalación en Windows.
- [`data/evals/README.md`](data/evals/README.md): los conjuntos de preguntas.
