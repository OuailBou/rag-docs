# Anexos

## Anexo A. Configuración final

Fichero `configs/default.yaml`, con todos los parámetros del sistema medido. La configuración para Windows (`configs/ollama.yaml`) solo cambia `llm.backend` a `ollama` y el modelo a `qwen2.5-coder:7b`.

```yaml
# Configuración final del sistema (la que se ha medido).
# Otra configuración: RAG_CONFIG=configs/ollama.yaml

data:
  spec: data/raw/spec3.json          # spec OpenAPI de Stripe (2026-09-30.endive)
  objects: data/objects_16.txt       # objetos del corpus (escala: data/objects_50.txt)

extract:
  open_active_objects: false         # false: no copiar dentro de un objeto otros objetos de la lista

chunker:
  strategy: hierarchical             # hierarchical | fixed
  max_tokens: 512                    # tamaño máximo de un fragmento (límite del modelo de embeddings)
  overlap: 64                        # solape entre fragmentos (solo fixed)
  header: true                       # cabecera con la ruta en cada fragmento
  small_to_big: false                # buscar con un fragmento por campo y devolver su fragmento grande

embedding:
  model: BAAI/bge-small-en-v1.5      # también se usa su tokenizador para contar tokens
  query_prefix: ""                   # texto que se antepone a cada pregunta

retrieval:
  method: semantic                   # semantic | lexical | hybrid
  object_filter: true                # priorizar los objetos que nombra la pregunta
  top_k: 5                           # fragmentos que recibe el modelo
  rerank: false                      # reordenar con un cross-encoder
  reranker_model: BAAI/bge-reranker-base

llm:
  backend: openai                    # openai (LM Studio u otro servidor compatible) | ollama
  base_url: http://localhost:1234/v1 # solo openai
  model: google/gemma-4-e4b          # alternativa ligera en LM Studio: qwen2.5-coder-7b-instruct
  temperature: 0
  seed: 0
  num_ctx: 8192
  timeout: 300                       # segundos
```

## Anexo B. Reproducción de los resultados

Todos los comandos se ejecutan desde la raíz del proyecto. En macOS, antes de ejecutarlos: `export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`.

| Paso | Comando | Resultado |
|---|---|---|
| Instalar dependencias | `uv sync` | entorno fijado por `uv.lock` |
| Construir el índice | `uv run python -m rag.extract`, `rag.chunk`, `rag.index` | `out/table.parquet`, `out/chunks.jsonl`, `out/rag.faiss`, `out/rag.db` |
| Experimentos de desarrollo (E0–E11) | `uv run python scripts/experiment.py` | `out/experiments/summary.md` (tablas 5.2–5.4) |
| Control, test y test 2 (búsqueda) | `uv run python scripts/heldout.py` | `out/experiments/heldout_<conjunto>.md` (tablas 5.5–5.12 y 5.22) |
| Generación | `uv run python -m rag.eval_generation --dataset data/evals/test.jsonl` | `out/results/gen_test.jsonl` y su resumen |
| Juez a ciegas | `uv run python scripts/judge.py prepare` y, tras juzgar, `score` | `out/judge/` |
| Taxonomía de errores | `uv run python scripts/error_taxonomy.py --model google/gemma-4-e4b` | `out/analysis/error_taxonomy_gemma_4_e4b.md` |
| Pruebas | `uv run pytest` | 42 pruebas |
| Interfaz | `uv run python -m rag.app` | `http://127.0.0.1:7860` |

La generación necesita un servidor de modelos en marcha (`lms server start` y `lms load google/gemma-4-e4b --context-length 8192` en LM Studio). Las variantes y comparaciones de los dos primeros pasos están en `configs/experiments.yaml`.

## Anexo C. Instrucciones del modelo

*Prompt* de sistema que recibe el modelo de lenguaje (`src/rag/generate.py`). El mensaje de usuario contiene los cinco fragmentos numerados (`[1] (object: charge)` seguido del texto del fragmento) y la pregunta. Con LM Studio, la respuesta se restringe además a un esquema JSON con las cuatro claves (`response_format`).

```text
You answer questions about the Stripe API using ONLY the numbered documentation excerpts provided.
Rules:
- Use only information that appears in the excerpts. Never use outside knowledge.
- If the answer is not in the excerpts, set "found" to false and say that it could not be found in the documentation.
- "field_paths" must list the exact field paths as written in the excerpts (between backticks), without the object name.
- "cited_chunks" must list the numbers of the excerpts that support the answer.
- Keep "answer" to 1-3 sentences.
Reply with a JSON object with keys: found, answer, field_paths, cited_chunks.
```

## Anexo D. Ejemplo de fragmento jerárquico

Fragmento real del objeto `charge` (número 6 de 744). La caja `payment_method_details.ach_debit` cabe entera en el presupuesto de 512 *tokens*, así que se emite como un fragmento propio. Las dos primeras líneas son la cabecera, y cada campo lleva su ruta completa, su tipo y su descripción.

```text
Charge → payment method details → ach debit
Path: payment_method_details.ach_debit

- `payment_method_details.ach_debit` (object)

  - `payment_method_details.ach_debit.account_holder_type` (enum, nullable)
    Type of entity that holds the account. This can be either `individual` or `company`.
    Possible values: company, individual.

  - `payment_method_details.ach_debit.bank_name` (string, nullable)
    Name of the bank associated with the bank account.

  - `payment_method_details.ach_debit.country` (string, nullable)
    Two-letter ISO code representing the country the bank account is located in.

  - `payment_method_details.ach_debit.fingerprint` (string, nullable)
    Uniquely identifies this particular bank account. You can use this attribute to check whether two bank accounts are the same.

  - `payment_method_details.ach_debit.last4` (string, nullable)
    Last four digits of the bank account number.

  - `payment_method_details.ach_debit.routing_number` (string, nullable)
    Routing transit number of the bank account.
```

Sin la cabecera y las rutas, la línea `country` de este fragmento sería indistinguible de los otros 18 campos `country` del objeto `charge`.

## Anexo E. Ejemplo de respuesta y verificación

Pregunta del test `test-006`, *"What reconciliation statuses can a payout report?"*, respondida por Gemma 4 E4B con la configuración final.

Respuesta del modelo:

```json
{
  "found": true,
  "answer": "A payout can report three possible reconciliation statuses: `completed`, `in_progress`, or `not_applicable`.",
  "field_paths": [
    "reconciliation_status"
  ],
  "cited_chunks": [
    1
  ]
}
```

Resultado del verificador: ningún campo inventado (`invented_paths` vacío), ninguno sin respaldo en el contexto (`unsupported_paths` vacío), todas las citas válidas, y el campo coincide con el esperado (`path_hit: true`).

```json
{
  "invented_paths": [],
  "unsupported_paths": [],
  "invalid_citations": [],
  "path_hit": true
}
```

Ante la pregunta sin respuesta `test-018`, *"What phone number do I call to talk to Stripe support about a dispute?"*, el modelo devuelve `found: false` y responde: *"The provided documentation excerpts do not contain any information regarding a phone number to call Stripe support about a dispute."*

## Anexo F. Muestra de las preguntas de evaluación

Ejemplos reales de cada tipo de pregunta. Las preguntas `test-*` las escribió un agente de IA sin acceso al sistema. Las `heldout-*` son del conjunto de control, generado con plantillas. La columna de rutas indica los campos que se aceptan como respuesta.

| Id | Categoría | Objeto | Pregunta | Rutas aceptadas |
|---|---|---|---|---|
| test-001 | test_nested | `payment_link` | What is the largest number of units of an optional item a customer can add on a payment link? | `optional_items.adjustable_quantity.maximum`, `optional_items.adjustable_quantity` |
| test-002 | test_descriptive | `checkout.session` | If a customer abandons a checkout session and it times out, can Stripe give me a link that recreates it so I can email it to them? | `after_expiration.recovery.url`, `after_expiration.recovery`, `after_expiration.recovery.enabled` |
| test-006 | test_enum | `payout` | What reconciliation statuses can a payout report? | `reconciliation_status` |
| test-011 | test_direct | `invoice` | What is the due date of an invoice? | `due_date` |
| test-018 | test_unanswerable | `dispute` | What phone number do I call to talk to Stripe support about a dispute? | — |
| heldout-001 | heldout_desc | `payment_link` | On a payment link object, which field holds the specified behavior after the purchase is complete? | `after_completion.type` |
| heldout-002 | heldout_name | `payment_link` | What is the enabled in the managed payments of a payment link? | `managed_payments.enabled` |
| heldout-003 | heldout_ident | `invoice` | What does `default_tax_rates.country` mean on a invoice? | `default_tax_rates.country` |

La pregunta `test-011` es uno de los fallos analizados en la taxonomía de errores (sección 5.8). El campo `due_date` está en un grupo de campos sueltos de primer nivel del objeto `invoice`, que mezcla temas sin relación.

## Anexo G. Estructura del repositorio

```text
configs/            default.yaml (final), ollama.yaml (Windows), experiments.yaml (variantes)
data/raw/           spec3.json, la especificación OpenAPI de Stripe
data/               objects_16.txt y objects_50.txt (objetos del corpus)
data/evals/         dev_charge, dev_objects, control, test, test2 (JSONL)
src/rag/            extract, chunk, index, retrieve, generate, eval_retrieval,
                    eval_generation, app, common
scripts/            experiment, heldout, failures, make_control_set, judge, error_taxonomy
scripts/windows/    setup, run_app, eval_generation, run_experiments (PowerShell)
tests/              42 pruebas con pytest
docs/               arquitectura, experimentos, windows y esta memoria
out/                todo lo generado (se regenera con los comandos del anexo B)
```
