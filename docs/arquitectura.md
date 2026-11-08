# Arquitectura

Asistente que responde preguntas sobre la API de Stripe con un modelo de lenguaje local, usando como fuente la especificación OpenAPI oficial. Todo funciona en local: búsqueda, modelo y evaluación.

## El flujo completo

```
data/raw/spec3.json          spec OpenAPI de Stripe (versión 2026-09-30.endive)
        │
        │  1. extract   recorre los objetos de data/objects_16.txt siguiendo los $ref
        ▼
out/table.parquet            una fila por campo: ruta, tipo, nullable, enum, descripción...
        │
        │  2. chunk     agrupa los campos en fragmentos (troceado jerárquico)
        ▼
out/chunks.jsonl             un fragmento por línea: objeto, caja, texto
        │
        │  3. index     embeddings (bge-small) + búsqueda literal (SQLite FTS5)
        ▼
out/rag.faiss  out/rag.sqlite3  out/rag.metadata.jsonl
        │
        │  4. retrieve  pregunta → 5 fragmentos más parecidos (+ filtro por objeto)
        ▼
fragmentos
        │
        │  5. generate  el modelo local responde en JSON citando campos y fragmentos;
        │               el verificador comprueba cada campo citado contra la tabla
        ▼
respuesta + verificación     (rag.app la muestra en el navegador)
```

Cada paso lee de disco lo que dejó el anterior y escribe su resultado. Se puede ejecutar cualquiera por separado.

## Módulos (`src/rag/`)

Orden recomendado para leerlos:

| Fichero | Qué hace | Entra → sale | Comando |
|---|---|---|---|
| `common.py` | Utilidades compartidas: configuración, rutas, formato de un campo, cabecera, contar tokens | — | — |
| `extract.py` | Recorre la spec siguiendo los `$ref` y saca una fila por campo. No abre los objetos que ya tienen documento propio | `spec3.json` → `table.parquet` | `python -m rag.extract` |
| `chunk.py` | Troceado **jerárquico** (por el árbol de campos) o **fijo** (LangChain, de referencia) | `table.parquet` → `chunks.jsonl` | `python -m rag.chunk` |
| `index.py` | Calcula los embeddings y crea el índice semántico (FAISS) y el literal (FTS5) | `chunks.jsonl` → `rag.*` | `python -m rag.index` |
| `retrieve.py` | Búsqueda semántica, literal o híbrida, más el filtro por objeto y el *reranker* opcionales | pregunta → fragmentos | `python -m rag.retrieve "..."` |
| `generate.py` | Llama al modelo (LM Studio u Ollama), lee su JSON y **verifica** los campos citados | pregunta + fragmentos → respuesta | `python -m rag.generate "..."` |
| `eval_retrieval.py` | Evalúa la búsqueda: `Recall@k`, `nDCG`, `MRR`, con intervalos de confianza | preguntas → resultados | `python -m rag.eval_retrieval --dataset ...` |
| `eval_generation.py` | Evalúa el modelo: acierto del campo, alucinación, abstención, latencia. Se puede interrumpir y continuar | preguntas → resultados | `python -m rag.eval_generation --dataset ...` |
| `app.py` | Interfaz web (Gradio): respuesta, verificación y fragmentos usados | — | `python -m rag.app` |

### Las piezas clave por dentro

**`extract.walk`** decide qué es cada campo y si hay que entrar en él:

- **valor** (`string`, `integer`...): una fila.
- **caja** (`$ref`, `anyOf` con un solo `$ref`, `object` con `properties`): una fila y se baja a sus campos.
- **expandable** (`anyOf` con un `string` y un `$ref`, es decir, un ID que se puede expandir): una fila y no se baja.
- **lista** (`array`): se baja a lo que hay dentro de `items`.
- Si la caja ya está en el camino por el que se viene, o es otro objeto de la lista activa, no se entra: así se evitan los bucles y las copias.

**`chunk.HierChunker.level`** recorre un nivel del árbol con presupuesto de 512 tokens:

1. Los **valores sueltos** se van juntando en un grupo; cuando no caben más, el grupo se cierra y es un fragmento.
2. Una **caja que cabe entera** es un fragmento propio.
3. Una **caja que no cabe** deja su línea en el grupo y se baja un nivel.
4. Un campo que **él solo** no cabe se parte por frases, comas o espacios, repitiendo su línea.
5. Cada fragmento lleva una **cabecera** con la ruta: `Charge → payment method details → ach debit`.

**`generate.verify`** comprueba cada campo que cita el modelo:

- **inventado**: no existe en ningún objeto de la tabla (alucinación);
- **no respaldado**: existe, pero no está en los fragmentos que recibió el modelo;
- **cita no válida**: cita un fragmento que no existe.

## Formatos de datos

**Fila de `table.parquet`**: `object`, `title`, `path` (relativa al objeto, p. ej. `payment_method_details.ach_debit.country`), `name`, `depth`, `type`, `nullable`, `expandable`, `enum` (valores separados por `|`), `description`, `has_children`.

**Fragmento de `chunks.jsonl`**: `{"id", "object", "parent", "text"}`. `parent` es la caja de la que sale (vacío en el primer nivel).

**Pregunta de evaluación** (`data/evals/*.jsonl`): `{"id", "query", "object", "category", "difficulty", "answerable", "relevant_paths"}`.

**Respuesta del modelo** (esquema JSON fijo): `{"found", "answer", "field_paths", "cited_chunks"}`.

## Configuración

Todos los parámetros están en `configs/`. El código no tiene valores escondidos: lo que no está en la configuración no se puede cambiar.

| Fichero | Para qué |
|---|---|
| `configs/default.yaml` | Configuración final medida (LM Studio con `google/gemma-4-e4b`). Se usa por defecto |
| `configs/ollama.yaml` | Igual, pero con Ollama y `qwen2.5-coder:7b`. La usan los scripts de `scripts/windows/` |
| `configs/experiments.yaml` | Variantes de los experimentos: cada una cambia una cosa respecto a `default.yaml` |

Para usar otra configuración: `RAG_CONFIG=configs/ollama.yaml`. Para cambiar la carpeta de salida (por defecto `out/`): `RAG_OUT=...`.

| Sección | Clave | Valor final | Qué controla |
|---|---|---|---|
| `data` | `spec` | `data/raw/spec3.json` | la spec de Stripe |
| | `objects` | `data/objects_16.txt` | objetos del corpus (escala: `data/objects_50.txt`) |
| `extract` | `open_active_objects` | `false` | si se copian dentro de un objeto otros objetos de la lista |
| `chunker` | `strategy` | `hierarchical` | `hierarchical` o `fixed` |
| | `max_tokens` | `512` | tamaño máximo de un fragmento |
| | `overlap` | `64` | solape entre fragmentos (solo `fixed`) |
| | `header` | `true` | cabecera con la ruta |
| `embedding` | `model` | `BAAI/bge-small-en-v1.5` | modelo de embeddings (y su tokenizador para contar tokens) |
| | `query_prefix` | `""` | texto antepuesto a cada pregunta |
| `retrieval` | `method` | `semantic` | `semantic`, `lexical` o `hybrid` |
| | `object_filter` | `true` | priorizar los objetos que nombra la pregunta |
| | `top_k` | `5` | fragmentos que recibe el modelo |
| | `rerank` / `reranker_model` | `false` / `BAAI/bge-reranker-base` | reordenar con un cross-encoder |
| `llm` | `backend` / `base_url` | `openai` / `http://localhost:1234/v1` | servidor del modelo (`openai` para LM Studio, u `ollama`) |
| | `model` | `google/gemma-4-e4b` | modelo de lenguaje |
| | `temperature` / `seed` / `num_ctx` / `timeout` | `0` / `0` / `8192` / `300` | generación |

Los comandos `retrieve`, `generate`, `eval_retrieval` y `eval_generation` comparten las mismas opciones de búsqueda (`--method`, `--object-filter/--no-object-filter`, `--rerank/--no-rerank`, `--top-k`), definidas una sola vez en `retrieve.add_retrieval_args`. Si no se indican, toman el valor de la configuración.

## Evaluación

| Conjunto | Fichero | Para qué |
|---|---|---|
| Desarrollo | `dev_charge.jsonl` + `dev_objects.jsonl` (60) | decidir qué cambios se quedan |
| Control | `control.jsonl` (120, sintético) | comprobar una vez que las decisiones generalizan |
| Test | `test.jsonl` (70, escrito sin ver el sistema) | resultado final, evaluado una sola vez |

| Script (`scripts/`) | Qué hace |
|---|---|
| `experiment.py` + `experiments.yaml` | Ejecuta cada variante (una sola diferencia cada vez) y compara con intervalos de confianza |
| `heldout.py` | Evalúa la búsqueda en control, test y test2 (sección `heldout` de `experiments.yaml`) |
| `failures.py` | Lista las preguntas falladas de una variante y qué se recuperó en su lugar |
| `make_control_set.py` | Genera el conjunto de control |
| `judge.py` | Corrección de las respuestas con un juez a ciegas: `prepare` y `score` |
| `error_taxonomy.py` | Clasifica cada fallo del test por fase (búsqueda o generación) y tipo |

Los resultados y su interpretación están en [`experimentos.md`](experimentos.md).

## Mapa del repositorio

```
configs/
  default.yaml                     configuración final
  ollama.yaml                      la misma con Ollama (Windows)
  experiments.yaml                 variantes de los experimentos
data/
  raw/spec3.json                   fuente: spec OpenAPI de Stripe
  objects_16.txt                   objetos del corpus principal
  objects_50.txt                   objetos del experimento de escala
  evals/                           preguntas: dev_charge, dev_objects, control, test, test2
src/rag/                           el sistema (9 módulos)
scripts/                           experimentos y evaluación
  windows/                         scripts de PowerShell
tests/                             42 tests
docs/
  arquitectura.md                  este documento
  experimentos.md                  diario de experimentos y resultados
  windows.md                       guía para Windows
out/                               todo lo generado (no va a git; se regenera)
```
