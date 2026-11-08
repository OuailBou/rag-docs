# 4. Diseño e implementación

Este capítulo describe el sistema construido: los requisitos que guiaron su diseño, la arquitectura general y cada una de sus etapas, desde la extracción de la especificación OpenAPI hasta la interfaz web, junto con las herramientas de evaluación y las medidas de calidad del código. El código fuente está en `src/rag/` (nueve módulos), los scripts de evaluación en `scripts/` y las pruebas en `tests/`.

## 4.1. Requisitos

### 4.1.1. Requisitos funcionales

| Id | Requisito |
|---|---|
| RF1 | Extraer de la especificación OpenAPI de Stripe los campos de una lista configurable de objetos, con su ruta, tipo, nulabilidad, valores posibles y descripción |
| RF2 | Trocear esa información en fragmentos aptos para un modelo de embeddings, con una estrategia jerárquica y otra fija de referencia |
| RF3 | Construir un índice semántico y otro literal sobre los fragmentos |
| RF4 | Recuperar los fragmentos más relevantes para una pregunta en lenguaje natural, con búsqueda semántica, literal o híbrida, y con filtro por objeto y reordenación opcionales |
| RF5 | Generar una respuesta con un modelo de lenguaje local que cite los campos y los fragmentos que la respaldan, o que indique que la respuesta no está en la documentación |
| RF6 | Verificar automáticamente cada campo citado contra la especificación y contra los fragmentos recibidos |
| RF7 | Evaluar la búsqueda y la generación con conjuntos de preguntas y comparar variantes con intervalos de confianza |
| RF8 | Ofrecer una interfaz web que muestre la respuesta, su verificación y los fragmentos usados |

### 4.1.2. Requisitos no funcionales

| Id | Requisito | Cómo se cumple |
|---|---|---|
| RNF1 | **Local** | Embeddings, índices y modelo de lenguaje se ejecutan en la máquina del usuario; no se envía nada a servicios externos |
| RNF2 | **Determinista** | Búsqueda exacta, temperatura 0 y semilla 0 en el modelo, semillas fijas en los remuestreos y en los muestreos |
| RNF3 | **Reproducible** | Cada etapa lee de disco lo que deja la anterior; versiones fijadas en `uv.lock`; los experimentos se repiten con un comando |
| RNF4 | **Configurable** | Todos los parámetros están en `configs/*.yaml`; no hay valores escondidos en el código |
| RNF5 | **Verificable** | Respuesta en un esquema JSON fijo y verificador de rutas; pruebas automáticas de cada módulo |
| RNF6 | **Portable** | Funciona en macOS (LM Studio) y Windows (Ollama) con la misma base de código |

## 4.2. Arquitectura general

El sistema es una **cadena de cinco etapas** (figura 4.1). Cada una es un módulo de Python que se puede ejecutar por separado con `python -m rag.<etapa>`, lee de disco el resultado de la anterior y escribe el suyo en la carpeta de salida (`out/` por defecto, o la que indique la variable de entorno `RAG_OUT`).

> **Figura 4.1** — Arquitectura del sistema: flujo extract → chunk → index → retrieve → generate y ficheros intermedios (pendiente de dibujar).

```
data/raw/spec3.json          especificación OpenAPI de Stripe (2026-09-30.endive)
        │  1. extract   recorre los objetos de data/objects_16.txt siguiendo los $ref
        ▼
out/table.parquet            una fila por campo
        │  2. chunk     agrupa los campos en fragmentos (troceado jerárquico)
        ▼
out/chunks.jsonl             un fragmento por línea: id, objeto, caja, texto
        │  3. index     embeddings (bge-small) + índice literal (SQLite FTS5)
        ▼
out/rag.faiss  out/rag.sqlite3  out/rag.metadata.jsonl
        │  4. retrieve  pregunta → 5 fragmentos (+ filtro por objeto)
        ▼
fragmentos
        │  5. generate  el modelo responde en JSON; el verificador comprueba cada campo
        ▼
respuesta + verificación     (rag.app la muestra en el navegador)
```

Las tres primeras etapas se ejecutan una vez para construir el índice; las dos últimas, en cada pregunta. Los módulos auxiliares son `common.py` (configuración, rutas, formato de un campo, cabecera y recuento de tokens), `eval_retrieval.py` y `eval_generation.py` (evaluación) y `app.py` (interfaz). La separación en ficheros intermedios tiene tres ventajas: cada etapa se puede inspeccionar y probar por separado, los experimentos pueden reconstruir solo lo que cambia (sección 4.8.5) y un error en una etapa no obliga a repetir las anteriores.

## 4.3. Corpus y extracción desde OpenAPI

### 4.3.1. Fuente y selección de objetos

La fuente es el fichero `spec3.json` del repositorio oficial de Stripe [StripeOpenAPI], en la versión `2026-09-30.endive`, que sigue el estándar OpenAPI 3 [OpenAPI]. Los objetos de la API están en `components.schemas`; los que tienen documento propio en la referencia de Stripe llevan la marca `x-resourceId`, y se pueden listar con `python -m rag.extract --list`. El corpus principal lo forman los 16 objetos de `data/objects_16.txt` (`charge`, `customer`, `payment_intent`, `payment_method`, `refund`, `invoice`, `subscription`, `product`, `price`, `dispute`, `payout`, `balance_transaction`, `setup_intent`, `checkout.session`, `coupon` y `payment_link`); para el experimento de escala se usa `data/objects_50.txt`, con 50 objetos.

### 4.3.2. Recorrido del árbol: `extract.walk`

Cada objeto es un árbol: sus propiedades pueden ser valores simples o referencias (`$ref`) a otros esquemas, que a su vez tienen propiedades. La función `walk` recorre ese árbol en profundidad y añade una fila por campo. Para cada propiedad, `classify` decide de qué tipo es y, por tanto, si hay que entrar en ella:

```python
def classify(p):
    if "anyOf" in p:
        if any(obj.get("type") == "string" for obj in p["anyOf"]):
            return "expandable", None
        refs = [obj["$ref"] for obj in p["anyOf"] if "$ref" in obj]
        return ("box", refs[0]) if refs else ("value", None)
    if "$ref" in p:
        return "box", p["$ref"]
    if p.get("type") == "array":
        return "list", None
    if p.get("type") == "object" and p.get("properties"):
        return "inline", None
    return "value", None
```

**Tabla 4.1.** Clasificación de propiedades en `extract`.

| Clase | Cuándo | Qué se hace |
|---|---|---|
| Valor (`value`) | `string`, `integer`, `boolean`, `enum`... | Una fila; no hay hijos |
| Caja (`box`) | `$ref`, o `anyOf` con un solo `$ref` y sin alternativa `string` | Una fila de tipo `object` y se baja a sus campos |
| Caja en línea (`inline`) | `object` con `properties` definidas en el sitio | Igual que una caja |
| Expandible (`expandable`) | `anyOf` con una alternativa `string`: en Stripe, un identificador que se puede expandir al objeto completo (p. ej. `customer`) | Una fila de tipo `string` marcada como `expandable`; **no** se baja |
| Lista (`list`) | `array` | Una fila (`array of objects`, `array of strings`...) y se baja a lo que hay en `items` si es una caja |

Que los expandibles no se abran es una decisión importante: un `charge` tiene un campo `customer` que puede expandirse al objeto `customer` completo, pero ese objeto ya tiene su propio documento y copiarlo dentro del cargo solo añadiría ruido. En las listas, la ruta de los hijos no lleva índice: los elementos de `tax_ids.data` producen rutas como `tax_ids.data.type`.

**Ciclos.** La especificación tiene referencias circulares (un esquema que, a través de otros, vuelve a referirse a sí mismo). Para evitar recorridos infinitos, `walk` lleva un conjunto `stack` con los esquemas del camino actual, y no entra en una caja que ya esté en él. Las cajas en línea se identifican por su ruta.

**No copiar objetos de la lista.** Algunos objetos contienen, a través de cajas (no de expandibles), copias completas de otros objetos del corpus: el error de un `payment_intent` (`last_payment_error`) contiene un `setup_intent` entero. Con la opción `extract.open_active_objects: false`, el extractor recibe en `skip` el resto de objetos de la lista activa y no entra en ellos; su línea se mantiene, pero sus campos solo aparecen en el documento propio. Esta es la diferencia entre las variantes E0 (con copias) y E1 (sin copias), que mejoró `Recall@5` en +0,125 en desarrollo (sección de resultados).

```python
if child is not None and child.get("properties") and key not in stack and key not in skip:
    walk(schemas, child, obj, title, path, depth + 1, stack | {key}, rows, skip)
```

La descripción se limpia de enlaces Markdown y espacios repetidos (`common.clean`); si la propiedad no tiene descripción propia, se toma la del esquema al que apunta.

### 4.3.3. Tabla resultante

El resultado es `out/table.parquet`, con una fila por campo. Con los 16 objetos, la especificación produce **3.535 filas**.

**Tabla 4.2.** Columnas de `table.parquet`.

| Columna | Contenido |
|---|---|
| `object` | Objeto al que pertenece el campo (`charge`) |
| `title` | Nombre legible del objeto (`Charge`, `Checkout Session`) |
| `path` | Ruta relativa al objeto (`payment_method_details.ach_debit.country`) |
| `name` | Último componente de la ruta (`country`) |
| `depth` | Profundidad (0 en el primer nivel) |
| `type` | `string`, `integer`, `enum`, `object`, `array of objects`... |
| `nullable` | Si el campo puede ser nulo |
| `expandable` | Si es un identificador expandible |
| `enum` | Valores posibles, separados por `\|` |
| `description` | Descripción limpia |
| `has_children` | Si hay filas cuya ruta empieza por la de este campo (se calcula al final) |

Esta tabla es la **fuente de verdad** del sistema: de ella salen los fragmentos, el verificador comprueba contra ella las rutas citadas, el conjunto de control se genera a partir de ella y el test se escribió usándola como referencia.

### 4.3.4. Validación contra la documentación publicada

Para comprobar que la extracción no pierde ni inventa campos, se comparó el objeto `charge` extraído de la especificación con la referencia de ese mismo objeto en la documentación web de Stripe, exportada a Markdown en marzo de 2026 (`Charges.md`). La comparación se hizo sobre las rutas de la sección de atributos del documento web y las rutas de la tabla, en una primera versión del extractor que todavía copiaba los objetos anidados (igual que hace la documentación web, que muestra, por ejemplo, los reembolsos dentro del cargo).

| | Documentación web (marzo de 2026) | Especificación (2026-09-30) |
|---|---:|---:|
| Campos de `charge` | 577 | 600 |
| Campos de primer nivel | 45 | 45 |
| Niveles de profundidad | 6 | 6 |
| Campos comunes | 575 | 575 |
| Solo en uno de los dos | 2 | 25 |

Las dos fuentes coinciden en 575 campos, en el primer nivel y en la profundidad. Las diferencias se explican por los seis meses que las separan: los 25 campos que solo están en la especificación corresponden sobre todo a **métodos de pago añadidos después de marzo** (`payment_method_details.bizum`, `scalapay`, `sequra`...), y los 2 que solo están en la web (`payment_method_details.card.wallet.apple_pay.type` y `payment_method_details.crypto.fingerprint`) desaparecieron de la especificación posterior. La extracción, por tanto, reproduce fielmente la documentación publicada.

## 4.4. Troceado jerárquico

### 4.4.1. Motivación

Los modelos de embeddings tienen un límite de entrada: `bge-small-en-v1.5` [Xiao2023] trunca a 512 tokens. Un objeto como `invoice` o `checkout.session` tiene cientos de campos y no cabe en un solo fragmento, así que hay que trocearlo. Un troceado fijo corta el texto cada cierto número de tokens sin tener en cuenta su estructura, y es habitual que un fragmento empiece a mitad de una caja y termine en otra distinta [Smith2024]. La idea del troceado jerárquico es **cortar por el árbol de atributos**, de modo que cada fragmento hable de una sola cosa: un grupo de campos de un mismo nivel o una caja completa.

### 4.4.2. Formato de un campo

Cada campo se escribe con `common.format_field` como una **línea de campo** seguida de su descripción y, si los tiene, sus valores posibles, con una sangría de dos espacios por nivel:

```
- `payment_method_details.ach_debit.country` (string, nullable)
  Two-letter ISO code representing the country the bank account is located in.
```

El formato `` - `ruta` (tipo, nullable, expandable) `` no es solo presentación: la expresión regular `FIELD_LINE` lo reconoce, y es lo que usan la métrica de recuperación (sección 3.4.1), el verificador (sección 4.7) y *small-to-big* (sección 4.4.6) para saber qué campos contiene un fragmento.

### 4.4.3. Algoritmo: `HierChunker.level`

`HierChunker` agrupa las filas por objeto, construye para cada uno un diccionario de hijos por ruta padre y llama a `level("")` sobre el primer nivel. El método `level(parent)` recorre los hijos de una caja con un **presupuesto de 512 tokens** (`chunker.max_tokens`), contados con el propio tokenizador del modelo de embeddings, y aplica cuatro reglas:

1. **Valores sueltos.** Los campos sin hijos se van añadiendo a un grupo. Cuando el siguiente ya no cabe, el grupo se cierra y se emite como fragmento.
2. **Caja que cabe entera.** Si una caja, con todos sus descendientes y su cabecera, cabe en el presupuesto, se emite como un fragmento propio, cuyo `parent` es la ruta de la caja.
3. **Caja que no cabe.** Se deja solo su línea en el grupo del nivel actual y se llama recursivamente a `level` con la ruta de la caja, que se trocea con las mismas reglas.
4. **Campo que solo no cabe.** Si un único campo, con su descripción y sus valores, supera el presupuesto, se parte (sección 4.4.4).

```python
for row in self.kids.get(parent, []):
    if not row["has_children"]:
        add(self.subtree(row, base))
        continue
    box_head = header(self.title, row["path"]) if self.use_header else ""
    box_text = self.compose(box_head, [self.subtree(row, base)])
    if self.tokens(box_text) <= self.budget:
        self.emit(row["path"], box_text)
    else:
        add(format_field(row, base))
        self.level(row["path"])
flush()
```

Con estas reglas, **cada campo aparece exactamente una vez** en el conjunto de fragmentos (las líneas de los campos partidos se repiten en cada trozo, pero el contenido no se duplica). Los fragmentos se guardan en `out/chunks.jsonl` con las claves `id`, `object`, `parent` (la caja de la que sale; vacío en el primer nivel) y `text`. Con los 16 objetos se obtienen **744 fragmentos**, ninguno por encima del límite de 512 tokens; con 50 objetos, 1.103.

**Cabecera.** Cada fragmento empieza con una cabecera (`common.header`) que da la ruta legible y la ruta exacta. Su objetivo es que el embedding sepa de qué objeto y de qué caja es el fragmento: sin ella, campos repetidos en muchas cajas (`country`, `last4`) son indistinguibles. Se puede desactivar con `chunker.header: false` (variante E3).

**Ejemplo real.** El fragmento de la caja `payment_method_details.ach_debit` del objeto `charge`, tal como queda en `out/chunks.jsonl`, es un caso de la regla 2 (caja que cabe entera):

```
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
    Uniquely identifies this particular bank account. You can use this attribute
    to check whether two bank accounts are the same.

  - `payment_method_details.ach_debit.last4` (string, nullable)
    Last four digits of the bank account number.

  - `payment_method_details.ach_debit.routing_number` (string, nullable)
    Routing transit number of the bank account.
```

(Se ha partido una línea larga para la maqueta.) La caja padre `payment_method_details` no cabe entera, así que su línea queda en un grupo del primer nivel y sus subcajas (`ach_debit`, `card`...) se trocean una a una.

### 4.4.4. Campos que no caben: partición por frases, comas y espacios

Algunos campos tienen descripciones o listas de valores enormes. El caso extremo son los tipos de identificador fiscal (`tax_ids.data.type` en `customer`), cuya descripción enumera más de cien códigos (`ad_nrt`, `ae_trn`, `al_tin`...) en una sola frase y cuya lista de valores posibles repite la enumeración. La primera versión del troceador solo partía por frases, y estas listas generaban fragmentos de hasta 1.447 tokens que el embedder truncaba en silencio (iteración 0).

La versión final (`split_long` y `fit`) separa la línea de campo del resto del texto e intenta partir ese resto con una lista de separadores de grano cada vez más fino:

```python
SEPARATORS = [r"(?<=[.!?])\s+", r"(?<=,)\s+", r"\s+"]
```

Primero por final de frase; si una frase sigue sin caber, por comas; y si un trozo entre comas sigue sin caber, por espacios. Cada trozo se emite con la cabecera y **la línea del campo repetida**, para que no pierda el contexto. En el corpus final, `tax_ids.data.type` queda repartido en cuatro fragmentos consecutivos de la caja `tax_ids.data`; el tercero, por ejemplo, empieza así:

```
Customer → tax ids → data
Path: tax_ids.data

- `tax_ids.data.type` (enum)
Note that some legacy tax IDs have type `unknown` Possible values: ad_nrt, ae_trn, al_tin, ...
```

### 4.4.5. Troceado fijo de referencia

Para contrastar la hipótesis H1 se implementó un troceado fijo (`chunker.strategy: fixed`) que representa la práctica habitual. `render_documents` escribe un documento Markdown por objeto con todos sus campos en orden (`# Charge`, `## Attributes` y las líneas de campo con su sangría), que se guarda en `out/docs/`, y lo trocea con el `RecursiveCharacterTextSplitter` de LangChain [LangChain], configurado con el mismo tokenizador, un tamaño de `max_tokens − 2` (para dejar sitio a los tokens especiales) y un solape de 64 tokens (`chunker.overlap`). Los fragmentos no llevan cabecera ni `parent`. Así, la única diferencia con el jerárquico es la forma de cortar: el texto de cada campo, el tokenizador y el presupuesto son los mismos.

### 4.4.6. *Small-to-big* como opción

La taxonomía de errores del test (sección 4.8.4) mostró que el fallo de búsqueda más frecuente era el «cajón de sastre»: un campo de primer nivel como `due_date` comparte grupo con campos sin relación y su significado se diluye en el embedding del grupo. Para atacarlo se implementó la técnica *small-to-big* o *parent document retrieval* [LlamaIndex], activable con `chunker.small_to_big: true` (variante E11).

`field_chunks` crea un **fragmento pequeño por campo** (cabecera con la ruta del campo, línea y descripción) que apunta, mediante `chunk_id`, al primer fragmento grande que contiene la línea de ese campo. Se guardan en `out/fields.jsonl`. La búsqueda semántica se hace sobre los fragmentos de campo, pero al modelo se le entregan los fragmentos grandes (sección 4.6.5). La opción solo es compatible con el troceado jerárquico. Mejoró el orden de la búsqueda en desarrollo pero no las respuestas finales en el segundo test, así que se mantiene desactivada en la configuración final.

## 4.5. Indexado

`rag.index` (`build_index`) construye dos índices a partir de `chunks.jsonl`:

- **Índice semántico.** Los fragmentos se codifican con `sentence-transformers` [Reimers2019] usando `BAAI/bge-small-en-v1.5` [Xiao2023], un modelo de 384 dimensiones elegido por su buen equilibrio entre tamaño y calidad en MTEB [Muennighoff2023] y porque se ejecuta con soltura en CPU. Los vectores se normalizan y se guardan en un índice `faiss.IndexFlatIP` [Johnson2019]: con vectores normalizados, el producto escalar es la similitud coseno, y el índice plano hace una **búsqueda exacta**, sin aproximaciones. Con menos de dos mil fragmentos, la búsqueda exhaustiva es instantánea y garantiza resultados deterministas.
- **Índice literal.** Una base SQLite (`rag.sqlite3`) con una tabla `chunks` y una tabla virtual **FTS5** con contenido externo sobre el texto, que ordena por la función de relevancia BM25 de FTS5 [Robertson2009].

Junto al índice FAISS se escribe `rag.metadata.jsonl`, con los fragmentos en el mismo orden que los vectores, para traducir posiciones del índice en fragmentos. Con *small-to-big*, los vectores y los metadatos son los de los fragmentos de campo, y los fragmentos grandes se guardan aparte en `rag.chunks.jsonl`; el índice literal sigue conteniendo los fragmentos grandes.

## 4.6. Recuperación

El módulo `retrieve.py` implementa cada estrategia como una clase con un método `retrieve(query, top_k)`, de modo que se pueden **componer** envolviendo unas en otras. `build_retriever` monta la combinación pedida:

```python
if rerank:
    retriever = RerankRetriever(retriever)
if object_filter:
    retriever = ObjectFilterRetriever(retriever, {m["object"] for m in load_metadata(metadata_path)})
```

### 4.6.1. Búsqueda semántica

`SemanticRetriever` codifica la pregunta con el mismo modelo, antepone el prefijo de consulta configurado (`embedding.query_prefix`, vacío en la configuración final; la variante E5 prueba el prefijo recomendado por BGE) y devuelve los `top_k` fragmentos más cercanos del índice FAISS.

### 4.6.2. Búsqueda léxica e híbrida

`LexicalRetriever` convierte la pregunta en una consulta FTS5 que une con `OR` todas sus palabras entre comillas y ordena por `rank`. `HybridRetriever` combina ambas búsquedas con **fusión por rango recíproco** (RRF) [Cormack2009]: pide a cada una `max(4·top_k, 20)` candidatos y asigna a cada fragmento la suma de 1/(60 + r) sobre las listas en las que aparece, con r su posición. En los experimentos, la búsqueda literal resultó mucho peor que la semántica en estas preguntas y la fusión empeoró el resultado, por lo que la configuración final usa solo la semántica.

### 4.6.3. Filtro por objeto

Muchas preguntas nombran el objeto por el que preguntan (*"Why can a setup intent be canceled?"*). `ObjectFilterRetriever` aprovecha esa información con un algoritmo sencillo:

1. Pide a la búsqueda base **50 candidatos** (o `top_k`, si es mayor).
2. `mentioned_objects` normaliza la pregunta (minúsculas, solo letras y dígitos, separados por espacios) y busca cada nombre de objeto con `_` y `.` cambiados por espacios, como frase completa y admitiendo plural en `s` o `es`: `setup_intent` se reconoce en *"setup intents"* y `checkout.session` en *"checkout session"*.
3. Si la pregunta nombra algún objeto, **reordena** los candidatos poniendo primero, en su orden original, los de los objetos nombrados, y después el resto.
4. Devuelve los `top_k` primeros.

```python
found = mentioned_objects(query, self.objects)
if found:
    rows = [r for r in rows if r["object"] in found] + [r for r in rows if r["object"] not in found]
return rows[:top_k]
```

El filtro tiene **límites** conocidos: solo reordena, así que no puede rescatar un fragmento que no esté entre los 50 candidatos; no entiende sinónimos (*"bill"* no activa `invoice`); y si la pregunta nombra varios objetos, los prioriza todos por igual, lo que provoca el tipo de error «varios objetos en la pregunta» de la taxonomía. Aun así, mejora el orden de los resultados (`MRR` +0,074 en desarrollo) y hace al sistema robusto al crecimiento del corpus.

### 4.6.4. Reordenación (*reranker*)

`RerankRetriever` pide 20 candidatos a la búsqueda base y los reordena con un *cross-encoder* [Nogueira2019] que puntúa cada pareja (pregunta, fragmento). El modelo es configurable (`retrieval.reranker_model`): se probaron `BAAI/bge-reranker-base` y `cross-encoder/ms-marco-MiniLM-L-6-v2` (variantes E7 y E8). Mejoró `Recall@1` y `MRR`, pero la diferencia no fue concluyente con 56 preguntas, así que está desactivado por defecto. Cuando se combina con el filtro, el reordenamiento se aplica primero y el filtro por objeto después.

### 4.6.5. *Small-to-big* en la búsqueda

Si los metadatos del índice son fragmentos de campo (llevan `chunk_id`), `SemanticRetriever` busca a más profundidad (`max(40·top_k, 200)` campos), recorre los resultados en orden y sustituye cada campo por su fragmento grande, descartando los repetidos, hasta reunir `top_k` fragmentos distintos. Cada resultado conserva en `matched_field` el campo que lo hizo aparecer.

### 4.6.6. Opciones compartidas

Los cuatro comandos que buscan (`retrieve`, `generate`, `eval_retrieval` y `eval_generation`) comparten las mismas opciones, definidas una sola vez en `add_retrieval_args`. Sus valores por defecto salen de la sección `retrieval` de la configuración, de modo que la línea de comandos solo sobrescribe lo que se indique:

```bash
uv run python -m rag.retrieve "Why can a setup intent be canceled?" \
    --method semantic --object-filter --no-rerank --top-k 5
```

La función `describe` traduce las opciones a un nombre de método (`semantic+objfilter`), que es el que usan los ficheros de resultados y `configs/experiments.yaml`.

## 4.7. Generación y verificación

### 4.7.1. Prompt y contexto

`generate.py` construye dos mensajes. El **mensaje de sistema** fija las reglas: responder solo con la información de los fragmentos, sin conocimiento externo; si la respuesta no está, poner `found` a falso; listar en `field_paths` las rutas exactas tal como aparecen entre comillas invertidas y sin el nombre del objeto; citar en `cited_chunks` los números de los fragmentos que respaldan la respuesta; y limitar la respuesta a una a tres frases. El **mensaje de usuario** contiene los fragmentos numerados, cada uno con su objeto, seguidos de la pregunta:

```
Documentation excerpts:

[1] (object: charge)
Charge → payment method details → ach debit
...

Question: What country is the bank account of an ACH debit charge in?   (ejemplo ilustrativo)
```

La numeración permite que el modelo cite y que el verificador compruebe las citas. Se entregan 5 fragmentos, una cantidad suficiente para cubrir la mayoría de preguntas sin alargar el contexto en exceso, lo que podría perjudicar el uso que el modelo hace de la información intermedia [Liu2024].

### 4.7.2. Esquema de salida

La respuesta debe ajustarse a un esquema JSON fijo (`ANSWER_SCHEMA`) con cuatro claves obligatorias:

```json
{"found": true, "answer": "...", "field_paths": ["payment_method_details.ach_debit.country"],
 "cited_chunks": [1]}
```

El esquema se pasa al servidor para que restrinja la generación (salida estructurada), lo que en la práctica elimina los errores de formato: no hubo ninguno en las evaluaciones.

### 4.7.3. Backends

El sistema admite dos servidores locales, elegidos con `llm.backend`:

- **`openai`**: cualquier servidor con la API de OpenAI, como LM Studio (el usado en el Mac), llama.cpp [llamacpp] o vLLM. `OpenAICompatibleClient` hace una petición `POST /chat/completions` con `httpx`, envía `temperature` y `seed` y pide `response_format` de tipo `json_schema` estricto. El tamaño de contexto no se envía: se fija al cargar el modelo (`lms load ... --context-length 8192`).
- **`ollama`**: el cliente oficial de Ollama, que recibe el esquema en `format` y las opciones `temperature`, `seed` y `num_ctx`.

Los dos clientes exponen la misma interfaz (`chat`), así que el resto del código no depende del servidor. `call_llm` traduce los errores habituales a mensajes con instrucciones concretas: servidor no disponible, modelo no cargado (error 404) o tiempo de espera agotado (`llm.timeout`, 300 s).

### 4.7.4. Lectura de la respuesta: `parse_answer`

Aunque el esquema se impone en el servidor, `parse_answer` no confía en él. Extrae el primer objeto JSON del texto con una expresión regular, comprueba que tiene las cuatro claves y **normaliza los tipos**: `found` acepta booleanos o cadenas como `"true"`; `field_paths` se convierte en lista de cadenas sin vacíos; `cited_chunks` se convierte en lista de enteros sin repeticiones, extrayendo los dígitos si el modelo escribe `"[1]"`. Si algo falla, devuelve una respuesta de «no encontrado» con `parse_error: true` y el texto original, de modo que un error de formato nunca se cuenta como acierto.

### 4.7.5. Verificación: `verify`

El verificador es la pieza que permite medir la alucinación sin intervención humana. Para cada respuesta:

1. **Normaliza las rutas** (`normalize_path`): quita comillas invertidas y espacios y, si la ruta empieza por el nombre de un objeto seguido de un punto y el resto es una ruta conocida de ese objeto, quita el prefijo (`charge.amount` → `amount`).
2. Clasifica cada ruta citada:
   - **inventada** si no existe en ningún objeto de la tabla: es una alucinación;
   - **no respaldada** si existe pero ni ella ni ningún descendiente aparece como línea de campo en los fragmentos que recibió el modelo;
3. Marca como **citas no válidas** los números de fragmento fuera del rango 1..n.
4. Considera la respuesta **fundamentada** si el modelo dice haberla encontrado y no hay ningún problema de los anteriores.

```python
invented = [p for p in paths if p not in all_known]
unsupported = [p for p in paths if p in all_known and not is_supported(p, chunk_paths)]
invalid = [n for n in result.get("cited_chunks", []) if not 1 <= n <= len(chunks)]
grounded = bool(result.get("found")) and not (invented or unsupported or invalid)
```

Como compara la ruta completa, el verificador detecta alucinaciones sutiles: en desarrollo, el modelo citó `payment_method_details.reader`, un campo que existe pero un nivel más abajo (dentro de `card_present`), y se marcó como inventado. Una limitación es que una ruta se considera conocida si existe en *cualquier* objeto, no necesariamente en el de la pregunta.

## 4.8. Evaluación automática

### 4.8.1. Búsqueda: `eval_retrieval`

`rag.eval_retrieval` recupera 5 fragmentos para cada pregunta del conjunto y aplica la definición de acierto de la sección 3.4.1. `chunk_paths` devuelve las líneas de campo de un fragmento solo si es del objeto esperado (si no, un conjunto vacío que ocupa su puesto), y `matched` aplica la regla de los hijos:

```python
def matched(expected, paths):
    return sorted(e for e in expected if any(p == e or p.startswith(e + ".") for p in paths))
```

Con esas coincidencias calcula `Recall@1/3/5`, `nDCG@1/3/5` y `MRR` por pregunta, excluye las preguntas sin respuesta y las de la categoría `metadata`, y resume la media global, por categoría y el intervalo de confianza de `Recall@5` con `bootstrap_ci` (5.000 remuestreos, semilla 0). El informe completo, con lo recuperado en cada pregunta, se guarda en `out/results/<conjunto>_<método>.json`.

### 4.8.2. Generación: `eval_generation`

`rag.eval_generation` recorre las preguntas, recupera los fragmentos, llama al modelo, verifica la respuesta y calcula para cada una el acierto del campo, si la búsqueda acertó (reutilizando `evaluate_question`), la abstención, la alucinación y la latencia. Como cada pregunta tarda decenas de segundos, la evaluación **se puede interrumpir y reanudar**:

- cada resultado se añade al fichero `out/results/gen_<conjunto>.jsonl` en cuanto se obtiene, y se vuelca a disco;
- cada registro lleva la identificación de la ejecución (modelo, método de búsqueda y `top_k`), de modo que al volver a lanzar el mismo comando se saltan las preguntas ya respondidas **con esa misma configuración**, y en el mismo fichero pueden convivir resultados de varios modelos;
- si la ejecución se cortó a mitad de una línea, la línea truncada se ignora al leer y se añade un salto de línea antes de seguir escribiendo.

El resumen (`gen_<conjunto>_summary.json`) se recalcula siempre con los últimos registros de la ejecución e incluye intervalos de confianza para cada tasa.

### 4.8.3. Juez a ciegas: `judge.py`

`scripts/judge.py` implementa la evaluación de la corrección descrita en la sección 3.4.3 en dos pasos:

- `prepare` lee las respuestas del test (`out/results/gen_test.jsonl`), las **baraja con una semilla fija** (2026), les asigna identificadores neutros (`J001`...) y escribe `out/judge/items.jsonl` con, para cada una, la pregunta, la respuesta, si el modelo dijo haberla encontrado, las filas de la especificación de los campos esperados y de los citados y las rutas inventadas. El nombre del modelo solo se guarda en `mapping.json`, que el juez no lee.
- `score` lee los veredictos (`judgments.jsonl`, con `item_id`, `verdict` y `reason`), comprueba que no falte ninguno y que todos sean válidos, deshace la mezcla y calcula, por modelo, la proporción de cada veredicto con intervalos de confianza, los aciertos por categoría y el trato de las preguntas sin respuesta. Añade una comprobación automática para las preguntas de valores posibles: qué fracción de los valores del `enum` aparece en el texto de la respuesta.

### 4.8.4. Taxonomía de errores: `error_taxonomy.py`

`scripts/error_taxonomy.py` toma cada pregunta con respuesta del test cuyo veredicto no es *correcta* y la clasifica con reglas deterministas, aplicadas en este orden:

| Fase | Tipo | Regla |
|---|---|---|
| Búsqueda | Varios objetos en la pregunta | La pregunta nombra varios objetos, entre ellos el correcto, y el primer fragmento es de otro |
| | Objeto equivocado | Ninguno de los 5 fragmentos es del objeto correcto |
| | Campo gemelo en otra caja | Un fragmento del objeto correcto contiene un campo con el mismo nombre final en otra ruta |
| | Cajón de sastre | El mejor fragmento que contiene la respuesta es un grupo de primer nivel (`parent` vacío) |
| | Caja equivocada | Cualquier otro fallo de búsqueda |
| Generación | Campo inventado | El verificador detectó una ruta inexistente |
| | Falsa abstención | El modelo dice no encontrarlo o el juez dice que no responde |
| | Incompleta | Veredicto *parcial* |
| | Campo equivocado del contexto | El resto |

Para los fallos de búsqueda calcula además el **puesto del fragmento correcto entre los de su propio objeto**, ordenando por similitud todos los fragmentos de ese objeto. Ese dato mostró que en 5 de los 8 fallos de búsqueda con Gemma el fragmento correcto estaba entre los 10 primeros de su objeto.

### 4.8.5. Experimentos: `experiment.py` y `heldout.py`

`scripts/experiment.py` ejecuta las variantes de `configs/experiments.yaml`. Para cada una, mezcla su configuración con la base (`merge`, recursivo por secciones), escribe el resultado en `out/experiments/<variante>/config.yaml` y lanza los módulos como subprocesos con `RAG_OUT` y `RAG_CONFIG` apuntando a esa carpeta, de modo que cada variante tiene sus propios ficheros intermedios y no interfiere con las demás ni con `out/`.

Construir un índice cuesta minutos, así que el script tiene una **caché de índices**: compara las claves que afectan al índice (`data`, `extract`, `chunker` y el modelo de embeddings) con las de la configuración anterior de la variante y, si coinciden y el índice existe, solo evalúa. Cambiar el prefijo de consulta o el *reranker*, que solo afectan a la búsqueda, no obliga a reindexar.

```python
def index_settings(cfg):
    return {"data": cfg["data"], "extract": cfg["extract"], "chunker": cfg["chunker"],
            "embedding_model": cfg["embedding"]["model"]}
```

Después evalúa cada método sobre los dos conjuntos de desarrollo, junta las puntuaciones por pregunta y genera `summary.md` con la tabla de variantes y las comparaciones de la sección `compare`, cada una con la diferencia media emparejada de `Recall@5` y `MRR` y su intervalo de confianza. Un script complementario, `failures.py`, lista las preguntas falladas de una variante y qué se recuperó en su lugar, y es la herramienta del paso de análisis de fallos del ciclo iterativo.

`scripts/heldout.py` aplica el mismo mecanismo a los conjuntos reservados. Lee la sección `heldout` de `configs/experiments.yaml`, que indica para cada conjunto (control, test y test2) qué ejecuciones evaluar, qué parejas comparar y cuál es la configuración final que se desglosa por categoría. Reutiliza los índices de cada variante (si alguno falta, lo construye con `experiment.py`) y escribe una tabla por conjunto en `out/experiments/heldout_<conjunto>.md`. Las cifras de control, test y test2 del capítulo 5 salen de estas tablas.

## 4.9. Interfaz web

`rag.app` ofrece una interfaz web local construida con Gradio (figura 4.2), que se abre en `http://127.0.0.1:7860`. La columna izquierda tiene el cuadro de la pregunta, una casilla para activar el filtro por objeto, un control deslizante para el número de fragmentos (de 1 a 10) y cuatro preguntas de ejemplo; debajo aparece la respuesta. La columna derecha muestra los fragmentos recuperados.

> **Figura 4.2** — Captura de la interfaz web con una respuesta verificada y sus fragmentos (pendiente de capturar).

La respuesta incluye el texto del modelo, el resultado de la verificación («fundamentada» o «no fundamentada», con los campos inventados, no respaldados o las citas no válidas), los campos citados, la latencia y el modelo. Cada fragmento se muestra en un bloque desplegable con su número, objeto, caja y puntuación; los que el modelo cita aparecen marcados como «citado» y desplegados.

La interfaz está pensada para **no caerse**: una pregunta vacía devuelve un aviso; un error en la recuperación se muestra como mensaje; y si falla la generación (por ejemplo, porque LM Studio no está en marcha), se muestra el error con la instrucción para resolverlo y **los fragmentos recuperados igualmente**, porque siguen siendo útiles. El índice, el filtro y la tabla de rutas conocidas se cargan una sola vez y se guardan en una caché protegida por un cerrojo, ya que Gradio puede atender peticiones en paralelo. Por coherencia con el requisito de funcionamiento local, la interfaz se lanza sin enlace público, sin analítica y sin el servidor MCP de Gradio.

## 4.10. Configuración

Todos los parámetros están en `configs/default.yaml`, agrupados por etapa. Otra configuración completa se elige con la variable `RAG_CONFIG` (por ejemplo, `configs/ollama.yaml`, idéntica salvo la sección `llm`) y la carpeta de salida con `RAG_OUT`. Si se cambia algo de `data`, `extract`, `chunker` o `embedding`, hay que volver a ejecutar `extract`, `chunk` e `index`.

**Tabla 4.3.** Claves de configuración y valores finales.

| Sección | Clave | Valor final | Qué controla |
|---|---|---|---|
| `data` | `spec` | `data/raw/spec3.json` | Especificación de Stripe |
| | `objects` | `data/objects_16.txt` | Objetos del corpus (escala: `data/objects_50.txt`) |
| `extract` | `open_active_objects` | `false` | Si se copian dentro de un objeto los otros objetos de la lista |
| `chunker` | `strategy` | `hierarchical` | `hierarchical` o `fixed` |
| | `max_tokens` | `512` | Tamaño máximo de un fragmento |
| | `overlap` | `64` | Solape entre fragmentos (solo `fixed`) |
| | `header` | `true` | Cabecera con la ruta |
| | `small_to_big` | `false` | Buscar por fragmentos de campo y devolver el fragmento grande |
| `embedding` | `model` | `BAAI/bge-small-en-v1.5` | Modelo de embeddings y tokenizador para contar tokens |
| | `query_prefix` | `""` | Texto antepuesto a cada pregunta |
| `retrieval` | `method` | `semantic` | `semantic`, `lexical` o `hybrid` |
| | `object_filter` | `true` | Priorizar los objetos que nombra la pregunta |
| | `top_k` | `5` | Fragmentos que recibe el modelo |
| | `rerank` | `false` | Reordenar con un *cross-encoder* |
| | `reranker_model` | `BAAI/bge-reranker-base` | Modelo de reordenación |
| `llm` | `backend` | `openai` | `openai` (LM Studio) u `ollama` |
| | `base_url` | `http://localhost:1234/v1` | Dirección del servidor (solo `openai`) |
| | `model` | `google/gemma-4-e4b` | Modelo de lenguaje (`qwen2.5-coder:7b` en `ollama.yaml`) |
| | `temperature` / `seed` | `0` / `0` | Generación determinista |
| | `num_ctx` | `8192` | Tamaño de contexto (solo lo envía el cliente de Ollama) |
| | `timeout` | `300` | Tiempo máximo de espera, en segundos |

## 4.11. Calidad

### 4.11.1. Pruebas automáticas

El proyecto tiene **42 pruebas** con `pytest` (39 funciones, una de ellas parametrizada con cuatro casos), que se ejecutan con `uv run pytest`. Las pruebas no dependen de un servidor de modelos: el modelo de lenguaje, el recuperador y el verificador se sustituyen por dobles de prueba cuando hace falta.

**Tabla 4.4.** Pruebas por módulo.

| Fichero | Pruebas | Qué cubren |
|---|---:|---|
| `test_generate.py` | 19 | Numeración del contexto; `parse_answer` con JSON válido y cuatro casos inválidos; `verify` con respuestas fundamentadas, rutas inventadas, no respaldadas, normalización de prefijos y comillas, hijos que respaldan una caja, citas no válidas y respuestas «no encontrado»; errores de conexión y de modelo ausente; cliente compatible con OpenAI (envío del esquema y lectura de la respuesta) y elección del backend |
| `test_eval_generation.py` | 8 | Regla de acierto del campo; separación entre fallos de búsqueda y de generación; preguntas sin respuesta y `metadata`; reanudación por modelo; líneas truncadas; salida limpia si el servidor no responde; una abstención nunca es acierto |
| `test_app.py` | 7 | Formato de la respuesta y de la verificación; marcado de fragmentos citados; pregunta vacía; errores de JSON; fragmentos visibles aunque el servidor esté caído |
| `test_eval_retrieval.py` | 5 | Métricas y resumen; los fragmentos de otros objetos conservan su puesto y no cuentan; las menciones en descripciones no cuentan; exclusión de `metadata`; detección de objetos nombrados |
| `test_retrieval.py` | 2 | Construcción del índice y búsqueda de extremo a extremo; *small-to-big* busca por campo y devuelve su fragmento |
| `test_imports.py` | 1 | El paquete se importa y encuentra la raíz del proyecto |

Varias pruebas fijan precisamente los defectos corregidos en la iteración 0 (posiciones de otros objetos y menciones en descripciones), para que no vuelvan a aparecer.

### 4.11.2. Análisis estático

El código se revisa con `ruff`, configurado en `pyproject.toml` con una longitud de línea de 120 caracteres. Las pocas excepciones a sus reglas están anotadas en el propio código con su justificación; por ejemplo, la captura genérica de excepciones de la interfaz (`# noqa: BLE001 (la interfaz no debe caerse)`).

### 4.11.3. Portabilidad a Windows

El sistema se desarrolló en macOS, pero debía funcionar también en un portátil con Windows 11 y GPU NVIDIA. Se resolvieron tres tipos de problemas:

- **Codificación.** Windows usa `cp1252` por defecto y los fragmentos contienen caracteres como `→`. Todo el código abre los ficheros con `encoding="utf-8"` explícito, los comandos reconfiguran la salida estándar a UTF-8, los ficheros JSONL se escriben con fin de línea LF y los scripts de PowerShell activan `PYTHONUTF8=1`. El propio fichero común de PowerShell se mantiene en ASCII.
- **Scripts de PowerShell** (`scripts/windows/`): `setup.ps1` comprueba `uv`, instala las dependencias, arranca Ollama, descarga el modelo de `configs/ollama.yaml`, construye el índice y pasa las pruebas, deteniéndose con un mensaje claro si algo falla; `run_app.ps1`, `eval_generation.ps1` y `run_experiments.ps1` lanzan la interfaz, la evaluación de generación y los experimentos. Todos usan `configs/ollama.yaml` salvo que `RAG_CONFIG` indique otra cosa. La guía `docs/windows.md` incluye una lista de comprobación con los números que deben coincidir con los del Mac (3.535 filas, 744 fragmentos, 384 dimensiones y las métricas de desarrollo).
- **Choque entre `faiss` y `torch` en macOS.** Ambas bibliotecas cargan su propia versión de la biblioteca OpenMP, y en macOS el proceso puede terminar sin mensaje al buscar. Se evita con `OMP_NUM_THREADS=1` y `KMP_DUPLICATE_LIB_OK=TRUE`; `experiment.py` las fija automáticamente en macOS y el README indica cómo hacerlo a mano. En Windows, el mismo aviso de OpenMP se resuelve con la segunda variable.
