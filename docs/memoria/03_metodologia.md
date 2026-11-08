# 3. Metodología

## 3.1. Enfoque

Este trabajo sigue un enfoque de **ingeniería con evaluación experimental**. El objetivo no es solo construir un sistema de generación aumentada por recuperación (RAG, *Retrieval-Augmented Generation*) [Lewis2020] [Gao2023] que responda preguntas sobre la API de Stripe, sino poder **justificar cada decisión de diseño con una medida**. Por eso el sistema se ha construido como una cadena de etapas independientes (extracción, troceado, indexado, recuperación y generación), cada una configurable desde un único fichero, y se ha acompañado desde el principio de un banco de evaluación automática.

### 3.1.1. Hipótesis

La hipótesis central del trabajo es la siguiente:

> **H1.** En una documentación de API estructurada como un árbol de atributos, un troceado *jerárquico* que respeta ese árbol (cada fragmento habla de un objeto o de una caja de campos concreta) recupera mejor el fragmento que responde a la pregunta que un troceado *fijo* por número de tokens, como el que ofrecen por defecto bibliotecas como LangChain [LangChain].

Junto a ella se plantearon hipótesis secundarias, todas sobre la etapa de recuperación salvo la última:

- **H2.** No copiar dentro de un objeto otros objetos que ya tienen documento propio mejora la recuperación, porque evita que las copias compitan con el original.
- **H3.** Una cabecera con la ruta del fragmento (`Charge → payment method details → ach debit`) ayuda al modelo de embeddings a distinguir campos con el mismo nombre en cajas distintas.
- **H4.** Priorizar los fragmentos del objeto que nombra la pregunta mejora el orden de los resultados.
- **H5.** Un modelo de lenguaje local de unos 7.000 millones de parámetros, con temperatura 0 y salida estructurada, responde correctamente cuando recibe el fragmento adecuado, y un verificador automático puede detectar los campos que invente.

La métrica principal para contrastar H1–H4 es `Recall@5` (sección 3.4). Para H5 se usan las métricas de generación de la sección 3.4.2.

### 3.1.2. Ciclo iterativo

El desarrollo se ha organizado en un ciclo que se repite:

1. **Medir** la configuración actual con el conjunto de desarrollo.
2. **Analizar los fallos** pregunta a pregunta: qué se esperaba, qué se recuperó en su lugar y por qué (`scripts/failures.py`, y más adelante `scripts/error_taxonomy.py`).
3. **Proponer un cambio** concreto que ataque la causa observada, expresado como una opción de configuración.
4. **Volver a medir** la variante con el cambio frente a la configuración de referencia y decidir si se queda.

Este ciclo se aplicó también a la propia evaluación. En la iteración 0, antes de comparar variantes, el análisis de fallos reveló tres defectos que distorsionaban los números: el evaluador filtraba los resultados por objeto *antes* de calcular las posiciones (lo que inflaba `Recall@1`, `Recall@3` y `MRR`), la métrica aceptaba como acierto cualquier mención de una ruta entre comillas invertidas dentro de una descripción, y el troceador generaba fragmentos de hasta 1.447 tokens que el modelo de embeddings truncaba en silencio. Se corrigieron los tres antes de medir. En la iteración 1, el análisis de fallos llevó a dos ajustes de la evaluación (sección 3.3.5 y la regla de los hijos de la sección 3.4.1), y en la iteración 4 la taxonomía de errores del test motivó el experimento *small-to-big*.

## 3.2. Diseño experimental

### 3.2.1. Una variable por variante

Cada experimento compara dos configuraciones que **difieren en una sola cosa**. Las variantes están declaradas en `configs/experiments.yaml`: cada una indica solo las claves que cambia, y esas claves se mezclan con `configs/default.yaml` para obtener la configuración completa. Por ejemplo:

```yaml
E2_troceado_fijo:            # troceado de referencia (LangChain)
  config: {chunker: {strategy: fixed}}
  methods: [semantic, semantic+objfilter]
```

El campo `methods` indica con qué métodos de búsqueda se evalúa la variante (`semantic`, `lexical` o `hybrid`, con los modificadores `+objfilter` y `+rerank`). La sección `compare` del mismo fichero lista las parejas que se comparan, siempre de la forma «variante con el cambio − referencia». En total se definieron doce variantes (E0–E11), agrupadas en dos rondas de diseño (E0–E5 y E6–E8), un experimento de escala de 16 a 50 objetos (E9–E10) y la iteración *small-to-big* (E11). La tabla 3.1 resume qué cambia cada una.

**Tabla 3.1.** Variantes experimentales.

| Variante | Cambio respecto a la referencia |
|---|---|
| E0_base | Se copian dentro de un objeto los otros objetos de la lista (`open_active_objects: true`) |
| E1_sin_copias | Configuración final de referencia |
| E2_troceado_fijo | Troceado fijo con LangChain |
| E3_sin_cabecera | Fragmentos sin cabecera de ruta |
| E4_bge_base | Modelo de embeddings `bge-base-en-v1.5` en lugar de `bge-small` |
| E5_bge_base_prefijo | E4 con el prefijo de consulta recomendado por BGE |
| E6_tokens_256 | Presupuesto de 256 tokens por fragmento |
| E7_rerank_minilm | Reordenación con `ms-marco-MiniLM-L-6-v2` |
| E8_rerank_bge | Reordenación con `bge-reranker-base` |
| E9_escala_50 | Corpus de 50 objetos |
| E10_fijo_escala_50 | Corpus de 50 objetos con troceado fijo |
| E11_small_to_big | Búsqueda por fragmentos de campo y entrega del fragmento grande |

### 3.2.2. Reproducibilidad

Todos los experimentos de búsqueda sobre el conjunto de desarrollo se reproducen con un único comando:

```bash
uv run python scripts/experiment.py            # todas las variantes
uv run python scripts/experiment.py E1_sin_copias E2_troceado_fijo
```

El script crea una carpeta por variante en `out/experiments/<variante>/`, escribe allí la configuración mezclada (`config.yaml`), reconstruye el índice si hace falta y evalúa cada método. Al final genera `out/experiments/summary.md` con la tabla de resultados y las comparaciones con sus intervalos de confianza. Las evaluaciones de los conjuntos reservados (control, test y test2, sección 3.3) se reproducen igual con `uv run python scripts/heldout.py`, que lee de la sección `heldout` del mismo fichero qué ejecuciones y comparaciones calcular y escribe `out/experiments/heldout_<conjunto>.md`. La reproducibilidad se apoya además en tres decisiones de diseño:

- **Sin valores escondidos.** Todo parámetro que afecta al resultado está en `configs/`; lo que no está en la configuración no se puede cambiar.
- **Determinismo.** La búsqueda es exacta (sección 4.5), los remuestreos usan semillas fijas y el modelo de lenguaje se invoca con temperatura 0 y semilla 0.
- **Dependencias fijadas.** Las versiones de todas las bibliotecas están fijadas en `uv.lock` y el entorno se recrea con `uv sync`.

## 3.3. Conjuntos de preguntas

Para evitar ajustar el sistema a las mismas preguntas con las que se informa el resultado, se separaron cuatro tipos de conjuntos con funciones distintas. Todos están en `data/evals/` en formato JSONL, con una pregunta por línea y los campos `id`, `query` (en inglés, como la documentación), `object`, `category`, `difficulty`, `answerable` y `relevant_paths` (las rutas de los campos que responden la pregunta, relativas al objeto).

**Tabla 3.2.** Conjuntos de evaluación.

| Conjunto | Fichero | Preguntas | Origen | Uso |
|---|---|---:|---|---|
| Desarrollo | `dev_charge.jsonl` | 30 (2 sin respuesta) | Escritas a mano sobre `charge` | Decidir qué cambios se quedan |
| Desarrollo | `dev_objects.jsonl` | 30 | Escritas a mano, 2 por cada uno de los otros 15 objetos | Decidir qué cambios se quedan |
| Control | `control.jsonl` | 120 | Generadas con plantillas | Comprobar una vez que las decisiones generalizan |
| Test | `test.jsonl` | 70 (10 sin respuesta) | Escritas por un agente de IA sin ver el sistema | Resultado final, una sola evaluación |
| Test 2 | `test2.jsonl` | 70 (10 sin respuesta) | Mismo procedimiento que el test | Evaluación limpia de *small-to-big* |

### 3.3.1. Desarrollo

El conjunto de desarrollo reúne 60 preguntas escritas a mano por el autor. Las 30 de `dev_charge` se centran en el objeto `charge` y cubren varias categorías (campos directos, objetos anidados, valores posibles, preguntas semánticas, búsqueda por ruta, preguntas sin respuesta y preguntas de metadatos). Las 30 de `dev_objects` tienen dos preguntas directas por cada uno de los otros 15 objetos del corpus. Descontando las 2 preguntas sin respuesta y las 2 de la categoría `metadata` (sección 3.3.5), quedan **56 preguntas puntuadas** en las métricas de búsqueda; en generación se evalúan 58, porque las preguntas sin respuesta sí cuentan para medir la abstención. Es el único conjunto con el que se toman decisiones.

### 3.3.2. Control

El conjunto de control (`control.jsonl`) se generó con `scripts/make_control_set.py` **después de cerrar las decisiones de diseño**, a partir de la tabla de campos. El script elige al azar (semilla 2026) 120 campos sin hijos, de profundidad máxima 2, con una descripción de al menos cinco palabras y excluyendo campos genéricos (`object`, `id`, `livemode`, `metadata`). Con cada campo se construye una pregunta de uno de tres tipos, 40 de cada uno:

- `desc`: describe el campo con la primera frase de su descripción, sin nombrarlo (*"On a payment link object, which field holds the specified behavior after the purchase is complete?"*);
- `name`: lo nombra en lenguaje natural (*"What is the enabled in the managed payments of a payment link?"*);
- `ident`: usa la ruta exacta (*"What does `after_completion.type` mean on a payment link?"*).

Es un conjunto **sintético**: mide si las decisiones generalizan a preguntas no vistas, pero no sustituye a preguntas escritas por personas. El script se niega a sobrescribir el fichero si ya existe, para que no se regenere por accidente.

### 3.3.3. Test

El conjunto de test (`test.jsonl`) lo escribió **un agente de IA sin acceso a los fragmentos, a los resultados ni a la ejecución del sistema**, usando como única fuente de verdad la tabla de campos extraída de la especificación. Tiene 60 preguntas con respuesta, repartidas en cuatro categorías de 15 (`test_direct`, `test_descriptive`, `test_nested` y `test_enum`) sobre los 16 objetos, y 10 preguntas sin respuesta: 6 sobre información que no está en la especificación (tarifas, límites, soporte...) y 4 sobre campos que no existen.

### 3.3.4. Segundo test

La iteración 4 (*small-to-big*) se motivó a partir de los fallos del primer test, por lo que ese conjunto ya no servía para evaluarla sin sesgo. Se escribió entonces un **segundo test a ciegas** (`test2.jsonl`), con 70 preguntas nuevas, la misma composición (15 por categoría y 10 sin respuesta) y las mismas reglas de independencia. La decisión sobre *small-to-big* se tomó con el conjunto de desarrollo y el segundo test se evaluó una sola vez.

### 3.3.5. Reglas de uso

- **El test se evalúa una sola vez**, con la configuración ya cerrada. Lo mismo se aplica al conjunto de control y al segundo test. Ningún resultado de estos conjuntos se ha usado para cambiar la configuración final.
- **Reformulación de preguntas ambiguas.** Las preguntas de `dev_charge` se escribieron cuando el corpus solo contenía `charge`. Al pasar a 16 objetos, algunas dejaron de tener una única respuesta válida: *"How can I return money to a customer?"* también se responde con el objeto `refund`, pero la evaluación solo aceptaba `charge`. Para no corregirlas caso a caso (lo que equivaldría a ajustar la evaluación a los fallos), se aplicó una **regla objetiva a todas**: a las 15 preguntas respondibles de `charge` que no nombraban el objeto se les añadió *"on a charge"*. Además, `charge-026` pasó a preguntar por los estados del objeto `refund`, porque sin copias ese campo vive en `refund.status`. Estos cambios se hicieron sobre el conjunto de desarrollo, antes de la segunda ronda, y se aplicaron igual a todas las variantes.
- **Categoría `metadata`.** Las preguntas `charge-023` y `charge-024` (*"Which charge fields are expandable?"*) tienen la respuesta repartida por todo el objeto. No son un problema de búsqueda, sino de consulta de la tabla, y se excluyen de las métricas de recuperación y de generación.

## 3.4. Métricas

### 3.4.1. Recuperación

Para cada pregunta con respuesta se recuperan los 5 primeros fragmentos y se calcula, para cada posición, qué rutas esperadas cubre ese fragmento. La **definición de acierto** es estricta y es la misma para todas las variantes:

1. El fragmento debe ser **del objeto correcto**. Un fragmento de otro objeto ocupa su puesto en la lista y cuenta como fallo; no se filtra antes de calcular las posiciones.
2. La ruta debe aparecer como **línea de campo**, es decir, con el formato `` - `ruta` (tipo) `` que genera el sistema; una mención de la ruta dentro de una descripción no cuenta.
3. Una ruta esperada `e` queda cubierta por una línea `p` si `p == e` o si `p` empieza por `e.`: una pregunta sobre una caja también la responde un fragmento que contiene sus **campos hijos**. Esta regla se introdujo tras la iteración 1, porque cuando una caja no cabe entera en un fragmento su propia línea queda separada de sus hijos (sección 4.4).

Con esa definición se calculan, siguiendo la formulación habitual en recuperación de información [Manning2008]:

- **`Recall@k`** (k = 1, 3, 5): fracción de las rutas esperadas cubiertas por los k primeros fragmentos. Es la métrica principal, con k = 5 porque es el número de fragmentos que recibe el modelo.
- **`MRR`** (*Mean Reciprocal Rank*): media de 1/r, donde r es la posición del primer fragmento que cubre alguna ruta esperada (0 si ninguno).
- **`nDCG@k`** [Jarvelin2002]: con relevancia binaria por fragmento, DCG = Σ 1/log₂(r + 1) sobre las posiciones relevantes, normalizado por el DCG ideal con min(|esperadas|, k) fragmentos relevantes en las primeras posiciones.

### 3.4.2. Generación

`rag.eval_generation` mide la respuesta del modelo sin intervención humana, a partir de la salida JSON y del verificador (sección 4.7):

| Métrica | Definición |
|---|---|
| Acierto del campo | El modelo responde (`found = true`) y cita al menos una ruta esperada o un hijo suyo. Decir «no encontrado» nunca es acierto |
| Acierto con recuperación correcta | Lo mismo, restringido a las preguntas en las que la búsqueda trajo el fragmento correcto. Aísla el error del modelo del de la búsqueda |
| Alucinación | Cita una ruta que no existe en ningún objeto de la tabla |
| No respaldado | Cita una ruta real que no está en los fragmentos que recibió |
| Abstención | En las preguntas sin respuesta, responde `found = false` |
| Fallos de búsqueda / de generación | Sin el fragmento correcto en el top-5 / con él, pero sin acierto del campo |

También se registran los errores de formato JSON y la latencia media y mediana.

### 3.4.3. Corrección con juez a ciegas

El acierto del campo indica si el modelo señala el campo correcto, pero no si lo que **dice** sobre él es cierto. Para medirlo se usó un modelo como juez (*LLM-as-judge*) [Zheng2023], en la línea de los marcos de evaluación automática de RAG [Es2024], con tres precauciones (`scripts/judge.py`):

- **A ciegas.** Las respuestas de los dos modelos comparados se mezclan al azar con una semilla fija y reciben identificadores neutros (`J001`...). La correspondencia con el modelo se guarda en un fichero aparte que el juez no lee.
- **Contra la especificación.** El juez ve la pregunta, la respuesta y lo que dice la especificación de los campos esperados y de los citados (tipo, nullable, valores, descripción), pero no los fragmentos recuperados.
- **Rúbrica fija** de cuatro niveles: *correcta* (campo correcto y nada contradice la especificación), *parcial* (campo correcto pero incompleta o con un error secundario), *incorrecta* (campo equivocado o algo que contradice la especificación) y *no responde*.

El juez fue un agente de IA, por lo que no equivale a una evaluación humana; se puede validar revisando a mano una muestra de veredictos.

### 3.4.4. Taxonomía de errores

Para entender dónde falla el sistema, `scripts/error_taxonomy.py` clasifica cada pregunta del test que el juez no considera correcta en una **fase** y un **tipo**, con reglas deterministas. Si la búsqueda no trajo el fragmento correcto, el fallo es de **búsqueda** y su tipo se decide con los datos: varios objetos en la pregunta, objeto equivocado, campo gemelo en otra caja, cajón de sastre (la respuesta estaba en un grupo de campos de primer nivel) o caja equivocada del objeto correcto. Si la búsqueda acertó, el fallo es de **generación** y su tipo sale del veredicto del juez: campo inventado, campo equivocado del contexto, respuesta incompleta o falsa abstención. Las reglas se detallan en la sección 4.8.

## 3.5. Análisis estadístico

Los conjuntos de preguntas son pequeños (56 preguntas puntuadas en desarrollo y 60 en test), así que cada media se acompaña de un **intervalo de confianza al 95 %** calculado por remuestreo *bootstrap* de percentiles [Efron1993]: se generan **5.000 remuestreos** con reemplazo de las puntuaciones por pregunta, se calcula la media de cada uno y se toman los percentiles 2,5 y 97,5. La semilla es fija (0), de modo que los intervalos se reproducen exactamente.

Para comparar dos variantes se usan **diferencias emparejadas**: para cada pregunta presente en ambas se calcula la diferencia de su puntuación, y el *bootstrap* se aplica a esas diferencias. Al emparejar se elimina la variabilidad debida a la dificultad de cada pregunta, lo que da intervalos más estrechos que comparar dos intervalos independientes.

El criterio de decisión es conservador: **si el intervalo de una diferencia cruza el cero, no se afirma que una variante sea mejor que la otra**, y el resultado se informa como «no concluyente». Con 56 preguntas los intervalos de `Recall@5` tienen una anchura de unos ±0,07, por lo que algunas diferencias pequeñas (el prefijo de consulta o el *reranker*) no se pueden resolver con este tamaño. Cuando había que elegir entre dos opciones, el criterio se fijó de antemano; por ejemplo, para elegir el modelo de lenguaje se usó el acierto del campo en desarrollo.

## 3.6. Entorno y herramientas

El desarrollo y todas las mediciones se hicieron en un **Mac con chip M4 Pro**. Los modelos de lenguaje se sirven en local con **LM Studio**, que expone una API compatible con la de OpenAI en `http://localhost:1234/v1`; el modelo principal es `google/gemma-4-e4b` [Gemma4] y la alternativa ligera `qwen2.5-coder-7b-instruct` con cuantización Q4_K_M [Hui2024] [llamacpp]. Como alternativa se preparó una instalación en **Windows 11** (portátil con GPU NVIDIA RTX 3070 Ti de 8 GB) con **Ollama** y `qwen2.5-coder:7b`, descrita en `docs/windows.md`.

**Tabla 3.3.** Herramientas y bibliotecas principales.

| Herramienta | Uso |
|---|---|
| Python 3.13 y `uv` | Lenguaje y gestión del entorno y las dependencias (`uv.lock`) |
| `pandas`, `pyarrow` | Tabla de campos en formato Parquet |
| `transformers`, `sentence-transformers` | Tokenizador, embeddings [Reimers2019] y *cross-encoders* |
| `faiss-cpu` | Índice vectorial exacto [Johnson2019] |
| SQLite FTS5 | Índice literal (búsqueda por palabras) |
| `langchain-text-splitters` | Troceado fijo de referencia [LangChain] |
| LM Studio / Ollama | Servidores locales del modelo de lenguaje |
| `gradio` | Interfaz web |
| `pytest`, `ruff` | Pruebas automáticas y análisis estático |

La fuente de datos es la especificación OpenAPI [OpenAPI] oficial de Stripe, versión `2026-09-30.endive` [StripeOpenAPI].

## 3.7. Planificación temporal

El proyecto se desarrolló aproximadamente entre **septiembre y noviembre de 2026**, en las fases que resume la tabla 3.4. Las fases de experimentación se solaparon, porque el ciclo iterativo obligaba a volver a etapas anteriores cuando el análisis de fallos lo indicaba.

**Tabla 3.4.** Fases del proyecto (planificación aproximada).

| Periodo aproximado | Fase | Resultado |
|---|---|---|
| Septiembre de 2026 | Estudio del problema y del estado del arte; análisis de la especificación OpenAPI | Objetivos, hipótesis y elección del corpus |
| Septiembre – inicio de octubre | Primera versión de la cadena: extracción, troceado jerárquico, indexado y búsqueda | Sistema de búsqueda funcional |
| Inicio de octubre | Conjunto de desarrollo y evaluador de búsqueda; correcciones de la iteración 0 | Métricas fiables |
| Octubre | Rondas de experimentos (E0–E8), conjunto de control y experimento de escala (E9–E10) | Configuración de búsqueda cerrada |
| Octubre | Generación con modelo local, verificador, evaluación de generación y comparación de modelos | Sistema completo |
| Finales de octubre | Test final, juez a ciegas, taxonomía de errores e iteración *small-to-big* con el segundo test | Resultados finales |
| Octubre – noviembre | Interfaz web, pruebas, portabilidad a Windows y redacción de la memoria | Entrega |
