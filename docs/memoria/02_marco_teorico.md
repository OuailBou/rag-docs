# 2. Marco teórico y estado del arte

Este capítulo presenta los fundamentos sobre los que se construye el sistema. Se parte de los modelos de lenguaje grandes y de su principal limitación para esta tarea, la alucinación (sección 2.1); se describe la generación aumentada por recuperación como respuesta a esa limitación (2.2); se repasan las técnicas de representación y búsqueda de documentos (2.3) y las estrategias de troceado (2.4); se explica la estructura de una especificación OpenAPI y las particularidades de la de Stripe (2.5); se revisan los métodos de evaluación de sistemas RAG y las herramientas estadísticas necesarias para comparar variantes (2.6); y se describen las técnicas que permiten ejecutar modelos en un ordenador personal (2.7). El capítulo concluye situando este trabajo respecto a los enfoques existentes (2.8).

## 2.1. Modelos de lenguaje grandes

### 2.1.1. La arquitectura *transformer*

Los modelos de lenguaje actuales se basan en la arquitectura *transformer* [Vaswani2017], que sustituyó a las redes recurrentes como base del procesamiento del lenguaje natural. Su elemento central es el mecanismo de **atención**: para cada posición de la secuencia, el modelo calcula una combinación ponderada de las representaciones de todas las demás posiciones, de modo que cada *token* puede «mirar» directamente a cualquier otro, sin pasar por un estado intermedio que se degrada con la distancia. La atención se aplica en paralelo con varias proyecciones distintas (atención multicabeza) y se combina con capas *feed-forward*, conexiones residuales y normalización. Como la atención no tiene noción de orden, se añade a cada *token* una codificación de su posición.

La arquitectura original constaba de un codificador y un decodificador. De ella derivan dos familias que aparecen en este trabajo: los modelos **solo codificador**, como BERT, que producen representaciones contextuales de un texto completo y se usan para clasificar, comparar o representar textos como vectores (secciones 2.3.2 y 2.3.5); y los modelos **solo decodificador**, que generan texto *token* a *token* prediciendo el siguiente a partir de los anteriores, y que constituyen la base de los LLM conversacionales.

### 2.1.2. Preentrenamiento y aprendizaje en contexto

Un LLM se **preentrena** con un objetivo sencillo —predecir el siguiente *token*— sobre colecciones de texto de enorme tamaño. Brown et al. [Brown2020] mostraron con GPT-3 que, al escalar el número de parámetros y de datos, los modelos adquieren la capacidad de resolver tareas nuevas a partir de unas pocas instrucciones o ejemplos incluidos en la propia entrada (*few-shot* o aprendizaje en contexto), sin modificar sus pesos. Tras el preentrenamiento, los modelos se suelen ajustar para seguir instrucciones y mantener conversaciones.

Esta capacidad de aprendizaje en contexto es la que hace posible RAG: si el modelo puede usar la información que aparece en su entrada, basta con colocar ahí los fragmentos de documentación relevantes para que responda a partir de ellos.

### 2.1.3. Ventana de contexto

La **ventana de contexto** es el número máximo de *tokens* que el modelo puede procesar a la vez, sumando la entrada y la salida. Aunque los modelos recientes admiten ventanas muy amplias, ampliar el contexto tiene costes: el cálculo de la atención crece con la longitud de la secuencia, la memoria necesaria aumenta y la latencia de respuesta se dispara, algo especialmente relevante al ejecutar el modelo en local. Además, como se discute en la sección 2.4.6, disponer de un contexto largo no garantiza que el modelo use bien toda la información que contiene [Liu2024]. En este trabajo se usa una ventana de 8.192 *tokens* y se entregan al modelo cinco fragmentos de, como máximo, 512 *tokens* cada uno.

### 2.1.4. Alucinación

Se denomina **alucinación** a la generación de contenido que no es fiel a la fuente o que no se corresponde con la realidad, aunque se presente con la misma fluidez y seguridad que el contenido correcto [Ji2023]. La literatura distingue entre alucinaciones **intrínsecas**, que contradicen la fuente proporcionada, y **extrínsecas**, que añaden información que no puede verificarse a partir de ella. Las causas son diversas: datos de entrenamiento ruidosos o desactualizados, el propio objetivo de maximizar la verosimilitud del texto y la tendencia del modelo a completar con lo plausible cuando le falta información.

En la documentación de una API, la alucinación adopta una forma muy concreta: **nombres de campos o rutas inexistentes**. Un modelo puede proponer un campo con un nombre verosímil que no existe en la especificación, o combinar dos niveles del árbol en una ruta que no existe. Un ejemplo observado durante este trabajo es una respuesta que citó `payment_method_details.reader`: el campo `reader` existe, pero dentro de `payment_method_details.card_present`, y el modelo se saltó un nivel de la ruta. Este tipo de error tiene una ventaja respecto a la alucinación en texto libre: **se puede comprobar de forma automática**, porque la especificación enumera exhaustivamente las rutas válidas. Esta observación motiva el verificador descrito en el capítulo 4.

## 2.2. Generación aumentada por recuperación

### 2.2.1. Esquema general

La **generación aumentada por recuperación** (RAG) combina un modelo generativo con un mecanismo de búsqueda sobre una colección externa de documentos. Lewis et al. [Lewis2020] la formularon como la combinación de una **memoria paramétrica**, el conocimiento almacenado en los pesos de un modelo generativo preentrenado, y una **memoria no paramétrica**, un índice denso de documentos al que se accede con un recuperador neuronal. En su propuesta, recuperador y generador se entrenaban conjuntamente; en la práctica actual, y en este trabajo, lo habitual es componer piezas preentrenadas sin reentrenarlas.

El esquema de un sistema RAG típico consta de dos fases:

1. **Indexación** (fuera de línea): los documentos se dividen en fragmentos, cada fragmento se transforma en una representación apta para la búsqueda (un vector denso, un índice invertido o ambos) y se almacena en un índice.
2. **Consulta** (en línea): la pregunta del usuario se representa de la misma forma, se recuperan los *k* fragmentos más similares, se construye una instrucción (*prompt*) que contiene esos fragmentos y la pregunta, y el modelo genera una respuesta apoyada en ellos.

> **Figura 2.1** — Esquema de un sistema RAG: fase de indexación (documentos → troceado → *embeddings* → índice) y fase de consulta (pregunta → recuperación → *prompt* → LLM → respuesta) (pendiente de dibujar).

### 2.2.2. Por qué reduce las alucinaciones

RAG ataca varias de las causas de la alucinación descritas en la sección 2.1.4. En primer lugar, **aporta información actualizada**: la documentación de Stripe cambia con cada versión de la API, y un modelo entrenado meses antes no puede conocer los cambios, mientras que el índice puede regenerarse en minutos. En segundo lugar, **reduce la dependencia de la memoria paramétrica**: si el fragmento correcto está en el contexto, el modelo no necesita recordar el nombre exacto del campo, sino copiarlo. En tercer lugar, **hace las respuestas citables**: al saber qué fragmentos se usaron, es posible comprobar si la respuesta está respaldada.

RAG no elimina, sin embargo, el problema. Si la recuperación falla, el modelo recibe un contexto que no contiene la respuesta y puede rellenar el hueco con conocimiento propio o con un campo cercano pero incorrecto. Por eso, en un sistema RAG la **calidad de la recuperación condiciona la de la respuesta**, y conviene evaluar ambas fases por separado (sección 2.6). Además, es útil instruir al modelo para que **se abstenga** cuando el contexto no contenga la respuesta, y medir si lo hace.

### 2.2.3. Componentes y variantes

Gao et al. [Gao2023] organizan la evolución de RAG en tres paradigmas. El **RAG ingenuo** (*naive RAG*) sigue literalmente el esquema anterior: trocear, indexar, recuperar y generar. El **RAG avanzado** añade optimizaciones antes de la recuperación (mejor troceado, enriquecimiento de los fragmentos con metadatos, reescritura de la consulta) y después de ella (reordenación, compresión o filtrado del contexto). El **RAG modular** descompone el sistema en módulos intercambiables —búsqueda, memoria, enrutamiento, fusión— que pueden reorganizarse según la tarea.

Desde el punto de vista de la ingeniería, los componentes que deben decidirse en cualquier sistema RAG son:

| Componente | Decisión | Opciones habituales |
|---|---|---|
| Fuente | De dónde salen los documentos | Páginas web, PDF, bases de datos, especificaciones formales |
| Troceado | Cómo se dividen en fragmentos | Longitud fija, separadores, estructura, semántica (2.4) |
| Representación | Cómo se representa cada fragmento | Índice invertido (léxica), *embeddings* (densa), ambas (2.3) |
| Recuperación | Cómo se seleccionan los *k* fragmentos | Similitud, fusión de rankings, filtros por metadatos, reordenación |
| Generación | Cómo se construye el *prompt* y qué formato tiene la respuesta | Texto libre, salida estructurada, citas |
| Evaluación | Cómo se mide cada fase | Métricas de recuperación, fidelidad, corrección (2.6) |

Este trabajo se sitúa en el paradigma de RAG avanzado: su principal aportación se encuentra en la fase previa a la recuperación (construcción del corpus y troceado), complementada con un filtro por metadatos y con una fase posterior de verificación de la respuesta.

## 2.3. Representación y búsqueda de documentos

### 2.3.1. Búsqueda léxica: BM25 y FTS5

La **búsqueda léxica** representa documentos y consultas como conjuntos de términos y puntúa cada documento según los términos que comparte con la consulta. El modelo de referencia es **BM25**, derivado del marco probabilístico de relevancia [Robertson2009]. Para una consulta *q* y un documento *d*, BM25 suma, para cada término *t* de la consulta, un producto de dos factores:

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}$$

donde *f(t, d)* es la frecuencia del término en el documento, |*d*| su longitud, avgdl la longitud media de la colección, y *k₁* y *b* son parámetros que controlan la saturación de la frecuencia y la normalización por longitud. El factor IDF (*inverse document frequency*) da más peso a los términos raros en la colección. La saturación hace que repetir un término muchas veces aporte cada vez menos, y la normalización evita favorecer a los documentos largos solo por serlo.

BM25 es rápido, no requiere entrenamiento y es muy eficaz cuando la consulta contiene los mismos términos que el documento, como un identificador exacto. Su debilidad es el **desajuste de vocabulario**: si el usuario describe un concepto con palabras distintas a las de la documentación, la búsqueda léxica no lo encuentra. En la documentación de una API este desajuste es frecuente: una pregunta como «¿cuándo vence una factura?» no contiene el término `due_date`.

En este trabajo, la búsqueda léxica se implementa con **FTS5**, la extensión de búsqueda de texto completo de SQLite, que mantiene un índice invertido y ofrece una función de ordenación basada en BM25. Su ventaja práctica es que no requiere ningún servicio adicional: el índice es un fichero.

### 2.3.2. Búsqueda densa y *embeddings* de frase

La **búsqueda densa** representa cada texto como un vector de dimensión fija (un *embedding*) en un espacio donde los textos de significado parecido quedan próximos. La búsqueda consiste en encontrar los vectores de documentos más cercanos al vector de la consulta. Al comparar significados y no términos, puede recuperar un documento relevante aunque no comparta palabras con la consulta.

Karpukhin et al. [Karpukhin2020] mostraron con DPR (*Dense Passage Retrieval*) que un recuperador denso basado en dos codificadores BERT —uno para preguntas y otro para pasajes—, entrenado con pares pregunta-pasaje, podía superar a BM25 en la recuperación de pasajes para responder preguntas de dominio abierto. Paralelamente, Reimers y Gurevych [Reimers2019] propusieron Sentence-BERT, que ajusta BERT con una arquitectura siamesa para producir *embeddings* de frase comparables directamente mediante similitud coseno, evitando el coste de comparar cada par de frases con el modelo completo.

Desde entonces han aparecido numerosas familias de modelos de *embeddings* de propósito general. Este trabajo usa los modelos **BGE** (*BAAI General Embedding*), publicados junto con el conjunto de recursos C-Pack [Xiao2023], en concreto `bge-small-en-v1.5`, un modelo pequeño para inglés que produce vectores de 384 dimensiones y admite textos de hasta 512 *tokens*. Ese límite de entrada es el que fija el tamaño máximo de los fragmentos del sistema: un fragmento más largo sería truncado en silencio por el modelo. Para comparar modelos de *embeddings* de forma sistemática, Muennighoff et al. [Muennighoff2023] propusieron MTEB (*Massive Text Embedding Benchmark*), que evalúa los modelos en un amplio abanico de tareas —recuperación, clasificación, agrupamiento, similitud semántica, entre otras— y que se ha convertido en la referencia habitual para elegir un modelo. Uno de sus hallazgos es que ningún modelo domina en todas las tareas, lo que obliga a validar la elección en el dominio concreto.

Una limitación de la búsqueda densa, relevante para este trabajo, es que los modelos de *embeddings* generalistas no conocen la semántica específica de una API: para el modelo, dos líneas que dicen `country` en cajas distintas son casi idénticas, aunque una se refiera al país de emisión de una tarjeta y otra al país de una dirección de facturación.

### 2.3.3. Similitud coseno e índices exactos

La medida de proximidad más habitual entre *embeddings* es la **similitud coseno**, el coseno del ángulo entre dos vectores:

$$\cos(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert}$$

Si los vectores se normalizan previamente a norma unidad, la similitud coseno coincide con el producto escalar, que es más barato de calcular.

Para buscar los vecinos más próximos entre muchos vectores se usan bibliotecas especializadas. **FAISS** [Johnson2019] ofrece tanto índices **exactos**, que comparan la consulta con todos los vectores, como índices **aproximados** (basados en particiones, cuantización de productos o grafos), que sacrifican algo de precisión a cambio de velocidad en colecciones de millones de vectores. En este trabajo se usa el índice exacto `IndexFlatIP` (producto interno sin compresión) sobre vectores normalizados, lo que equivale a una búsqueda exacta por similitud coseno. Con un corpus de algo más de mil fragmentos, la búsqueda exhaustiva es instantánea y elimina el error de aproximación como fuente de variabilidad en los experimentos.

### 2.3.4. Búsqueda híbrida y fusión de rankings

Como la búsqueda léxica y la densa fallan en casos distintos, una estrategia frecuente es combinarlas en una **búsqueda híbrida**. El problema es que sus puntuaciones no son comparables: BM25 produce valores no acotados que dependen de la colección, mientras que la similitud coseno está acotada. Un método sencillo y robusto para combinarlas es la **fusión por rango recíproco** (*Reciprocal Rank Fusion*, RRF) [Cormack2009], que ignora las puntuaciones y usa solo las posiciones:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

donde *R* es el conjunto de rankings que se fusionan, rank*ᵣ*(*d*) es la posición del documento *d* en el ranking *r* y *k* es una constante (60 en la propuesta original, valor que también se usa aquí) que suaviza la influencia de las primeras posiciones. Cormack et al. mostraron que esta fusión, pese a su sencillez, superaba a métodos más elaborados de combinación de rankings.

La búsqueda híbrida no es una mejora garantizada: si uno de los dos métodos es claramente peor en la tarea, la fusión puede arrastrar hacia abajo los resultados del mejor. Esta posibilidad se comprueba experimentalmente en el capítulo 5.

### 2.3.5. Reordenación con *cross-encoders*

Los recuperadores densos son **bi-codificadores**: codifican pregunta y documento por separado y los comparan al final. Esto permite precalcular los vectores de los documentos, pero limita la capacidad del modelo para relacionar finamente la pregunta con el documento. Un **cross-encoder**, en cambio, recibe la pregunta y el documento concatenados y produce directamente una puntuación de relevancia, de modo que la atención puede relacionar cada palabra de la pregunta con cada palabra del documento. Nogueira y Cho [Nogueira2019] mostraron que un modelo BERT ajustado de esta forma mejoraba notablemente la reordenación de pasajes respecto a métodos anteriores.

El coste de un *cross-encoder* es que debe ejecutarse una vez por cada par pregunta-documento, por lo que no puede usarse sobre toda la colección. Lo habitual es un esquema en dos etapas: un recuperador rápido selecciona unas decenas de candidatos y el *cross-encoder* los reordena. En este trabajo se evalúa esta opción con los modelos `bge-reranker-base` y un modelo MiniLM, reordenando los 20 primeros candidatos.

## 2.4. Estrategias de troceado (*chunking*)

El **troceado** decide qué unidades de texto se indexan y, por tanto, qué recibe el modelo de *embeddings* y qué recibe el LLM. Es una decisión con dos tensiones opuestas: fragmentos demasiado grandes mezclan varios temas en un mismo vector y consumen contexto; fragmentos demasiado pequeños pierden el contexto necesario para entenderlos. A continuación se describen las estrategias principales.

### 2.4.1. Tamaño fijo

La estrategia más sencilla divide el texto en bloques de un número fijo de caracteres o *tokens*, normalmente con un **solape** entre bloques consecutivos para que la información situada en un borde aparezca completa en al menos uno de ellos. Es independiente del contenido y fácil de implementar, pero corta sin tener en cuenta la estructura: una frase, una tabla o, en el caso que nos ocupa, la descripción de un campo pueden quedar partidas en dos.

### 2.4.2. Recursivo por separadores

Una mejora muy extendida es el troceado **recursivo por separadores**, popularizado por el `RecursiveCharacterTextSplitter` de LangChain [LangChain]. El algoritmo intenta dividir el texto por el separador de mayor nivel (por defecto, dobles saltos de línea, es decir, párrafos); si algún trozo sigue superando el tamaño máximo, lo divide por el siguiente separador (saltos de línea simples, después espacios) y así sucesivamente, fusionando trozos pequeños hasta acercarse al tamaño objetivo. El resultado respeta mejor los párrafos y las frases que el corte fijo, y puede medir el tamaño en *tokens* del propio modelo de *embeddings*. Por ser la opción por defecto en muchas guías y bibliotecas, se toma en este trabajo como **troceado de referencia**, aplicado sobre un documento en Markdown que lista todos los campos de cada objeto, con 512 *tokens* y 64 de solape.

### 2.4.3. Troceado estructural

Cuando el documento tiene una estructura explícita —secciones de Markdown, etiquetas HTML, funciones en código fuente—, se puede trocear siguiéndola: un fragmento por sección o por función, subdividiendo solo las que exceden el tamaño máximo. Esta familia de estrategias aprovecha que el autor del documento ya agrupó el contenido por temas. El troceado jerárquico propuesto en este TFG pertenece a esta familia, con una particularidad: la estructura no se infiere de marcas de formato, sino que procede directamente del **árbol de esquemas de la especificación OpenAPI**, de modo que cada fragmento corresponde a un nodo (una «caja») del árbol de atributos o a un grupo de campos hermanos.

### 2.4.4. Troceado semántico

El **troceado semántico** decide los puntos de corte según el contenido: se calculan *embeddings* de frases consecutivas y se corta donde la similitud entre una frase y la siguiente cae por debajo de un umbral, señal de un cambio de tema. No requiere estructura explícita, pero añade coste de cálculo y depende de un umbral que hay que ajustar. En la documentación de una API, donde cada campo es una unidad con su propio tema, la estructura explícita del árbol ofrece puntos de corte más fiables que cualquier medida de similitud.

### 2.4.5. Contexto añadido al fragmento y *small-to-big*

Un problema común a todas las estrategias es que un fragmento aislado pierde el contexto del documento del que procede. Anthropic [Anthropic2024] propuso la **recuperación contextual** (*contextual retrieval*): antes de indexar cada fragmento, un LLM genera un breve texto que lo sitúa dentro del documento completo, y ese texto se antepone al fragmento tanto para calcular su *embedding* como para el índice léxico. La cabecera con la ruta que usa este trabajo (`Charge → payment method details → ach debit`) persigue el mismo objetivo, pero se obtiene de forma **determinista** a partir de la ruta del campo en la especificación, sin coste de generación y sin riesgo de que el contexto añadido contenga errores.

Otra técnica relacionada es la recuperación ***small-to-big*** o de **documento padre** (*parent document retrieval*), descrita, entre otros, en la documentación de LlamaIndex [LlamaIndex]: se indexan unidades pequeñas, cuyo significado es preciso y cuyo *embedding* no se diluye, pero al modelo se le entrega la unidad mayor a la que pertenecen, que aporta contexto suficiente para responder. Este trabajo evalúa esta técnica como variante, indexando cada campo por separado y entregando el fragmento jerárquico que lo contiene (capítulo 5).

### 2.4.6. Evaluación de estrategias y dilución del significado

Pese a su importancia, el troceado se ha evaluado de forma sistemática con relativamente poca frecuencia. Smith y Troynikov [Smith2024] compararon diversas estrategias populares y mostraron que la elección de la estrategia y de sus parámetros afecta de forma apreciable a la calidad de la recuperación, y que las opciones por defecto de las bibliotecas no son necesariamente las mejores. Su trabajo subraya la necesidad de medir el troceado en cada dominio en lugar de adoptar una configuración genérica.

Detrás de esta sensibilidad hay un fenómeno que conviene nombrar: la **dilución del significado en un único vector**. Un *embedding* resume todo el fragmento en un solo punto del espacio; si el fragmento habla de muchas cosas, el vector queda en una posición intermedia que no se parece especialmente a ninguna de ellas. Un ejemplo de este trabajo lo ilustra: en el objeto `invoice`, los campos de primer nivel sin hijos se agrupan en fragmentos por orden alfabético, y el campo `due_date` (fecha de vencimiento) comparte fragmento con campos sin relación como `customer_phone`, `customer_tax_ids` o `default_source`. Ante la pregunta «*What is the due date of an invoice?*», ese fragmento no queda entre los primeros, porque su vector representa una mezcla de temas, y el modelo acaba respondiendo con un campo vecino, `next_payment_attempt`. El análisis detallado se presenta en el capítulo 5.

El problema tiene una contrapartida en el lado de la generación. Liu et al. [Liu2024] mostraron que los modelos de lenguaje usan peor la información situada en mitad de un contexto largo que la situada al principio o al final (*lost in the middle*). Por tanto, la tentación de compensar una mala recuperación entregando al modelo muchos más fragmentos tiene límites: un contexto más largo no equivale a un contexto mejor aprovechado. Ambos fenómenos apuntan en la misma dirección: **fragmentos coherentes, que traten de una sola cosa, y pocos fragmentos bien elegidos**.

> **Figura 2.2** — Comparación entre el troceado por longitud y el troceado jerárquico sobre un mismo subárbol del objeto `charge` (pendiente de dibujar).

## 2.5. Documentación de APIs y OpenAPI

### 2.5.1. Estructura de una especificación OpenAPI

La **especificación OpenAPI** [OpenAPI] es un estándar para describir APIs HTTP de forma legible por máquinas, en JSON o YAML. Una especificación describe las rutas (*paths*) y operaciones de la API, sus parámetros, sus respuestas y, en la sección `components.schemas`, los **esquemas** de los objetos que la API intercambia. Los esquemas siguen un dialecto de JSON Schema. Los elementos que aparecen en este trabajo son:

| Elemento | Significado | Ejemplo en Stripe |
|---|---|---|
| `components.schemas` | Diccionario de esquemas con nombre | `charge`, `customer`, `invoice` |
| `properties` | Campos de un objeto, cada uno con su propio esquema | `charge.properties.amount` |
| `type` | Tipo del valor: `string`, `integer`, `boolean`, `object`, `array`... | `amount`: `integer` |
| `$ref` | Referencia a otro esquema, que se «incluye» en ese punto | `#/components/schemas/refund` |
| `anyOf` | El valor puede ajustarse a cualquiera de varios esquemas | Un ID (`string`) o el objeto completo |
| `array` + `items` | Lista de elementos con el esquema indicado en `items` | `refunds.data` |
| `enum` | Conjunto cerrado de valores permitidos | `status`: `failed`, `pending`, `succeeded` |
| `nullable` | El campo puede valer `null` (OpenAPI 3.0) | `customer` |
| `description` | Texto en lenguaje natural que explica el campo | «*The status of the payment...*» |

Gracias a `$ref`, un esquema puede contener a otros, que a su vez contienen a otros, formando un **árbol de atributos** (en rigor, un grafo, porque las referencias pueden formar ciclos). La ruta de un campo dentro de ese árbol, como `payment_method_details.card.country`, es su identificador inequívoco: el nombre `country` por sí solo no lo es.

### 2.5.2. La especificación de Stripe

Stripe publica su especificación OpenAPI en un repositorio público [StripeOpenAPI]. Este trabajo usa el fichero `spec3.json` correspondiente a la versión de la API `2026-09-30.endive`, que declara la versión 3.0.0 de OpenAPI y contiene más de 1.500 esquemas. Varias características de esta especificación son relevantes para el diseño del sistema:

- **Profundidad y repetición.** Los objetos principales tienen cientos de campos anidados en varios niveles. Muchos nombres se repiten en ramas distintas: el objeto `charge` contiene 19 campos llamados `country`, y lo mismo ocurre con `last4`, `brand` o `exp_month` en los distintos métodos de pago. Sin la ruta, estos campos son indistinguibles.
- **Campos expandibles.** Muchos campos que apuntan a otro objeto devuelven por defecto solo su identificador, pero pueden «expandirse» para obtener el objeto completo mediante el parámetro `expand` de la petición. En la especificación se representan con un `anyOf` que combina un `string` (el ID) y uno o varios `$ref` al objeto, acompañados de la extensión `x-expansionResources`. Por ejemplo, `charge.customer` es un ID de cliente o un objeto `customer`. Si un extractor siguiera ingenuamente estos `$ref`, cada cargo incluiría una copia completa del cliente.
- **Objetos que contienen copias de otros.** Algunos esquemas incluyen objetos completos de forma no expandible. Por ejemplo, el error asociado a un `payment_intent` contiene un `setup_intent` entero, y la lista `charge.refunds.data` contiene objetos `refund` completos. Al construir el corpus, estas copias generan fragmentos casi idénticos a los del objeto original, que compiten con él en la búsqueda. Una de las decisiones evaluadas en el capítulo 5 es precisamente no abrir dentro de un objeto otros objetos que ya tienen su propio documento.
- **Descripciones con valores.** Las descripciones suelen enumerar los valores posibles de un campo o explicar su relación con otros campos, y a veces son muy largas: algunas listas de valores superan el centenar de elementos, lo que obliga a partir un único campo en varios fragmentos.

Estas características hacen de la especificación de Stripe un caso de estudio adecuado para la hipótesis del trabajo: es grande, profunda, repetitiva y, a la vez, completamente formal, lo que permite usarla como verdad de referencia.

## 2.6. Evaluación de sistemas RAG

Un sistema RAG se puede evaluar de extremo a extremo, juzgando solo la respuesta final, o por fases, midiendo por separado la recuperación y la generación. La evaluación por fases permite saber **dónde** falla el sistema, y es la que se adopta en este trabajo.

### 2.6.1. Métricas de recuperación

Las métricas de recuperación proceden de la tradición de la recuperación de información [Manning2008]. Dada una pregunta con un conjunto de documentos relevantes conocidos y el ranking devuelto por el sistema:

- **Recall@k** es la fracción de los documentos relevantes que aparecen entre los *k* primeros resultados. En un sistema RAG que entrega *k* fragmentos al modelo, Recall@k responde directamente a la pregunta de si el modelo tuvo acceso a la respuesta. Es la métrica principal de este trabajo, con *k* = 5.
- **MRR** (*Mean Reciprocal Rank*) es la media, sobre todas las preguntas, del inverso de la posición del primer resultado relevante (1 si está en primera posición, 1/2 si está en segunda, etc., y 0 si no aparece). Mide si la respuesta aparece **arriba**, lo que importa porque los modelos aprovechan mejor la información situada al principio del contexto [Liu2024].
- **nDCG** (*normalized Discounted Cumulative Gain*) [Jarvelin2002] suma la relevancia de cada resultado descontada por un factor logarítmico de su posición, y la normaliza dividiendo por el valor del ranking ideal. Admite grados de relevancia y es la métrica principal de bancos de pruebas como BEIR [Thakur2021].

BEIR [Thakur2021] es relevante aquí por otra razón: al evaluar recuperadores en conjuntos de datos heterogéneos, mostró que BM25 sigue siendo una referencia difícil de batir fuera del dominio de entrenamiento de los modelos densos. Por eso no se puede dar por hecho que la búsqueda densa sea mejor que la léxica en un dominio especializado como la documentación de una API; hay que comprobarlo.

Para aplicar estas métricas hace falta definir qué es un «documento relevante». En este trabajo, cada pregunta lleva anotadas las **rutas** de los campos que la responden, y un fragmento cuenta como acierto si es del objeto correcto y contiene la línea de definición de esa ruta. Al estar las rutas definidas en la especificación, el juicio de relevancia es **determinista** y se aplica por igual a cualquier estrategia de troceado.

### 2.6.2. Evaluación de la generación

Evaluar la respuesta generada es más difícil, porque es texto libre. Los marcos más extendidos distinguen varias dimensiones:

- **Fidelidad** (*faithfulness*): si todo lo que afirma la respuesta está respaldado por el contexto recuperado.
- **Relevancia de la respuesta**: si la respuesta contesta a lo que se preguntó.
- **Relevancia del contexto**: si los fragmentos recuperados son pertinentes y no contienen ruido innecesario.

RAGAS [Es2024] propone medir estas dimensiones sin respuestas de referencia, usando un LLM para descomponer la respuesta en afirmaciones y comprobar si cada una se deduce del contexto, entre otros procedimientos. Este enfoque, y en general el uso de un **LLM como juez** (*LLM-as-a-judge*), se ha generalizado por su bajo coste frente a la evaluación humana. Zheng et al. [Zheng2023] estudiaron este uso y encontraron que un juez basado en un modelo potente puede alcanzar un acuerdo con evaluadores humanos comparable al acuerdo entre humanos, pero también documentaron **sesgos** sistemáticos: preferencia por la respuesta que aparece en una posición determinada, por las respuestas más largas y por las generadas por el propio modelo juez, además de limitaciones en razonamiento matemático y lógico.

La documentación de una API permite reducir la dependencia del juez. Si se pide al modelo que responda en un **formato estructurado** que incluya las rutas de los campos que cita, varias propiedades de la respuesta se pueden comprobar sin juicio alguno: si el campo citado **existe** en la especificación (alucinación), si estaba **en el contexto** recuperado (respaldo), si coincide con el campo esperado (acierto) y si el modelo **se abstiene** cuando la pregunta no tiene respuesta. En este trabajo, estas métricas deterministas son las principales, y el juez automático se reserva para la dimensión que no pueden cubrir: si lo que la respuesta **dice** sobre el campo es correcto. Para mitigar los sesgos señalados, ese juez trabaja a ciegas respecto al modelo evaluado, compara contra la especificación y aplica una rúbrica fija (capítulo 5).

### 2.6.3. Estadística: *bootstrap* y comparaciones emparejadas

Los conjuntos de preguntas de evaluación de un TFG son necesariamente pequeños (decenas de preguntas), por lo que las métricas tienen una incertidumbre considerable. Reportar una sola cifra sin su incertidumbre invita a sacar conclusiones de diferencias que pueden deberse al azar.

El ***bootstrap*** [Efron1993] estima la variabilidad de un estadístico remuestreando los datos con reemplazo: se generan muchas muestras del mismo tamaño que la original, se calcula el estadístico en cada una y se toman los percentiles 2,5 y 97,5 de la distribución resultante como **intervalo de confianza al 95 %**. Su atractivo es que no exige suponer una distribución concreta para la métrica.

Cuando se comparan dos variantes del sistema sobre **las mismas preguntas**, conviene una **comparación emparejada**: en cada remuestreo se seleccionan las mismas preguntas para ambas variantes y se calcula la diferencia. Así se elimina la variabilidad debida a la dificultad de cada pregunta, que afecta por igual a las dos variantes, y el intervalo de la diferencia es más estrecho que el que se obtendría comparando dos intervalos independientes. El criterio que se adopta es sencillo: **si el intervalo de la diferencia incluye el cero, no se puede afirmar que una variante sea mejor que la otra**.

### 2.6.4. Separación entre desarrollo y test

Un riesgo menos visible que el azar es el **sobreajuste al conjunto de evaluación**. Cada vez que se observa un resultado y se modifica el sistema en consecuencia, el sistema se adapta un poco más a esas preguntas concretas, y la métrica deja de ser una estimación honesta del rendimiento con preguntas nuevas. Es el mismo problema que, en aprendizaje automático, obliga a separar los datos de entrenamiento, validación y test, trasladado a las decisiones de diseño.

La práctica recomendada es separar al menos dos conjuntos: uno de **desarrollo**, con el que se toman todas las decisiones, y uno de **test**, que se evalúa una sola vez con la configuración ya cerrada. Este trabajo añade un tercer conjunto, de **control**, generado de forma sintética después de cerrar las decisiones, para comprobar que las conclusiones se mantienen con preguntas de otro origen. La caída de rendimiento al pasar de desarrollo a test es, en sí misma, una medida informativa del grado de sobreajuste.

## 2.7. Modelos locales

### 2.7.1. Motivación

Ejecutar el modelo de lenguaje en local tiene ventajas claras para este trabajo: no se envían datos a servicios externos, no hay coste por consulta, los experimentos son reproducibles al no depender de un modelo remoto que puede cambiar sin aviso, y el sistema puede usarse sin conexión. La contrapartida es que el modelo debe caber en la memoria de un ordenador personal y responder en un tiempo razonable, lo que limita su tamaño.

### 2.7.2. Cuantización

La **cuantización** reduce la memoria que ocupa un modelo representando sus pesos con menos bits: en lugar de 16 bits por parámetro, 8, 4 o incluso menos. Un modelo de 7.000 millones de parámetros, que en 16 bits ocupa unos 14 GB, ocupa aproximadamente una cuarta parte con 4 bits, lo que lo hace viable en un portátil.

Las técnicas de **cuantización posterior al entrenamiento** cuantizan un modelo ya entrenado sin reentrenarlo. **GPTQ** [Frantar2023] cuantiza los pesos capa a capa usando información aproximada de segundo orden para compensar el error introducido, y mostró que es posible reducir modelos grandes a 3 o 4 bits por peso con una pérdida de calidad pequeña. En el ecosistema de ejecución en CPU y en ordenadores personales, el proyecto **llama.cpp** [llamacpp] y su formato de fichero **GGUF** se han convertido en el estándar de facto. GGUF almacena en un único fichero los pesos cuantizados y los metadatos del modelo, y admite varios esquemas de cuantización. Entre ellos, los denominados *k-quants* cuantizan los pesos por bloques con escalas propias; la variante **Q4_K_M** usa en torno a 4 bits por peso, manteniendo con más precisión algunos tensores especialmente sensibles, y suele considerarse un buen equilibrio entre tamaño y calidad.

### 2.7.3. Modelos utilizados

Se han utilizado dos modelos abiertos de tamaño similar y propósito distinto:

- **Gemma 4 E4B** [Gemma4], de Google DeepMind, un modelo de propósito general diseñado para ejecutarse en dispositivos con recursos limitados. La «E» de su nombre hace referencia a sus parámetros «efectivos»; el número total de parámetros es mayor (en torno a 7.500 millones según la información que muestra el servidor de inferencia) (verificar en la ficha del modelo). Es el modelo principal del sistema.
- **Qwen2.5-Coder 7B Instruct** [Hui2024], de la familia Qwen2.5-Coder, especializada en código y publicada en varios tamaños. Se incluye para comprobar si un modelo orientado a la programación responde mejor sobre documentación de API que uno generalista, y se usa cuantizado en Q4_K_M.

### 2.7.4. Servidores de inferencia y salida estructurada

Para usar un modelo local desde una aplicación se recurre a un **servidor de inferencia** que carga el modelo y expone una API. **Ollama** y **LM Studio** son dos de los más extendidos en ordenadores personales; ambos se apoyan en motores como llama.cpp y ofrecen, entre otras, una **API compatible con la de OpenAI** (el *endpoint* `/v1/chat/completions`). Esta compatibilidad permite escribir el cliente una sola vez y cambiar de servidor o de modelo modificando solo la URL y el nombre del modelo, que es como se ha implementado el sistema.

Una funcionalidad clave para este trabajo es la **salida estructurada**: el cliente envía, junto con la petición, un **JSON Schema** que describe el formato de la respuesta, y el servidor restringe la generación para que la salida sea un JSON válido conforme a ese esquema. Técnicamente, esto se consigue limitando en cada paso los *tokens* que el modelo puede elegir a los que mantienen la salida dentro de la gramática del esquema. En el sistema, la respuesta debe tener cuatro claves —si se encontró la respuesta, el texto de la respuesta, las rutas de los campos citados y los números de los fragmentos citados—, lo que hace posible la verificación automática descrita en la sección 2.6.2 y elimina los errores de formato.

## 2.8. Síntesis y posicionamiento del trabajo

La revisión anterior muestra que los componentes de un sistema RAG están bien estudiados por separado, pero que la mayoría de las propuestas y herramientas están pensadas para **texto en prosa** y para **evaluación con juicios humanos o de un LLM**. La documentación de una API descrita en OpenAPI se aparta de ambos supuestos: su contenido es un árbol formal de atributos, y ese mismo árbol ofrece una verdad de referencia contra la que comprobar tanto la recuperación como la generación.

La siguiente tabla resume los enfoques revisados y lo que aporta este trabajo en cada aspecto:

| Aspecto | Enfoque habitual | Limitación en documentación de API | Aportación de este TFG |
|---|---|---|---|
| Fuente del corpus | Páginas web o documentos convertidos a texto | Pierde la estructura y duplica objetos referenciados | Extracción directa de `components.schemas`, siguiendo `$ref` sin copiar objetos con documento propio |
| Troceado | Longitud fija o recursivo por separadores [LangChain]; semántico | Parte cajas por la mitad y separa los campos de su contexto | Troceado jerárquico por el árbol de atributos, con presupuesto de *tokens* |
| Contexto del fragmento | Contexto generado por un LLM [Anthropic2024] | Coste de generación y posibles errores | Cabecera determinista con la ruta del fragmento |
| Unidad de búsqueda | Fragmento único; *small-to-big* [LlamaIndex] | Dilución del significado en grupos heterogéneos | Evaluación de *small-to-big* como variante sobre el troceado jerárquico |
| Recuperación | Densa, léxica o híbrida con RRF [Cormack2009]; reordenación [Nogueira2019] | Campos repetidos entre objetos | Comparación de los tres métodos y filtro por el objeto que nombra la pregunta |
| Verificación de la respuesta | Juez humano o LLM [Zheng2023]; RAGAS [Es2024] | Caro, subjetivo y con sesgos | Verificador determinista de campos contra la especificación; juez a ciegas solo para el contenido |
| Protocolo de evaluación | Un único conjunto de preguntas, cifras sin intervalo | Sobreajuste y conclusiones no significativas | Desarrollo, control y test; *bootstrap* emparejado [Efron1993] |
| Ejecución | Modelos en la nube | Privacidad, coste, reproducibilidad | Sistema 100 % local con modelos cuantizados [llamacpp] |

En resumen, este trabajo no propone un nuevo modelo, sino una forma de **construir, trocear, verificar y evaluar** un sistema RAG que aprovecha la estructura formal de la documentación de una API. Los capítulos siguientes describen la metodología (capítulo 3) y el diseño e implementación del sistema (capítulo 4), y contrastan experimentalmente las hipótesis planteadas en la introducción (capítulo 5).
