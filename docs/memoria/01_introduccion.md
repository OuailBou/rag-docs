# 1. Introducción

## 1.1. Contexto

En los últimos años, los modelos de lenguaje grandes (*Large Language Models*, LLM) se han incorporado de forma generalizada al trabajo diario de los desarrolladores de software. Asistentes integrados en el editor, interfaces conversacionales y agentes que escriben código se consultan hoy de manera rutinaria para resolver dudas que antes exigían leer la documentación oficial. Una parte importante de esas consultas trata sobre APIs de terceros: qué campo devuelve un objeto, qué valores puede tomar un estado, si un atributo puede ser nulo o en qué nivel de la respuesta se encuentra un dato concreto.

Las APIs de pago son un caso especialmente exigente. Stripe, una de las plataformas de pagos más extendidas, publica una API con cientos de objetos, cada uno con decenas o cientos de campos anidados, y la actualiza con frecuencia mediante versiones fechadas. Un error al interpretar un campo no es un detalle menor: puede traducirse en una integración que lee un importe en la unidad equivocada, que consulta un estado que no existe o que no tiene en cuenta un valor posible de una enumeración. En este contexto, un asistente que responda preguntas sobre la API debe ser, ante todo, **fiable y verificable**.

La técnica más extendida para conectar un LLM con una fuente de conocimiento concreta es la **generación aumentada por recuperación** (*Retrieval-Augmented Generation*, RAG) [Lewis2020][Gao2023]. En lugar de confiar en lo que el modelo memorizó durante su entrenamiento, un sistema RAG busca en una colección de documentos los fragmentos más relevantes para la pregunta y se los entrega al modelo como contexto, de modo que la respuesta se apoye en información actual y citable. Este Trabajo Fin de Grado diseña, implementa y evalúa un sistema RAG especializado en la documentación de la API de Stripe, construido a partir de su especificación OpenAPI oficial [StripeOpenAPI] y ejecutado íntegramente en local.

## 1.2. Motivación y problema

El punto de partida del trabajo son tres problemas observados al aplicar un LLM, con o sin RAG, a la documentación de una API.

**Los LLM alucinan nombres de campos.** Un modelo de lenguaje genera el texto más plausible dada la pregunta, no necesariamente el verdadero [Ji2023]. En una API, lo plausible y lo real están muy cerca: un modelo puede responder con un campo `refund_status` que suena razonable pero no existe, o con una ruta que mezcla dos niveles del árbol de atributos. Estos errores son difíciles de detectar para quien pregunta precisamente porque no conoce la respuesta.

**La documentación de una API es jerárquica y repetitiva, y el troceado genérico la rompe.** Un sistema RAG divide los documentos en fragmentos (*chunks*) antes de indexarlos. Las estrategias habituales cortan el texto por longitud, con algo de solape y respetando, en el mejor de los casos, saltos de párrafo o de línea [LangChain]. Esta forma de trocear es razonable para prosa, pero no para una referencia de API, cuyo contenido es un **árbol de atributos**. El objeto `charge` de Stripe, por ejemplo, contiene **19 campos llamados `country`**, repartidos entre las direcciones de facturación, los distintos métodos de pago (tarjeta, adeudo bancario, etc.) y otras estructuras anidadas. Si un fragmento corta el árbol por la mitad, la línea que dice `country` queda separada de la caja a la que pertenece y deja de ser posible saber de qué país se habla. El problema no es solo de presentación: el modelo de *embeddings* que representa cada fragmento como un vector recibe un texto sin el contexto que lo desambigua.

**Evaluar un sistema RAG es caro y subjetivo.** La práctica habitual consiste en pedir a evaluadores humanos, o a otro LLM actuando como juez, que valoren si las respuestas son correctas y fieles al contexto [Es2024][Zheng2023]. Ambas opciones tienen costes: la evaluación humana es lenta y difícil de repetir, y los jueces automáticos tienen sesgos conocidos. Sin embargo, la documentación de una API ofrece una oportunidad poco aprovechada: la **especificación OpenAPI** [OpenAPI] es un documento formal que enumera de forma exhaustiva qué campos existen, de qué tipo son y qué valores admiten. Esto permite comprobar de forma **determinista** si un campo citado por el modelo existe, si estaba en el contexto recuperado y si la búsqueda trajo el fragmento correcto, sin depender de juicios subjetivos.

## 1.3. Objetivos

### Objetivo general

Diseñar, implementar y evaluar de forma rigurosa un sistema de generación aumentada por recuperación que responda preguntas sobre la API de Stripe a partir de su especificación OpenAPI, que funcione íntegramente en local y cuyas respuestas puedan verificarse automáticamente contra la propia especificación.

### Objetivos específicos

1. **Construir el corpus a partir de la especificación OpenAPI**, recorriendo las referencias entre esquemas para obtener una tabla con una fila por campo (ruta, tipo, nulabilidad, valores posibles y descripción), sin duplicar dentro de un objeto otros objetos que ya tienen documento propio.
2. **Diseñar un troceado jerárquico** que siga el árbol de atributos de cada objeto, respete un presupuesto de tokens compatible con el modelo de *embeddings* y añada a cada fragmento una cabecera con su ruta, y compararlo con un troceado por longitud de referencia.
3. **Implementar y comparar estrategias de recuperación** —semántica, léxica e híbrida—, junto con mejoras como un filtro por objeto y la reordenación con un *cross-encoder*.
4. **Integrar un modelo de lenguaje local** que responda en un formato estructurado, citando los campos y fragmentos que usa, y **un verificador** que detecte campos inventados o no respaldados por el contexto.
5. **Definir un protocolo de evaluación reproducible**, con conjuntos separados de desarrollo, control y test, métricas de recuperación y de generación e intervalos de confianza, que evite el sobreajuste al conjunto con el que se toman las decisiones.
6. **Analizar los errores** del sistema por fase (búsqueda o generación) para identificar su cuello de botella y orientar el trabajo futuro.

## 1.4. Hipótesis

La hipótesis principal del trabajo es la siguiente:

> **H1.** En la documentación de una API descrita con OpenAPI, un troceado jerárquico que sigue el árbol de atributos de cada objeto mejora la recuperación —medida como `Recall@5`— frente a un troceado por longitud fija como el que ofrecen las bibliotecas habituales de RAG.

Además de la hipótesis principal, se contrastan otras de carácter secundario, ligadas a decisiones de diseño concretas: que añadir a cada fragmento una cabecera con su ruta mejora la recuperación, que eliminar las copias de objetos dentro de otros objetos reduce la competencia entre fragmentos, y que priorizar los fragmentos del objeto que nombra la pregunta mejora el orden de los resultados. Todas ellas se evalúan con el mismo protocolo, cambiando una sola variable cada vez.

## 1.5. Contribuciones

Las aportaciones principales de este trabajo son:

- **Extracción del corpus desde OpenAPI sin copias.** Un extractor que recorre `components.schemas` siguiendo los `$ref`, distingue valores, cajas, listas y campos expandibles, y no abre dentro de un objeto otros objetos del corpus, evitando así fragmentos duplicados que compiten con el original.
- **Troceado jerárquico con cabecera.** Un algoritmo que agrupa los campos según el árbol de atributos, mantiene enteras las cajas que caben en el presupuesto de tokens, desciende de nivel cuando no caben y antepone a cada fragmento la ruta que lo sitúa (por ejemplo, `Charge → payment method details → ach debit`).
- **Filtro por objeto.** Un mecanismo sencillo que detecta los objetos que nombra la pregunta y prioriza sus fragmentos en el ranking.
- **Verificador de alucinaciones contra la especificación.** Un componente que comprueba cada campo citado por el modelo y lo clasifica como inventado, no respaldado por el contexto o correcto, de forma completamente determinista.
- **Protocolo de evaluación con desarrollo, control y test.** Una metodología que separa el conjunto con el que se toman decisiones de los conjuntos con los que se confirman, evaluados una sola vez, con intervalos de confianza por *bootstrap* y comparaciones emparejadas.
- **Sistema 100 % local.** Búsqueda, modelo de lenguaje y evaluación se ejecutan en un ordenador personal, con modelos abiertos y cuantizados, sin enviar datos a servicios externos.

## 1.6. Alcance y limitaciones

El trabajo se centra en la **referencia de objetos** de la API de Stripe: qué campos tiene cada objeto, cómo se anidan, de qué tipo son y qué valores admiten. El corpus principal está formado por 16 objetos de uso frecuente (`charge`, `customer`, `payment_intent`, `invoice`, `subscription`, etc.), y se amplía a 50 en un experimento de escala. Quedan fuera del alcance:

- **Los *endpoints* y sus parámetros de petición**, así como las guías conceptuales y los tutoriales de la documentación web, que no forman parte de los esquemas de objetos.
- **Las preguntas de tipo agregado** (por ejemplo, «¿qué campos pueden ser nulos?»), cuya respuesta está repartida por todo el objeto y se resolvería mejor consultando la tabla de campos que mediante búsqueda.
- **La generación de código** de integración: el sistema responde qué campo contiene un dato y qué significa, no cómo programar la llamada.
- **La evaluación con usuarios reales.** Las preguntas de evaluación se han escrito por el autor, generado con plantillas o redactado por un agente de IA sin acceso al sistema; no proceden de usuarios de la API.
- **El idioma.** La especificación está en inglés y las preguntas de evaluación también; no se estudia el comportamiento con preguntas en otros idiomas.

Además, el uso de modelos locales de tamaño moderado y de un modelo de *embeddings* pequeño es una decisión deliberada, orientada a que el sistema pueda ejecutarse en un portátil; los resultados no deben extrapolarse directamente a modelos de mayor tamaño servidos en la nube.

## 1.7. Estructura del documento

El resto de la memoria se organiza de la siguiente manera:

- **Capítulo 2, Marco teórico y estado del arte.** Presenta los fundamentos de los modelos de lenguaje, la generación aumentada por recuperación, la búsqueda léxica y densa, las estrategias de troceado, la especificación OpenAPI, la evaluación de sistemas RAG y la ejecución de modelos en local, y sitúa este trabajo respecto a los enfoques existentes.
- **Capítulo 3, Metodología.** Expone el enfoque de trabajo, el diseño experimental (una sola variable cada vez), los conjuntos de preguntas de desarrollo, control y test, las métricas, el análisis estadístico, el entorno y la planificación.
- **Capítulo 4, Diseño e implementación.** Describe la arquitectura y cada una de sus etapas: extracción, troceado, indexación, recuperación, generación y verificación, junto con las decisiones técnicas y la configuración.
- **Capítulo 5, Experimentos y resultados.** Presenta los resultados de cada experimento en desarrollo, control y test, el experimento de escala, la evaluación de la generación, la taxonomía de errores y la variante *small-to-big*.
- **Capítulo 6, Discusión.** Interpreta los resultados frente a la hipótesis y los objetivos, explica por qué funciona el troceado jerárquico y dónde falla, analiza las amenazas a la validez y las consideraciones éticas.
- **Capítulo 7, Conclusiones y trabajo futuro.** Resume las aportaciones y propone líneas de continuación.
