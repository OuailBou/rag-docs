# 5. Experimentos y resultados

Este capítulo presenta, en el orden en que se obtuvieron, los experimentos realizados sobre el sistema y sus resultados. Todos siguen el mismo principio metodológico: **cada variante cambia una sola cosa** respecto a aquella con la que se compara, de modo que cualquier diferencia se puede atribuir a ese cambio. Las variantes están definidas en `configs/experiments.yaml` y se reproducen con un único comando (`scripts/experiment.py`).

Antes de entrar en los resultados conviene recordar cómo se mide (véase el capítulo de metodología):

- **Conjuntos de preguntas.** Se usan cuatro conjuntos con papeles distintos. El de **desarrollo** (`dev_charge` y `dev_objects`, 60 preguntas escritas a mano, de las que se puntúan 56 en búsqueda) sirve para decidir qué cambios se quedan. El de **control** (120 preguntas sintéticas generadas con plantillas) se evalúa una sola vez, al cerrar las decisiones. El **test** (70 preguntas, 60 con respuesta y 10 sin ella, escritas por un agente de IA sin ver el sistema) es el resultado final y también se evalúa una sola vez. Por último, un **segundo test** (`test2`, 70 preguntas nuevas con las mismas reglas) se usó solo para evaluar la iteración 4.
- **Métricas de búsqueda.** La principal es `Recall@5`: la fracción de rutas correctas que aparecen entre los 5 primeros fragmentos recuperados. Se acompaña de `Recall@1` (si el primer fragmento ya es correcto) y de `MRR` (*mean reciprocal rank*), que premia que el fragmento correcto aparezca cuanto más arriba mejor [Manning2008].
- **Intervalos de confianza al 95 %** calculados por remuestreo (*bootstrap*) [Efron1993]. Para comparar dos variantes se calcula el intervalo de la **diferencia**: si cruza el cero, no se puede afirmar que una variante sea mejor que la otra.
- Un fragmento solo cuenta como acierto si pertenece al **objeto correcto** y la ruta aparece como **línea de campo** (`` - `ruta` (tipo) ``), no si simplemente se menciona en una descripción.
- Las preguntas de tipo `metadata` (por ejemplo, *"¿qué campos son nullable?"*) no se puntúan en búsqueda, porque su respuesta está repartida por todo el objeto y corresponde a una consulta sobre la tabla, no a un problema de recuperación.

## 5.1 Iteración 0: correcciones antes de medir

Antes de comparar ninguna variante se revisó el propio sistema de evaluación y el troceador. Se encontraron tres problemas que distorsionaban los números y que, de no haberse corregido, habrían invalidado las conclusiones posteriores.

**Tabla 5.1.** Problemas corregidos en la iteración 0.

| Problema | Efecto | Arreglo |
|---|---|---|
| `eval_retrieval.py` filtraba los resultados por objeto **antes** de calcular las posiciones | Un fragmento correcto en el puesto 4, detrás de tres de otros objetos, contaba como puesto 1. `Recall@1`, `Recall@3` y `MRR` salían inflados. `Recall@5` no se veía afectado | Los fragmentos de otros objetos ocupan su puesto y cuentan como fallo |
| La métrica aceptaba cualquier palabra entre comillas invertidas | Un fragmento que solo *mencionaba* `` `customer` `` en una descripción contaba como acierto | Solo cuentan las líneas de campo |
| El troceador cortaba los campos largos solo por frases | Listas de más de 100 valores en una sola frase generaban fragmentos de hasta 1.447 tokens, que el modelo de embeddings truncaba en silencio | Si una frase no cabe, se corta por comas y después por espacios, repitiendo la línea del campo en cada trozo |

Los dos primeros problemas eran de la **métrica** y tenían el mismo efecto: hacían que el sistema pareciera mejor de lo que era. El primero afectaba sobre todo a las métricas sensibles a la posición (`Recall@1` y `MRR`), precisamente las que después permiten distinguir entre variantes que recuperan lo mismo pero en distinto orden. El segundo era más grave conceptualmente: premiaba fragmentos que no contenían la respuesta, sino solo una referencia de pasada a un campo con el mismo nombre. El tercero era un problema del **troceador**: un fragmento de 1.447 tokens supera con mucho la ventana del modelo de embeddings, de modo que una parte del texto nunca se representaba en el índice, sin que nada avisara de ello.

Además se añadieron tres opciones de configuración para poder experimentar:

- `extract.open_active_objects`: si es `false`, el extractor no copia dentro de un objeto otros objetos que ya tienen su propio documento.
- `chunker.strategy: fixed`: el troceador de referencia, con el `RecursiveCharacterTextSplitter` de LangChain aplicado al documento completo [LangChain].
- `--object-filter` en la búsqueda: si la pregunta nombra un objeto (*"a setup intent"*), se priorizan los fragmentos de ese objeto.

La lección de esta iteración es que **medir mal es peor que no medir**: un error en la evaluación no solo da números incorrectos, sino que puede llevar a elegir la variante equivocada con plena confianza.

## 5.2 Comparación de variantes en desarrollo

### 5.2.1 Primera ronda y ajustes de la evaluación

La primera ronda (variantes E0 a E5) dio conclusiones que se mantuvieron en la segunda, por lo que aquí se presentan directamente los números definitivos. Lo relevante de la primera ronda es lo que enseñó el análisis de fallos:

1. **Preguntas ambiguas.** Las preguntas de `charge` se escribieron cuando el corpus solo contenía ese objeto. Con 16 objetos, una pregunta como *"How can I return money to a customer?"* tiene más de una respuesta válida (por ejemplo, el objeto `refund`), pero la evaluación solo aceptaba `charge`.
2. **La métrica penalizaba las cajas abiertas.** En *"Which fields describe the payment method details?"* el primer resultado era el fragmento de `payment_method_details`, pero como esa caja se había abierto por no caber entera, su propia línea había quedado en otro fragmento y no contaba como acierto.
3. **Fallos reales** de búsqueda, como `cancellation_reason` de `setup_intent`.

En consecuencia se hicieron dos cambios, ambos con reglas generales y no caso a caso. En la **métrica**, una pregunta sobre una caja también se considera respondida por un fragmento que contiene sus campos hijos; la regla se aplica por igual a todas las variantes, incluido el troceado fijo. En las **preguntas**, a todas las preguntas respondibles de `charge` que no nombraban el objeto se les añadió *"on a charge"* (15 preguntas), y `charge-026` pasó a preguntar por el objeto `refund`, porque sin copias el estado de un reembolso vive en `refund.status`. La regla se aplicó a todas las preguntas afectadas, no solo a las que fallaban; aun así, es una decisión tomada sobre el conjunto de desarrollo y se discute como amenaza a la validez en el capítulo 6.

### 5.2.2 Resultados definitivos (56 preguntas)

La Tabla 5.2 recoge los resultados de todas las variantes sobre el conjunto de desarrollo. E1 (sin copias, troceado jerárquico con cabecera, 512 tokens y `bge-small-en-v1.5`) es la referencia; cada una de las demás cambia un único aspecto. Las filas E9 y E10 corresponden al experimento de escala (sección 5.5) y se incluyen para tener la tabla completa.

**Tabla 5.2.** Búsqueda en desarrollo (56 preguntas puntuadas).

| Variante | Método | R@1 | R@5 | R@5 IC95% | MRR |
|---|---|---:|---:|---|---:|
| E0 base (con copias) | semántico | 0.473 | 0.777 | [0.66, 0.88] | 0.608 |
| E0 base (con copias) | literal | 0.274 | 0.542 | [0.41, 0.67] | 0.408 |
| E0 base (con copias) | híbrido | 0.455 | 0.774 | [0.67, 0.88] | 0.587 |
| E1 sin copias | semántico | 0.562 | 0.902 | [0.82, 0.97] | 0.703 |
| E1 sin copias | literal | 0.318 | 0.604 | [0.48, 0.73] | 0.454 |
| E1 sin copias | híbrido | 0.562 | 0.827 | [0.73, 0.91] | 0.674 |
| **E1 sin copias** | **semántico + filtro** | **0.652** | **0.911** | **[0.83, 0.97]** | **0.777** |
| E2 troceado fijo | semántico | 0.464 | 0.622 | [0.49, 0.75] | 0.549 |
| E3 sin cabecera | semántico | 0.411 | 0.756 | [0.64, 0.86] | 0.557 |
| E4 `bge-base` | semántico | 0.464 | 0.848 | [0.75, 0.93] | 0.631 |
| E5 `bge-base` + prefijo | semántico | 0.509 | 0.893 | [0.81, 0.96] | 0.670 |
| E6 fragmentos de 256 tokens | semántico | 0.500 | 0.795 | [0.70, 0.88] | 0.660 |
| E7 reranker MiniLM | semántico + rerank | 0.604 | 0.902 | [0.82, 0.96] | 0.748 |
| E8 reranker BGE | semántico + filtro + rerank | 0.738 | 0.917 | [0.85, 0.97] | 0.838 |
| E9 50 objetos | semántico + filtro | 0.652 | 0.911 | [0.83, 0.97] | 0.779 |
| E10 fijo, 50 objetos | semántico | 0.375 | 0.604 | [0.47, 0.73] | 0.494 |

Leída en palabras: con la configuración final (E1 con filtro por objeto), en el **91,1 %** de los casos la ruta correcta aparece entre los cinco fragmentos que recibe el modelo, y en el **65,2 %** el primer fragmento ya es el correcto. El intervalo [0.83, 0.97] indica que, con 56 preguntas, el valor real podría estar razonablemente en cualquier punto de ese rango. Destaca el troceado fijo (E2), que solo encuentra la respuesta en el 62,2 % de los casos, y la búsqueda literal, claramente por debajo de la semántica en todas las variantes.

> **Figura 5.1** — Recall@5 por variante en desarrollo, con intervalos de confianza al 95 % (pendiente de generar). Gráfico de puntos horizontales, una fila por variante y método de la Tabla 5.2, ordenadas de mayor a menor; cada punto con su barra de error y una línea vertical de referencia en el valor de E1 semántico (0.902). Debe hacer visible que E2 y la búsqueda literal quedan claramente separadas y que la mayoría de las demás variantes se solapan con E1.

Como los intervalos de cada variante por separado se solapan mucho, la forma correcta de compararlas es calcular el intervalo de la diferencia sobre las mismas preguntas (Tabla 5.3).

**Tabla 5.3.** Comparaciones en desarrollo (diferencia de la primera menos la segunda).

| Comparación | ΔR@5 | IC95% | ΔMRR | IC95% | Conclusión |
|---|---:|---|---:|---|---|
| Jerárquico − troceado fijo | +0.280 | [+0.15, +0.40] | +0.154 | [+0.05, +0.27] | **Gana el jerárquico** |
| Con cabecera − sin cabecera | +0.146 | [+0.06, +0.24] | +0.145 | [+0.07, +0.22] | **La cabecera ayuda** |
| Sin copias − con copias | +0.125 | [+0.04, +0.23] | +0.094 | [+0.02, +0.16] | **No copiar objetos** |
| Semántico − híbrido | +0.074 | [+0.02, +0.15] | +0.028 | [−0.05, +0.11] | **El híbrido sobra** |
| Filtro por objeto − sin filtro | +0.009 | [+0.00, +0.03] | +0.074 | [+0.03, +0.13] | Mejora el orden (MRR) |
| `bge-base` − `bge-small` | −0.054 | [−0.14, +0.04] | −0.072 | [−0.15, +0.01] | El modelo grande no mejora |
| Prefijo − sin prefijo (`bge-base`) | +0.045 | [−0.02, +0.12] | +0.039 | [−0.02, +0.10] | No concluyente |
| 256 tokens − 512 tokens | −0.107 | [−0.20, −0.04] | −0.043 | [−0.13, +0.04] | **Fragmentos más pequeños empeoran** |
| Reranker BGE + filtro − filtro | +0.006 | [−0.09, +0.11] | +0.061 | [−0.03, +0.15] | Prometedor, no concluyente |

> **Figura 5.2** — Diferencias de Recall@5 y MRR respecto a la referencia, con intervalos de confianza (pendiente de generar). Gráfico de tipo *forest plot*: una fila por comparación de la Tabla 5.3, con dos marcadores (ΔR@5 y ΔMRR) y sus intervalos, y una línea vertical en cero. Debe permitir ver de un vistazo qué intervalos quedan enteramente a la derecha del cero (efecto demostrado) y cuáles lo cruzan.

A continuación se interpreta cada comparación.

**Troceado jerárquico frente a fijo (+0.280).** Es la hipótesis central del trabajo y el resultado más fuerte del capítulo. Cortar los documentos siguiendo el árbol de atributos de cada objeto, en lugar de hacerlo cada 512 tokens con el divisor de LangChain [LangChain], aumenta en 28 puntos la fracción de preguntas cuya respuesta llega al modelo. El intervalo [+0.15, +0.40] está lejos del cero: incluso en el peor caso compatible con los datos, la mejora es de 15 puntos. La explicación es que el troceado jerárquico mantiene cada caja entera y hace que cada fragmento hable de una sola cosa, mientras que el fijo mezcla en un fragmento el final de una caja y el principio de la siguiente. La ventaja se mantiene en control, en test y con 50 objetos (secciones 5.3 a 5.5).

**La cabecera (+0.146).** La ruta escrita al principio de cada fragmento (`Charge → payment method details → ach debit`) indica al modelo de embeddings a qué objeto y a qué caja pertenece el texto. Sin ella, los campos que se repiten en muchos sitios (`country`, `last4`...) son indistinguibles. Es una idea próxima a la de enriquecer cada fragmento con su contexto antes de indexarlo [Anthropic2024]. Su efecto en desarrollo es claro, aunque en test se reduce (sección 5.4).

**Sin copias (+0.125).** Algunos objetos de la spec contienen dentro copias completas de otros: por ejemplo, el error de `payment_intent` incluye un `setup_intent` entero. Al indexarlas, esas copias competían con el documento original y a menudo lo desplazaban del top-5. No copiar los objetos que ya tienen documento propio mejora la búsqueda en 12,5 puntos.

**El híbrido sobra (−0.074 respecto al semántico).** La búsqueda literal con SQLite FTS5 es mucho peor que la semántica en estas preguntas (0.604 frente a 0.902 en E1), y al combinarlas mediante *Reciprocal Rank Fusion* [Cormack2009] la literal arrastra el resultado hacia abajo. La diferencia en `Recall@5` es significativa a favor de la búsqueda semántica sola; en `MRR` el intervalo cruza el cero.

**El filtro por objeto (+0.009 en R@5, +0.074 en MRR).** Prácticamente no cambia cuántas respuestas entran en el top-5, pero sí las sube de puesto: `Recall@1` pasa de 0.562 a 0.652. Su efecto depende mucho de la cabecera. Sin cabecera, el filtro eleva `Recall@5` de 0.756 a 0.854; con cabecera, de 0.902 a 0.911. La razón es que la cabecera ya transmite el objeto, de modo que ambas técnicas se solapan parcialmente (Tabla 5.4).

**Tabla 5.4.** Efecto del filtro por objeto en cada variante (desarrollo).

| Variante | R@5 sin filtro | R@5 con filtro | MRR sin filtro | MRR con filtro |
|---|---:|---:|---:|---:|
| E1 sin copias | 0.902 | 0.911 | 0.703 | 0.777 |
| E2 troceado fijo | 0.622 | 0.801 | 0.549 | 0.667 |
| E3 sin cabecera | 0.756 | 0.854 | 0.557 | 0.754 |
| E4 `bge-base` | 0.848 | 0.929 | 0.631 | 0.770 |
| E5 `bge-base` + prefijo | 0.893 | 0.929 | 0.670 | 0.760 |
| E6 256 tokens | 0.795 | 0.839 | 0.660 | 0.768 |

La tabla muestra que el filtro ayuda más cuanto peor es la variante de partida: al troceado fijo le aporta casi 18 puntos, a la variante sin cabecera casi 10 y a la configuración final menos de uno. Dicho de otro modo, el filtro compensa en parte la información que le falta a un fragmento sobre su objeto, y esa información la aporta ya la cabecera en la configuración final.

**Modelo de embeddings, prefijo, 256 tokens y reranker.** Ninguna de estas variantes mejora de forma demostrable a E1:

- **`bge-base` frente a `bge-small`** [Xiao2023]: el modelo grande obtiene un `Recall@5` 5,4 puntos inferior, aunque el intervalo [−0.14, +0.04] cruza el cero. Con filtro por objeto, `bge-base` queda ligeramente por encima (0.929 frente a 0.911), pero esa diferencia tampoco es concluyente (+0.018 [−0.04, +0.07] en R@5; −0.007 [−0.07, +0.06] en MRR). No hay motivo, por tanto, para pagar el coste de un modelo mayor.
- **Prefijo de consulta** recomendado para BGE: +0.045 con un intervalo que cruza el cero. No concluyente.
- **Fragmentos de 256 tokens**: empeoran de forma significativa (−0.107 [−0.20, −0.04]). Con un presupuesto menor, cajas que con 512 tokens cabían enteras se parten, y se pierde la ventaja del troceado jerárquico. Es una confirmación indirecta de la hipótesis principal: lo que importa es que el fragmento respete la estructura.
- **Reordenación con *cross-encoder*** [Nogueira2019]: el reranker BGE combinado con el filtro obtiene el mejor `Recall@1` (0.738) y `MRR` (0.838) de toda la tabla, pero la diferencia frente al filtro solo (+0.006 en R@5, +0.061 en MRR) tiene intervalos que cruzan el cero. Respecto a E1 semántico sin filtro, la mejora en `MRR` sí es significativa (+0.135 [+0.04, +0.23]), pero esa comparación mezcla el efecto del filtro y el del reranker. El MiniLM no cambia `Recall@5` respecto a E1 (+0.000 [−0.07, +0.07]). Se considera un resultado prometedor que no se puede demostrar con 56 preguntas, y no se incorpora a la configuración final.

Con estos resultados se cerró la configuración final de búsqueda: **extracción sin copias, troceado jerárquico con cabecera, 512 tokens, `bge-small-en-v1.5`, búsqueda semántica con filtro por objeto y 5 fragmentos.**

## 5.3 Conjunto de control

Un riesgo evidente de decidir con 56 preguntas es ajustar el sistema a ellas. Para comprobarlo se generó el conjunto de control (`scripts/make_control_set.py`) **después** de cerrar las decisiones, y se evaluó una sola vez. Contiene 120 preguntas, 40 de cada tipo: describiendo el campo sin nombrarlo (`desc`), nombrándolo en lenguaje natural (`name`) y con la ruta exacta (`ident`).

**Tabla 5.5.** Búsqueda en control (120 preguntas, una sola evaluación).

| Variante | R@1 | R@5 | R@5 IC95% | MRR |
|---|---:|---:|---|---:|
| E0 como estaba | 0.458 | 0.767 | [0.69, 0.84] | 0.580 |
| E2 troceado fijo | 0.308 | 0.608 | [0.52, 0.69] | 0.416 |
| E3 sin cabecera | 0.433 | 0.742 | [0.66, 0.82] | 0.551 |
| E1 sin copias | 0.558 | 0.825 | [0.76, 0.89] | 0.663 |
| **Final: E1 + filtro por objeto** | **0.692** | **0.892** | **[0.83, 0.94]** | **0.768** |

**Tabla 5.6.** Comparaciones en control.

| Comparación en control | ΔR@5 | IC95% | ΔMRR | IC95% |
|---|---:|---|---:|---|
| Jerárquico − troceado fijo | +0.217 | [+0.13, +0.30] | +0.247 | [+0.16, +0.33] |
| Con cabecera − sin cabecera | +0.083 | [+0.04, +0.13] | +0.112 | [+0.06, +0.16] |
| Sin copias − con copias | +0.058 | [+0.03, +0.10] | +0.083 | [+0.05, +0.12] |
| Filtro por objeto − sin filtro | +0.067 | [+0.03, +0.12] | +0.105 | [+0.07, +0.14] |
| Final − como estaba | +0.125 | [+0.07, +0.18] | +0.188 | [+0.14, +0.24] |

**Tabla 5.7.** Configuración final en control, por tipo de pregunta.

| Tipo de pregunta | n | R@5 | MRR |
|---|---:|---:|---:|
| Describe el campo sin nombrarlo | 40 | 0.750 | 0.551 |
| Nombra el campo | 40 | 0.975 | 0.871 |
| Ruta exacta | 40 | 0.950 | 0.882 |

Las cuatro decisiones de diseño se confirman con preguntas que el sistema no había visto, y en el mismo orden de importancia que en desarrollo: troceado jerárquico, cabecera, sin copias y filtro. Todos los intervalos de las diferencias quedan por encima del cero. La configuración final pasa de 0.911 en desarrollo a 0.892 en control, una pérdida pequeña que indica que **no se ha sobreajustado** a las preguntas de desarrollo. El filtro por objeto, que en desarrollo apenas cambiaba `Recall@5`, aquí aporta 6,7 puntos demostrables.

El desglose por tipo identifica el punto débil del sistema: cuando la pregunta nombra el campo o da su ruta, la respuesta llega al modelo en el 95–97,5 % de los casos; cuando solo describe lo que se necesita sin nombrarlo, en el 75 %.

## 5.4 Test final: búsqueda

El test (`test.jsonl`) lo escribió un agente de IA **sin acceso a los fragmentos, a los resultados ni a la ejecución del sistema**, usando como única fuente de verdad la tabla de campos. Contiene 60 preguntas con respuesta (15 directas, 15 descriptivas, 15 sobre campos anidados y 15 sobre valores posibles) y 10 sin respuesta, repartidas entre los 16 objetos. Se evaluó una única vez, con la configuración ya cerrada.

**Tabla 5.8.** Búsqueda en test (60 preguntas con respuesta, una sola evaluación).

| Variante | R@1 | R@5 | R@5 IC95% | MRR |
|---|---:|---:|---|---:|
| Como estaba (E0) | 0.417 | 0.700 | [0.58, 0.82] | 0.542 |
| Troceado fijo (E2) | 0.353 | 0.533 | [0.41, 0.66] | 0.436 |
| Sin cabecera (E3) | 0.433 | 0.708 | [0.59, 0.82] | 0.547 |
| Sin copias (E1) | 0.500 | 0.767 | [0.66, 0.87] | 0.627 |
| **Final: E1 + filtro por objeto** | **0.617** | **0.850** | **[0.76, 0.93]** | **0.724** |

**Tabla 5.9.** Comparaciones en test.

| Comparación en test | ΔR@5 | IC95% | ΔMRR | IC95% |
|---|---:|---|---:|---|
| Jerárquico − troceado fijo | +0.233 | [+0.12, +0.35] | +0.191 | [+0.08, +0.30] |
| Con cabecera − sin cabecera | +0.058 | [−0.03, +0.15] | +0.079 | [+0.02, +0.15] |
| Sin copias − con copias | +0.067 | [+0.02, +0.13] | +0.085 | [+0.04, +0.14] |
| Filtro por objeto − sin filtro | +0.083 | [+0.02, +0.17] | +0.097 | [+0.04, +0.16] |
| Final − como estaba | +0.150 | [+0.07, +0.25] | +0.182 | [+0.11, +0.26] |

En el test, la configuración final pone la respuesta entre los cinco fragmentos recuperados en el **85 %** de las preguntas, con un intervalo de [0.76, 0.93]. **Este 0.850 es el número que se reporta como resultado final** del sistema de búsqueda. La caída gradual (0.911 en desarrollo, 0.892 en control, 0.850 en test) es la esperable al alejarse de las preguntas usadas para decidir, y es la razón de reservar un conjunto independiente.

Las comparaciones confirman la hipótesis principal: el troceado jerárquico supera al fijo en 23,3 puntos, con un intervalo lejos del cero. También se confirman la eliminación de copias y el filtro por objeto. La **cabecera se confirma solo a medias**: mejora claramente el orden (`MRR` +0.079, intervalo por encima del cero), pero su efecto sobre `Recall@5` (+0.058) no se distingue del ruido con 60 preguntas. Su efecto real es, por tanto, menor que el medido en desarrollo (+0.146).

> **Figura 5.3** — Recall@5 de las cinco variantes en desarrollo, control y test (pendiente de generar). Gráfico de barras agrupadas: un grupo por variante (E0, E2, E3, E1, final) y tres barras por grupo (desarrollo, control, test), con intervalos de confianza. Debe mostrar que el orden de las variantes se mantiene en los tres conjuntos y la caída gradual de la configuración final.

**Tabla 5.10.** Configuración final en test, por categoría.

| Categoría | n | R@5 | MRR |
|---|---:|---:|---:|
| Descriptivas (sin nombrar el campo) | 15 | 0.733 | 0.639 |
| Directas | 15 | 0.867 | 0.644 |
| Campos anidados | 15 | 0.800 | 0.833 |
| Valores posibles (enum) | 15 | 1.000 | 0.780 |

Por categorías, las preguntas sobre **valores posibles** se resuelven siempre (15 de 15 en el top-5), porque los valores de un `enum` están escritos literalmente en la línea del campo. Las **descriptivas** vuelven a ser el punto débil (0.733), coherente con lo observado en control (0.750): cuando la pregunta describe una necesidad sin nombrar el campo, la similitud entre pregunta y fragmento es menor. Las preguntas sobre **campos anidados** tienen un `Recall@5` intermedio (0.800) pero el `MRR` más alto (0.833): cuando se encuentran, suelen estar en primera posición, lo que es coherente con que cada caja anidada sea un fragmento propio con su ruta en la cabecera. Con 15 preguntas por categoría, estos valores deben leerse como tendencias.

## 5.5 Escala: de 16 a 50 objetos

Para comprobar si el método escala se repitió la evaluación con los 50 objetos de `data/objects_50.txt`, con la misma configuración y las mismas preguntas (variantes E9, jerárquico, y E10, troceado fijo). El índice pasa de 744 a 1.103 fragmentos (+48 %). El crecimiento no es proporcional al triple de objetos porque muchos de los objetos nuevos son pequeños y, al formar parte de la lista activa, dejan de copiarse dentro de otros.

**Tabla 5.11.** `Recall@5` al pasar de 16 a 50 objetos.

| `Recall@5` | Desarrollo 16 → 50 | Test 16 → 50 | Δ en test | IC95% |
|---|---|---|---:|---|
| Final (semántico + filtro) | 0.911 → 0.911 | 0.850 → 0.867 | +0.017 | [−0.03, +0.07] |
| Semántico sin filtro | 0.902 → 0.866 | 0.767 → 0.758 | −0.008 | [−0.05, +0.03] |
| Troceado fijo | 0.622 → 0.604 | 0.533 → 0.525 | −0.008 | [−0.07, +0.06] |

**Tabla 5.12.** Ventaja del troceado jerárquico sobre el fijo según el tamaño del corpus.

| Jerárquico − fijo | 16 objetos | 50 objetos |
|---|---:|---|
| Desarrollo | +0.280 | +0.262 [+0.12, +0.40] |
| Test | +0.233 | +0.233 [+0.13, +0.34] |

Al triplicar el número de objetos, **la configuración final no pierde rendimiento**: en desarrollo se queda exactamente igual (0.911) y en test sube ligeramente (+0.017), con un intervalo que incluye el cero, es decir, sin cambio demostrable. La ventaja del troceado jerárquico sobre el fijo es prácticamente la misma con 16 y con 50 objetos. El único descenso aparece en desarrollo **sin** el filtro por objeto (de 0.902 a 0.866, −0.036 con intervalo [−0.09, +0.00]; en `MRR`, −0.035 [−0.07, −0.01]): con más objetos hay más fragmentos parecidos que compiten, y es el filtro lo que hace al sistema robusto frente a un corpus mayor.

> **Figura 5.4** — Recall@5 con 16 y con 50 objetos (pendiente de generar). Gráfico de pendientes (*slope chart*) con dos columnas (16 y 50 objetos) y una línea por configuración (final, semántico sin filtro, fijo), un panel para desarrollo y otro para test. Debe mostrar que las líneas son casi horizontales y que la separación entre jerárquico y fijo se mantiene.

## 5.6 Generación

### 5.6.1 Cómo se mide

La generación se evalúa sin juez humano, con métricas automáticas que comparan los campos citados por el modelo con la spec (`rag.eval_generation`). La búsqueda es la configuración final (semántica con filtro, 5 fragmentos) y la temperatura es 0.

**Tabla 5.13.** Métricas de generación.

| Métrica | Qué significa |
|---|---|
| Acierto del campo | El modelo respondió y citó el campo correcto (o uno de sus hijos). Decir "no encontrado" nunca cuenta como acierto |
| Acierto cuando la búsqueda trajo el fragmento correcto | Lo mismo, pero solo cuando la búsqueda sí trajo el fragmento correcto. Mide al modelo sin culparle de los fallos de búsqueda |
| Alucinación | Cita un campo que no existe en la spec de Stripe |
| No presente en el contexto | Cita un campo real que no estaba en los fragmentos recuperados |
| Abstención | En las preguntas sin respuesta, dice que no lo encuentra |
| Fallos de búsqueda / del modelo | Sin el fragmento correcto en el top-5 / con él, pero respuesta incorrecta |

### 5.6.2 Desarrollo con Gemma

El modelo principal es `google/gemma-4-e4b` [Gemma4], un modelo general servido con LM Studio en un Mac M4 Pro. Se evaluó sobre las 58 preguntas de desarrollo (las 60 menos las 2 de tipo `metadata`).

**Tabla 5.14.** Generación en desarrollo con Gemma.

| Métrica | `dev_charge` | `dev_objects` |
|---|---:|---:|
| Acierto del campo | 0.923 [0.81, 1.00] | 0.933 [0.83, 1.00] |
| Acierto cuando la búsqueda trajo el fragmento correcto | 1.000 | 1.000 |
| Alucinación (campo que no existe) | 0.036 (1 de 28) | 0.000 |
| Campo real pero no presente en el contexto | 0.000 | 0.000 |
| Abstención en preguntas sin respuesta | 1.000 (2 de 2) | — |
| Errores de formato JSON | 0 | 0 |
| Fallos de búsqueda / fallos del modelo | 2 / 0 | 2 / 0 |
| Latencia media (Mac M4 Pro) | 25 s | 24 s |

El resultado más importante de esta tabla es la segunda fila: **cuando la búsqueda trae el fragmento correcto, el modelo acierta siempre**. Los cuatro fallos (dos en cada conjunto) son todos de búsqueda. Cuando la búsqueda falla, el modelo no se inventa la respuesta: o bien dice que no la encuentra (*"Why can a setup intent be canceled?"*), o responde con un campo real del contexto que no es el pedido.

La única alucinación es sutil. En *"Which fields describe the payment method details on a charge?"* el modelo citó `payment_method_details.reader` y `payment_method_details.read_method`. Esos campos existen, pero dentro de `card_present` e `interac_present`: el modelo se saltó un nivel de la ruta. El verificador la detecta porque comprueba la ruta completa contra la spec, no solo el nombre del campo. No se ajustó el prompt, porque no había errores del modelo que corregir.

### 5.6.3 Test con Gemma

**Tabla 5.15.** Generación en test con Gemma (70 preguntas, una sola evaluación).

| Métrica | Test |
|---|---:|
| Acierto del campo | 0.850 [0.75, 0.93] (n=60) |
| Acierto cuando la búsqueda trajo el fragmento correcto | 0.981 [0.94, 1.00] (51 de 52) |
| Alucinación (campo que no existe) | 0.000 (0 de 70) |
| Campo real pero no presente en el contexto | 0.000 |
| Abstención en preguntas sin respuesta | 1.000 (10 de 10) |
| Errores de formato JSON | 0 |
| Fallos de búsqueda / fallos del modelo | 8 / 1 |
| Latencia media / mediana | 16.4 s / 15.4 s |

En test, el modelo cita el campo correcto en el 85 % de las 60 preguntas con respuesta. De las 52 preguntas en las que la búsqueda le entregó el fragmento correcto, responde bien en 51. No inventa ningún campo en las 70 preguntas y reconoce las 10 preguntas sin respuesta, incluidas las 4 que preguntan por campos que no existen. De los 9 fallos, 8 son de búsqueda y solo 1 del modelo. La conclusión de desarrollo se mantiene: **el cuello de botella del sistema es la recuperación, no la generación**.

### 5.6.4 Comparación con Qwen-Coder

Se comparó Gemma con `qwen2.5-coder-7b-instruct` [Hui2024], un modelo especializado en código, cuantizado en formato Q4_K_M [llamacpp]. Las condiciones son idénticas: misma máquina, misma búsqueda, mismas preguntas, temperatura 0 y contexto de 8.192 tokens.

**Tabla 5.16.** Gemma frente a Qwen-Coder.

| | `google/gemma-4-e4b` | `qwen2.5-coder-7b-instruct` (Q4_K_M) |
|---|---:|---:|
| Tipo | general, 7.5B | especializado en código, 7B |
| Tamaño | 6.3 GB | 4.7 GB |
| Acierto del campo, desarrollo (56) | **0.929** | 0.857 |
| Acierto del campo, test (60) | 0.850 | 0.867 |
| Acierto con el fragmento correcto, test | 0.981 | 0.981 |
| Alucinación, test | 0.000 | 0.000 |
| Abstención en preguntas sin respuesta, test | 10 de 10 | 10 de 10 |
| Latencia media, test | 16.4 s | **6.5 s** |

**Tabla 5.17.** Diferencia en acierto del campo (Qwen − Gemma).

| Conjunto | Diferencia | IC95% |
|---|---:|---|
| Desarrollo | −0.071 | [−0.14, −0.02] |
| Test | +0.017 | [+0.00, +0.05] |

En **desarrollo, Gemma es mejor** con una diferencia real (el intervalo queda por debajo del cero). Los errores de Qwen son de un tipo concreto: cita campos del nivel equivocado (en *"Which fields describe a refund on a charge?"* da los campos del objeto `refund` en lugar del campo `refunds` del cargo), dos veces dice "no encontrado" teniendo la respuesta delante, y una vez fusiona dos rutas en una que no existe (`total_details.amount_total`). En **test los dos modelos empatan** en la práctica: Qwen queda 1,7 puntos por encima, una diferencia mínima cuyo intervalo llega hasta el cero. La ventaja de Gemma en desarrollo no se reproduce con preguntas nuevas.

En cambio, **Qwen es entre 2,5 y 4 veces más rápido** y ocupa 1,6 GB menos, lo que importa en un portátil con 8 GB de memoria de GPU. Con ambos modelos, la generación no es el cuello de botella: los dos responden bien en 51 de 52 preguntas cuando la búsqueda trae el fragmento correcto. Las latencias deben tomarse como orientativas, porque se midieron en el Mac a veces con otros procesos en marcha.

**Decisión.** El criterio fijado de antemano para elegir el modelo era el acierto en desarrollo, de modo que **Gemma es el modelo principal**. **Qwen queda como alternativa ligera** para la demostración y para el portátil con 8 GB de GPU, con la misma calidad en test y respuestas en unos 6 segundos. Mantener el criterio fijado antes de ver el test, aunque en test los modelos empaten, evita elegir el modelo a posteriori.

## 5.7 Corrección de las respuestas con un juez a ciegas

El acierto del campo mide si el modelo señala el campo correcto, pero no si lo que **dice** de él es cierto. Para medirlo, un agente de IA juzgó las 140 respuestas del test (70 de cada modelo) con `scripts/judge.py`, siguiendo el enfoque de modelo como juez (*LLM-as-judge*) [Zheng2023]:

- **A ciegas**: las respuestas de los dos modelos se mezclaron al azar con identificadores neutros (`J001`...), de modo que el juez no sabía qué modelo había escrito cada una.
- **Contra la spec**: el juez ve la pregunta, la respuesta y lo que dice la spec de los campos esperados y de los citados (tipo, nullable, valores, descripción). No ve los fragmentos recuperados.
- **Rúbrica fija**: *correcta* (campo correcto y nada contradice la spec), *parcial* (campo correcto pero incompleta o con un error secundario), *incorrecta* (campo equivocado o algo que contradice la spec) y *no responde*.

Es una evaluación con un modelo como juez, no una evaluación humana; sus limitaciones se discuten en el capítulo 6.

**Tabla 5.18.** Veredictos del juez sobre las 60 preguntas con respuesta del test.

| | Gemma (principal) | Qwen-Coder |
|---|---:|---:|
| **Correcta** | **0.817** [0.72, 0.90] | 0.750 [0.63, 0.85] |
| Parcial | 0.050 | 0.100 |
| Incorrecta | 0.100 | 0.067 |
| No responde | 0.033 | 0.083 |
| Correcta o parcial | 0.867 [0.77, 0.95] | 0.850 [0.75, 0.93] |
| Preguntas sin respuesta bien tratadas | 10 de 10 | 10 de 10 |
| Respuestas `enum` con todos los valores (comprobación automática) | 15 de 15 | 14 de 15 |

**Tabla 5.19.** Respuestas correctas por categoría.

| Categoría | Gemma | Qwen |
|---|---:|---:|
| Descriptivas | 0.73 | 0.67 |
| Directas | 0.80 | 0.73 |
| Anidadas | 0.73 | 0.67 |
| Valores posibles | 1.00 | 0.93 |

Con Gemma, **el 81,7 % de las respuestas son correctas en contenido**, algo menos que el 85 % de acierto del campo, porque ahora también se juzga la explicación. Si se cuentan también las parciales, el 86,7 % de las respuestas señalan el campo correcto sin errores graves. Los errores típicos son tres: usar un campo vecino (`next_payment_attempt` en lugar de `due_date`), dar respuestas incompletas (por ejemplo, una sola de las tres pruebas de entrega de una disputa) y algún matiz falso (afirmar que 99 es el máximo fijo cuando la spec dice que es el valor por defecto).

Gemma vuelve a quedar por delante de Qwen en respuestas correctas (0.817 frente a 0.750) y en todas las categorías, aunque los intervalos de ambos modelos se solapan. La diferencia está sobre todo en las respuestas parciales y en las abstenciones: Qwen da más respuestas incompletas y más veces no responde. Ambos tratan bien las 10 preguntas sin respuesta.

> **Figura 5.5** — Distribución de veredictos del juez por modelo (pendiente de generar). Barras apiladas horizontales al 100 %, una por modelo, con los cuatro segmentos (correcta, parcial, incorrecta, no responde) en colores ordenados de mejor a peor. Debe mostrar que la mayor parte de la diferencia entre modelos está en los segmentos parcial y no responde.

## 5.8 Taxonomía de errores por fase

Para saber dónde actuar a continuación, `scripts/error_taxonomy.py` clasifica cada pregunta fallada del test en una **fase** y un **tipo**, con reglas fijas. Si la búsqueda no trajo el fragmento correcto, el error es de búsqueda; si lo trajo, es de generación. El tipo de error de búsqueda se decide con los datos: qué objetos nombra la pregunta, de qué objeto son los cinco fragmentos recuperados, si existe un campo con el mismo nombre en otra ruta y si la respuesta estaba en un grupo de primer nivel. El tipo de error de generación sale del veredicto del juez. Se considera fallo toda respuesta que el juez no calificó como correcta.

**Tabla 5.20.** Fallos del test por fase y tipo (60 preguntas con respuesta).

| Fase | Tipo de error | Gemma | Qwen |
|---|---|---:|---:|
| Búsqueda | Cajón de sastre (grupo de primer nivel) | 4 | 4 |
| | Varios objetos en la pregunta | 2 | 2 |
| | Campo gemelo en otra caja | 1 | 1 |
| | Caja equivocada del objeto correcto | 1 | 1 |
| Generación | Elige un campo equivocado del contexto | 1 | 1 |
| | Incompleta o con un matiz falso | 1 | 5 |
| | Falsa abstención | 1 | 1 |
| | Campo inventado | 0 | 0 |
| **Total** | | **11 de 60** | **15 de 60** |

Con Gemma hay 11 fallos: 8 de búsqueda y 3 de generación. Los tipos de búsqueda se ilustran con ejemplos reales del test:

- **Cajón de sastre.** Los campos sueltos de primer nivel de un objeto se agrupan en orden alfabético, y un mismo grupo puede mezclar campos sin relación entre sí. En *"What is the due date of an invoice?"*, `due_date` está en un fragmento junto con `customer_phone`, `customer_tax_ids`, `default_source`... y queda 8.º de los 81 fragmentos de `invoice`. El modelo recibe otros cinco grupos de la factura y responde con `next_payment_attempt`, un campo vecino.
- **Varios objetos en la pregunta.** En *"Is there a Stripe-hosted page so a customer can look at an invoice and pay it?"*, el filtro prioriza `customer` e `invoice`, y sale primero `customer`.
- **Campo gemelo.** En *"At what time did an invoice transition to paid?"* llega `payments.data.status_transitions.paid_at` (el del pago) en lugar de `status_transitions.paid_at` (el de la factura): dos campos con el mismo nombre y significado parecido en cajas distintas.
- **Caja equivocada.** El enlace de recuperación de una sesión de checkout caducada está en `after_expiration.recovery`, y llegan otras cajas de `checkout.session`.

Tres observaciones se desprenden de la tabla:

1. **El cuello de botella es la búsqueda** (8 de 11 fallos con Gemma) y, dentro de ella, el **cajón de sastre** (4 de 8). Es la limitación principal de la regla que agrupa los valores sueltos de primer nivel, y afecta sobre todo a objetos grandes como `invoice` (tres de los fallos de búsqueda).
2. **Al fragmento correcto le faltó poco.** Su puesto entre los fragmentos de su propio objeto fue 2, 6, 6, 8, 10, 12, 14 y 26. En 5 de los 8 fallos estaba entre los 10 primeros, de modo que pasar al modelo 10 fragmentos en lugar de 5 probablemente recuperaría varios de ellos.
3. **Los dos modelos comparten exactamente los mismos fallos de búsqueda**, como era de esperar con la misma búsqueda, y la diferencia entre ellos está en la generación: Qwen da 5 respuestas incompletas o con matices falsos frente a 1 de Gemma.

> **Figura 5.6** — Puesto del fragmento correcto entre los de su objeto en los 8 fallos de búsqueda (pendiente de generar). Gráfico de puntos sobre un eje horizontal de puestos (1 a 30), un punto por fallo (2, 6, 6, 8, 10, 12, 14 y 26) coloreado por tipo de error, con líneas verticales en los puestos 5 y 10. Debe mostrar que la mayoría de los fallos están justo por encima del corte de 5.

## 5.9 Iteración 4: small-to-big y segundo test

La taxonomía de errores motivó una última iteración. La idea, conocida como *small-to-big* o recuperación por documento padre [LlamaIndex], consiste en indexar cada campo por separado (cabecera, línea y descripción) para la búsqueda, y entregar al modelo el fragmento grande al que pertenece. Ataca directamente el cajón de sastre: en un fragmento pequeño, el significado de `due_date` no se diluye entre campos sin relación. Se implementó como la opción `chunker.small_to_big` (variante E11).

Como la idea surgió del análisis de los fallos del test, evaluarla sobre ese mismo test habría sido hacer trampa. Por eso se decidió con desarrollo y se evaluó **una sola vez** en un **segundo test a ciegas** (`test2.jsonl`, 70 preguntas nuevas escritas por un agente con las mismas reglas que el primero).

**Tabla 5.21.** Búsqueda con small-to-big (semántica + filtro).

| Búsqueda | Desarrollo R@5 / MRR | Test 2 R@5 / MRR |
|---|---|---|
| Actual (E1) | 0.911 / 0.777 | 0.958 / 0.821 |
| Small-to-big (E11) | 0.970 / 0.960 | 0.975 / 0.868 |
| Diferencia | +0.060 [−0.00, +0.13] / **+0.183 [+0.10, +0.27]** | +0.017 [−0.04, +0.07] / +0.047 [−0.04, +0.14] |

**Tabla 5.22.** Test 2 en detalle (60 preguntas con respuesta, semántica + filtro).

| Variante | R@1 | R@5 | R@5 IC95% | MRR |
|---|---:|---:|---|---:|
| Troceado fijo | 0.533 | 0.808 | [0.71, 0.90] | 0.684 |
| Actual (E1) | 0.681 | 0.958 | [0.91, 1.00] | 0.821 |
| Small-to-big (E11) | 0.758 | 0.975 | [0.93, 1.00] | 0.868 |

En el test 2, el sistema actual frente al troceado fijo mantiene una ventaja demostrable (+0.150 [+0.05, +0.25] en `Recall@5`), lo que añade una cuarta confirmación de la hipótesis principal, esta vez con el filtro por objeto aplicado a ambos.

**Tabla 5.23.** Generación en test 2 con Gemma.

| Métrica | Actual | Small-to-big |
|---|---:|---:|
| Acierto del campo | 0.950 | 0.950 |
| Campos inventados | 0 de 70 | 2 de 70 |
| Sin respuesta reconocidas | 10 de 10 | 10 de 10 |

En desarrollo, small-to-big mejora de forma clara el **orden** de los resultados (`MRR` +0.183, intervalo por encima del cero) y sube `Recall@5` en 6 puntos, aunque este último intervalo roza el cero. En el primer test, analizado a posteriori y por tanto no como evaluación limpia, arregla 6 de los 8 fallos de búsqueda, incluidos los tres de `invoice`, pero rompe 4 preguntas que antes acertaban: `Recall@5` pasa de 0.850 a 0.900.

En la evaluación limpia del test 2, sin embargo, las diferencias en búsqueda (+0.017 en R@5 y +0.047 en MRR) no son demostrables, y en generación el acierto del campo es idéntico (0.950 en ambos casos), con dos campos inventados con small-to-big frente a ninguno con el sistema actual. La razón es que el sistema actual ya trae el fragmento correcto en el 95,8 % de las preguntas del test 2, y queda muy poco margen de mejora.

**Decisión.** Small-to-big mejora el orden de la búsqueda y resuelve los fallos de cajón de sastre, pero **no mejora las respuestas finales** en una evaluación limpia y añade dos alucinaciones. Por tanto, **se mantiene E1 como configuración final** y small-to-big queda como opción documentada y disponible en la configuración.

Cabe señalar que el test 2 resultó más fácil para el sistema que el primer test (0.958 frente a 0.850 en `Recall@5` con la misma configuración). Las dos cifras son legítimas, pero el 0.850 del primer test sigue siendo el resultado final que se reporta, por ser la evaluación prevista desde el principio.

## 5.10 Resumen de resultados

**Tabla 5.24.** Resumen por conjunto.

| | Desarrollo | Control | **Test** |
|---|---:|---:|---:|
| Preguntas puntuadas | 56 | 120 | 60 |
| `Recall@5`, configuración final | 0.911 | 0.892 | **0.850** |
| `Recall@5`, troceado fijo | 0.622 | 0.608 | 0.533 |
| Jerárquico − fijo | +0.280 | +0.217 | **+0.233** |
| Acierto del campo (Gemma) | 0.929 | — | **0.850** |
| Alucinación (Gemma) | 1 de 58 | — | **0 de 70** |
| Respuesta correcta según el juez (Gemma) | — | — | **0.817** |

En resumen:

1. **Hipótesis principal confirmada.** El troceado jerárquico por el árbol de atributos supera al troceado fijo en los tres conjuntos (+0.22 a +0.28 en `Recall@5`), también con 50 objetos y en el segundo test.
2. **Construcción del corpus.** No copiar objetos dentro de otros mejora la búsqueda (+0.06 a +0.13).
3. **Filtro por objeto.** Mejora el orden de los resultados y hace al sistema robusto cuando crece el corpus.
4. **Cabecera con la ruta.** Mejora el orden (`MRR`); su efecto sobre `Recall@5` solo se demuestra en desarrollo y control.
5. **Corrección de las respuestas.** El 82 % de las respuestas son correctas en contenido según el juez a ciegas, y se reconocen todas las preguntas sin respuesta.
6. **El modelo no es el cuello de botella.** Con el fragmento correcto responde bien en el 98 % de los casos y no inventa campos en el test. Los errores que quedan son de búsqueda, sobre todo en preguntas que describen el campo sin nombrarlo y en los grupos de campos sueltos de objetos grandes.

> **Figura 5.7** — Embudo de la configuración final en el test (pendiente de generar). Diagrama de barras decrecientes: 60 preguntas con respuesta → 52 con el fragmento correcto en el top-5 (búsqueda) → 51 con el campo correcto (generación) → 49 correctas según el juez. Debe hacer visible que la mayor pérdida se produce en la etapa de búsqueda.
