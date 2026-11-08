# Diario de experimentos

Cada experimento cambia **una sola cosa**. Se reproducen todos con:

```bash
uv run python scripts/experiment.py
```

La lista de variantes está en `configs/experiments.yaml` y los resultados quedan en `out/experiments/<variante>/`.

Las tablas de los conjuntos reservados (control, test y test2) se reproducen con `uv run python scripts/heldout.py`, que escribe `out/experiments/heldout_<conjunto>.md`.

## Cómo se mide

- **Preguntas de desarrollo**: `dev_charge.jsonl` + `dev_objects.jsonl`. Con ellas se decide qué cambios se quedan.
- **Preguntas de control**: `control.jsonl`. Se generan una vez y se evalúan **una sola vez, al final**, para comprobar que lo ganado no es un ajuste a las preguntas de desarrollo.
- **Métrica principal**: `Recall@5`, la fracción de rutas correctas que aparecen en los 5 primeros chunks. También `Recall@1` y `MRR`.
- **Intervalo de confianza al 95%** por remuestreo (bootstrap). Si el intervalo de una diferencia cruza el cero, no se puede afirmar que una variante gana.
- Un chunk **solo cuenta como acierto si es del objeto correcto** y la ruta aparece como **línea de campo** (`` - `ruta` (tipo) ``), no si solo se menciona dentro de una descripción.
- Las preguntas de la categoría `metadata` (*"¿qué campos son nullable?"*) no se cuentan: su respuesta está repartida por todo el objeto y no es un problema de búsqueda, sino de consultar la tabla.

## Iteración 0 · Correcciones antes de medir

Antes de experimentar se corrigieron tres problemas que distorsionaban los números:

| Problema | Efecto | Arreglo |
|---|---|---|
| `eval_retrieval.py` filtraba los resultados por objeto **antes** de calcular las posiciones | Un chunk correcto en el puesto 4, detrás de tres de otros objetos, contaba como puesto 1. `Recall@1`, `Recall@3` y `MRR` salían inflados. `Recall@5` no se veía afectado | Los chunks de otros objetos ocupan su puesto y cuentan como fallo |
| La métrica aceptaba cualquier palabra entre comillas invertidas | Un chunk que solo *mencionaba* `` `customer` `` en una descripción contaba como acierto | Solo cuentan las líneas de campo |
| El troceador cortaba los campos largos solo por frases | Listas de más de 100 valores en una sola frase generaban chunks de hasta 1.447 tokens, que el embedder truncaba en silencio | Si una frase no cabe, se corta por comas y después por espacios, repitiendo la línea del campo en cada trozo |

Además se añadieron tres opciones para poder experimentar:

- `extract.open_active_objects`: si es `false`, el extractor no copia dentro de un objeto otros objetos que ya tienen su propio documento.
- `chunker.strategy: fixed`: el troceador de referencia (el "tonto"), con el `RecursiveCharacterTextSplitter` de LangChain sobre el documento completo.
- `--object-filter` en la búsqueda: si la pregunta nombra un objeto (*"a setup intent"*), se priorizan los chunks de ese objeto.

## Iteración 1 · Primera ronda (E0–E5)

Primera comparación de las variantes. Sus conclusiones se mantuvieron en la ronda 2, así que los números definitivos están en la tabla de la iteración 2. Lo que se aprendió del análisis de fallos:

1. **Preguntas ambiguas.** Las preguntas de `charge` se escribieron cuando el corpus solo tenía `charge`. Con 16 objetos, *"How can I return money to a customer?"* tiene más de una respuesta válida (el objeto `refund`, por ejemplo), pero la evaluación solo aceptaba `charge`.
2. **La métrica penalizaba las cajas abiertas.** En *"Which fields describe the payment method details?"* el primer resultado era el chunk de `payment_method_details`, pero como esa caja se abrió por no caber, su propia línea quedó en otro chunk y no contaba.
3. **Fallos reales** de búsqueda, como `cancellation_reason` de `setup_intent`.

Cambios para la ronda 2:

- **Métrica:** una pregunta sobre una caja también la responde un chunk que contiene sus campos hijos. Se aplica igual a todas las variantes, incluido el troceado fijo.
- **Preguntas:** regla objetiva, no caso a caso. A **todas** las preguntas respondibles de `charge` que no nombran el objeto se les añade *"on a charge"* (15 preguntas). `charge-026` pasa a preguntar por el objeto `refund`, porque sin copias el campo de estado de un reembolso vive en `refund.status`.

## Iteración 2 · Resultados definitivos en desarrollo (56 preguntas)

| Variante | Método | R@1 | R@5 | R@5 IC95% | MRR |
|---|---|---:|---:|---|---:|
| E0 base (con copias) | semántico | 0.473 | 0.777 | [0.66, 0.88] | 0.608 |
| E0 base (con copias) | híbrido | 0.455 | 0.774 | [0.67, 0.88] | 0.587 |
| E1 sin copias | semántico | 0.562 | 0.902 | [0.82, 0.97] | 0.703 |
| E1 sin copias | literal | 0.318 | 0.604 | [0.48, 0.73] | 0.454 |
| E1 sin copias | híbrido | 0.562 | 0.827 | [0.73, 0.91] | 0.674 |
| **E1 sin copias** | **semántico + filtro** | **0.652** | **0.911** | **[0.83, 0.97]** | **0.777** |
| E2 troceado fijo | semántico | 0.464 | 0.622 | [0.49, 0.75] | 0.549 |
| E3 sin cabecera | semántico | 0.411 | 0.756 | [0.64, 0.86] | 0.557 |
| E4 `bge-base` | semántico | 0.464 | 0.848 | [0.75, 0.93] | 0.631 |
| E5 `bge-base` + prefijo | semántico | 0.509 | 0.893 | [0.81, 0.96] | 0.670 |
| E6 chunks de 256 tokens | semántico | 0.500 | 0.795 | [0.70, 0.88] | 0.660 |
| E7 reranker MiniLM | semántico + rerank | 0.604 | 0.902 | [0.82, 0.96] | 0.748 |
| E8 reranker BGE | semántico + filtro + rerank | 0.738 | 0.917 | [0.85, 0.97] | 0.838 |

| Comparación | ΔR@5 | IC95% | ΔMRR | IC95% | Conclusión |
|---|---:|---|---:|---|---|
| Jerárquico − troceado fijo | +0.280 | [+0.15, +0.40] | +0.154 | [+0.05, +0.27] | **Gana el jerárquico** |
| Con cabecera − sin cabecera | +0.146 | [+0.06, +0.24] | +0.145 | [+0.07, +0.22] | **La cabecera ayuda** |
| Sin copias − con copias | +0.125 | [+0.04, +0.23] | +0.094 | [+0.02, +0.16] | **No copiar objetos** |
| Semántico − híbrido | +0.074 | [+0.02, +0.15] | +0.028 | [−0.05, +0.11] | **El híbrido sobra** |
| Filtro por objeto − sin filtro | +0.009 | [+0.00, +0.03] | +0.074 | [+0.03, +0.13] | Mejora el orden (MRR) |
| `bge-base` − `bge-small` | −0.054 | [−0.14, +0.04] | −0.072 | [−0.15, +0.01] | El modelo grande no mejora |
| Prefijo − sin prefijo (`bge-base`) | +0.045 | [−0.02, +0.12] | +0.039 | [−0.02, +0.10] | No concluyente |
| 256 tokens − 512 tokens | −0.107 | [−0.20, −0.04] | −0.043 | [−0.13, +0.04] | **Chunks más pequeños empeoran** |
| Reranker BGE + filtro − filtro | +0.006 | [−0.09, +0.11] | +0.061 | [−0.03, +0.15] | Prometedor, no concluyente |

### Qué significa cada resultado

- **Jerárquico frente a fijo (+0.28).** Es la hipótesis central del TFG. Cortar por el árbol de atributos, en vez de cada 512 tokens, mantiene cada caja entera y hace que cada chunk hable de una sola cosa.
- **La cabecera (+0.15).** La ruta escrita arriba (`Charge → payment method details → ach debit`) le dice al embedder de qué objeto y de qué caja es el chunk. Sin ella, los campos repetidos (`country`, `last4`...) son indistinguibles.
- **Sin copias (+0.13).** Algunos objetos llevan dentro copias completas de otros: el error de `payment_intent` contiene un `setup_intent` entero. Esas copias competían con el original y lo tapaban.
- **El híbrido sobra.** La búsqueda literal (FTS5) es mucho peor que la semántica en estas preguntas, y al mezclarlas con RRF arrastra el resultado hacia abajo.
- **El filtro por objeto** no cambia cuántas respuestas aparecen en el top-5, pero las sube de puesto. Ayuda mucho más sin cabecera (0.756 → 0.854) que con ella: la cabecera ya transmite el objeto.
- **Modelo grande, prefijo, chunks pequeños y reranker** no mejoran de forma clara. Los chunks de 256 tokens empeoran porque parten cajas que con 512 cabían enteras.

## Comprobación final · Conjunto de control (120 preguntas, una sola evaluación)

`control.jsonl` se generó con `scripts/make_control_set.py` **después** de cerrar las decisiones, y se evaluó una sola vez. Tiene tres tipos de pregunta, 40 de cada uno: describiendo el campo sin nombrarlo (`desc`), nombrándolo en lenguaje natural (`name`) y con la ruta exacta (`ident`).

| Variante | R@1 | R@5 | R@5 IC95% | MRR |
|---|---:|---:|---|---:|
| E0 como estaba | 0.458 | 0.767 | [0.69, 0.84] | 0.580 |
| E2 troceado fijo | 0.308 | 0.608 | [0.52, 0.69] | 0.416 |
| E3 sin cabecera | 0.433 | 0.742 | [0.66, 0.82] | 0.551 |
| E1 sin copias | 0.558 | 0.825 | [0.76, 0.89] | 0.663 |
| **Final: E1 + filtro por objeto** | **0.692** | **0.892** | **[0.83, 0.94]** | **0.768** |

| Comparación en control | ΔR@5 | IC95% | ΔMRR | IC95% |
|---|---:|---|---:|---|
| Jerárquico − troceado fijo | +0.217 | [+0.13, +0.30] | +0.247 | [+0.16, +0.33] |
| Con cabecera − sin cabecera | +0.083 | [+0.04, +0.13] | +0.112 | [+0.06, +0.16] |
| Sin copias − con copias | +0.058 | [+0.03, +0.10] | +0.083 | [+0.05, +0.12] |
| Filtro por objeto − sin filtro | +0.067 | [+0.03, +0.12] | +0.105 | [+0.07, +0.14] |
| Final − como estaba | +0.125 | [+0.07, +0.18] | +0.188 | [+0.14, +0.24] |

| Final por tipo de pregunta | n | R@5 | MRR |
|---|---:|---:|---:|
| Describe el campo sin nombrarlo | 40 | 0.750 | 0.551 |
| Nombra el campo | 40 | 0.975 | 0.871 |
| Ruta exacta | 40 | 0.950 | 0.882 |

**Conclusión:** las cuatro decisiones de diseño se confirman con preguntas que el sistema no había visto, en el mismo orden que en desarrollo. La configuración final pierde muy poco al pasar de desarrollo (0.911) a control (0.892), así que no se ha sobreajustado a las preguntas de desarrollo. El punto débil son las preguntas que describen un campo sin nombrarlo.

## Iteración 3 · Generación con un modelo local (desarrollo, 58 preguntas)

Modelo `google/gemma-4-e4b` (7.5B, general) servido con LM Studio en el Mac (`configs/default.yaml`). Búsqueda final: semántica + filtro por objeto, 5 fragmentos. Temperatura 0.

Cómo se mide (`rag.eval_generation`), sin juez humano:

| Métrica | Qué significa |
|---|---|
| Acierto del campo | El modelo respondió y citó el campo correcto (o uno de sus hijos). Decir "no encontrado" nunca cuenta como acierto |
| Acierto cuando la búsqueda trajo el fragmento correcto | Lo mismo, pero solo cuando la búsqueda sí trajo el fragmento correcto. Mide al modelo sin culparle de los fallos de búsqueda |
| Alucinación | Cita un campo que no existe en la spec de Stripe |
| No presente en el contexto | Cita un campo real que no estaba en los fragmentos recuperados |
| Abstención | En las preguntas sin respuesta, dice que no lo encuentra |
| Fallos de búsqueda / del modelo | Sin el fragmento correcto en el top-5 / con él, pero respuesta incorrecta |

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

**Qué significa:**

- **El modelo no falla cuando tiene el fragmento correcto.** Todos los errores vienen de la búsqueda. El cuello de botella del sistema es la recuperación, no la generación.
- **Cuando la búsqueda falla, el modelo no se inventa la respuesta:** dice que no lo encuentra (*"Why can a setup intent be canceled?"*) o responde con un campo real del contexto que no es el pedido.
- **La única alucinación es sutil:** en *"Which fields describe the payment method details on a charge?"* el modelo citó `payment_method_details.reader` y `payment_method_details.read_method`. Esos campos existen, pero dentro de `card_present` e `interac_present`: se saltó un nivel de la ruta. El verificador la detecta porque comprueba la ruta completa contra la spec.
- No se ha ajustado el prompt: no había errores del modelo que corregir.

## Test final · 70 preguntas, evaluadas una sola vez

`test.jsonl` lo escribió un agente de IA **sin acceso a los fragmentos, a los resultados ni a la ejecución del sistema**, usando solo la tabla de campos como fuente de verdad. Son 60 preguntas con respuesta (15 directas, 15 descriptivas, 15 de campos anidados y 15 de valores posibles) y 10 sin respuesta, sobre los 16 objetos. Se evaluó una única vez, con la configuración ya cerrada.

### Búsqueda

| Variante | R@1 | R@5 | R@5 IC95% | MRR |
|---|---:|---:|---|---:|
| Como estaba (E0) | 0.417 | 0.700 | [0.58, 0.82] | 0.542 |
| Troceado fijo (E2) | 0.353 | 0.533 | [0.41, 0.66] | 0.436 |
| Sin cabecera (E3) | 0.433 | 0.708 | [0.59, 0.82] | 0.547 |
| Sin copias (E1) | 0.500 | 0.767 | [0.66, 0.87] | 0.627 |
| **Final: E1 + filtro por objeto** | **0.617** | **0.850** | **[0.76, 0.93]** | **0.724** |

| Comparación en test | ΔR@5 | IC95% | ΔMRR | IC95% |
|---|---:|---|---:|---|
| Jerárquico − troceado fijo | +0.233 | [+0.12, +0.35] | +0.191 | [+0.08, +0.30] |
| Con cabecera − sin cabecera | +0.058 | [−0.03, +0.15] | +0.079 | [+0.02, +0.15] |
| Sin copias − con copias | +0.067 | [+0.02, +0.13] | +0.085 | [+0.04, +0.14] |
| Filtro por objeto − sin filtro | +0.083 | [+0.02, +0.17] | +0.097 | [+0.04, +0.16] |
| Final − como estaba | +0.150 | [+0.07, +0.25] | +0.182 | [+0.11, +0.26] |

| Final por categoría | n | R@5 | MRR |
|---|---:|---:|---:|
| Descriptivas (sin nombrar el campo) | 15 | 0.733 | 0.639 |
| Directas | 15 | 0.867 | 0.644 |
| Campos anidados | 15 | 0.800 | 0.833 |
| Valores posibles (enum) | 15 | 1.000 | 0.780 |

**Qué significa:**

- **El troceado jerárquico gana al fijo en los tres conjuntos** (desarrollo +0.28, control +0.22, test +0.23). Es el resultado principal.
- **Sin copias y filtro por objeto se confirman** en test.
- **La cabecera se confirma solo a medias:** mejora claramente el orden (`MRR` +0.079), pero su efecto sobre `Recall@5` (+0.058) no se distingue del ruido con 60 preguntas. Su efecto real es menor que el medido en desarrollo.
- **Las preguntas descriptivas son el punto débil** en control (0.750) y en test (0.733): cuando la pregunta describe lo que se necesita sin nombrar el campo, la búsqueda falla más.
- **`Recall@5` baja de forma gradual**: 0.911 en desarrollo, 0.892 en control y 0.850 en test. Es lo esperable al alejarse de las preguntas usadas para decidir, y **0.850 es el número que se debe reportar como resultado final**.

### Generación (Gemma, una sola evaluación)

| Métrica | Test (70 preguntas) |
|---|---:|
| Acierto del campo | 0.850 [0.75, 0.93] (n=60) |
| Acierto cuando la búsqueda trajo el fragmento correcto | 0.981 [0.94, 1.00] (51 de 52) |
| Alucinación (campo que no existe) | 0.000 (0 de 70) |
| Campo real pero no presente en el contexto | 0.000 |
| Abstención en preguntas sin respuesta | 1.000 (10 de 10) |
| Errores de formato JSON | 0 |
| Fallos de búsqueda / fallos del modelo | 8 / 1 |
| Latencia media / mediana | 16.4 s / 15.4 s |

**Qué significa:** el modelo responde bien prácticamente siempre que la búsqueda le da el fragmento correcto (51 de 52), no se inventa ningún campo en las 70 preguntas y reconoce las 10 preguntas sin respuesta, incluidas las 4 que preguntan por campos que no existen. De los 9 fallos, 8 son de la búsqueda. **El cuello de botella del sistema es la recuperación.**

## Experimento de escala · de 16 a 50 objetos

Misma configuración y mismas preguntas, con los 50 objetos de `data/objects_50.txt` (`data.objects` en la configuración). Variantes E9 (jerárquico) y E10 (troceado fijo).

El índice pasa de 744 a 1.103 chunks (+48%) y no de forma proporcional al triple de objetos: muchos de los objetos nuevos son pequeños y, al estar activos, ya no se copian dentro de otros.

| `Recall@5` | Desarrollo 16 → 50 | Test 16 → 50 | Δ en test | IC95% |
|---|---|---|---:|---|
| Final (semántico + filtro) | 0.911 → 0.911 | 0.850 → 0.867 | +0.017 | [−0.03, +0.07] |
| Semántico sin filtro | 0.902 → 0.866 | 0.767 → 0.758 | −0.008 | [−0.05, +0.03] |
| Troceado fijo | 0.622 → 0.604 | 0.533 → 0.525 | −0.008 | [−0.07, +0.06] |

| Jerárquico − fijo | 16 objetos | 50 objetos |
|---|---:|---|
| Desarrollo | +0.280 | +0.262 [+0.12, +0.40] |
| Test | +0.233 | +0.233 [+0.13, +0.34] |

**Qué significa:** al triplicar el corpus, **la configuración final no pierde rendimiento**, y la ventaja del troceado jerárquico sobre el fijo es la misma. En desarrollo, sin el filtro por objeto hay una caída pequeña (−0.036): el filtro es lo que hace al sistema robusto a un corpus más grande. El método escala.

## Comparación de modelos · Gemma (general) frente a Qwen-Coder

Misma máquina (Mac M4 Pro, LM Studio), misma búsqueda, mismas preguntas, temperatura 0, contexto de 8.192 tokens.

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

| Qwen − Gemma (acierto del campo) | Diferencia | IC95% |
|---|---:|---|
| Desarrollo | −0.071 | [−0.14, −0.02] |
| Test | +0.017 | [+0.00, +0.05] |

**Qué significa:**

- **En desarrollo, Gemma es mejor** con una diferencia real. Los errores de Qwen son de un tipo concreto: cita campos del nivel equivocado (en *"Which fields describe a refund on a charge?"* da los campos del objeto `refund` en lugar del campo `refunds` del cargo), dos veces dice "no encontrado" teniendo la respuesta delante, y una vez fusiona dos rutas en una que no existe (`total_details.amount_total`).
- **En test empatan.** La ventaja de Gemma en desarrollo no se reproduce con preguntas nuevas.
- **Qwen es 2,5-4 veces más rápido** y ocupa 1,6 GB menos, lo que importa en un portátil con 8 GB de GPU.
- Con los dos modelos, la generación no es el cuello de botella: ambos responden bien en 51 de 52 preguntas cuando la búsqueda trae el fragmento correcto.

**Decisión.** El criterio fijado de antemano era el acierto en desarrollo, así que **Gemma es el modelo principal** (`configs/default.yaml`). **Qwen es la alternativa ligera** para la demo y para el portátil con 8 GB de GPU (en LM Studio, `llm.model: qwen2.5-coder-7b-instruct`; con Ollama, `configs/ollama.yaml`): misma calidad en test y respuestas en unos 6 segundos.

Las latencias son orientativas: se midieron en el Mac, a veces con otros procesos en marcha.

## Corrección de las respuestas · juez a ciegas (test)

El acierto del campo mide si el modelo señala el campo correcto, pero no si lo que **dice** de él es cierto. Para medirlo, un agente de IA juzgó las 140 respuestas del test (70 de cada modelo) con `scripts/judge.py`:

- **A ciegas**: las respuestas de los dos modelos se mezclaron al azar con identificadores neutros (`J001`...). El juez no vio qué modelo escribió cada una.
- **Contra la spec**: el juez ve la pregunta, la respuesta y lo que dice la spec de los campos esperados y de los citados (tipo, nullable, valores, descripción). No ve los fragmentos recuperados.
- **Rúbrica fija**: *correcta* (campo correcto y nada contradice la spec), *parcial* (campo correcto pero incompleta, o con un error secundario), *incorrecta* (campo equivocado o algo que contradice la spec), *no responde*.

Es una evaluación con **un modelo como juez** (*LLM-as-judge*), no una evaluación humana. Se puede validar revisando a mano una muestra de veredictos y midiendo el acuerdo.

| | Gemma (principal) | Qwen-Coder |
|---|---:|---:|
| **Correcta** | **0.817** [0.72, 0.90] | 0.750 [0.63, 0.85] |
| Parcial | 0.050 | 0.100 |
| Incorrecta | 0.100 | 0.067 |
| No responde | 0.033 | 0.083 |
| Correcta o parcial | 0.867 [0.77, 0.95] | 0.850 [0.75, 0.93] |
| Preguntas sin respuesta bien tratadas | 10 de 10 | 10 de 10 |
| Respuestas `enum` con todos los valores (comprobación automática) | 15 de 15 | 14 de 15 |

| Correctas por categoría | Gemma | Qwen |
|---|---:|---:|
| Descriptivas | 0.73 | 0.67 |
| Directas | 0.80 | 0.73 |
| Anidadas | 0.73 | 0.67 |
| Valores posibles | 1.00 | 0.93 |

**Qué significa:** con Gemma, **el 82% de las respuestas son correctas en contenido**, algo menos que el 85% de acierto del campo, porque ahora también cuenta la explicación. Los errores típicos: usar un campo vecino (`next_payment_attempt` en lugar de `due_date`), respuestas incompletas (una de las tres pruebas de entrega de una disputa) y algún matiz falso (decir que 99 es el máximo fijo cuando la spec dice que es el valor por defecto). Gemma vuelve a quedar por delante de Qwen, aunque los intervalos se solapan.

## Taxonomía de errores por fase (test)

`scripts/error_taxonomy.py` clasifica cada pregunta fallada del test en una **fase** y un **tipo** con reglas fijas. Si la búsqueda no trajo el fragmento correcto, el error es de búsqueda; si lo trajo, es de generación. El tipo de búsqueda se decide con los datos (qué objetos nombra la pregunta, de qué objeto son los 5 fragmentos, si hay un campo con el mismo nombre en otra ruta, si la respuesta estaba en un grupo de primer nivel). El tipo de generación sale del veredicto del juez.

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

Qué es cada tipo de búsqueda, con un ejemplo real:

- **Cajón de sastre.** Los campos sueltos de primer nivel de un objeto se agrupan en orden alfabético, y un grupo mezcla campos sin relación. *"What is the due date of an invoice?"*: `due_date` está en un fragmento con `customer_phone`, `customer_tax_ids`, `default_source`... y queda 8º de los 81 fragmentos de `invoice`. El modelo recibe otros cinco grupos de la factura y responde con `next_payment_attempt`.
- **Varios objetos en la pregunta.** *"Is there a Stripe-hosted page so a customer can look at an invoice and pay it?"*: el filtro prioriza `customer` e `invoice` y sale primero `customer`.
- **Campo gemelo.** *"At what time did an invoice transition to paid?"*: llega `payments.data.status_transitions.paid_at` (el del pago) en lugar de `status_transitions.paid_at` (el de la factura).
- **Caja equivocada.** El enlace de recuperación de una sesión de checkout caducada está en `after_expiration.recovery`, y llegan otras cajas de `checkout.session`.

**Qué significa:**

1. **El cuello de botella es la búsqueda** (8 de 11 fallos con Gemma), y dentro de ella el **cajón de sastre** (4 de 8). Es el límite principal de la regla de agrupar los valores sueltos.
2. **Al fragmento correcto le faltó poco:** su puesto entre los fragmentos de su propio objeto fue 2, 6, 6, 8, 10, 12, 14 y 26. En 5 de los 8 fallos estaba entre los 10 primeros, así que pasar al modelo 10 fragmentos en lugar de 5 probablemente recuperaría varios.
3. **Los dos modelos comparten los mismos fallos de búsqueda**, y la diferencia entre ellos está en la generación: Qwen da 5 respuestas incompletas frente a 1 de Gemma.

## Iteración 4 · Small-to-big (motivada por la taxonomía de errores)

**Idea.** Indexar cada campo por separado (cabecera + línea + descripción) para buscar, y entregar al modelo el fragmento grande al que pertenece. Ataca el "cajón de sastre": en un fragmento pequeño, el significado de `due_date` no se diluye. Opción `chunker.small_to_big` (variante E11).

**Evaluación.** Como la idea salió de los fallos del test, se decidió con desarrollo y se evaluó **una vez** en un **segundo test a ciegas** (`test2.jsonl`, 70 preguntas nuevas escritas por un agente con las mismas reglas).

| Búsqueda (semántico + filtro) | Desarrollo R@5 / MRR | Test 2 R@5 / MRR |
|---|---|---|
| Actual (E1) | 0.911 / 0.777 | 0.958 / 0.821 |
| Small-to-big (E11) | 0.970 / 0.960 | 0.975 / 0.868 |
| Diferencia | +0.060 [−0.00, +0.13] / **+0.183 [+0.10, +0.27]** | +0.017 [−0.04, +0.07] / +0.047 [−0.04, +0.14] |

| Generación en test 2 (Gemma) | Actual | Small-to-big |
|---|---:|---:|
| Acierto del campo | 0.950 | 0.950 |
| Campos inventados | 0 de 70 | 2 de 70 |
| Sin respuesta reconocidas | 10 de 10 | 10 de 10 |

En el primer test (análisis posterior, no evaluación limpia) arregla 6 de los 8 fallos de búsqueda, incluidos los tres de `invoice`, pero rompe 4 preguntas que antes acertaban: `Recall@5` pasa de 0.850 a 0.900.

**Qué significa:** small-to-big **mejora el orden de la búsqueda** (de forma clara en desarrollo) y arregla los fallos de cajón de sastre, pero **no mejora las respuestas finales** en una evaluación limpia: el sistema actual ya trae el fragmento correcto en el 96% del test 2, y queda poco margen. **Se mantiene E1 como configuración final**, y small-to-big queda como opción documentada.

## Configuración final

- **Búsqueda** (`configs/default.yaml`): extracción sin copias, troceado jerárquico con cabecera, 512 tokens, `bge-small-en-v1.5`, búsqueda semántica con filtro por objeto, 5 fragmentos.
- **Modelo**: `google/gemma-4-e4b` como principal (`configs/default.yaml`); `qwen2.5-coder` 7B como alternativa ligera (con Ollama, `configs/ollama.yaml`).

## Limitaciones

- **56 preguntas de desarrollo** dan intervalos de unos ±0.07. Las diferencias pequeñas (prefijo, reranker) no se pueden resolver con este tamaño.
- **El conjunto de control es sintético**: está generado con plantillas a partir de la tabla. Mide si los resultados se generalizan, pero no sustituye a preguntas reales escritas a mano.
- **El test final lo escribió un agente de IA** sin acceso a los fragmentos, a los resultados ni a la ejecución del sistema, usando solo la tabla de campos. Es más independiente que las preguntas de desarrollo, pero no equivale a preguntas de usuarios reales.
- **Las 15 preguntas reformuladas** con *"on a charge"* forman parte del conjunto de desarrollo. Están documentadas arriba y la regla se aplicó a todas, no solo a las que fallaban.
- **Las preguntas de tipo `metadata`** (*"¿qué campos son nullable?"*) se excluyen: no son de búsqueda y se deberían responder consultando la tabla.

## Resumen final

| | Desarrollo | Control | **Test** |
|---|---:|---:|---:|
| Preguntas puntuadas | 56 | 120 | 60 |
| `Recall@5`, configuración final | 0.911 | 0.892 | **0.850** |
| `Recall@5`, troceado fijo | 0.622 | 0.608 | 0.533 |
| Jerárquico − fijo | +0.280 | +0.217 | **+0.233** |
| Acierto del campo (Gemma) | 0.929 | — | **0.850** |
| Alucinación (Gemma) | 1 de 58 | — | **0 de 70** |
| Respuesta correcta según el juez (Gemma) | — | — | **0.817** |

1. **Hipótesis principal confirmada:** el troceado jerárquico por el árbol de atributos supera al troceado fijo en los tres conjuntos (+0.22 a +0.28 en `Recall@5`), también con 50 objetos.
2. **Construcción del corpus:** no copiar objetos dentro de otros mejora la búsqueda (+0.06 a +0.13).
3. **Filtro por objeto:** mejora el orden de los resultados y hace al sistema robusto cuando crece el corpus.
4. **Cabecera con la ruta:** mejora el orden (`MRR`); su efecto sobre `Recall@5` solo se demuestra en desarrollo y control.
5. **Las respuestas son correctas en contenido en el 82% de los casos** (juez a ciegas sobre el test); las preguntas sin respuesta se reconocen todas.
6. **El modelo no es el cuello de botella:** con el fragmento correcto responde bien en el 98% de los casos, no inventa campos en el test y reconoce las preguntas sin respuesta. **Los errores que quedan son de búsqueda**, sobre todo en preguntas que describen el campo sin nombrarlo.

## Trabajo futuro

- **Agrupar los campos sueltos de primer nivel por tema** (o en grupos más pequeños en los objetos grandes), que es la causa principal de los fallos de búsqueda.
- **Pasar más fragmentos al modelo** (`retrieval.top_k: 10`): en 5 de los 8 fallos de búsqueda, el fragmento correcto estaba entre los 10 primeros de su objeto.
- **Un filtro por campo**, además del filtro por objeto: priorizar los fragmentos que contienen un campo cuyo nombre aparece en la pregunta (`due_date` en *"due date"*).
- Mejorar la búsqueda en preguntas descriptivas (0.73 en test), por ejemplo enriqueciendo cada chunk con preguntas o sinónimos generados.
- Responder las preguntas de tipo `metadata` consultando la tabla directamente.
- Medir el *reranker* con más preguntas: mejoró `Recall@1` y `MRR`, pero no se pudo demostrar con 56.
- Repetir la evaluación con preguntas de usuarios reales.
