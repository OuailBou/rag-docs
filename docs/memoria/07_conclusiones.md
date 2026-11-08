# 7. Conclusiones y trabajo futuro

Este capítulo resume lo conseguido frente a los objetivos y la hipótesis planteados en el capítulo 1 (sección 7.1), recoge las aportaciones del trabajo (7.2) y propone líneas de continuación (7.3).

## 7.1. Cumplimiento de los objetivos

El **objetivo general** era diseñar, implementar y evaluar de forma rigurosa un sistema RAG que respondiera preguntas sobre la API de Stripe a partir de su especificación OpenAPI, que funcionara en local y cuyas respuestas pudieran verificarse contra la especificación. El sistema construido cumple las tres condiciones. Toda la cadena (extracción, troceado, indexado, búsqueda, generación y verificación) se ejecuta en un portátil, con LM Studio en macOS u Ollama en Windows, y cada campo citado se comprueba contra la tabla extraída de la especificación.

Los objetivos específicos se han cubierto así:

1. **Corpus desde OpenAPI.** El extractor recorre los esquemas siguiendo los `$ref` y produce una tabla de 3.535 campos para los 16 objetos del corpus principal. La comparación con la documentación web (sección 4.3.4) muestra que coinciden 575 campos y que las diferencias se deben a la fecha de cada fuente. No copiar dentro de un objeto otros objetos del corpus mejora `Recall@5` entre +0,06 y +0,13 según el conjunto.
2. **Troceado jerárquico.** El algoritmo produce 744 fragmentos que respetan el árbol de atributos y llevan una cabecera con su ruta. Se ha comparado con el troceado recursivo de LangChain en las mismas condiciones (mismo texto, tokenizador y presupuesto).
3. **Estrategias de recuperación.** Se compararon la búsqueda semántica, la léxica y la híbrida, el filtro por objeto y la reordenación con dos *cross-encoders*. La configuración final, búsqueda semántica con filtro por objeto, es la más sencilla de las que funcionan. La híbrida empeoró los resultados, y el *reranker* no demostró mejorar sobre el filtro.
4. **Modelo local y verificador.** Gemma 4 E4B responde en JSON con los campos y fragmentos citados. El verificador no detectó ningún campo inventado en las 70 preguntas del test, y el sistema reconoció las 10 preguntas sin respuesta.
5. **Protocolo de evaluación.** Se han usado cuatro tipos de conjuntos (desarrollo, control, test y segundo test), métricas de búsqueda, de generación y de corrección con un juez a ciegas, e intervalos de confianza emparejados. Todo se reproduce con tres comandos: `experiment.py`, `heldout.py` y `eval_generation`.
6. **Análisis de errores.** La taxonomía por fase clasifica los 11 fallos de Gemma en el test: 8 son de búsqueda y 3 de generación. Identifica el cuello de botella (el grupo de campos sueltos de primer nivel en objetos grandes) y motivó la iteración *small-to-big*.

Respecto a la **hipótesis H1**, los datos la confirman: el troceado jerárquico mejora `Recall@5` frente al troceado por longitud fija en desarrollo (+0,280), control (+0,217), test (+0,233), test con 50 objetos (+0,233) y segundo test (+0,150, con el filtro aplicado a ambos). Ningún intervalo de esas diferencias se acerca al cero. De las **hipótesis secundarias**, se confirman la eliminación de copias y el filtro por objeto. La cabecera se confirma a medias: mejora el orden (`MRR`) en todos los conjuntos, pero su efecto sobre `Recall@5` en el test no es demostrable. El capítulo 6 lo explica porque la cabecera y el filtro se solapan.

El resultado final del sistema es un `Recall@5` de **0,850** [0,76; 0,93] en el test. El modelo cita el campo correcto en el 85 % de las preguntas, y el juez considera correctas el **81,7 %** de las respuestas.

## 7.2. Conclusiones principales

- **La estructura de los datos es la mejor señal.** Las mejoras más grandes vinieron de respetar la estructura de la especificación: trocear por el árbol, poner la ruta en el fragmento y no duplicar objetos. Un modelo de *embeddings* más grande, un prefijo en la consulta o un *reranker* no consiguieron mejoras demostrables. En documentación con una estructura formal tan rica como OpenAPI, aprovecharla rinde más que añadir capacidad de cómputo.
- **En esta tarea, el modelo de lenguaje no es lo que limita.** Gemma y Qwen-Coder, dos modelos de unos 7.000 millones de parámetros, responden bien en 51 de 52 casos cuando la búsqueda les trae el fragmento correcto. Para mejorar el sistema conviene invertir en la búsqueda.
- **Verificar es barato y útil.** Comprobar cada ruta citada contra la tabla de campos es determinista, instantáneo y no necesita otro modelo. Permite mostrar al usuario qué partes de la respuesta están respaldadas.
- **Una mejora en búsqueda no garantiza mejores respuestas.** *Small-to-big* mejoró claramente el orden de los resultados en desarrollo, pero en la evaluación limpia del segundo test no cambió el acierto y añadió dos campos inventados. Por eso las decisiones se han tomado con la métrica final y con conjuntos no usados antes.
- **El rigor de la evaluación cambia las conclusiones.** Los errores de medida corregidos en la iteración 0 inflaban los resultados. Las diferencias pequeñas (prefijo, *reranker*, Gemma frente a Qwen en test) solo pueden declararse no demostradas con este número de preguntas. Separar desarrollo, control y test hace que la caída de 0,911 a 0,850 sea una medida honesta y no una sorpresa.

## 7.3. Trabajo futuro

Las líneas siguientes salen directamente del análisis de errores y de las limitaciones del capítulo 6, ordenadas de más a menos inmediatas:

1. **Agrupar los campos sueltos de primer nivel por tema.** Es la causa principal de los fallos de búsqueda que quedan. En lugar de agruparlos por orden alfabético, se podrían agrupar por similitud de su descripción, o en grupos más pequeños en los objetos grandes como `invoice`.
2. **Pasar más fragmentos al modelo.** En 5 de los 8 fallos de búsqueda del test, el fragmento correcto estaba entre los 10 primeros de su objeto. Subir `retrieval.top_k` de 5 a 10 cabe en el contexto de 8.192 *tokens* y solo requiere medir que no empeore la generación.
3. **Un filtro por campo.** Además del filtro por objeto, se podrían priorizar los fragmentos que contienen un campo cuyo nombre aparece en la pregunta (por ejemplo, `due_date` en *"due date"*).
4. **Mejorar las preguntas descriptivas**, las que describen el campo sin nombrarlo y son el tipo más difícil (0,733 en test y 0,750 en control). Una opción es enriquecer cada fragmento con preguntas o sinónimos generados al indexar.
5. **Medir el *reranker* con más preguntas.** Obtuvo el mejor `Recall@1` y `MRR` de desarrollo, pero con 56 preguntas no se pudo distinguir del ruido.
6. **Responder las preguntas de tipo agregado consultando la tabla.** Preguntas como «¿qué campos son expandibles?» no se resuelven bien con búsqueda. Se resolverían con una consulta directa a la tabla de campos, decidida por un clasificador de intención.
7. **Ampliar el alcance de la documentación**: los *endpoints* y sus parámetros, que también están en la especificación, y las guías conceptuales de la documentación web.
8. **Conversaciones de varios turnos**, en las que una pregunta de seguimiento («¿y en un *invoice*?») dependa de la anterior. Requiere reescribir la consulta antes de buscar.
9. **Evaluar con preguntas de usuarios reales** y con especificaciones OpenAPI de otras APIs, para comprobar si los resultados se generalizan fuera de Stripe.
