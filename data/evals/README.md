# Preguntas de evaluación

Una pregunta por línea (JSONL):

| Clave | Significado |
|---|---|
| `id` | identificador estable |
| `query` | la pregunta, en inglés |
| `object` | objeto de Stripe al que se refiere (`charge`, `payment_intent`...) |
| `category` | tipo de pregunta |
| `difficulty` | `easy`, `medium` o `hard` |
| `answerable` | si la spec contiene la respuesta |
| `relevant_paths` | rutas de los campos que la responden, relativas al objeto |

## Conjuntos

| Fichero | Preguntas | Origen | Uso |
|---|---:|---|---|
| `dev_charge.jsonl` | 30 (2 sin respuesta) | escritas a mano sobre `charge` | desarrollo |
| `dev_objects.jsonl` | 30 | escritas a mano, 2 por cada uno de los otros 15 objetos | desarrollo |
| `control.jsonl` | 120 | generadas con plantillas (`scripts/make_control_set.py`) | control, evaluado una vez |
| `test.jsonl` | 70 (10 sin respuesta) | escritas por un agente de IA sin ver el sistema | **test final**, evaluado una vez |

## Notas

- **Categoría `metadata`** (`charge-023`, `charge-024`): preguntas como *"¿qué campos son nullable?"*, cuya respuesta está repartida por todo el objeto. No cuentan en las métricas de búsqueda.
- **Cambios en desarrollo:** a las preguntas de `charge` que no nombraban el objeto se les añadió *"on a charge"* (con un corpus de 16 objetos eran ambiguas), y `charge-026` pregunta ahora por `refund.status`. Se explica en `docs/experimentos.md`.
- **Test:** 60 preguntas con respuesta repartidas en `test_direct`, `test_descriptive`, `test_nested` y `test_enum` (15 de cada), sobre los 16 objetos, y 10 sin respuesta: 6 sobre información que no está en la spec (tarifas, límites, soporte...) y 4 sobre campos que no existen. El agente que las escribió no tuvo acceso a los fragmentos, a los resultados ni a la ejecución del sistema; solo a la tabla de campos. Conviene revisar a mano una muestra.
