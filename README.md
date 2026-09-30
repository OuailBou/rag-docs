# Stripe RAG

Proyecto para extraer información de esquemas de Stripe y prepararla para un pipeline RAG.

## Requisitos

- Python 3.11+
- uv

## Configuración

```bash
uv sync
```

## Uso

```bash
uv run python -m stripe_rag.extract --list
uv run python -m stripe_rag.extract
uv run python -m stripe_rag.chunk
```

## Estructura

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── objects.txt
├── notebooks/
├── scripts/
├── src/
│   └── stripe_rag/
├── tests/
├── config.yaml
├── pyproject.toml
├── README.md
└── .python-version
```
