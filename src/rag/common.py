import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("RAG_OUT", ROOT / "out"))
CONFIG = Path(os.environ.get("RAG_CONFIG", ROOT / "configs" / "default.yaml"))
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
FIELD_LINE = re.compile(r"^\s*- `([^`]+)` \(", re.MULTILINE)
_tokenizer = None


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def field_paths(text):
    return set(FIELD_LINE.findall(text))


def read_objects():
    path = ROOT / load_config()["data"]["objects"]
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def load_table():
    import pandas as pd

    return pd.read_parquet(OUT / "table.parquet").to_dict("records")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def clean(text):
    return " ".join(LINK.sub(r"\1", text or "").split())


def format_field(row, base_depth=0):
    pad = "  " * (row["depth"] - base_depth)
    flags = [row["type"]] + (["nullable"] if row["nullable"] else []) + (["expandable"] if row["expandable"] else [])
    lines = [f"{pad}- `{row['path']}` ({', '.join(flags)})"]
    if row["description"]:
        lines.append(f"{pad}  {row['description']}")
    if row["enum"]:
        lines.append(f"{pad}  Possible values: {', '.join(row['enum'].split('|'))}.")
    return "\n".join(lines)


def header(title, path):
    parts = [title] + [p.replace("_", " ") for p in path.split(".") if p]
    return " → ".join(parts) + f"\nPath: {path or '(top level)'}"


def count_tokens(text, model):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model)
        _tokenizer.model_max_length = 10**9
    return len(_tokenizer(text)["input_ids"])
