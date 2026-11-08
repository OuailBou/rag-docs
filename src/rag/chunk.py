import re
import statistics
from collections import defaultdict

from .common import OUT, count_tokens, field_paths, format_field, header, load_config, load_table, write_jsonl

SEPARATORS = [r"(?<=[.!?])\s+", r"(?<=,)\s+", r"\s+"]


class HierChunker:
    def __init__(self, cfg):
        self.budget = cfg["chunker"]["max_tokens"]
        self.use_header = cfg["chunker"]["header"]
        self.model = cfg["embedding"]["model"]
        self.out = []

    def tokens(self, text):
        return count_tokens(text, self.model)

    def compose(self, head, parts):
        body = "\n\n".join(parts)
        return f"{head}\n\n{body}" if head else body

    def subtree(self, row, base):
        parts = [format_field(row, base)]
        for kid in self.kids.get(row["path"], []):
            parts.append(self.subtree(kid, base))
        return "\n\n".join(parts)

    def emit(self, parent, text):
        self.out.append({"object": self.obj, "parent": parent, "text": text})

    def fit(self, text, wrap, level=0):
        if self.tokens(wrap(text)) <= self.budget or level == len(SEPARATORS):
            return [text]
        pieces, cur = [], ""
        for unit in re.split(SEPARATORS[level], text):
            cand = f"{cur} {unit}".strip()
            if self.tokens(wrap(cand)) <= self.budget:
                cur = cand
                continue
            if cur:
                pieces.append(cur)
            if self.tokens(wrap(unit)) > self.budget:
                pieces.extend(self.fit(unit, wrap, level + 1))
                cur = ""
            else:
                cur = unit
        if cur:
            pieces.append(cur)
        return pieces

    def split_long(self, parent, head, part):
        lead, _, rest = part.partition("\n")
        wrap = lambda t: self.compose(head, [f"{lead}\n{t}"])
        for piece in self.fit(" ".join(rest.split()), wrap):
            self.emit(parent, wrap(piece))

    def level(self, parent):
        head = header(self.title, parent) if self.use_header else ""
        base = parent.count(".") + 1 if parent else 0
        group = []

        def flush():
            if group:
                self.emit(parent, self.compose(head, group))
                group.clear()

        def add(part):
            if group and self.tokens(self.compose(head, group + [part])) > self.budget:
                flush()
            if self.tokens(self.compose(head, [part])) > self.budget:
                self.split_long(parent, head, part)
            else:
                group.append(part)

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

    def run(self, rows):
        by_object = defaultdict(list)
        for row in rows:
            by_object[row["object"]].append(row)
        for obj, obj_rows in by_object.items():
            self.obj, self.title = obj, obj_rows[0]["title"]
            self.kids = defaultdict(list)
            for row in obj_rows:
                self.kids[row["path"].rsplit(".", 1)[0] if "." in row["path"] else ""].append(row)
            self.level("")
        return self.out


def render_documents(rows):
    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object"]].append(row)
    return {
        obj: f"# {obj_rows[0]['title']}\n\n## Attributes\n\n" + "\n\n".join(format_field(r) for r in obj_rows)
        for obj, obj_rows in by_object.items()
    }


def fixed_chunks(cfg, rows):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["embedding"]["model"])
    tok.model_max_length = 10**9
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tok, chunk_size=cfg["chunker"]["max_tokens"] - 2, chunk_overlap=cfg["chunker"]["overlap"])
    docs_dir = OUT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for obj, text in render_documents(rows).items():
        (docs_dir / f"{obj}.md").write_text(text + "\n", encoding="utf-8")
        out += [{"object": obj, "parent": "", "text": piece} for piece in splitter.split_text(text)]
    return out


def field_chunks(rows, chunks):
    """Small-to-big: un fragmento pequeño por campo, que apunta al fragmento grande donde está ese campo."""
    owner = {}
    for c in chunks:
        for path in field_paths(c["text"]):
            owner.setdefault((c["object"], path), c["id"])
    fields = []
    for r in rows:
        if (r["object"], r["path"]) in owner:
            fields.append({"id": len(fields), "object": r["object"], "parent": r["path"],
                           "chunk_id": owner[(r["object"], r["path"])],
                           "text": header(r["title"], r["path"]) + "\n\n" + format_field(r, r["depth"])})
    return fields


def main():
    cfg = load_config()
    rows = load_table()
    strategy = cfg["chunker"]["strategy"]
    chunks = fixed_chunks(cfg, rows) if strategy == "fixed" else HierChunker(cfg).run(rows)
    for index, chunk in enumerate(chunks):
        chunk["id"] = index
    write_jsonl(OUT / "chunks.jsonl", chunks)
    fields_path = OUT / "fields.jsonl"
    if cfg["chunker"]["small_to_big"]:
        if strategy != "hierarchical":
            raise SystemExit("small_to_big solo funciona con strategy: hierarchical")
        fields = field_chunks(rows, chunks)
        write_jsonl(fields_path, fields)
        print(f"small-to-big: {len(fields)} fragmentos de campo -> {fields_path}")
    elif fields_path.exists():
        fields_path.unlink()

    sizes = [count_tokens(chunk["text"], cfg["embedding"]["model"]) for chunk in chunks]
    over = sum(size > cfg["chunker"]["max_tokens"] for size in sizes)
    print(f"{strategy}: {len(chunks)} chunks -> {OUT / 'chunks.jsonl'}")
    print(f"tokens: min {min(sizes)}  mediana {statistics.median(sizes):.0f}  max {max(sizes)}  pasan del limite: {over}")


if __name__ == "__main__":
    main()
