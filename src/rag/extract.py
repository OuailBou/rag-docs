import argparse
import json

import pandas as pd

from .common import OUT, ROOT, clean, load_config, read_objects


def classify(p):
    if "anyOf" in p:
        if any(obj.get("type") == "string" for obj in p["anyOf"]):
            return "expandable", None
        refs = [obj["$ref"] for obj in p["anyOf"] if "$ref" in obj]
        return ("box", refs[0]) if refs else ("value", None)
    if "$ref" in p:
        return "box", p["$ref"]
    if p.get("type") == "array":
        return "list", None
    if p.get("type") == "object" and p.get("properties"):
        return "inline", None
    return "value", None


def field_type(p, kind, item_kind):
    if kind in ("box", "inline"):
        return "object"
    if kind == "expandable":
        return "string"
    if kind == "list":
        if item_kind in ("box", "inline"):
            return "array of objects"
        if item_kind == "expandable":
            return "array of strings"
        return f"array of {p.get('items', {}).get('type', 'value')}s"
    if "enum" in p:
        return "enum"
    return p.get("type", "string")


def walk(schemas, schema, obj, title, prefix, depth, stack, rows):
    for name, p in schema.get("properties", {}).items():
        path = f"{prefix}.{name}" if prefix else name
        kind, ref = classify(p)
        item_kind, child, key = None, None, None
        if kind == "box":
            key = ref.split("/")[-1]
            child = schemas[key]
        elif kind == "inline":
            child, key = p, path
        elif kind == "list":
            item_kind, item_ref = classify(p.get("items", {}))
            if item_kind == "box":
                key = item_ref.split("/")[-1]
                child = schemas[key]
            elif item_kind == "inline":
                child, key = p["items"], path
        rows.append({
            "object": obj,
            "title": title,
            "path": path,
            "name": name,
            "depth": depth,
            "type": field_type(p, kind, item_kind),
            "nullable": bool(p.get("nullable")),
            "expandable": kind == "expandable" or item_kind == "expandable",
            "enum": "|".join(str(v) for v in p.get("enum", [])),
            "description": clean(p.get("description") or (child or {}).get("description", "")),
        })
        if child is not None and child.get("properties") and key not in stack:
            walk(schemas, child, obj, title, path, depth + 1, stack | {key}, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    spec = json.loads((ROOT / cfg["spec"]).read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]
    if args.list:
        print("\n".join(sorted(k for k, value in schemas.items() if "x-resourceId" in value)))
        return

    rows = []
    for obj in read_objects():
        title = obj.replace(".", " ").replace("_", " ").title()
        walk(schemas, schemas[obj], obj, title, "", 0, {obj}, rows)

    df = pd.DataFrame(rows)
    parents = {(obj, path.rsplit(".", 1)[0]) for obj, path in zip(df["object"], df["path"]) if "." in path}
    df["has_children"] = [(obj, path) in parents for obj, path in zip(df["object"], df["path"])]
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "table.parquet", index=False)

    print(f"spec {spec['info']['version']} -> {len(df)} filas")
    for obj, group in df.groupby("object", sort=False):
        print(f"  {obj:30} filas={len(group):5}  primer_nivel={(group.depth == 0).sum():3}  "
              f"niveles={group.depth.max() + 1}  cajas={group.has_children.sum():4}  expandables={group.expandable.sum():3}")


if __name__ == "__main__":
    main()
