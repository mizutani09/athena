#!/usr/bin/env python3
"""Expand a compact RSG resolution file into a normal Athena++ input file."""
import argparse
from pathlib import Path


def read_variant(path):
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (x.strip() for x in line.split("=", 1))
            values[key] = value
    return values


def replace(text, block, key, value):
    lines, active, found = [], None, False
    for raw in text.splitlines():
        clean = raw.split("#", 1)[0].strip()
        if clean.startswith("<"):
            active = clean.strip("<>")
        elif active == block and "=" in clean and clean.split("=", 1)[0].strip() == key:
            raw = f"{key} = {value}"
            found = True
        lines.append(raw)
    if not found:
        raise ValueError(f"missing {block}/{key} in base input")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    cfg = read_variant(args.variant)
    for key in ("base", "mesh_nx", "meshblock_nx", "problem_id"):
        if key not in cfg:
            raise ValueError(f"variant lacks {key}")
    mesh_nx = int(cfg["mesh_nx"])
    block_nx = int(cfg["meshblock_nx"])
    if block_nx <= 0 or block_nx & (block_nx-1):
        raise ValueError("NR-FLD multigrid requires a power-of-two MeshBlock size")
    if mesh_nx % block_nx:
        raise ValueError("mesh_nx must be divisible by meshblock_nx")
    base = args.variant.parent/cfg["base"]
    text = base.read_text()
    for axis in (1, 2, 3):
        text = replace(text, "mesh", f"nx{axis}", str(mesh_nx))
        text = replace(text, "meshblock", f"nx{axis}", str(block_nx))
    text = replace(text, "job", "problem_id", cfg["problem_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
