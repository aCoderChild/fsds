#!/usr/bin/env python3
"""Offline project validation; does not claim a GPU test or fetch the dataset."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "notebooks").glob("0*.ipynb"))
assert len(files) == 3, f"Expected three notebooks, got: {files}"
for path in files:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", ""))
        code = "\n".join(line for line in src.splitlines()
                         if not line.lstrip().startswith(("%", "!")))
        ast.parse(code, filename=f"{path.name}:cell{i}")
    print("VALID", path.name, len(nb["cells"]), "cells")
print("Only syntax and notebook structure checked. GPU/model tests NOT RUN.")
