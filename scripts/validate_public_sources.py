#!/usr/bin/env python3
"""Validate the actual public-source MQuAKE large split on a CPU runner.

This script downloads only benchmark data, never model weights or translated
questions. It runs the exact source-loader cell from Notebook 01 so this check
cannot silently differ from training. Public facts are never fabricated.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks/01_mquake_t_source_translate_sft.ipynb").read_text(encoding="utf-8"))
source_cells = [
    "".join(c.get("source", [])) for c in nb["cells"]
    if c.get("cell_type") == "code" and "def build_cases(" in "".join(c.get("source", []))
]
assert len(source_cells) == 1, f"Expected one data-loader cell, found {len(source_cells)}"
namespace = {"__name__": "__main__"}
exec(compile(source_cells[0], "notebook:public_dataset_loader", "exec"), namespace)
raw, sha = namespace["load_source"]()
n_forget, n_retain, n_holdout = 512, 2048, 512
cases, reasons = namespace["build_cases"](raw, n_forget, n_retain, n_holdout, seed=namespace["SEED"])
assert len(cases) == n_forget + n_retain + n_holdout
assert len({r["fact_id"] for r in cases}) == len(cases)
assert all(r["provenance_scope"] == "original_pre_edit_only" for r in cases)
assert all(r["depth"] >= 2 for r in cases)
assert all(r["support_steps"] for r in cases)
from collections import Counter
print("PUBLIC_RECORDS", namespace["SOURCE_MANIFEST"]["dataset_counts"])
print("SOURCE_CHECKSUMS", namespace["SOURCE_MANIFEST"]["dataset_sha256"])
print("COMBINED_SHA256", sha)
print("SELECTED_BY_SPLIT", dict(Counter(r["split"] for r in cases)))
print("SELECTED_BY_SOURCE", dict(Counter(r["source_dataset"] for r in cases)))
print("SELECTED_BY_HOP_DEPTH", dict(Counter(r["depth"] for r in cases)))
print("EXCLUSION_REASONS", reasons)
print("VALIDATION_OK: selected all original-fact, published-QA cases, no new facts generated")
