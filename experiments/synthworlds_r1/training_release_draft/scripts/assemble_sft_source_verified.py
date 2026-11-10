#!/usr/bin/env python3
"""Run the frozen release assembler and require source-bound SFT validation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import validate_sft_source_integrity as source
import assemble_training_release as legacy


def main() -> None:
    canonical, _ = source.validate_prepared_inputs()
    rebuilt, _ = legacy.build_corpus(legacy.checked_translation_reviews())
    source.validate_corpus_source_rows(rebuilt, canonical)
    legacy.validate_split_and_evaluation(rebuilt)
    legacy.main()
    corpus = PROJECT / "sft_training/corpus.jsonl"
    manifest = PROJECT / "sft_training/manifest.json"
    proof = source.validate_release(corpus, manifest)
    output = corpus.parent / "source_integrity_validation.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": proof["status"], "proof": str(output)}, indent=2))


if __name__ == "__main__":
    main()
