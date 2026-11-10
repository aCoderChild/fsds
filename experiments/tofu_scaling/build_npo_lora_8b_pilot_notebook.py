#!/usr/bin/env python3
"""Build reproducible smoke-only or locked full-pilot Kaggle notebooks."""

from __future__ import annotations

import hashlib
import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "kaggle" / "npo_lora_8b_pilot"
NOTEBOOK_PATH = PACKAGE / "tofu_8b_npo_lora_pilot.ipynb"
PROTOCOL_PATH = ROOT / "npo_lora_8b_pilot_protocol.json"
REQUIREMENTS_PATH = ROOT / "requirements-npo-pilot.txt"
SOURCE_CELLS = {
    5: "tofu_official_eval.py",
    6: "verify_tofu_official_eval.py",
    7: "npo_objective.py",
    8: "verify_npo_objective.py",
    9: "npo_lora_train.py",
    10: "evaluate_npo_adapter.py",
    11: "verify_npo_run.py",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def writefile_cell(filename: str, body: str) -> list[str]:
    return (f"%%writefile /kaggle/working/{filename}\n" + body).splitlines(keepends=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Required for full mode; writes a separate package and preserves the canonical smoke package.",
    )
    args = parser.parse_args()
    if args.mode == "full" and args.output_dir is None:
        parser.error("--output-dir is required for full mode so the smoke package stays unchanged")
    if args.mode == "smoke" and args.output_dir is not None:
        parser.error("--output-dir is only supported for full mode")

    package = PACKAGE
    if args.mode == "full":
        package = args.output_dir.resolve()
        if package.exists():
            raise SystemExit(f"Refusing to overwrite existing full-pilot package: {package}")
        shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    notebook_path = package / NOTEBOOK_PATH.name

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 18:
        raise SystemExit("Unexpected 8B notebook layout; refusing to rewrite cells by index")
    mode_cell = "".join(notebook["cells"][1]["source"])
    if "RUN_FULL_TRAINING = False" not in mode_cell:
        raise SystemExit("Notebook is not smoke-only; refusing to publish a full-training default")
    if args.mode == "full":
        notebook["cells"][1]["source"] = mode_cell.replace(
            "RUN_FULL_TRAINING = False", "RUN_FULL_TRAINING = True", 1
        ).splitlines(keepends=True)

    embedded_sources: dict[str, str] = {}
    for index, filename in SOURCE_CELLS.items():
        body = (ROOT / filename).read_text(encoding="utf-8")
        notebook["cells"][index]["source"] = writefile_cell(filename, body)
        embedded_sources[filename] = sha256(body.encode("utf-8"))

    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    notebook["cells"][12]["source"] = writefile_cell("npo_lora_8b_pilot_protocol.json", protocol)
    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    notebook["cells"][13]["source"] = writefile_cell("requirements-npo-pilot.txt", requirements)

    notebook_bytes = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    notebook_path.write_bytes(notebook_bytes)

    baseline_manifest = package / "baseline_input" / "npo_8b_baseline_manifest.json"
    source_receipt = package / "baseline_input" / "source_verification_receipt.json"
    source_manifest = {
        "notebook_sha256": sha256(notebook_bytes),
        "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()),
        "baseline_manifest_sha256": sha256(baseline_manifest.read_bytes()),
        "source_verification_sha256": sha256(source_receipt.read_bytes()),
        "embedded_source_sha256": embedded_sources,
        "notebook_mode": args.mode,
        "full_training_enabled": args.mode == "full",
    }
    (package / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"package": str(package), **source_manifest}, indent=2))


if __name__ == "__main__":
    main()
