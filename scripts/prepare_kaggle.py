#!/usr/bin/env python3
"""Prepare Kaggle CLI folders from committed notebook files.

This step does NOT upload or run notebooks. Provenance: notebooks are stored in
this repository, and the original MQuAKE-T facts are downloaded at runtime.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = {
    "01": (
        "mquake-real-01-source-translation-sft",
        "MQuAKE Real 01 Source Translation SFT",
        "01_mquake_t_source_translate_sft.ipynb",
        [],
    ),
    "02": (
        "mquake-real-02-unlearning",
        "MQuAKE Real 02 Unlearning",
        "02_mquake_t_unlearning_baselines.ipynb",
        ["01"],
    ),
    "03": (
        "mquake-real-03-evaluation",
        "MQuAKE Real 03 Evaluation",
        "03_mquake_t_crosslingual_evaluation.ipynb",
        ["01", "02"],
    ),
}

def prepare(username, stage, execute, outdir=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Kaggle username must be an account slug, not an email or URL.")
    if stage == "all" and execute:
        raise ValueError("Run ONE stage at a time so the previous stage's saved outputs exist.")
    outdir = Path(outdir) if outdir else ROOT / ".kaggle" / "publish"
    outdir.mkdir(parents=True, exist_ok=True)
    stages = list(SPEC) if stage == "all" else [stage]
    for number in stages:
        slug, title, filename, parents = SPEC[number]
        source = ROOT / "notebooks" / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        notebook = json.loads(source.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not notebook.get("cells"):
            raise ValueError(f"Invalid notebook: {source}")
        dest = outdir / number
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest / filename)
        # For a no-run bulk upload, upstream versions might not exist yet.
        # Attach dependencies for single-stage operations after their source runs.
        sources = [f"{username}/{SPEC[p][0]}" for p in parents] if stage != "all" else []
        meta = {
            "id": f"{username}/{slug}",
            "title": title,
            "code_file": filename,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": sources,
            "model_sources": [],
        }
        (dest / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Prepared {number}: {meta['id']}, inputs={len(sources)}")
    return outdir, stages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--stage", choices=["01", "02", "03", "all"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    prepare(args.username, args.stage, args.execute, args.outdir)
