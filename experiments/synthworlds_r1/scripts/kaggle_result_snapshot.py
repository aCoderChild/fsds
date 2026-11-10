#!/usr/bin/env python3
"""Create and restore integrity-checked Kaggle stage snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = "synthworlds-r1-kaggle-snapshot-v1"
MANIFEST_MEMBER = "__snapshot__/manifest.json"
ALLOWED_STAGES = {"stageA_sft", "stageB_baseline", "stageC_unlearning", "stageD_post_unlearning"}
MAX_SNAPSHOT_BYTES = 5 * 1024**3


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_snapshot(results_root: Path, archive_path: Path, stage_name: str) -> Path:
    results_root = Path(results_root).resolve()
    archive_path = Path(archive_path)
    if stage_name not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported snapshot stage: {stage_name}")
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    files = []
    for path in sorted(p for p in results_root.rglob("*") if p.is_file()):
        if any(part.startswith("checkpoint-") for part in path.parts):
            continue
        relative = (Path(results_root.name) / path.relative_to(results_root)).as_posix()
        data_hash = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                data_hash.update(block)
                size += len(block)
        files.append({"path": relative, "sha256": data_hash.hexdigest(), "size_bytes": size})

    if not files:
        raise ValueError("Refusing to snapshot an empty results directory")
    manifest = {"schema": SCHEMA, "stage": stage_name, "results_root": results_root.name, "files": files}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(f".{archive_path.name}.tmp")
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for record in files:
                source = results_root.parent / record["path"]
                archive.write(source, record["path"])
            archive.writestr(MANIFEST_MEMBER, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temp_path, archive_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return archive_path


def restore_snapshot(archive_paths: list[Path], working_root: Path, results_root: Path) -> dict:
    if len(archive_paths) != 1:
        raise ValueError(f"Attach exactly one *_latest.zip result snapshot; found {len(archive_paths)}")
    archive_path = Path(archive_paths[0])
    working_root = Path(working_root).resolve()
    results_root = Path(results_root).resolve()
    if results_root.parent != working_root:
        raise ValueError("Results must restore directly under the declared Kaggle working directory")

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or MANIFEST_MEMBER not in names:
            raise ValueError("Snapshot has duplicate ZIP entries or no integrity manifest")
        manifest_info = archive.getinfo(MANIFEST_MEMBER)
        if stat.S_ISLNK(manifest_info.external_attr >> 16):
            raise ValueError("Snapshot manifest cannot be a symbolic link")
        manifest = json.loads(archive.read(MANIFEST_MEMBER).decode("utf-8"))
        if (manifest.get("schema") != SCHEMA or manifest.get("results_root") != results_root.name
                or manifest.get("stage") not in ALLOWED_STAGES or not isinstance(manifest.get("files"), list)):
            raise ValueError("Snapshot manifest schema, stage, or results root is invalid")

        records = {}
        total_size = 0
        for record in manifest["files"]:
            if not isinstance(record, dict):
                raise ValueError("Snapshot file entry is malformed")
            name = record.get("path")
            relative = PurePosixPath(str(name))
            if (not isinstance(name, str) or "\\" in name or relative.is_absolute()
                    or ".." in relative.parts or not relative.parts
                    or relative.parts[0] != results_root.name or len(relative.parts) < 2):
                raise ValueError(f"Unsafe or out-of-root snapshot path: {name!r}")
            if name in records or name not in names:
                raise ValueError(f"Duplicate or missing snapshot file: {name}")
            info = archive.getinfo(name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Snapshot entries must be regular files: {name}")
            total_size += info.file_size
            if total_size > MAX_SNAPSHOT_BYTES:
                raise ValueError(f"Snapshot exceeds the {MAX_SNAPSHOT_BYTES} byte safety limit")
            data = archive.read(name)
            if (len(data) != record.get("size_bytes")
                    or sha256_bytes(data) != record.get("sha256")):
                raise ValueError(f"Snapshot checksum/size mismatch: {name}")
            records[name] = record
        if set(names) != set(records) | {MANIFEST_MEMBER}:
            raise ValueError("Snapshot contains unindexed ZIP entries")

        existing_paths = list(results_root.rglob("*")) if results_root.exists() else []
        if any(path.is_symlink() for path in existing_paths):
            raise FileExistsError(f"Refusing to restore into a results directory containing symlinks: {results_root}")
        if any(path.is_file() for path in existing_paths):
            raise FileExistsError(f"Refusing to merge snapshot into non-empty results directory: {results_root}")
        for name in records:
            destination = working_root / PurePosixPath(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))

    return manifest
