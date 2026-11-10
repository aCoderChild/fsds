#!/usr/bin/env python3
"""Poll the corrected 8B Kaggle diagnostic, download, and verify its archive."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
KERNEL = "bunnoob2005/tofu-8b-npo-lora-corrected-diagnostic-seed-43"
PROTOCOL = ROOT / "protocol.json"
MANIFEST = ROOT / "source_manifest.json"
ATTEMPT = ROOT / "attempt_manifest.json"
OUTPUT = ROOT / "kaggle_output"
ARTIFACT = "tofu_8b_npo_lora_corrected_v2_artifacts.zip"
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status() -> tuple[str, str]:
    proc = subprocess.run(["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True, timeout=60)
    detail = (proc.stdout + proc.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", detail)
    if proc.returncode or not match:
        raise RuntimeError(f"Kaggle status failed (exit={proc.returncode}): {detail[-1200:]}")
    return match.group(1), detail


def update_attempt(state: str, detail: str, **fields: object) -> None:
    data = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    data.update({"status": state, "last_checked_local": timestamp, **fields})
    data.setdefault("events", []).append({"checked_at_local": timestamp, "status": state, "detail": detail[-1200:]})
    ATTEMPT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{timestamp} {state}: {detail[-500:]}", flush=True)


def download() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="corrected8b-kaggle-output-") as temporary:
        proc = subprocess.run(
            ["kaggle", "kernels", "output", KERNEL, "--path", temporary,
             "--file-pattern", r"^tofu_8b_npo_lora_corrected_v2_artifacts\.zip$"],
            capture_output=True, text=True, timeout=900,
        )
        matches = list(Path(temporary).rglob(ARTIFACT))
        if proc.returncode or len(matches) != 1:
            raise RuntimeError(
                f"Artifact download failed (exit={proc.returncode}, matches={len(matches)}): "
                f"{proc.stdout[-1000:]} {proc.stderr[-1000:]}"
            )
        destination = OUTPUT / ARTIFACT
        shutil.copy2(matches[0], destination)
    return destination


def verify_archive(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failed at {bad}")
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
            raise RuntimeError("Archive contains an unsafe path")
        roots = {PurePosixPath(name).parts[0] for name in names}
        if len(roots) != 1:
            raise RuntimeError(f"Expected a single run root, found {sorted(roots)}")
        root_name = next(iter(roots))
        if not re.fullmatch(r"tofu_8b_npo_corrected_v2_run_\d{8}T\d{6}Z", root_name):
            raise RuntimeError(f"Unexpected run root: {root_name}")
        with tempfile.TemporaryDirectory(prefix="corrected8b-verify-") as temporary:
            zf.extractall(temporary)
            run_root = Path(temporary) / root_name
            archived_protocol = run_root / "npo_lora_8b_corrected_v2_protocol.json"
            if not archived_protocol.is_file() or sha256(archived_protocol) != sha256(PROTOCOL):
                raise RuntimeError("Archived protocol does not match the preregistered protocol")
            training = json.loads((run_root / "npo_train" / "training_receipt.json").read_text())
            smoke = json.loads((run_root / "npo_smoke" / "training_receipt.json").read_text())
            expected_trainer = json.loads(MANIFEST.read_text())["source_sha256"]["npo_lora_train.py"]
            if sha256(run_root / "npo_lora_train.py") != expected_trainer:
                raise RuntimeError("Archived trainer hash differs from preregistered source")
            if training.get("seed") != 43 or training.get("global_steps") != 500 or training.get("status") != "TRAINING_COMPLETE":
                raise RuntimeError("Training receipt does not prove the locked seed-43, 500-step completion")
            if smoke.get("seed") != 43 or smoke.get("status") != "SMOKE_PASS" or smoke.get("global_steps") != 1:
                raise RuntimeError("Bounded real-data smoke receipt failed its locked criteria")
            if training.get("protocol_sha256") != sha256(PROTOCOL) or smoke.get("protocol_sha256") != sha256(PROTOCOL):
                raise RuntimeError("Training/smoke receipt protocol hash mismatch")
            objective = training.get("objective_equivalence")
            if not isinstance(objective, dict) or objective.get("status") != "PASS":
                raise RuntimeError("Objective-equivalence receipt missing or failed")
            objective_path = Path(temporary) / "objective_equivalence.json"
            objective_path.write_text(json.dumps(objective, indent=2) + "\n")
            train_dir = run_root / "npo_train"
            eval_dir = run_root / "tofu_official_eval"
            subprocess.run(
                ["python", str(Path(__file__).resolve().parents[1] / "verify_npo_run.py"),
                 "--model", "8b", "--training-dir", str(train_dir), "--eval-dir", str(eval_dir),
                 "--protocol", str(PROTOCOL), "--objective-check", str(objective_path)],
                check=True, capture_output=True, text=True,
            )
            verification = json.loads((eval_dir / "npo_verification.json").read_text())
            if verification.get("overall_status") != "PASS":
                raise RuntimeError(f"Independent local metric verification failed: {verification.get('problems')}")
            return {
                "status": verification["overall_status"],
                "research_diagnostic_gate_status": verification["research_diagnostic_gate_status"],
                "problems": verification["problems"],
                "primary_result": verification["primary_result"],
                "utility_gates": verification["utility_gates"],
                "training_steps": training["global_steps"],
                "seed": training["seed"],
                "trainer_sha256": expected_trainer,
                "protocol_sha256": sha256(PROTOCOL),
            }


def main() -> None:
    while True:
        state, detail = status()
        update_attempt(state, detail, kaggle_kernel_running=(state == "RUNNING"), result_available=False)
        if state in TERMINAL:
            if state != "COMPLETE":
                update_attempt(state, detail, kaggle_kernel_running=False, result_available=False)
                raise SystemExit(f"Kaggle run ended with {state}; inspect Kaggle output and logs.")
            archive = download()
            result = verify_archive(archive)
            update_attempt(
                "COMPLETE_VERIFIED_DIAGNOSTIC", detail,
                training_started=True, result_available=True,
                artifact={"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256(archive), "crc": "PASS"},
                verification=result,
                completed_at_utc=utc_now(),
            )
            print(json.dumps(result, indent=2), flush=True)
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
