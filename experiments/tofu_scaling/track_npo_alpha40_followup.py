#!/usr/bin/env python3
"""Track the Kaggle alpha-40 run and independently verify its downloaded output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNEL_ID = "bunnoob2005/tofu-3b-npo-lora-alpha-40-follow-up"
PROTOCOL = ROOT / "npo_lora_alpha40_followup_protocol.json"
DIAGNOSTICS = ROOT / "diagnostics" / "npo_lora_profile_alpha40_2026-11-08"
TRACKER = DIAGNOSTICS / "run_tracker.json"
DEFAULT_OUTPUT = DIAGNOSTICS / "kaggle_output"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_tracker() -> dict:
    if TRACKER.is_file():
        return json.loads(TRACKER.read_text(encoding="utf-8"))
    return {
        "protocol_id": "tofu_3b_npo_lora_profile_alpha40_followup_v1",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "kaggle_kernel": KERNEL_ID,
        "events": [],
    }


def write_tracker(value: dict, status: str, detail: str) -> None:
    value["last_status"] = status
    value["updated_at_utc"] = utc_now()
    value.setdefault("events", []).append({"at_utc": value["updated_at_utc"], "status": status, "detail": detail})
    TRACKER.parent.mkdir(parents=True, exist_ok=True)
    TRACKER.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"{value['updated_at_utc']} {status}: {detail}", flush=True)


def kernel_status() -> tuple[str, str]:
    result = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if result.returncode != 0 or not match:
        raise RuntimeError(f"Could not read Kaggle kernel state (exit={result.returncode}): {output}")
    return match.group(1), output


def download_and_verify(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle", "kernels", "output", KERNEL_ID, "-p", str(output), "-o",
        "--file-pattern", r"^(tofu_npo_alpha40_followup_artifacts\.zip|alpha40_independent_verification\.json)$",
    ]
    subprocess.run(command, check=True)
    archive = output / "tofu_npo_alpha40_followup_artifacts.zip"
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError(f"Kaggle completed without the expected artifact ZIP: {archive}")
    digest_obj = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_obj.update(chunk)
    digest = digest_obj.hexdigest()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Downloaded artifact archive CRC failed at {bad}")
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise RuntimeError("Downloaded artifact archive contains an unsafe path")
        roots = {Path(name).parts[0] for name in names}
        if len(roots) != 1:
            raise RuntimeError(f"Expected one run root in archive; found {roots}")
        root_name = next(iter(roots))
        with tempfile.TemporaryDirectory(prefix="tofu-alpha40-verify-") as temp:
            extract = Path(temp) / root_name
            zf.extractall(Path(temp))
            verification = output / "alpha40_independent_verification_local.json"
            command = [
                "python", str(ROOT / "verify_npo_profile_recovery_run.py"),
                "--run-root", str(extract), "--protocol", str(PROTOCOL),
                "--artifact-zip", str(archive), "--output", str(verification),
            ]
            subprocess.run(command, check=True)
            result = json.loads(verification.read_text(encoding="utf-8"))
    return {"archive_path": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": digest, "archive_crc": "PASS", "verification": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="poll until Kaggle reports COMPLETE or ERROR")
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--download-on-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    tracker = read_tracker()
    expected_protocol = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if tracker.get("protocol_sha256", expected_protocol) != expected_protocol:
        raise RuntimeError("Local protocol changed after run tracking began")
    while True:
        state, detail = kernel_status()
        write_tracker(tracker, state, detail)
        if state in {"COMPLETE", "ERROR", "CANCELLED"}:
            break
        if not args.watch:
            print(json.dumps(tracker, indent=2))
            return
        time.sleep(args.poll_seconds)
    if state == "COMPLETE" and args.download_on_complete:
        try:
            evidence = download_and_verify(args.output)
            tracker["downloaded_result"] = evidence
            write_tracker(tracker, evidence["verification"]["status"], "downloaded artifact independently verified locally")
        except Exception as error:
            write_tracker(tracker, "OUTPUT_OR_VERIFICATION_ERROR", f"{type(error).__name__}: {error}")
            raise
    if state == "ERROR":
        raise SystemExit("Kaggle kernel failed; inspect its output receipt/log artifacts before interpreting it")


if __name__ == "__main__":
    main()
