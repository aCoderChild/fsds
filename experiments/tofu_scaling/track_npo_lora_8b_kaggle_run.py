#!/usr/bin/env python3
"""Track the locked Kaggle 8B pilot and download only its result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath


KERNEL = "bunnoob2005/tofu-8b-npo-lora-pilot-2026-11-08"
ARTIFACT_PATTERN = r"tofu_8b_npo_lora_pilot_artifacts\.zip$"
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED"}
PROTOCOL = Path(__file__).resolve().parent / "npo_lora_8b_pilot_protocol.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def kaggle_status() -> tuple[str, str]:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    output = (proc.stdout + proc.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if proc.returncode != 0 or not match:
        raise RuntimeError(f"Could not parse Kaggle kernel status (exit={proc.returncode}): {output[-1000:]}")
    return match.group(1), output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_identity(artifact: Path, not_before: datetime) -> dict:
    """Bind an output ZIP to this full training attempt, not a smoke-only run."""
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member path: {name}")
        roots = {
            name.split("/", 1)[0]
            for name in names
            if name and "/" in name
        }
        parsed = []
        for root in roots:
            match = re.fullmatch(r"tofu_8b_npo_run_(\d{8}T\d{6}Z)", root)
            if match:
                started = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                parsed.append((root, started))
        if len(parsed) != 1:
            raise ValueError(f"Expected one timestamped run root in archive; found {sorted(roots)}")
        run_root, run_started = parsed[0]
        prefix = run_root + "/"
        protocol_name = prefix + "npo_lora_8b_pilot_protocol.json"
        smoke_name = prefix + "npo_smoke/training_receipt.json"
        train_name = prefix + "npo_train/training_receipt.json"
        for required in (protocol_name, smoke_name, train_name):
            if required not in names:
                raise ValueError(f"Current full-run archive is missing required member: {required}")
        expected_protocol_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        archive_protocol_sha = hashlib.sha256(archive.read(protocol_name)).hexdigest()
        if archive_protocol_sha != expected_protocol_sha:
            raise ValueError("Archive protocol differs from the locally locked protocol")

        smoke = json.loads(archive.read(smoke_name))
        training = json.loads(archive.read(train_name))
        smoke_started = datetime.fromisoformat(smoke["started_at_utc"].replace("Z", "+00:00"))
        smoke_completed = datetime.fromisoformat(smoke["completed_at_utc"].replace("Z", "+00:00"))
        training_started = datetime.fromisoformat(training["started_at_utc"].replace("Z", "+00:00"))
        # The run directory and bounded smoke may precede the formal full-training
        # cutoff because setup/smoke runs first. Bind them chronologically to the
        # completed training receipt instead of rejecting this same attempt.
        if run_started > smoke_started or training_started < smoke_completed:
            raise ValueError("Run root, bounded smoke, and full-training receipt are out of order")
        if training_started - run_started > timedelta(minutes=30):
            raise ValueError("Full-training receipt is too far after the run-root timestamp")
        if training_started < not_before:
            raise ValueError(f"Full-training receipt predates this attempt ({not_before.isoformat()})")
        for name, receipt, expected_type in (
            (smoke_name, smoke, "bounded_real_data_smoke"),
            (train_name, training, "exploratory_single_seed"),
        ):
            if receipt.get("run_type") != expected_type or receipt.get("seed") != 42:
                raise ValueError(f"Unexpected run type or seed in {name}")
            if receipt.get("protocol_sha256") != expected_protocol_sha:
                raise ValueError(f"Receipt protocol differs from the locked protocol in {name}")
        if smoke.get("status") != "SMOKE_PASS" or smoke.get("global_steps") != 1:
            raise ValueError("Bounded real-data smoke receipt did not pass")
        if (training.get("status") != "TRAINING_COMPLETE"
                or training.get("global_steps") != 500
                or training.get("adapter_saved") is not True):
            raise ValueError("Full-training receipt does not prove the locked 500-step run completed")
        return {
            "run_root": run_root,
            "run_started_at_utc": run_started.isoformat(),
            "smoke_receipt": {
                "status": smoke["status"],
                "started_at_utc": smoke["started_at_utc"],
                "completed_at_utc": smoke["completed_at_utc"],
                "global_steps": smoke["global_steps"],
            },
            "training_receipt": {
                "status": training["status"],
                "started_at_utc": training["started_at_utc"],
                "completed_at_utc": training.get("completed_at_utc"),
                "global_steps": training["global_steps"],
                "protocol_sha256": training["protocol_sha256"],
            },
        }


def download_archive(
    output_dir: Path,
    not_before: datetime,
    attempts: int = 4,
) -> tuple[Path | None, dict | None, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_rejections = 0
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory(prefix="npo_8b_output_attempt_") as temporary:
            temp_dir = Path(temporary)
            proc = subprocess.run(
                [
                    "kaggle", "kernels", "output", KERNEL,
                    "--path", str(temp_dir),
                    "--file-pattern", ARTIFACT_PATTERN,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
            matches = sorted(temp_dir.glob("tofu_8b_npo_lora_pilot_artifacts.zip"))
            if proc.returncode == 0 and matches:
                artifact = matches[-1]
                try:
                    identity = archive_identity(artifact, not_before)
                except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
                    stale_rejections += 1
                    print(json.dumps({
                        "event": "reject_stale_or_invalid_archive",
                        "attempt": attempt,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }), flush=True)
                else:
                    destination = output_dir / artifact.name
                    shutil.copy2(artifact, destination)
                    return destination, identity, stale_rejections
        print(
            json.dumps({
                "event": "artifact_download_retry",
                "attempt": attempt,
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout[-500:],
                "stderr_tail": proc.stderr[-500:],
            }),
            flush=True,
        )
        if attempt < attempts:
            time.sleep(60)
    return None, None, stale_rejections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument(
        "--not-before-utc",
        required=True,
        help="Reject result archives whose timestamped run folder predates this UTC launch time.",
    )
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=14)
    args = parser.parse_args()
    if args.interval_seconds < 10 or args.max_wait_hours <= 0:
        parser.error("interval must be >=10 seconds and max wait must be positive")
    try:
        not_before = datetime.fromisoformat(args.not_before_utc.replace("Z", "+00:00"))
    except ValueError:
        parser.error("--not-before-utc must be an ISO-8601 UTC timestamp")
    if not_before.tzinfo is None or not_before.utcoffset():
        parser.error("--not-before-utc must specify UTC")

    deadline = time.monotonic() + args.max_wait_hours * 3600
    last_status = None
    while time.monotonic() < deadline:
        try:
            status, raw = kaggle_status()
        except Exception as exc:
            status, raw = "STATUS_CHECK_ERROR", f"{type(exc).__name__}: {exc}"
        if status != last_status:
            event = {"observed_at_utc": now_utc(), "status": status, "kernel": KERNEL, "raw_status": raw[-1000:]}
            record(args.event_log, event)
            print(json.dumps(event), flush=True)
            last_status = status
        if status in TERMINAL:
            break
        if status == "STATUS_CHECK_ERROR":
            # Keep transient API/CLI failures from being mistaken for a terminal run state.
            event = {"observed_at_utc": now_utc(), "status": status, "kernel": KERNEL, "detail": raw[-1000:]}
            record(args.event_log, event)
        time.sleep(args.interval_seconds)

    if last_status not in TERMINAL:
        result = {
            "status": "MONITOR_TIMEOUT",
            "kernel": KERNEL,
            "checked_at_utc": now_utc(),
            "last_kernel_status": last_status,
            "artifact_downloaded": False,
        }
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return 2

    artifact, identity, stale_rejections = download_archive(args.output_dir, not_before)
    result = {
        "status": last_status,
        "kernel": KERNEL,
        "checked_at_utc": now_utc(),
        "artifact_downloaded": artifact is not None,
        "artifact_path": str(artifact) if artifact else None,
        "artifact_sha256": sha256(artifact) if artifact else None,
        "artifact_bytes": artifact.stat().st_size if artifact else None,
        "artifact_crc": "NOT_CHECKED",
        "artifact_identity": identity,
        "stale_or_invalid_archives_rejected": stale_rejections,
    }
    if artifact:
        try:
            with zipfile.ZipFile(artifact) as archive:
                bad_member = archive.testzip()
                result["artifact_crc"] = "PASS" if bad_member is None else f"FAIL:{bad_member}"
                result["artifact_members"] = len(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            result["artifact_crc"] = f"FAIL:{type(exc).__name__}:{exc}"
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.status_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    record(args.event_log, {"observed_at_utc": now_utc(), **result})
    print(json.dumps(result, indent=2), flush=True)
    return 0 if artifact and result["artifact_crc"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
