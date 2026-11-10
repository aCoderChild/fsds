#!/usr/bin/env python3
"""Record current status for all visible Kaggle kernels related to SynthWorlds R1."""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


USER = "bunnoob2005"
OUT = Path(__file__).resolve().parent
JSON_OUT = OUT / "kaggle_related_job_inventory_2026-11-05.json"
MD_OUT = OUT / "kaggle_related_job_inventory_2026-11-05.md"
PAGE_SIZE = 200
RELATED_MARKERS = ("synthworld", "sw-gap", "swsm", "sw-third")
EXPLICIT_REFS = {f"{USER}/experiments"}


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def parse_listing(raw: str) -> list[dict[str, str]]:
    lines = raw.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("ref,title,author,")), None)
    if header is None:
        nonempty = [line for line in lines if line.strip()]
        if (
            not nonempty
            or "Not found" in nonempty
            or all(line.startswith("Warning: Looks like you're using an outdated") for line in nonempty)
        ):
            return []
        raise RuntimeError("Kaggle kernel listing has no CSV header")
    return list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))


def list_all_pages() -> tuple[list[dict[str, str]], int]:
    all_rows: list[dict[str, str]] = []
    page = 1
    while True:
        proc = run([
            "kaggle", "kernels", "list", "--user", USER, "--page-size", str(PAGE_SIZE),
            "--page", str(page), "--sort-by", "dateRun", "--csv",
        ])
        if proc.returncode:
            raise RuntimeError(f"Kaggle kernel listing page {page} failed: {proc.stderr.strip()}")
        rows = parse_listing(proc.stdout)
        all_rows.extend(rows)
        if not rows:
            return all_rows, page
        page += 1


def query_status(ref: str) -> dict[str, str | int]:
    proc = run(["kaggle", "kernels", "status", ref])
    output = proc.stdout + proc.stderr
    match = re.search(r'has status "([^"]+)"', output)
    return {
        "status": match.group(1) if match else "STATUS_QUERY_FAILED",
        "exit_code": proc.returncode,
    }


def main() -> None:
    rows, pages = list_all_pages()
    selected = [
        row for row in rows
        if row.get("ref") in EXPLICIT_REFS
        or any(marker in row.get("ref", "").lower() for marker in RELATED_MARKERS)
    ]
    kernels = {}
    for row in selected:
        ref = row["ref"]
        kernels[ref] = {
            "title": row.get("title", ""),
            "last_run_time": row.get("lastRunTime", ""),
            **query_status(ref),
        }
    counts = Counter(item["status"] for item in kernels.values())
    started = datetime.now(timezone.utc).isoformat()
    result = {
        "observed_at_utc": started,
        "status": "PASS_SCOPED_LIVE_INVENTORY",
        "scope": (
            "All Kaggle kernels visible to the configured account in the retrieved pages whose refs "
            "contain SynthWorlds R1 naming markers, plus the legacy experiments alias. Status is "
            "Kaggle job state only; completed jobs require output-receipt inspection to determine "
            "whether research training occurred."
        ),
        "account": USER,
        "listing_pages": pages,
        "visible_kernels_scanned": len(rows),
        "related_kernels": len(kernels),
        "status_counts": dict(sorted(counts.items())),
        "kernels": kernels,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Kaggle SynthWorlds R1 job inventory",
        "",
        f"Observed: **{started}**",
        "",
        f"Scanned {len(rows)} visible kernels across {pages} listing page(s); {len(kernels)} matched the recorded R1 naming scope.",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    lines.extend([
        "",
        "| Kernel | Last run (Kaggle local time) | Status |",
        "|---|---|---|",
    ])
    lines.extend(
        f"| {ref} | {item['last_run_time']} | {item['status']} |"
        for ref, item in kernels.items()
    )
    lines.extend([
        "",
        "This inventory is a point-in-time status check, not evidence that a completed job produced a research result. "
        "Inspect each relevant output receipt/log before interpreting its scientific role.",
        "",
        "Reproduce with python -B audit_kaggle_job_inventory.py.",
        "",
    ])
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "scanned": len(rows),
        "related": len(kernels),
        "status_counts": dict(sorted(counts.items())),
        "json": str(JSON_OUT),
        "markdown": str(MD_OUT),
    }, indent=2))
    if any(item["status"] == "STATUS_QUERY_FAILED" or item["exit_code"] != 0 for item in kernels.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
