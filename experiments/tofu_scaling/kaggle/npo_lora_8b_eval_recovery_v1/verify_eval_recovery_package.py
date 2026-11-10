#!/usr/bin/env python3
"""Offline integrity checks for the evaluation-only Kaggle recovery notebook."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
NOTEBOOK = PACKAGE / "tofu_8b_npo_eval_recovery.ipynb"
PROTOCOL = ROOT / "npo_lora_8b_pilot_protocol.json"
OBJECTIVE = ROOT / "diagnostics" / "npo_lora_8b_pilot_2026-11-08" / "objective_equivalence.json"
SOURCE_MANIFEST = ROOT / "kaggle" / "npo_lora_8b_pilot_full" / "source_manifest.json"
BASELINE_INPUT = ROOT / "kaggle" / "npo_lora_8b_pilot_full" / "baseline_input"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture_archive(path: Path, *, corrupt_adapter: bool = False,
                         unsafe_member: bool = False, missing_evaluation: bool = False) -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_name = "tofu_8b_npo_run_20261108T210000Z"
    fixture = path.parent / (path.stem + "_root") / run_name
    if fixture.parent.exists():
        shutil.rmtree(fixture.parent)
    fixture.mkdir(parents=True)
    shutil.copy2(PROTOCOL, fixture / "npo_lora_8b_pilot_protocol.json")
    source_paths = {
        "tofu_official_eval.py": ROOT / "tofu_official_eval.py",
        "verify_tofu_official_eval.py": ROOT / "verify_tofu_official_eval.py",
        "npo_objective.py": ROOT / "npo_objective.py",
        "verify_npo_objective.py": ROOT / "verify_npo_objective.py",
        "npo_lora_train.py": ROOT / "npo_lora_train.py",
        "evaluate_npo_adapter.py": ROOT / "evaluate_npo_adapter.py",
        "verify_npo_run.py": ROOT / "verify_npo_run.py",
    }
    for name, source in source_paths.items():
        assert sha256(source) == manifest["embedded_source_sha256"][name]
        shutil.copy2(source, fixture / name)

    baseline_inputs = fixture / "baseline_inputs"
    shutil.copytree(BASELINE_INPUT, baseline_inputs)
    eval_dir = fixture / "tofu_official_eval"
    eval_dir.mkdir()
    mappings = {
        "llama_3_1_8b_full_official_eval.json": "llama_3_1_8b_full_official_eval.json",
        "llama_3_1_8b_retain90_official_eval.json": "llama_3_1_8b_retain90_official_eval.json",
        "tofu_readiness_audit.json": "audit/tofu_readiness_audit.json",
        "official_eval_status.json": "official_eval_status.json",
        "source_verification_receipt.json": "source_verification/official_eval_verification.json",
    }
    for source, target in mappings.items():
        destination = eval_dir / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_inputs / source, destination)

    train_dir = fixture / "npo_train"
    adapter_dir = train_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fixture-adapter-bytes")
    dataset = protocol["dataset"]
    training_data = dataset["training"]
    receipt = {
        "status": "TRAINING_COMPLETE",
        "run_type": "exploratory_single_seed",
        "started_at_utc": "2026-11-08T21:01:00+00:00",
        "model": protocol["base_checkpoint"],
        "dataset": {
            "repo": dataset["repo_id"],
            "revision": dataset["revision"],
            "forget10_sha256": training_data["forget_file"]["sha256"],
            "retain90_sha256": training_data["retain_file"]["sha256"],
            "forget10_rows": 400,
            "retain90_rows": 3600,
        },
        "method": {"beta": protocol["objective"]["beta"],
                   "epochs": protocol["adaptation"]["epochs"],
                   "lora_rank": protocol["adaptation"]["lora_rank"]},
        "protocol_sha256": manifest["protocol_sha256"],
        "training_script_sha256": manifest["embedded_source_sha256"]["npo_lora_train.py"],
        "seed": 42,
        "training_started": True,
        "global_steps": 500,
        "adapter_saved": True,
        "adapter_files_sha256": {p.name: sha256(p) for p in sorted(adapter_dir.iterdir())},
        "tokenization": {"truncated_examples": 0, "max_length_observed": 300},
        "runtime": {"cpu_or_disk_offload": False, "device_map": {"model.embed_tokens": "0", "model.layers.0": "1"}},
        "gpu_names": ["Tesla T4", "Tesla T4"],
        "training_schedule": {"planned_optimizer_steps": 500, "planned_loss_trace_rows": 4000},
    }
    train_dir.mkdir(exist_ok=True)
    (train_dir / "training_receipt.json").write_text(json.dumps(receipt) + "\n")
    if corrupt_adapter:
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"altered-after-receipt")
    (train_dir / "loss_trace.jsonl").write_text("".join(
        json.dumps({"global_step": index % 500 + 1}) + "\n" for index in range(4000)
    ))
    if not missing_evaluation:
        (eval_dir / "llama_3_1_8b_npo_lora_seed42_official_eval.json").write_text("{}\n")
        (eval_dir / "npo_comparison.json").write_text("{}\n")

    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(fixture.parent.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(fixture.parent))
        if unsafe_member:
            archive.writestr("../unsafe_escape.txt", "must never be extracted")


def exercise_setup_cell(setup_source: str, temp_root: Path, archive_path: Path) -> tuple[bool, str]:
    temp_root.mkdir(parents=True, exist_ok=True)
    input_dir = temp_root / "input"
    working_dir = temp_root / "working"
    input_dir.mkdir(exist_ok=True)
    working_dir.mkdir(exist_ok=True)
    shutil.copy2(archive_path, input_dir / "tofu_8b_npo_lora_pilot_artifacts.zip")
    source = setup_source.replace('Path("/kaggle/input")', f"Path({str(input_dir)!r})")
    source = source.replace('Path("/kaggle/working/eval_recovery_input")',
                            f"Path({str(working_dir / 'eval_recovery_input')!r})")
    source = source.replace('Path("/kaggle/working/npo_objective_equivalence.json")',
                            f"Path({str(working_dir / 'npo_objective_equivalence.json')!r})")
    output = io.StringIO()
    namespace = {"__name__": "recovery_fixture"}
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(source, "<notebook-setup-cell>", "exec"), namespace)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
    return bool(namespace["needs_inference"]), output.getvalue()


def main() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    assert len(notebook.get("cells", [])) == 4
    assert sha256(PROTOCOL) == manifest["protocol_sha256"]
    assert json.loads(OBJECTIVE.read_text(encoding="utf-8")).get("protocol_sha256") == manifest["protocol_sha256"]

    code_sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source)
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
        code_sources.append(source)

    setup_source, evaluation_source = code_sources[1:]
    for name, digest in manifest["embedded_source_sha256"].items():
        assert digest in setup_source, f"missing source pin: {name}"
    assert "RUN_FULL_TRAINING" not in "\n".join(code_sources)
    assert "npo_lora_train.py'," not in evaluation_source
    assert "evaluate_npo_adapter.py" in evaluation_source
    assert "verify_npo_run.py" in evaluation_source
    assert 'get("global_steps") != 500' in setup_source
    assert "adapter_files_sha256" in setup_source
    assert "archive.testzip()" in setup_source
    assert "Unsafe archive member rejected" in setup_source
    assert "LAUNCH_CUTOFF_UTC" in setup_source
    assert "loss_trace.jsonl" in setup_source
    assert "subprocess.Popen" in evaluation_source
    assert "capture_output=True" not in evaluation_source
    assert "enable_internet" in json.loads((PACKAGE / "kernel-metadata.json").read_text())
    assert json.loads((PACKAGE / "kernel-metadata.json").read_text()).get("is_private") is True

    with tempfile.TemporaryDirectory(prefix="npo_eval_recovery_fixture_") as temporary:
        temp_root = Path(temporary)
        good = temp_root / "good.zip"
        make_fixture_archive(good)
        needs_inference, output = exercise_setup_cell(setup_source, temp_root / "good_case", good)
        if needs_inference or "'preflight': 'PASS'" not in output:
            raise AssertionError(f"valid fixture was rejected: needs_inference={needs_inference}; output={output}")

        bad_adapter = temp_root / "bad_adapter.zip"
        make_fixture_archive(bad_adapter, corrupt_adapter=True)
        try:
            exercise_setup_cell(setup_source, temp_root / "bad_adapter_case", bad_adapter)
        except RuntimeError as error:
            assert "adapter files do not match" in str(error)
        else:
            raise AssertionError("preflight accepted adapter bytes that differ from the training receipt")

        unsafe = temp_root / "unsafe.zip"
        make_fixture_archive(unsafe, unsafe_member=True)
        try:
            exercise_setup_cell(setup_source, temp_root / "unsafe_case", unsafe)
        except RuntimeError as error:
            assert "Unsafe archive member" in str(error)
        else:
            raise AssertionError("preflight accepted an archive path traversal member")

        missing_eval = temp_root / "missing_eval.zip"
        make_fixture_archive(missing_eval, missing_evaluation=True)
        try:
            exercise_setup_cell(setup_source, temp_root / "missing_eval_case", missing_eval)
        except RuntimeError as error:
            assert "Select Kaggle GPU T4 x2" in str(error)
        else:
            raise AssertionError("preflight did not stop when inference was needed but no GPU was available")

    print(json.dumps({
        "status": "PASS_OFFLINE_PACKAGE_CHECKS",
        "notebook_sha256": sha256(NOTEBOOK),
        "protocol_sha256": sha256(PROTOCOL),
        "objective_check_status": json.loads(OBJECTIVE.read_text()).get("status"),
        "code_cells_compiled": len(code_sources),
        "archive_preflight_fixtures": {
            "valid_completed_archive_without_gpu": "PASS",
            "modified_adapter_rejected": "PASS",
            "zip_path_traversal_rejected": "PASS",
            "missing_evaluation_without_gpu_rejected": "PASS",
        },
        "training_invocation_present": False,
        "evaluation_stdout_streamed": True,
        "gpu_or_kaggle_execution": False,
    }, indent=2))


if __name__ == "__main__":
    main()
