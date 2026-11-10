#!/usr/bin/env python3
"""Verify the local-only streaming fallback without loading a model or GPU."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ERRORS: list[str] = []


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def main() -> int:
    notebook_path = HERE / 'tofu_8b_npo_lora_pilot.ipynb'
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    cells = [''.join(cell.get('source', [])) for cell in notebook['cells']]
    mode = cells[1]
    check('RUN_FULL_TRAINING = True' in mode, 'package is not full-training mode')
    protocol = json.loads((ROOT / 'npo_lora_8b_pilot_protocol.json').read_text())
    check(protocol.get('status') == 'PREREGISTERED_BEFORE_GPU_RUN', 'locked protocol status changed')
    check(protocol.get('protocol_id') == 'tofu_8b_npo_lora_pilot_v1', 'protocol ID changed')
    check(digest(HERE / 'npo_lora_8b_pilot_protocol.json') == digest(ROOT / 'npo_lora_8b_pilot_protocol.json'), 'package protocol file differs from the locked protocol')

    source_manifest = json.loads((HERE / 'source_manifest.json').read_text())
    check(source_manifest.get('notebook_sha256') == digest(notebook_path), 'notebook SHA-256 does not match source manifest')
    check(source_manifest.get('protocol_sha256') == digest(ROOT / 'npo_lora_8b_pilot_protocol.json'), 'protocol SHA-256 does not match source manifest')
    baseline_manifest_path = HERE / 'baseline_input' / 'npo_8b_baseline_manifest.json'
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    check(source_manifest.get('baseline_manifest_sha256') == digest(baseline_manifest_path), 'baseline manifest SHA-256 does not match source manifest')
    check(baseline_manifest.get('status') == 'PASS' and baseline_manifest.get('source_independent_verification_status') == 'PASS', 'baseline input is not independently verified')
    for relative, expected in baseline_manifest.get('files', {}).items():
        path = HERE / 'baseline_input' / relative
        check(path.is_file() and digest(path) == expected, f'baseline input missing or hash mismatch: {relative}')

    metadata = json.loads((HERE / 'kernel-metadata.json').read_text())
    check(metadata.get('id') == 'bunnoob2005/tofu-8b-npo-lora-pilot-streaming-v1', 'fallback kernel slug is not isolated')
    check(metadata.get('is_private') is True and metadata.get('enable_gpu') is True and metadata.get('enable_internet') is True, 'Kaggle runtime settings are not private/GPU/internet enabled')
    check(metadata.get('dataset_sources') == ['bunnoob2005/tofu-8b-npo-baseline-inputs-2026-11-08'], 'fallback uses a different baseline dataset')

    embedded_map = {
        5: 'tofu_official_eval.py',
        6: 'verify_tofu_official_eval.py',
        7: 'npo_objective.py',
        8: 'verify_npo_objective.py',
        9: 'npo_lora_train.py',
        10: 'evaluate_npo_adapter.py',
        11: 'verify_npo_run.py',
    }
    recorded_sources = source_manifest.get('embedded_source_sha256', {})
    for index, filename in embedded_map.items():
        prefix = f'%%writefile /kaggle/working/{filename}\n'
        check(cells[index].startswith(prefix), f'embedded notebook source cell {index} has unexpected prefix')
        if not cells[index].startswith(prefix):
            continue
        embedded = cells[index][len(prefix):]
        check(recorded_sources.get(filename) == hashlib.sha256(embedded.encode()).hexdigest(), f'embedded source hash mismatch for {filename}')
        try:
            compile(embedded, filename, 'exec')
        except SyntaxError as exc:
            ERRORS.append(f'embedded source does not compile for {filename}: {exc}')
        if filename != 'npo_lora_train.py':
            check(embedded == (ROOT / filename).read_text(), f'non-training implementation unexpectedly changed: {filename}')
        else:
            check(embedded == (HERE / filename).read_text(), 'notebook trainer differs from the fallback trainer file')

    cell12_prefix = '%%writefile /kaggle/working/npo_lora_8b_pilot_protocol.json\n'
    check(cells[12].startswith(cell12_prefix), 'embedded protocol cell prefix changed')
    if cells[12].startswith(cell12_prefix):
        check(json.loads(cells[12][len(cell12_prefix):]) == protocol, 'embedded notebook protocol changed')

    cell16, cell17 = cells[16], cells[17]
    check('def run_logged(command, log_path):' in cell16, 'streaming runner is missing')
    for required in ('subprocess.Popen(', 'stderr=subprocess.STDOUT', "print(line, end='', flush=True)", "env['PYTHONUNBUFFERED'] = '1'"):
        check(required in cell16, f'streaming runner is missing requirement: {required}')
    check('capture_output=True' not in cell16 + cell17, 'long subprocess output is still captured invisibly')
    check("run_logged(train_cmd, RUN_ROOT / 'training.stdout.log')" in cell17, 'training does not use streamed logs')
    check("run_logged(eval_cmd, RUN_ROOT / 'evaluation.stdout.log')" in cell17, 'evaluation does not use streamed logs')
    check("run_logged(verify_cmd, RUN_ROOT / 'npo_verification.stdout.log')" in cell17, 'verification does not use streamed logs')

    base_trainer = (ROOT / 'npo_lora_train.py').read_text()
    fallback_trainer = (HERE / 'npo_lora_train.py').read_text()
    progress_snippet = '''                        if global_step % 10 == 0 or global_step == planned_steps:
                            print(json.dumps({
                                "training_progress": {
                                    "global_step": global_step,
                                    "planned_steps": planned_steps,
                                    "epoch": epoch + 1,
                                    "runtime_seconds": round(time.monotonic() - started, 1),
                                    "mean_combined_loss_this_update": sum(row["combined_loss"] for row in pending_losses) / len(pending_losses),
                                    "learning_rate": scheduler.get_last_lr()[0],
                                }
                            }), flush=True)
'''
    check(fallback_trainer.count(progress_snippet) == 1, 'trainer progress instrumentation is missing or duplicated')
    check(fallback_trainer.replace(progress_snippet, '', 1) == base_trainer, 'trainer changed beyond the progress-only insertion')

    # Execute the exact notebook helper against tiny successful and failing
    # children. This tests streaming, log persistence, and exit-code handling.
    start = cell16.find('def run_logged(command, log_path):')
    end = cell16.find('\nsmoke_cmd = ', start)
    check(start >= 0 and end > start, 'could not isolate the streamed subprocess helper for fixture verification')
    if start >= 0 and end > start:
        helper_ns = {'deque': deque, 'os': os, 'subprocess': subprocess, 'Path': Path}
        exec(compile(cell16[start:end], '<run_logged_fixture>', 'exec'), helper_ns)
        run_logged = helper_ns['run_logged']
        with tempfile.TemporaryDirectory(prefix='npo8b-stream-check-') as temp:
            temp = Path(temp)
            ok_path = temp / 'ok.log'
            ok = run_logged([sys.executable, '-c', "print('progress-one'); print('progress-two')"], ok_path)
            check(ok.returncode == 0 and 'progress-one' in ok.stdout and 'progress-two' in ok.stdout, 'streamed success output/return code failed')
            check(ok_path.read_text() == 'progress-one\nprogress-two\n', 'streamed output was not fully preserved in the log')
            fail_path = temp / 'fail.log'
            fail = run_logged([sys.executable, '-c', "import sys; print('known-failure'); sys.exit(7)"], fail_path)
            check(fail.returncode == 7 and 'known-failure' in fail.stdout, 'streamed nonzero exit code was not preserved')
            check(fail_path.read_text() == 'known-failure\n', 'failed child output was not preserved')

    result = {
        'status': 'PASS' if not ERRORS else 'FAIL',
        'package': str(HERE),
        'notebook_sha256': digest(notebook_path),
        'trainer_sha256': digest(HERE / 'npo_lora_train.py'),
        'protocol_sha256': digest(ROOT / 'npo_lora_8b_pilot_protocol.json'),
        'instrumentation_only_trainer_change': not any('trainer changed beyond' in error for error in ERRORS),
        'stream_runner_fixture': 'PASS' if not any('streamed' in error for error in ERRORS) else 'FAIL',
        'errors': ERRORS,
        'model_loaded': False,
        'gpu_used': False,
        'kaggle_run_started': False,
    }
    output_path = HERE / 'streaming_package_verification.json'
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if not ERRORS else 1


if __name__ == '__main__':
    raise SystemExit(main())
