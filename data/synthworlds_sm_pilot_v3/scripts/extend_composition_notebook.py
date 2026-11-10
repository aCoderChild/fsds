#!/usr/bin/env python3
"""Add a guarded retained-chain composition-SFT follow-up to the Kaggle notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle/sft_lora_pilot.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if any("Conditional follow-up: retained-chain composition SFT" in "".join(cell.get("source", [])) for cell in cells):
        print("Composition follow-up is already present")
        return
    cells.extend([
        md("""## Conditional follow-up: retained-chain composition SFT

The completed atomic-fact pilot did not establish a usable multi-hop baseline. This follow-up follows R1's conditional plan: add multi-hop final-answer practice from the **retain split only**, in all five languages. Forget and holdout chains and labels stay out of the composition examples. Retain questions intentionally overlap retain evaluation, so only forget-split performance is used as unseen-chain generalization. The retained prompts were reviewed AI-only; no fluent human review was performed.

Run this notebook in the Kaggle Notebook Editor with the `HF_TOKEN` secret attached. Do not push an API-launched kernel for model training: Kaggle's CLI does not configure Notebook Secrets.
"""),
        code("""# Composition stage controls. Keep each stage disabled until its displayed preflight passes.
RUN_COMPOSITION_SFT = False
RUN_COMPOSITION_RETAIN_ONLY = False
RUN_COMPOSITION_UNLEARNING = False
RUN_COMPOSITION_EVALUATION = False
COMPOSITION_CORPUS = PROJECT_ROOT / 'sft_training' / 'composition_corpus.jsonl'
COMPOSITION_MANIFEST = PROJECT_ROOT / 'sft_training' / 'composition_manifest.json'
COMPOSITION_RUN_DIR = Path('/kaggle/working/r1_composition_sft')
COMPOSITION_RETAIN_DIR = Path('/kaggle/working/r1_composition_retain_only')
COMPOSITION_UNLEARN_ROOT = Path('/kaggle/working/r1_composition_unlearning')
COMPOSITION_EVAL_OUT = Path('/kaggle/working/r1_composition_diagnostics')
COMPOSITION_UNLEARN_INPUT = Path('/kaggle/working/r1_composition_first_hop_input')

if not COMPOSITION_CORPUS.is_file() or not COMPOSITION_MANIFEST.is_file():
    raise FileNotFoundError('The attached R1 dataset lacks the retained-chain composition corpus.')
composition_preflight_run = subprocess.run(
    [sys.executable, str(runner), '--corpus', str(COMPOSITION_CORPUS), '--manifest', str(COMPOSITION_MANIFEST),
     '--output-dir', str(COMPOSITION_RUN_DIR)], check=True, text=True, capture_output=True)
COMPOSITION_PREFLIGHT = json.loads(composition_preflight_run.stdout)
CAN_COMPOSITION_TRAIN = (COMPOSITION_PREFLIGHT.get('status') == 'READY_TO_TRAIN'
                         and COMPOSITION_PREFLIGHT.get('training_packages_ready') is True
                         and HF_ACCESS_OK and torch.cuda.is_available())
print(composition_preflight_run.stdout)
print('Composition GPU + secret + package gate:', 'READY' if CAN_COMPOSITION_TRAIN else 'BLOCKED')
"""),
        code("""if RUN_COMPOSITION_SFT:
    if not CAN_COMPOSITION_TRAIN:
        raise RuntimeError('Composition SFT requires successful GPU, secret/model-access, and corpus preflights.')
    subprocess.run([sys.executable, str(runner), '--corpus', str(COMPOSITION_CORPUS),
                    '--manifest', str(COMPOSITION_MANIFEST), '--output-dir', str(COMPOSITION_RUN_DIR), '--train'], check=True)
    comp_run = json.loads((COMPOSITION_RUN_DIR/'run_manifest.json').read_text())
    if comp_run.get('status') != 'TRAINING_COMPLETE' or comp_run.get('training_variant') != 'composition_sft':
        raise RuntimeError(f'Composition SFT did not complete as expected: {comp_run}')
    print('Composition SFT complete:', comp_run['rows'], 'rows; seed', comp_run['seed'])
else:
    print('Composition SFT not started; RUN_COMPOSITION_SFT=False.')
"""),
        code("""if RUN_COMPOSITION_RETAIN_ONLY:
    if not CAN_COMPOSITION_TRAIN:
        raise RuntimeError('Composition retain-only SFT requires successful runtime preflights.')
    subprocess.run([sys.executable, str(runner), '--corpus', str(COMPOSITION_CORPUS),
                    '--manifest', str(COMPOSITION_MANIFEST), '--output-dir', str(COMPOSITION_RETAIN_DIR),
                    '--include-splits', 'retain', '--train'], check=True)
    retain_run = json.loads((COMPOSITION_RETAIN_DIR/'run_manifest.json').read_text())
    if retain_run.get('status') != 'TRAINING_COMPLETE' or retain_run.get('training_variant') != 'composition_retain_only':
        raise RuntimeError(f'Composition retain-only SFT did not complete as expected: {retain_run}')
    print('Composition retain-only reference complete:', retain_run['rows'], 'rows')
else:
    print('Composition retain-only training not started.')
"""),
        code("""COMPOSITION_UNLEARN_RUNS = {}\nif RUN_COMPOSITION_UNLEARNING:\n    if not CAN_COMPOSITION_TRAIN:\n        raise RuntimeError('Composition unlearning requires successful runtime preflights.')\n    if not COMPOSITION_BASELINE_GATE:\n        raise RuntimeError('Pre-unlearning bilingual competence gate failed. Review baseline diagnostics; do not run unlearning.')\n    sft_run_path = COMPOSITION_RUN_DIR / 'run_manifest.json'\n    if not sft_run_path.is_file():\n        raise FileNotFoundError('Complete composition SFT before unlearning.')\n    unlearn_script = PROJECT_ROOT / 'scripts' / 'unlearn_lora.py'\n    # unlearn_lora.py selects English r1 rows from the full, checksummed corpus.\n    # It ignores r2/composition practice and excludes holdout before optimization.\n    for method in METHODS:\n        out = COMPOSITION_UNLEARN_ROOT / method\n        check = subprocess.run([sys.executable, str(unlearn_script), '--corpus', str(COMPOSITION_CORPUS),\n                                '--corpus-manifest', str(COMPOSITION_MANIFEST), '--sft-run-manifest', str(sft_run_path),\n                                '--output-dir', str(out), '--method', method],\n                               check=True, text=True, capture_output=True)\n        report = json.loads(check.stdout)\n        print(method.upper(), 'preflight:', report.get('status'), report.get('errors', []))\n        if report.get('status') != 'READY_TO_UNLEARN':\n            raise RuntimeError(f'{method.upper()} composition unlearning preflight failed: {report}')\n    for method in METHODS:\n        out = COMPOSITION_UNLEARN_ROOT / method\n        subprocess.run([sys.executable, str(unlearn_script), '--corpus', str(COMPOSITION_CORPUS),\n                        '--corpus-manifest', str(COMPOSITION_MANIFEST), '--sft-run-manifest', str(sft_run_path),\n                        '--output-dir', str(out), '--method', method, '--train'], check=True)\n        run = json.loads((out / 'run_manifest.json').read_text())\n        if (run.get('status') != 'UNLEARNING_COMPLETE' or run.get('holdout_rows_used') != 0\n                or run.get('english_forget_rows') != 76 or run.get('english_retain_rows') != 26\n                or run.get('source_corpus_sha256') != hashlib.sha256(COMPOSITION_CORPUS.read_bytes()).hexdigest()):\n            raise RuntimeError(f'{method.upper()} composition unlearning output failed validation: {run}')\n        COMPOSITION_UNLEARN_RUNS[method] = out / 'run_manifest.json'\n        print(method.upper(), 'complete; English r1 rows:', run['english_forget_rows'], 'forget +',\n              run['english_retain_rows'], 'retain; holdout rows used:', run['holdout_rows_used'])\nelse:\n    print('Composition unlearning not started.')"""),
        code("""if RUN_COMPOSITION_EVALUATION:
    if not EVAL_READY:
        raise RuntimeError('Evaluation translation gate is not ready.')
    sft_run_path = COMPOSITION_RUN_DIR / 'run_manifest.json'
    retain_run_path = COMPOSITION_RETAIN_DIR / 'run_manifest.json'
    if not sft_run_path.is_file() or not retain_run_path.is_file():
        raise FileNotFoundError('Composition SFT and retain-only run manifests are required.')
    sft_run = json.loads(sft_run_path.read_text())
    retain_run = json.loads(retain_run_path.read_text())
    checkpoints = ['--checkpoint', 'pretrained=base', '--checkpoint', f"sft={sft_run['adapter_dir']}",
                   '--checkpoint', f"retain_only={retain_run['adapter_dir']}"]
    for method in METHODS:
        run_path = COMPOSITION_UNLEARN_ROOT / method / 'run_manifest.json'
        if not run_path.is_file():
            raise FileNotFoundError(f'Missing composition unlearning manifest for {method}.')
        run = json.loads(run_path.read_text())
        checkpoints += ['--checkpoint', f"{method.upper()}={run['adapter_dir']}"]
    evaluator = PROJECT_ROOT / 'scripts' / 'evaluate_crosslingual_unlearning.py'
    subprocess.run([sys.executable, str(evaluator), '--eval-root', str(EVAL_ROOT), '--output-dir', str(COMPOSITION_EVAL_OUT),
                    '--contrast-model', 'GA', '--contrast-model', 'GD', '--contrast-model', 'NPO', *checkpoints], check=True)
    print('Composition diagnostic report:', COMPOSITION_EVAL_OUT/'diagnostic_report.md')
    print('Interpret forget split as unseen-chain generalization; retain examples are intentionally seen during training.')
else:
    print('Composition evaluation not started.')
"""),
        md("""### Composition follow-up interpretation

This stage is a separate pilot checkpoint, not a replacement for the completed atomic-fact pilot. Compare the composition-SFT multi-hop baseline against the earlier atomic-SFT result. If forget-split precompetence remains too small, stop before interpreting English-only unlearning. Retain scores are training-overlap controls; do not call them held-out generalization. The experiment remains a small 1B feasibility run with AI-only translations and no human speaker validation.
"""),
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK} with guarded composition follow-up")


if __name__ == "__main__":
    main()
