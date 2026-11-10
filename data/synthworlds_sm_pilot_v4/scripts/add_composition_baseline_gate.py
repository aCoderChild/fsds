#!/usr/bin/env python3
"""Require a forget-split competence check before composition unlearning."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "kaggle/sft_lora_pilot.ipynb"


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    setup_i = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "RUN_COMPOSITION_SFT = False" in "".join(c["source"]))
    setup = "".join(cells[setup_i]["source"])
    setup = setup.replace("RUN_COMPOSITION_EVALUATION = False", "RUN_COMPOSITION_BASELINE_EVALUATION = False\nRUN_COMPOSITION_POST_EVALUATION = False\nCOMPOSITION_BASELINE_GATE = False\nCOMPOSITION_BASELINE_COUNTS = {}")
    cells[setup_i]["source"] = setup.splitlines(keepends=True)

    baseline_code = """COMPOSITION_BASELINE_GATE = False
COMPOSITION_BASELINE_COUNTS = {}
if RUN_COMPOSITION_BASELINE_EVALUATION:
    if not EVAL_READY:
        raise RuntimeError('Evaluation translation gate is not ready.')
    sft_run_path = COMPOSITION_RUN_DIR / 'run_manifest.json'
    retain_run_path = COMPOSITION_RETAIN_DIR / 'run_manifest.json'
    if not sft_run_path.is_file() or not retain_run_path.is_file():
        raise FileNotFoundError('Composition SFT and retain-only run manifests are required before the baseline gate.')
    sft_run = json.loads(sft_run_path.read_text())
    retain_run = json.loads(retain_run_path.read_text())
    checkpoints = ['--checkpoint', 'pretrained=base', '--checkpoint', f\"sft={sft_run['adapter_dir']}\",
                   '--checkpoint', f\"retain_only={retain_run['adapter_dir']}\"]
    evaluator = PROJECT_ROOT / 'scripts' / 'evaluate_crosslingual_unlearning.py'
    subprocess.run([sys.executable, str(evaluator), '--eval-root', str(EVAL_ROOT), '--output-dir', str(COMPOSITION_EVAL_OUT),
                    '--baseline-only', *checkpoints], check=True)
    case_path = COMPOSITION_EVAL_OUT / 'case_metrics.jsonl'
    case_rows = [json.loads(line) for line in case_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_key = {(row['instance_id'], row['language']): row for row in case_rows if row['model'] == 'sft'}
    en_ids = {iid for (iid, lang), row in by_key.items() if lang == 'en' and row['split'] == 'forget'
              and row['direct_correct'] and row['multi_correct']}
    target_languages = ('fr', 'de', 'hi', 'th')
    for lang in target_languages:
        lang_ids = {iid for (iid, row_lang), row in by_key.items() if row_lang == lang and row['split'] == 'forget'
                    and row['direct_correct'] and row['multi_correct']}
        paired = en_ids & lang_ids
        groups = {by_key[(iid, lang)]['bootstrap_cluster_id'] for iid in paired}
        COMPOSITION_BASELINE_COUNTS[lang] = {'bilingual_precompetent_cases': len(paired), 'entity_groups': len(groups)}
    # This is a pilot feasibility floor, not a power claim. It prevents spending
    # unlearning GPU time when matched competence is too sparse to interpret.
    COMPOSITION_BASELINE_GATE = all(v['bilingual_precompetent_cases'] >= 15 and v['entity_groups'] >= 8
                                    for v in COMPOSITION_BASELINE_COUNTS.values())
    print('Forget-split bilingual precompetence:', COMPOSITION_BASELINE_COUNTS)
    print('Pre-unlearning pilot gate (>=15 matched cases and >=8 groups in each target language):',
          'PASS' if COMPOSITION_BASELINE_GATE else 'FAIL — stop before unlearning')
else:
    print('Composition baseline evaluation not started; unlearning gate remains closed.')
"""
    if not any("if RUN_COMPOSITION_BASELINE_EVALUATION:" in "".join(c.get("source", [])) and c["cell_type"] == "code" for c in cells):
        unlearn_i = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "COMPOSITION_UNLEARN_RUNS = {}" in "".join(c["source"]))
        cells.insert(unlearn_i, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": baseline_code.splitlines(keepends=True)})

    unlearn_i = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "COMPOSITION_UNLEARN_RUNS = {}" in "".join(c["source"]))
    unlearn = "".join(cells[unlearn_i]["source"])
    needle = "if not CAN_COMPOSITION_TRAIN:\n        raise RuntimeError('Composition unlearning requires successful runtime preflights.')\n"
    replacement = needle + "    if not COMPOSITION_BASELINE_GATE:\n        raise RuntimeError('Pre-unlearning bilingual competence gate failed. Review baseline diagnostics; do not run unlearning.')\n"
    if "if not COMPOSITION_BASELINE_GATE:" not in unlearn:
        if needle not in unlearn:
            raise RuntimeError("Could not locate the composition unlearning guard")
        unlearn = unlearn.replace(needle, replacement, 1)
        cells[unlearn_i]["source"] = unlearn.splitlines(keepends=True)

    eval_i = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and
                  ("if RUN_COMPOSITION_EVALUATION:" in "".join(c["source"]) or "if RUN_COMPOSITION_POST_EVALUATION:" in "".join(c["source"])))
    evaluation = "".join(cells[eval_i]["source"]).replace("if RUN_COMPOSITION_EVALUATION:", "if RUN_COMPOSITION_POST_EVALUATION:")
    cells[eval_i]["source"] = evaluation.splitlines(keepends=True)
    for cell in cells:
        if cell["cell_type"] == "markdown" and "### Composition follow-up interpretation" in "".join(cell.get("source", [])):
            source = "".join(cell["source"])
            source = source.replace("If forget-split precompetence remains too small, stop before interpreting English-only unlearning.", "Run baseline-only evaluation first. The notebook blocks unlearning unless every target language has at least 15 matched bilingual precompetent forget cases across at least 8 entity groups; this is a pilot feasibility floor, not a power claim.")
            cell["source"] = source.splitlines(keepends=True)

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{i}", "exec")
    print(f"Added baseline-only evaluation and competence gate to {NOTEBOOK}")


if __name__ == "__main__":
    main()
