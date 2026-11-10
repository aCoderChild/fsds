# R1 protocol-to-code audit — 2026-11-05

## Status

**No v5.1 SFT, baseline, or unlearning experiment has run.** The source release assembler fails its data gate, and the Kaggle jobs observed in this audit were completed smoke/review or legacy pilot jobs. The Llama 3.1 8B runtime receipt is a synthetic-data compatibility check only.

## Protocol-to-implementation checks

| R1 requirement | Implementation evidence | Audit result |
|---|---|---|
| Five-language SFT; Llama 3.1 8B main profile | `experiments/synthworlds_r1/kaggle/r1_multiseed_experiment.ipynb` selects `MAIN_PROFILE`; `scripts/train_sft_lora.py` enforces five-language balanced rows and a hash-bound source proof | Implemented; not run because no releasable corpus exists |
| Three full-SFT seeds and three retain-only references | Main notebook sets seeds 42, 43, 44 and has distinct full-SFT and retain-only stages | Implemented; not run |
| English-only first-hop unlearning | `scripts/unlearn_lora.py:first_hop_unlearning_rows` selects English `r1` forget/retain rows; preflight binds forget IDs to the three-seed baseline gate | Implemented; not run |
| GA, GD, and NPO | `scripts/unlearn_lora.py` uses forget-set ascent; GD adds retain CE; NPO uses the shared sequence-log-probability objective against a frozen SFT reference | Implemented; offline objective fixtures pass; not run on research data |
| Competence gate before unlearning | `scripts/evaluate_pilot_three_probe.py:baseline_gate` requires at least 15 matched cases and 8 leakage groups per target language, correct on all three probes in all three full-SFT seeds | Fail-closed; this is a feasibility threshold, not a power calculation |
| Paired target-language change and DID | `scripts/evaluate_pilot_three_probe.py:effects` pairs pre/post outcomes on the same bilingual precompetent cases and resamples leakage groups; the notebook requests 100,000 bootstrap replicates, the CLI rejects fewer, and the notebook reports seed variation separately | Code matches the estimands in R1; the intervals cannot be produced until the gate passes and runs exist |
| Question and reasoning coverage | Frozen evaluation has 1,200 prompts = 80 cases × 5 languages × 3 probes. There is one fixed wording per probe/case/language and one two-hop depth; the positive and negative direct probes differ by candidate answer | No paraphrase robustness or depth comparison. R1 was corrected to state this limitation |
| Answer language | R1 evaluation labels retain canonical entity names across languages; direct probes return YES/NO | Measures multilingual-input comprehension with canonical entity output, not fluent final-answer generation in French, German, Hindi, or Thai |

## Current numbers

| Evidence | Result | What it supports |
|---|---:|---|
| Source-bound SFT release | 5/117 forget, 3/39 retain, 0/44 composition chains eligible; assembler blocked | No approved five-language SFT corpus; no main experiment start |
| Multilingual candidate diagnostics | 36/187 rows passed the targeted AI-only screen (19.3%); 154/187 labels disagreed between Qwen and Mistral | Candidate translations remain quarantined. These purposive AI-only screens are not population error-rate estimates or human validation |
| Controlled mismatch probes | Mistral passed 2/15; Qwen passed 1/16 | Both reviewers failed these small obvious-mismatch checks; neither can certify candidate translations |
| Offline pipeline fixtures | 36/36 passed | Expected software behavior on synthetic fixtures; not evidence of corpus quality or model effects |
| Llama 3.1 8B smoke | One synthetic LoRA step, loss 0.02476; one synthetic NPO update, loss 13.86294; no research rows loaded | Runtime/stack compatibility only |
| Historical Llama 3.2 1B pilot | One seed, 7,740 predictions rescored; all 7 derived artifacts reproduced byte-for-byte; exact post-hoc sensitivity gives 12/12 primary and 12/12 secondary familywise intervals that include or touch zero | Reproducible exploratory output, not a confirmatory answer; only 5–7 groups per contrast |

## Reliability judgment

The **analysis and training controls are substantially fail-closed**: the data source is reconstructed and hash-bound; the main notebook requires the exact 8B runtime receipt; SFT checks five-language coverage and source integrity; unlearning is English-only, excludes holdout, and requires a passing baseline gate; scoring rejects incomplete, duplicate, or mismatched predictions; paired effects are clustered by leakage group. The 36/36 synthetic tests and byte-identical pilot rescore provide useful software evidence.

The **experiment is not currently runnable on an approved corpus**. The AI-only translation pipeline has inadequate coverage and fails its small mismatch probes. Relaxing the release gate would create data of unverified quality and would not make the resulting statistics reliable. The previous 1B pilot is the only hypothesis-adjacent result, and it is low-powered, one-seed, and below the predeclared competence floor.

Before treating a later run as confirmatory:

1. Resolve translations through a validation process that passes source alignment, semantic/fluency screening, and independent human or calibrated reviewer checks; rerun the frozen source-bound assembler.
2. Keep the current 15-case/8-group cutoff labeled as a feasibility gate; preregister a power or precision target separately.
3. Decide whether the intended claim concerns multilingual comprehension with canonical entity output (the current design) or target-language answer generation. These are different outcomes.
4. If paraphrase or hop-depth robustness is part of the claim, add those prompts and translations before freezing the evaluation set; the current cohort does not cover them.

**Reproducibility anchors:** `verification.json` in this directory records current hashes and counts; the frozen R1 evaluation is `experiments/synthworlds_r1/training_release_draft/evaluation/prompts.jsonl` and `labels.jsonl`; the main Kaggle runner is `experiments/synthworlds_r1/kaggle/r1_multiseed_experiment.ipynb`.
