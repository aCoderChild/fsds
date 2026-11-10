# Current Kaggle run and experiment-design audit

Checked 2026-11-04 at 06:27 UTC from the local source tree and Kaggle API.

## Live Kaggle state

The kernel bunnoob2005/experiments reports COMPLETE, with lastRunTime 2026-11-04 06:22:10. Its pulled metadata attaches only bunnoob2005/synthworlds-sm-sft-pilot-v3 and specifies a Tesla T4. The pulled notebook is the legacy 1B workflow, not the current v5.1 three-seed R1 workflow. The preserved source, metadata, and output log are in [the Kaggle evidence folder](kaggle_legacy_experiments_2026-11-04/). Their SHA-256 values are source 3c98b2773c56f63e83d55f1d9b009ac5c417f13d4a003aa0805fbe85c18cf85d, metadata 93f4c9b1dd22ec4f5ccd99f73b93ab713085ff59e0a7466184247bf57cc08ca4, and output log 03ddd345f4a48da7982a347a34fc4770ea3042722f3b78c115dcff081dfcf310.

The latest output log SHA-256 is 03ddd345f4a48da7982a347a34fc4770ea3042722f3b78c115dcff081dfcf310. It reports the old v3 project root, a Llama 3.2 1B preflight, and a Kaggle HF_TOKEN retrieval failure. The combined training gate is BLOCKED; training_started is false; all SFT, unlearning, and evaluation switches are off. No model training or hypothesis evaluation occurred in this run. The API run's COMPLETE status means the notebook process finished, not that a research run completed.

The separate Llama 3.1 8B smoke attempt remains blocked before model loading because this Hugging Face account lacks access. It downloaded no 8B weights and ran no training step; see kaggle_8b_runtime_smoke_2026-11-04.md.

## Fresh local verification

| Check | Result |
|---|---|
| Offline R1 pipeline fixtures | 29/29 pass |
| Source candidate and split reconstruction | PASS_DRAFT_SOURCE_AND_SPLIT_VALIDATION; explicitly not approved for training/evaluation |
| Prepared-source integrity | PASS_PREPARED_SOURCE_ALIGNMENT_ONLY |
| Source-verified training assembler | Stops at 5/117 forget and 3/39 retain eligible cases, below 117/39; it writes no approved corpus or release manifest |
| Default 8B SFT preflight | BLOCKED; corpus/manifest absent, and local PEFT, Accelerate, bitsandbytes, and HF token are unavailable |

These results verify software guards and data lineage, not model behavior. No current v5.1 SFT, baseline, or unlearning run exists.

## Design interpretation to preserve

The current declared primary contrast is target-language multi-hop change minus English multi-hop change on the same pre-unlearning competent cases. It estimates whether the effect differs from English. A zero contrast does not mean zero non-English effect if both languages lose accuracy by the same amount. Report the absolute target-language pre/post change and the forget-versus-retain change alongside that contrast; do not use a null target-minus-English contrast to claim no cross-lingual effect.

The evaluator already emits per-language pre/post accuracy, per-language change, paired group-bootstrap intervals, direct forgetting, retain-set change, and target-minus-English differences. The 29-fixture suite exercises the three-seed scoring and tamper checks. Before the main run, keep the analysis table explicit about which of these estimates answers the literal question “does non-English capability change?” and which answers “does it change differently from English?”

“Deep unlearning” is not yet a pass/fail manipulation criterion. The analysis measures direct forgetting, but the protocol does not require a minimum English direct-recall reduction before calling an intervention deep. Report the direct-forgetting estimate and avoid that label unless the target fact is measurably suppressed. The frozen v5.1 evaluation cohort is two-hop only; it cannot estimate a hop-depth effect.

The question remains plausible: prior multilingual-unlearning work reports language-dependent transfer, while MQuAKE and cross-lingual multi-hop editing study downstream propagation under editing rather than deletion. Shortcut-aware evaluation and controls for never-learned knowledge remain essential. See the primary sources in R1.tex, including Xiang et al. (2026), Waseem and Fawaz (2026), MQuAKE (2023), SOCRATES (2025), and CroLin-MQuAKE (2024).
