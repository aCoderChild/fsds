# NLLB candidate set for unresolved translation rows

This diagnostic produces fresh translations for the 148 question-language pairs still unresolved after the 187-row target review and its 17-row retry. It selects no previously AI-passed pair. The remote input includes only an anonymous row key, target language, row kind, English question, and source hash. It excludes answers, entity values, split labels, and prior candidate text.

NLLB `facebook/nllb-200-distilled-600M` at revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d` generates a target-language candidate and back-translation using collision-checked `Alice`/`Bob` aliases. Qwen2.5-14B-Instruct at revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` screens the source/candidate/back-translation pair. The NLLB candidate and back-translation come from the same model, so the round-trip is only supporting evidence. All review is AI-only.

The result is quarantined. It does not alter the canonical translation records, candidate merge, release manifest, or SFT corpus; it cannot approve training data.

## Run

```bash
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05/prepare_candidates.py
kaggle kernels push -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05 --accelerator NvidiaTeslaT4
```

After completion, pull the kernel output and its exact source, then verify:

```bash
kaggle kernels output bunnoob2005/sw-gap187-nllb-148-candidate-review-2026-11-05 -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05/remote_output -o
kaggle kernels pull -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05/remote_source bunnoob2005/sw-gap187-nllb-148-candidate-review-2026-11-05/1 -m
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05/verify_candidate_review.py --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05/remote_output --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_candidates_2026-11-05/remote_source/sw-gap187-nllb-148-candidate-review-2026-11-05.py
```

The verifier checks the unresolved-row selection, exact source/script hashes, model revisions, GPU, placeholder multiplicities, decision schema, and answer/split blindness. A passing local receipt remains an AI-only diagnostic, not translation certification.
