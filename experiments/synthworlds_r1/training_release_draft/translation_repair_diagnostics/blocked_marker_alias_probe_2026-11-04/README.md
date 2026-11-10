# Marker-safe re-screen of 31 blocked rows

This directory contains a diagnostic follow-up for the exact 31 rows that remained blocked in the source-bound 799-row repair-review merge. It does not modify the frozen merge, authoritative translations, release manifest, or SFT corpus.

## Results

| Screen | Pass | Flag | Blocked |
|---|---:|---:|---:|
| Alias-restored, back-translation assisted | 12 | 19 | 0 |
| Matched direct source-to-target | 30 | 1 | 0 |

The alias probe covered 28 rows and preserved the exact expected placeholder multiset on 28/28. The other three rows used their previously hash-bound back-translations. The first constrained Qwen run truncated three long rationales; a three-row retry with a shorter reason limit produced one pass and two flags.

The matched direct screen used the same Qwen2.5-14B model revision and NF4 setup on the same source/translation pairs, but omitted back-translations and earlier judgments. The paired pass/flag matrix was:

| Back-translation result | Direct pass | Direct flag | Total |
|---|---:|---:|---:|
| Pass | 12 | 0 | 12 |
| Flag | 18 | 1 | 19 |

Overall screen agreement was 13/31; semantic agreement was 12/31 and fluency agreement was 30/31. The back-translation-assisted review mode flags many more rows in this selected sample. Because both the available evidence and reviewer prompt changed, the comparison cannot isolate the back-translation text alone. It does not establish which judgments are right: both screens use the same AI reviewer, there are no fluent-human gold labels, and direct passes are not translation certification.

## Artifacts

- `remote_output/verification.json`: alias placeholder-integrity receipt.
- `qwen_remote_output/constrained_v2/verification.json`: 31-row back-translation screen receipt.
- `qwen_retry3_2026-11-04/remote_output/verification.json`: retry receipt for the three truncated records.
- `final_diagnostic_overlay_2026-11-04/rechecked_blocked_31_rows.jsonl`: complete per-row diagnostic judgments, source/target/back-translation text, hashes, and review method.
- `final_diagnostic_overlay_2026-11-04/verification.json`: overlay coverage and counts.
- `direct_same31_2026-11-04/remote_output/paired_diagnostic.json`: matched per-row direct-versus-round-trip comparison.
- `direct_same31_2026-11-04/remote_output/verification.json`: direct-screen hash/schema receipt and agreement summary.
- `kaggle_status.json`: authenticated Kaggle CLI status recheck; all four diagnostic kernels completed.
- Each `remote_source/` folder contains the Kaggle-pulled code, which the verifier checks against the submitted local script.

All Kaggle jobs are complete. The reviewed inputs exclude answers and split labels. All decisions are AI-only and release/training flags are false.

## Reproduce local verification

Run from the repository root:

```bash
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/verify_alias_probe.py \
  --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/remote_output \
  --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/remote_source/synthworlds-r1-placeholder-alias-probe-2026-11-04.py

python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/verify_qwen_recheck.py \
  --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/qwen_remote_output/constrained_v2 \
  --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/qwen_remote_source/constrained_v2/synthworlds-r1-blocked-qwen-recheck-2026-11-04.py

python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/qwen_retry3_2026-11-04/verify_retry.py \
  --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/qwen_retry3_2026-11-04/remote_output \
  --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/qwen_retry3_2026-11-04/remote_source/synthworlds-r1-qwen-retry-truncated-3-2026-11-04.py

python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/final_diagnostic_overlay_2026-11-04/build_overlay.py

python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/direct_same31_2026-11-04/verify_direct.py \
  --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/direct_same31_2026-11-04/remote_output \
  --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/direct_same31_2026-11-04/remote_source/synthworlds-r1-direct-same31-review-2026-11-04.py
```

The global training-release gate remains unchanged: 5/117 forget cases, 3/39 retain cases, and 0/44 required composition chains. These targeted diagnostics do not create a corpus or start SFT/unlearning.
