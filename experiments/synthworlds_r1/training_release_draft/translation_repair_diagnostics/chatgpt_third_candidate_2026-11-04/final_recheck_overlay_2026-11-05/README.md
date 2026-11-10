# 187-row third-candidate recheck overlay

This is a diagnostic-only combination of the frozen 187-row Qwen/NLLB screen and 18 bounded, source-bound retries. It does not change the canonical translations or admit examples to the training corpus.

## Result

| Group | Pass | Flag | Blocked | Rows |
|---|---:|---:|---:|---:|
| French | 8 | 13 | 0 | 21 |
| German | 3 | 16 | 0 | 19 |
| Hindi | 14 | 48 | 0 | 62 |
| Thai | 11 | 74 | 0 | 85 |
| Atomic | 26 | 74 | 0 | 100 |
| Composition | 10 | 77 | 0 | 87 |
| **Total** | **36** | **151** | **0** | **187** |

The 18 repaired decisions consist of 7 passes and 11 flags. “Pass” and “flag” refer only to AI screening, not human adjudication.

## Retry history

1. The original screen had 15 rows blocked because NLLB back-translation dropped opaque entity markers and 3 rows blocked by malformed Qwen outputs.
2. A bounded alias-safe run restored and hash-checked all 15 back-translations, but a local `NameError` interrupted its Qwen calls. The three format-retry decisions completed.
3. A Qwen-only follow-up judged the saved 15 back-translations without rerunning NLLB. Fourteen decisions were valid; one had the wrong key (`R0394` for `R0349`) and failed strict validation.
4. A one-row schema-constrained Qwen retry returned a valid `R0349` decision.
5. The final overlay was built from the unchanged 169 valid original decisions plus the 18 verified retries.

## Reproduce and verify

From the repository root:

```bash
python experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_third_candidate_2026-11-04/build_final_recheck_overlay.py
python experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_third_candidate_2026-11-04/verify_final_recheck_overlay.py
```

The verifier checks frozen input and output hashes, complete and unique row coverage, source and translation hashes, retry lineage, the exact restored marker multisets, the constrained `R0349` decision, and aggregate counts. `build_receipt.json` records the overlay SHA-256.

## Limits

- AI-only; no fluent-human translation review was performed.
- The screen has known reviewer disagreement and hard-negative failures. Its labels are not ground truth.
- No answer values or split labels were sent to the reviewers.
- `release_approved` and `training_eligible` are false. Do not use this overlay to train or to claim corpus release readiness.
