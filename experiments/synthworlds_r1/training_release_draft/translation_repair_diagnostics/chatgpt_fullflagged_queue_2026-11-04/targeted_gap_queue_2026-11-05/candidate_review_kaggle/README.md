# Targeted gap candidate review (Kaggle)

This private Kaggle kernel ran the answer-blind NLLB back-translation plus Qwen2.5-14B semantic/fluency screen on 187 masked assistant-generated candidates. The review input is embedded in `review.py`; it contains no answer labels, split assignments, source case IDs, or private entity map.

## Result

| Language | Screened | Joint semantic + fluency pass | Blocked |
| --- | ---: | ---: | ---: |
| French | 21 | 9 | 2 |
| German | 19 | 7 | 3 |
| Hindi | 62 | 8 | 10 |
| Thai | 85 | 12 | 2 |
| **Total** | **187** | **36** | **17** |

The verifier found 170 usable judge decisions and 17 blocked rows. Fifteen were blocked because NLLB altered opaque-marker multiplicities; two had invalid judge output. Of the 170 decisions, 36 passed both semantic and fluency checks. This is one AI review pass, not independent linguistic certification. The run also logged a Transformers warning about sliding-window attention under the selected SDPA backend, so the AI decisions are provisional.

The result and kernel log are preserved under `remote_output/`; `review_verification.json` records exact coverage and hashes. The output is diagnostic and remains quarantined: `release_approved=false`, `training_eligible=false`. A passing judgment is not translation approval or SFT eligibility. The generating model revision is unknown, and existing calibration found reviewer errors.
