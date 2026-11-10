# Alpha=10 independent recheck

**Result:** `FAIL_DIAGNOSTIC_GATES`. Training and evaluation completed (250 optimizer steps; 800 development rows). The saved results passed independent checks for training receipt and adapter hashes, source data hashes, row/block assignment, and all five paired 95% bootstrap calculations.

| Gate | Change (95% profile-block CI) | Outcome |
|---|---:|---|
| forget10.truth_ratio_score | +5.22 pp [+3.84, +6.53] | Pass |
| forget10.answer_probability | -32.45 pp [-35.04, -29.64] | Pass |
| retain90.answer_probability | -24.60 pp [-26.02, -23.30] | Fail |
| retain90.rougeL_recall | -25.57 pp [-28.37, -22.66] | Fail |
| holdout10.answer_probability | +5.36 pp [+4.84, +5.82] | Pass |

Alpha=10 shows a forgetting signal, but retain answer probability and retain ROUGE-L fall beyond the preregistered 5 pp noninferiority margin. It is therefore rejected for the final cohort. Alpha=20 remains the next preregistered development candidate. This is a one-seed exploratory profile-cohort result, not a confirmatory or standard full-split TOFU result.
