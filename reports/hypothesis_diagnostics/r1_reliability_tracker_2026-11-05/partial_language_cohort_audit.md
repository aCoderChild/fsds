# Partial-language cohort feasibility audit — 2026-11-05

**Status: diagnostic only; no training release or approval.** The audit rechecks all current translation-review hashes and asks whether relaxing the all-four-target-languages-per-case requirement would make the atomic forget cohort numerically large enough. It does not relax translation pass criteria, graph isolation, split labels, or the baseline competence gate.

## Atomic two-hop case coverage

| Language | Forget cases / groups | Retain cases / groups | Meets frozen forget count floor (15 cases, 8 groups)? |
|---|---:|---:|---|
| FR | 46 / 30 | 14 / 12 | Yes |
| DE | 49 / 31 | 20 / 11 | Yes |
| HI | 45 / 28 | 8 / 7 | Yes |
| TH | 24 / 15 | 14 / 9 | Yes |

The intersection with both atomic hops screened as passing in **all four** target languages is only **5 forget cases / 3 groups** and **3 retain cases / 3 groups**.

## Disjoint composition practice

| Language | Complete four-question chains | Chains with 3/4 questions passing |
|---|---:|---:|
| FR | 15 / 44 | 13 / 44 |
| DE | 9 / 44 | 12 / 44 |
| HI | 3 / 44 | 14 / 44 |
| TH | 0 / 44 | 8 / 44 |

A partial atomic release could satisfy the *sample-count* floor before model evaluation, but it does not solve multilingual composition training. With the three-seed model competence gate still untested, this is not evidence the experiment will pass. Hindi/Thai composition translations need additional validated coverage, and the AI-only reviewer calibration issue remains unresolved.

## Reproduction and limits

Run `python -B partial_language_cohort_audit.py`. The script imports the frozen release checker, verifies the hash-bound 488-row review for each language, reads the leakage-group manifest, and emits the JSON receipt alongside this report. This is not a population-level translation-quality estimate and does not authorize training.

No training corpus was written.
