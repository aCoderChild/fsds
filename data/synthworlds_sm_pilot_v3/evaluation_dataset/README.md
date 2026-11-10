# R1 cross-lingual evaluation dataset preparation

`prompts/en.jsonl` contains the English direct intermediate-fact and final
multi-hop questions for all 129 leakage-isolated assignments. The per-language
files in `translation_requests/` contain question text only, with explicit
translation instructions. They contain no gold answers or source passages.

`labels/gold_labels.jsonl` is evaluation-only and must never be included in a
model prompt or SFT corpus. It stores the target intermediate answer, final
answer aliases, split, and chain IDs. Keep it separate from training inputs.

NLLB machine-translation drafts now exist for all 258 questions in each of
FR/DE/HI/TH under `translations/`. An automatic consistency audit flagged 73
records (FR 66, DE 5, HI 1, TH 1), mostly because protected names/placeholders
were mishandled. All drafts remain **pending semantic AI review**; none is
approved for evaluation, and no fluent human review has occurred. The user
authorized AI-only review, but that authorization does not turn draft output
or a mechanical check into semantic validation. The earlier
`translation_pilot/` files contain only three cases per non-English language
and are not the full evaluation set.

The safe AI-review input files contain only `translation_id`, the English
question, and the NLLB draft. The initial review packet directory is marked
rejected because its extra entity-name list could reveal answers; its output
was not imported. After the corrected review returns, run
`scripts/import_crosslingual_eval_translations.py` to enforce exact row/ID
coverage, name preservation from the English question, placeholder rejection,
and explicit AI-only labels before evaluation is enabled.

Regenerate the English prompts and translation requests with:

```bash
python data/synthworlds_sm_pilot_v3/scripts/prepare_crosslingual_eval_requests.py
```
