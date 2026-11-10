# ChatGPT repair queue — diagnostic only

This queue covers the 799 prompts flagged by the frozen v5.1 translation review: FR 164, DE 179, HI 200, TH 256. It does not contain answers or labels. Source entity names and numbers are masked in the request files.

## Files

- `fr/`, `de/`, `hi/`, `th/`: 27 ChatGPT input batches, up to 32 questions each.
- `private_row_map.jsonl`: local mapping from short row IDs to source IDs and protected-name/number restoration values. **Do not upload this file.**
- `manifest.json` and `queue_validation.json`: source hashes and queue checks.
- `responses/`: place returned batch JSONL files here, preserving language and batch filename. For example, save the output for `fr/batch_01.txt` as `responses/fr/batch_01.jsonl`.
- `refresh_candidate_manifest.py`: refreshes `responses/candidate_draft_manifest.json` with source/candidate hashes and checks against frozen queue hashes, row count, unique IDs/order, schema, nonempty strings, and masked-marker preservation. These checks do not assess translation quality or approve training data.
- `parsed_repair_candidates.jsonl`: generated only by the parser; remains quarantined and is not an SFT corpus.

## Translation and parsing

1. The 32-row ChatGPT calibration and reviewer cross-check are complete: Qwen2.5-14B passed 23/32; Mistral-7B passed all nine Qwen-flagged examples, producing disagreement on all nine. Mistral also passed both review fields on 2/15 deliberately mismatched pairs in an earlier negative control. These are diagnostics, not gold judgments; the translation route is not calibrated for bulk release.
2. If generating drafts, upload one batch text file at a time to ChatGPT. Request exactly one JSONL row per question with `row_id` and `translation`; do not upload the private map or any answer data. Treat every response as a quarantined candidate, not as an approved translation.
3. Save each response unchanged under `responses/<lang>/batch_<nn>.jsonl`.
4. Run from the repository root:

   ```bash
   python -B experiments/synthworlds_r1/training_release_draft/scripts/parse_chatgpt_flagged_repair_responses.py
   ```

The parser requires exact row-ID coverage, restores markers only from the local map, checks names/numbers/script, and writes candidate/report files in this directory. Missing or malformed batches are reported or rejected. Every candidate still needs complete independent semantic and fluency review using a calibrated policy. The parser does not approve translations, and nothing in this directory authorizes training.
