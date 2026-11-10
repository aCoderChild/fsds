# Review guide for SFT translation candidates

The four files under `sft_translation_candidates/` contain 205 one-hop SFT examples per language (French, German, Hindi, Thai). They were drafted by `facebook/nllb-200-distilled-600M`. The model is licensed CC-BY-NC-4.0, so use it only under its noncommercial terms.

For each JSONL row, compare `question_en` with `question` and `answer_en` with `answer_translation`. Check that the translation is grammatical and natural for the requested language, preserves the exact fact and answer, and does not add or omit qualifiers. Preserve `protected_entity_names` character-for-character. Check dates and numbers. Pay special attention to any nonempty `protected_text_repairs`: the draft generator restored a protected string in parentheses after the translation model omitted or duplicated it, and a fluent reviewer should rewrite that question or answer naturally while retaining the exact name or number.

## AI-only pilot mode

The user explicitly authorized AI-only review for this feasibility pilot. The repository records that decision in `translation_review_policy.json`. The review back-translated every question and answer, checked entity names and numbers, checked target scripts for Hindi and Thai, inspected low lexical-overlap cases, and applied documented corrections. It is recorded as `APPROVED_BY_AI_REVIEW`, with `reviewer_type: AI` and `human_review_status: NOT_PERFORMED`. The corpus manifest and training run preserve this limitation. AI review is not a human fluency attestation; this pilot should not be described as using human-validated translations.

For a future human-reviewed release, a fluent reviewer should compare `question_en` with `question` and `answer_en` with `answer_translation`, verify grammar, naturalness, semantic equivalence, names, numbers, and qualifiers, then use `APPROVED_BY_FLUENT_REVIEW` and record their name and date. Leave uncertain records pending and record corrections. Do not modify IDs, source questions/answers, split labels, provenance arrays, or protected-name lists. The corpus builder checks provenance, approvals, review mode, and holdout exclusion.

Human-review approvals remain human attestations. Machine translation, back-translation, or automatic checks do not qualify as fluent-speaker review.
