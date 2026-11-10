# SFT translation batch prompt

Paste this prompt into ChatGPT, then paste the contents of exactly one `sft_translation_requests/<language>/batch_NNN.jsonl` file where indicated.

```text
Translate each JSONL record's English question and answer into the language named by `translate_to` (fr=French, de=German, hi=Hindi, th=Thai).

Return exactly one valid JSON object per input line, in the same order. Preserve every input field and value exactly except fill `question_translation` and `answer_translation`. Do not add Markdown fences, commentary, or extra lines. Do not change IDs, split, English source text, protected names, or any other metadata.

Translation requirements:
- Translate the question and answer naturally and accurately.
- Preserve every string in `protected_entity_names` character-for-character in either the question or answer where it appears in the source meaning. Do not translate or transliterate these names.
- Preserve numbers, dates, negation, qualifiers, and the factual relation. Do not infer missing information.
- Keep proper fictional entity names unchanged. Translate answer labels such as “(year)” naturally when appropriate, while preserving the number.
- Leave `translation_status` as `PENDING_FLUENT_REVIEW` and `review_status` as `PENDING`. You are producing a draft, not certifying it.

Input JSONL:
<PASTE ONE BATCH HERE>
```
