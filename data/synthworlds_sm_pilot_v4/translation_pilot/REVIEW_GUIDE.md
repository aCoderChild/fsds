# Fluent review guide

The four language folders each contain a `review_queue.jsonl` with 21 entries. Review the language you know best; review every question, passage, and verified chain fact in that language.

For each record, fill `reviewer_name` and `review_date`, then set `review_status` to `APPROVE`, `CORRECT`, or `REJECT`. If you choose `CORRECT`, write the corrected text in `correction` and explain any semantic change in `reviewer_notes`.

Check that:

- The translated question asks for the same relation and answer as its English source.
- Q1 identifies the intended first-hop answer; Q2 preserves the bridge entity and asks the intended second-hop relation.
- Entity names, dates, numbers, qualifiers, and fact direction remain correct.
- The prose is natural in the target language.
- Document translations do not introduce or omit claims. These passages are audit-only and must not be model inputs.

The `expected_answer_reference_only` field helps review question meaning. Never include it in a model prompt. Even when all entries are approved, this three-case sample validates the translation process only; it is too small to support the main experiment.
