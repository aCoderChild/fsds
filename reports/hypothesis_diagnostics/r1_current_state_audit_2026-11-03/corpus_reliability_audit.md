# Corpus reliability audit

**Decision:** The five-language SFT corpus is not ready for the R1 experiment. No approved corpus or training manifest exists. AI-only review was authorized for the pilot; this audit does not claim fluent-human review.

## Findings

| Check | Result | Interpretation |
|---|---:|---|
| Source-backed candidate cases | 195/195 pass the draft validator | Source facts, final answers, and specified negative controls were rechecked; this does not certify translations. |
| Candidate split | 117 forget, 39 retain, 39 holdout; 102 groups | The validator found no cross-split overlap in its QID, target-fact, entity-surface, or group checks. These are defined checks, not proof against every indirect disclosure. |
| Original training questions passing final AI review | FR 324, DE 309, HI 288, TH 232; total 1,153/1,952 | Recomputed protected-name, number, script and marker checks found zero failures among these 1,153; semantics and fluency remain AI-reviewed only. |
| Complete original atomic cases | 5/117 forget; 3/39 retain | The release assembler expects all 117 and 39, respectively. |
| Complete original composition chains | 0/44 | All four prompt forms and all four target languages are required by the current release policy. |
| Quarantined replacement candidates | 948 | The generation report incorrectly marked all 948 as passing its integrity checks. Rechecking against the original protected-name lists finds **456 failures** and **492 passes**. |
| Repair selection | 799 original questions are final-review flags; 213 of them initially passed | The earlier 948-row repair targeted first-stage flags: it missed the 213 final-only flags and included 362 questions subsequently accepted in final review. |
| Gemma3 diagnostic review | 948 decisions; 947 source questions contained an unmasked protected name | The reviewer used incomplete entity metadata. Gemma3 shares a model family with the TranslateGemma repair translator. Its decisions are diagnostic only. |
| Original passes plus replacements passing Qwen7B, Gemma3, and recomputed integrity | 11/117 forget, 6/39 retain; 0/44 complete composition chains | This is an optimistic diagnostic bound, not an approved corpus. The replacements have not completed the frozen final review. |

For example, the Thai replacement for `0001-qa-sm:r1` translates the source name `Jorith Luque` into Thai even though the source request requires preserving that name. Its saved `protected_names_preserved` field is `true`. For the French version of the same prompt, Gemma3 accepted “Jorith Luque a-t-il fréquenté quelle école ?” as fluent, while Qwen7B flagged it. These examples demonstrate why a saved model pass is not sufficient evidence of reliable translation.

## Cause and effect

The all-flagged repair script reads the first-stage review JSONL files. Those records retain question text and recorded checks, but omit `protected_entity_names`. The repair script uses `row.get("protected_entity_names", [])`, so its name masking and name-preservation checks can silently operate on an empty list. The Gemma3 diagnostic similarly obtains names from final review records that omit that field. The source translation requests and reusable rows still contain the lists, so this audit rejoined candidates to those source rows and recomputed the checks.

The original final-review translations were also rechecked using the actual protected lists. None of the final-pass rows failed the same deterministic checks. The reviewed translations are nevertheless insufficient for the frozen release coverage, and automatic checks cannot establish semantic equivalence, fluency, or completeness of the source name lists. The two original Qwen reviewers share a model family. No fluent-human assessment was performed.

The saved repair outputs were not modified or promoted. Their historical model decisions and reported counts are retained as evidence of what ran; the recorded name-preservation result and the description of all review inputs as entity-masked must not be used as evidence of integrity. Earlier separate repair calibrations have not been revalidated by this audit.

## Verification and next work

Run from the repository root:

```sh
python experiments/synthworlds_r1/scripts/validate_candidate_draft.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/translation_coverage_diagnosis.py
python experiments/synthworlds_r1/training_release_draft/scripts/assemble_training_release.py
```

The first command verifies the source-backed draft and split; it writes `LOCAL_VALIDATION.json`. The second recomputes candidate provenance, name/number/script/marker checks, and coverage into `translation_coverage_diagnosis.json`. The third currently stops with the expected atomic-coverage error before writing a release. These checks do not run a model or establish language quality.

A new source-bound repair script, `experiments/synthworlds_r1/training_release_draft/scripts/repair_all_flagged_training_translations_v2.py`, retrieves protected-name lists from the hash-bound source rows, checks key and text equality, masks names before sending questions for review, and recomputes integrity on actual generated text. It selects the 799 final-review flags, including the 213 that passed the initial screen. Offline checks exercise source joins, name masking, altered-name rejection, sample selection and safe resume. The historical Gemma review script now refuses input review rows missing protected names. A 32-row stratified calibration found 11 passes under the existing 14B mask and 12 with a collision-free alternative; both are diagnostic and cannot approve a release. A further eight-row feedback probe yielded one same-reviewer adjudicated pass, six flags and one generation error. See `source_bound_calibration.md`. Existing repair candidates require revalidation or replacement, followed by the frozen final review and release assembly. After a release passes, SFT still needs to demonstrate baseline competence before unlearning results can answer the research question.
