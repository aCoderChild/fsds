# Source-bound translation repair calibration

**Status:** diagnostic only. This is not an approved SFT corpus or an experiment result for the unlearning hypothesis.

The corrected repair script selects from all 799 questions flagged by the final AI review. The previous repair sampled 948 first-stage flags and therefore missed 213 final-only flags, while including 362 questions later accepted by the final reviewer. A fixed sample of four final flags per language and question type produced 32 candidates, with the models, inputs, scripts and outputs bound by hashes.

| Language | Candidates | Protected-item checks pass | Direct Qwen7 screen pass | Round-trip Qwen7 pass | Existing Qwen14 mask pass | Alternative mask pass |
|---|---:|---:|---:|---:|---:|---:|
| French | 8 | 8 | 5 | 4 | 5 | 5 |
| German | 8 | 8 | 8 | 4 | 4 | 3 |
| Hindi | 8 | 8 | 7 | 0 | 1 | 3 |
| Thai | 8 | 8 | 4 | 0 | 1 | 1 |
| **Total** | **32** | **32** | **24** | **8** | **11** | **12** |

The final policy count consists of seven candidates passing both round-trip Qwen7 and Qwen14 checks and four candidates initially flagged by Qwen7 but accepted by Qwen14. The other 21 were flagged by Qwen14. This deliberately stratified sample is small; it does not estimate the full-corpus acceptance rate, prove translation quality, or justify changing the release gate. All checks were AI-only and no fluent-human review was performed. TranslateGemma generated and back-translated the questions; the two Qwen reviewers share a model family.

The direct and round-trip screens sometimes disagreed. For one German question, an English source asking for an **editor** was back-translated as asking for a **publisher**. For one French question, a Qwen7 explanation incorrectly described the past-tense phrase “a-t-il remplacé” as present tense. These examples show that review explanations and back-translations can themselves be wrong; individual decisions require contextual scrutiny.

All 32 candidates passed recomputed name, number, script and marker checks, and none contained an exact normalized source answer under the assembler's check. Back-translations passed their name and number checks. A separate source-bound verifier reproduced the decisions and hashes. The first attempt at Qwen14 review produced 32 runtime errors because its model server returned empty responses and a compute error. Unloading the other models and reloading Qwen14 restored inference; only Qwen14 decisions were retried, with the original checkpoint and the first error record preserved.

The existing Qwen14 masking function produces nested labels such as `[ENTITY_[NUMBER_1]]`, because it replaces numeric characters inside entity placeholders. A synthetic example with repeated names and a reordered number shows that this can lead to different substitutions across the source and target. All 1,952 current English training questions contain no digits or repeated protected names; in the 32 calibration examples the nested labels had matching counts across source, translation and back-translation. A collision-free masking helper was tested on both that synthetic example and all 32 sample triplets. A repeat using the same Qwen14 model and the alternative mask changed three decisions: one German pass became flagged and two Hindi flags became passes. The alternative-mask count is six two-model passes and six 14B adjudicated passes. This comparison shows variation, but it cannot establish masking as the sole cause or independently establish translation accuracy. Historical final review outputs retain their original hash-bound implementation.

## Feedback repair probe

The first two alternative-mask flags per language were selected for a separate eight-row diagnostic. TranslateGemma received the source question, its previous translation and the 14B reviewer's critique as explicitly unverified feedback, with entity names protected. Four revisions passed the direct Qwen7 screen, three were flagged, and one failed because the translation duplicated a protected marker. The seven usable revisions were then back-translated and reviewed with Qwen7 and Qwen14 using the collision-free mask. One German revision received a 14B adjudicated pass, and six remained flagged. The generation error remained blocked. These are same-reviewer rescues: Qwen14 generated the feedback and judged the revisions, so the single pass is not independent confirmation. The sample is too small to estimate general effectiveness, but these results do not justify scaling this exact feedback route without a better demonstrated review outcome.

## Prior NLLB calibration integrity recheck

An earlier 64-row diagnostic selected 55 NLLB 1.3B candidates and eight TranslateGemma fallbacks; one row had no selected candidate. All 28 candidates that passed its final AI screen were NLLB outputs. Rechecking source-bound entity and number preservation found two failures among the 63 selected outputs, both in TranslateGemma fallbacks that the final reviewer had flagged. The 28 screened passes also pass these recomputed integrity checks. The original set was a different stratified sample, and the fallbacks came from the flawed earlier repair run, so it should not be treated as a direct comparison of translators.

## Matched NLLB and TranslateGemma comparison

NLLB 1.3B generated a replacement for each of the same 32 questions; all 32 passed the source-bound name, number, script and marker checks. They were back-translated and screened using Qwen7 and Qwen14 with the collision-free mask. The paired comparison uses the same review policy for the TranslateGemma candidates, and all counts below are **AI-screen passes**, not independently verified translations.

| Language | Questions | NLLB pass | TranslateGemma pass | Either passes | Neither passes |
|---|---:|---:|---:|---:|---:|
| French | 8 | 6 | 5 | 6 | 2 |
| German | 8 | 2 | 3 | 3 | 5 |
| Hindi | 8 | 4 | 3 | 5 | 3 |
| Thai | 8 | 2 | 1 | 3 | 5 |
| **Total** | **32** | **14** | **12** | **17** | **15** |

Nine questions passed for both translators, five only for NLLB, three only for TranslateGemma, and 15 for neither. Qwen7 round-trip checks passed 13 NLLB candidates and eight TranslateGemma candidates; Qwen14's adjudication changed the final counts. The selection of a candidate based on these same reviews is adaptive, and the small sample does not estimate corpus-wide success or demonstrate linguistic correctness. NLLB and TranslateGemma are distinct translation families, but TranslateGemma also performs back-translation and the Qwen reviewers share a model family. No candidate was merged or approved.

Both sets of 32 candidates had no exact normalized source-answer mentions under the assembler's check. This does not rule out paraphrased, aliased or indirect answer disclosure.

## Reuse and full-release coverage

An independent source-bound coverage calculation checked the existing calibration records against the final review and protected-item requirements. It found 42 distinct provisional candidate keys across the matched comparison and the earlier NLLB calibration. The results below are hypothetical: the candidates remain quarantined and have not passed the production release procedure. The calculations combine historical and alternative-mask decisions, so they are not a new review policy.

| Candidate set | Complete forget cases | Complete retain cases | Complete composition chains | Unresolved flagged pairs |
|---|---:|---:|---:|---:|
| Original final review | 5/117 | 3/39 | 0/44 | 799 |
| If all 42 provisional keys were accepted | 6/117 | 4/39 | 0/44 | 757 |

The existing samples therefore cannot make the full dataset ready. In particular, none completes a composition chain, and the hypothetical complete cases cover only three forget evaluation cases in two groups and two retain evaluation cases in two groups. Further translation repair and independently screened review are necessary; no candidate is released by this calculation. The estimates concern coverage, not model competence or translation accuracy.

Recompute with `python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/reviewed_candidate_coverage.py`; its hash-bound inputs and detailed breakdown are recorded in `reviewed_candidate_coverage.json`.

## Assistant-generated diagnostic candidates

For the 15 matched questions flagged for both translation methods, the assistant generated a third set of candidate translations in `assistant_candidate_probe.json`. Source-bound deterministic checks pass for all 15, including the existing checks for protected names present in the question. These are unreviewed candidates, not replacements accepted for SFT. In particular, relation direction, ambiguous terms such as “editor” and “performed,” and fluency still require independent AI review. The selection was conditioned on earlier reviews, the exact assistant model revision was not recorded, and no fluent-human review was performed. This small probe cannot establish a full-corpus success rate.

`python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_assistant_candidate_probe.py` checks exact membership in the double-flagged set, source-bound protected items, and exact normalized source-answer mentions; it found no exact answer mentions in these 15 candidates. This check does not detect aliases or indirect disclosures and is not semantic review. Run `verify_nllb_matched32.py` as well to verify the source review evidence.

The Kaggle Qwen2.5-7B review completed on all 15, with 9 model passes and 6 flags (FR 0/2, DE 4/5, HI 2/3, TH 3/5 passes). The first Kaggle version failed before review because the installed bitsandbytes package lacked a CUDA 13 binary; the second used half precision. The result and log are in `evidence/assistant_kaggle_review_v2/`. `verify_assistant_kaggle_review.py` checks the 15 identifiers, raw JSON decisions, model revision and masked input hash, but cannot establish judgment accuracy.

The reviewer has clear explanatory errors: for the French source “Who is Calix Silverwood the godparent of?”, it says the English makes Calix the godchild, reversing the English relation. For a German stepparent question it proposes “Schritteltern” as the correct term. These errors cast doubt on individual flags, and agreement or confident explanations do not validate passes. This is one Qwen-family reviewer without a separate back-translation or adjudication; all 15 remain quarantined and the review does not satisfy the release gate. The sample is highly selected and does not measure corpus-wide translation quality.

A different-family Mistral 7B model reviewed the same masked 15 questions independently of the Qwen decisions. It passed all 15: nine agreed with Qwen's passes and six contradicted Qwen's flags. Mistral also made an obvious error when it described the French pronoun “il” as gender-neutral. Its blanket-pass behavior and unverified explanations make it unsuitable as standalone confirmation. There are no human gold labels for this sample, so neither agreement nor disagreement establishes accuracy. `verify_mistral_review_probe.py` checks both model revisions, matching input hashes, row identities and raw decision records. The Mistral output and log are in `evidence/mistral_review_probe/`.

A negative control retained those same 15 translated questions but replaced each English source with “What color is [ENTITY_A]?”, deliberately creating obvious mismatches. Mistral flagged 13/15 on at least one field but marked semantics as passing for 6/15; two of those received both semantic and fluency passes. Four other semantic passes were rejected only through the fluency field. The input and decisions are reproducibly checked with `verify_mistral_negative_control.py`; output and log are under `evidence/mistral_negative_control/`. The fixed color question is a simple synthetic test, not a representative accuracy estimate, but two outright passes and six semantic passes on these mismatches provide direct evidence that this model cannot be trusted alone to certify candidate equivalence. No candidate was promoted.

The pinned Qwen2.5 7B reviewer screened the identical 15 negative-control inputs and flagged all 15 on semantics. `verify_qwen_negative_control.py` verifies the identical input, model revision, row identities and raw decisions; output and log are under `evidence/qwen_negative_control/`. This demonstrates detection of these obvious mismatches only. It does not measure sensitivity to subtle relationship, gender, grammatical or factual errors, nor negate the observed Qwen reasoning mistakes. The 619 full-screen passes remain quarantined.

An additional NLLB 1.3B back-translation on Kaggle used the same masked inputs. Four of 15 outputs did not preserve the exact protected markers, including three Thai and one French output; they cannot be used as intact back-translations. Among the other 11, the German doctoral-student question came back as a student “at” the person, and the Hindi question about what is on a boundary came back as what the boundary is. These shifts may arise in either translation direction and do not identify which translation is wrong. The raw output and log are in `evidence/assistant_nllb_backtranslation/`; `verify_assistant_nllb_backtranslation.py` checks model revision, input hash, row identity and independently recomputed marker counts. NLLB was also one of the prior candidate generators, so this is a different family from Qwen but not independent of the whole calibration pipeline. No result establishes fluent-human correctness or satisfies the release gate.

A second NLLB run substituted neutral names before back-translation and restored markers afterward. All 15 outputs preserved marker counts after restoration; the doctoral-student and boundary questions now back-translated more faithfully. This does not prove that the substitute names caused those improvements. Other outputs remain problematic: a Thai noble-title question returned “gods” instead of a title, and a German stepparent question returned a phrase about a boot. These may be back-translation errors, candidate errors, or both. The saved second run is in `evidence/assistant_nllb_backtranslation_surrogates/`; `verify_assistant_nllb_surrogates.py` independently reconstructs the substitutions and checks model, input hash, identifiers, restoration and marker counts. No semantic passes or release approvals are assigned.

## Full NLLB candidate generation

The pinned NLLB 1.3B model generated 799 answer-blind candidate translations covering the complete final-review flag queue. A local validator reconstructed source bindings and name substitutions, checked output ordering, model/input hashes, source names, numbers, scripts, markers and exact normalized answer mentions. It found no exact answer mentions. Results are quarantined, without semantic or fluency review:

| Language | Structural checks pass | Structural checks fail |
|---|---:|---:|
| French | 164 | 0 |
| German | 179 | 0 |
| Hindi | 12 | 188 |
| Thai | 17 | 239 |
| **Total** | **372** | **427** |

All 427 failures involve nonpreservation of substitute names and consequently protected names; no failed item is admitted for review as a valid candidate. Structural success is not translation quality. The 372 passes still require independent semantic and fluency review, while the 427 need a robust name-preserving repair method. Outputs and log are under `evidence/full_nllb_candidates/`; reproduce the deterministic checks with `python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_full_nllb_candidates.py`. No release file or training corpus was modified.

An exploratory post-hoc restoration matched a small manually curated list of exact Hindi and Thai transliterations of the two surrogate names and restored the source names. With those heuristics, all 188 Hindi and 239 Thai previously failing rows pass the existing deterministic checks; no exact normalized answer mentions were found. The variant list was chosen after inspecting model outputs and has not been independently validated. Correct counts do not show that a replacement denotes the intended entity or that the resulting sentence is grammatical. These 427 remain **provisional and quarantined**, as do the other 372; no production data was changed. Reproduce the experiment with `python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/diagnose_transliterated_surrogates.py`, after the full-candidate verifier. It writes `transliteration_restoration_diagnostic.json` with the exact variant list and source hashes.

A private Kaggle direct-screen job was submitted for all 799 candidates. It uses masked, answer-blind source/candidate pairs and the pinned Qwen2.5 7B model, and records which 427 candidates use heuristic name restoration. It is a **diagnostic first screen**, not the frozen two-stage release review: the 15-row probe already demonstrated model reasoning errors, and this job does not include independent back-translation or adjudication. No candidate is approved by a passing decision. The source-bound builder is `build_kaggle_full_candidate_review.py` and the kernel is `bunnoob2005/synthworlds-full-candidate-review`.

The full direct screen finished and `verify_full_candidate_review.py` reconstructed all 799 masked inputs and checked model revision, decisions, input hashes and restoration provenance. Qwen returned 619 passes and 180 flags: French 143/164 passes, German 146/179, Hindi 141/200, and Thai 189/256. Hindi and Thai include 427 heuristic restorations. These are **model judgments, not approved translations**. Even if all 619 passes were accepted hypothetically, complete coverage would be only 64/117 forget cases, 25/39 retain cases and 8/44 composition chains; the current actual release coverage remains 5, 3 and 0. These hypothetical counts do not apply the full release protocol and should not be used for training. The saved output is `evidence/full_candidate_review/review.json`.

An NLLB 600M job generated alternative drafts for all 180 direct-screen flags. Source-bound deterministic checks found 85 with exact substitute preservation, 56 recoverable with the same post-hoc transliteration heuristics, and 39 unresolved (one Hindi and 38 Thai). The 141 restored drafts are unreviewed and neither checkpoint is independent of the other. Output and log are in `evidence/nllb600_flagged_repairs/`; `verify_nllb600_flagged_repairs.py` recomputes selection, source bindings, model/input hashes, protected-item checks and exact answer matching. No draft is approved, and no training files were changed.

Of the 141 structurally restored drafts, 41 were identical to the previously flagged text and 100 differed. A Qwen direct screen of the 100 different drafts produced 33 passes and 67 flags: French 5/12 passes, German 7/19, Hindi 8/40 and Thai 13/29. `verify_nllb600_repair_review.py` reconstructs the masked questions from source, verifies candidate bindings and model revision, and parses the recorded decisions. The outputs are in `evidence/nllb600_repair_review/`. All decisions are diagnostic and adaptive: the same Qwen reviewer determined which drafts were regenerated and then scored the replacements, with no independent back-translation or adjudication. The 33 passes are not approved for release. Even provisionally counting all of them leaves 147 of the original 180 Qwen-flagged pairs unresolved; actual release status is unchanged.

`build_candidate_review_ledger.py` reconstructs an evidence ledger for all 799 originally flagged keys after running the underlying verifiers. Each entry binds its selected draft and review record by SHA-256, distinguishes the first screen from adaptively screened alternatives and unresolved flags, and explicitly sets `release_approved` to false. It produces 619 first-screen candidates, 33 adaptive candidates and 147 unresolved records in `evidence/candidate_review_ledger.json`. This ledger is a diagnostic inventory only: the legacy assembler and source-integrity proof continue to require the original frozen review, and the ledger cannot authorize or create an SFT release.

Verification commands:

```sh
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_source_bound_calibration.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_calibration_roundtrip.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_calibration_masking_comparison.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_feedback_calibration.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_prior_nllb_calibration.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_nllb_matched32.py
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/audit_matched_candidate_answers.py
```

The candidate and review records are under `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/source_bound_finalflagged_calibration32/` and `reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/evidence/source_bound_calibration_roundtrip_retry/`. No candidate was merged into the release, and SFT and unlearning have not started.
