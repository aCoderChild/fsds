#!/usr/bin/env python3
"""Assemble a fail-closed, hash-bound five-language pilot SFT release.

Release requires all 1,952 target-language training questions to pass the
round-trip screen and a second Qwen review, with all 117 forget and 39 retain
atomic cases plus all 44 disjoint training-only composition chains intact.
This is not a claim of fluent-human translation review.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
V51 = REPO / "experiments/synthworlds_r1"
OUT = V51 / "training_release_draft"
REVIEW_DIR = OUT / "translation_reviews_roundtrip_qwen14b"
TRAINING = V51 / "sft_training"
LANGS = ("en", "fr", "de", "hi", "th")
TARGET_LANGS = LANGS[1:]
SPLITS = ("forget", "retain", "train_only")
EXPECTED_CANDIDATE_QUESTIONS_PER_LANG = 488
EXPECTED_ATOMIC_CASES_BY_SPLIT = {"forget": 117, "retain": 39}
EXPECTED_COMPOSITION_CHAINS = 44
PASS_STATUSES = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_adjudication_record(lang: str, key: tuple[str, str], row: dict) -> None:
    """Check a row's two-stage decision without treating a downgrade as corruption."""
    initial_status = row.get("review_status")
    final_status = row.get("final_review_status")
    primary_pass = initial_status == "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS"
    if initial_status not in {"AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS", "AI_ONLY_REVIEW_FLAGGED"}:
        raise ValueError(f"{lang}/{key}: first-stage review is blocked or incomplete")
    if type(row.get("qwen14b_semantic_pass")) is not bool or type(row.get("qwen14b_fluency_pass")) is not bool:
        raise ValueError(f"{lang}/{key}: missing second-stage semantic/fluency decision")
    if primary_pass and (row.get("qwen_semantic_pass") is not True or row.get("qwen_fluency_pass") is not True):
        raise ValueError(f"{lang}/{key}: first-stage pass lacks semantic or fluency evidence")

    target_checks = row.get("target_deterministic_checks")
    backtranslation_checks = row.get("backtranslation_checks")
    integrity_pass = (
        isinstance(target_checks, dict) and bool(target_checks) and all(target_checks.values())
        and isinstance(backtranslation_checks, dict) and bool(backtranslation_checks) and all(backtranslation_checks.values())
    )
    qwen14_pass = row.get("qwen14b_semantic_pass") is True and row.get("qwen14b_fluency_pass") is True

    if final_status == "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS":
        if not primary_pass:
            raise ValueError(f"{lang}/{key}: two-model pass lacks a first-stage pass")
        if not qwen14_pass or not integrity_pass:
            raise ValueError(f"{lang}/{key}: two-model pass lacks second-stage or integrity evidence")
    elif final_status == "AI_ONLY_14B_ADJUDICATED_PASS":
        if primary_pass:
            raise ValueError(f"{lang}/{key}: 14B adjudication does not correspond to an initial flagged row")
        if not qwen14_pass or not integrity_pass:
            raise ValueError(f"{lang}/{key}: 14B pass lacks semantic, fluency, or integrity evidence")
    elif final_status == "AI_ONLY_REVIEW_FLAGGED":
        # A stronger reviewer may downgrade a first-stage pass. A flagged row
        # is inconsistent only when every condition for a final pass is met.
        if qwen14_pass and integrity_pass:
            raise ValueError(f"{lang}/{key}: final review flags a row that satisfies every pass condition")
    else:
        raise ValueError(f"{lang}/{key}: blocked or unexpected final review status")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def source_hashes_for_lang(lang: str) -> dict:
    # Recompute the same source manifest used by collect_rows; the imported
    # checker is the only place that defines the translation-question set.
    import sys
    script_dir = SCRIPT.parent
    sys.path.insert(0, str(script_dir))
    from review_training_translations_qwen import collect_rows, REVIEW_PROTOCOL
    rows, hashes = collect_rows(lang)
    hashes.pop("review_protocol", None)
    hashes = {**hashes,
              "roundtrip_protocol": "translategemma_backtranslation_plus_qwen2.5_7b_primary_and_14b_all_candidate_second_pass_v1",
              "adjudication_script_sha256": sha(script_dir / "adjudicate_training_translations_qwen14b.py")}
    if len(rows) != EXPECTED_CANDIDATE_QUESTIONS_PER_LANG:
        raise ValueError(f"{lang}: source question set has {len(rows)} rows, expected {EXPECTED_CANDIDATE_QUESTIONS_PER_LANG}")
    return hashes


def adjudication_script_provenance(recorded_sha256: str) -> dict:
    """Verify the recorded reviewer source, allowing one exact root-path rewrite.

    The prior script used the compatibility path under ``data/``. That path is
    now a symlink to this canonical project. Accept the old receipt only when
    reconstructing the old file from the current source by changing that one
    assignment reproduces the recorded digest exactly; all per-language input
    hashes and row/source checks are still enforced by checked_translation_reviews.
    """
    script = SCRIPT.parent / "adjudicate_training_translations_qwen14b.py"
    current_bytes = script.read_bytes()
    current_sha256 = sha_bytes(current_bytes)
    if recorded_sha256 == current_sha256:
        return {
            "status": "PASS_CURRENT_SCRIPT_EXACT",
            "recorded_script_sha256": recorded_sha256,
            "current_script_sha256": current_sha256,
        }

    current_assignment = b'V51 = REPO / "experiments/synthworlds_r1"'
    legacy_assignment = b'V51 = REPO / "data/synthworlds_sm_pilot_v5_1"'
    if current_bytes.count(current_assignment) != 1:
        raise ValueError("The current adjudicator no longer has the expected canonical root assignment")
    reconstructed = current_bytes.replace(current_assignment, legacy_assignment, 1)
    reconstructed_sha256 = sha_bytes(reconstructed)
    legacy_root = REPO / "data/synthworlds_sm_pilot_v5_1"
    try:
        alias_matches = legacy_root.is_symlink() and legacy_root.resolve(strict=True) == V51.resolve(strict=True)
    except OSError:
        alias_matches = False
    if recorded_sha256 != reconstructed_sha256 or not alias_matches:
        raise ValueError("The adjudication report was produced by a different script version")
    return {
        "status": "PASS_EXACT_LEGACY_ROOT_PATH_ALIAS",
        "recorded_script_sha256": recorded_sha256,
        "current_script_sha256": current_sha256,
        "reconstructed_legacy_script_sha256": reconstructed_sha256,
        "only_source_change": "V51 root assignment from data/synthworlds_sm_pilot_v5_1 to experiments/synthworlds_r1",
        "legacy_root_resolves_to_canonical_project": str(legacy_root.resolve(strict=True)) == str(V51.resolve(strict=True)),
    }


def checked_translation_reviews() -> dict[tuple[str, str, str], dict]:
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    from review_training_translations_qwen import collect_rows

    review_dir = REVIEW_DIR
    report_path = review_dir / "report.json"
    report = read_json(report_path)
    if (report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
            or report.get("release_ready_for_assembly") is not True
            or report.get("do_not_train") is not True
            or report.get("reviewed_every_candidate") is not True
            or report.get("adjudicator_model") != "ollama:qwen2.5:14b"):
        raise ValueError("The two-stage AI-only training translation screen/adjudication is incomplete")
    if report.get("human_translation_reviewed") is not False or report.get("answers_sent_to_reviewer") is not False:
        raise ValueError("Translation review metadata misstates human review or answer blinding")
    expected_protocol = "translategemma_backtranslation_plus_qwen2.5_7b_primary_and_14b_all_candidate_second_pass_v1"
    if report.get("protocol") != expected_protocol:
        raise ValueError("Unexpected translation review protocol")
    initial_report_path = OUT / "translation_reviews_roundtrip_qwen/report.json"
    if report.get("initial_screen_report_sha256") != sha(initial_report_path):
        raise ValueError("The adjudication report is not bound to the current first-stage screen")
    script_provenance = adjudication_script_provenance(report.get("adjudication_script_sha256", ""))

    translations = {}
    initial_report = read_json(initial_report_path)
    for lang in TARGET_LANGS:
        rows_path = review_dir / f"{lang}.jsonl"
        detail = report.get("languages", {}).get(lang, {})
        initial_detail = initial_report.get("languages", {}).get(lang, {})
        if detail.get("adjudication_sha256") != sha(rows_path):
            raise ValueError(f"{lang}: review file hash mismatch")
        # The script hash comes from the receipt only after exact-byte
        # compatibility verification above. The rest of this mapping is
        # recomputed from current inputs and compared in full.
        current_sources = source_hashes_for_lang(lang)
        current_sources["adjudication_script_sha256"] = script_provenance["recorded_script_sha256"]
        if detail.get("source_hashes") != current_sources:
            raise ValueError(f"{lang}: reviewer source hashes differ from current source rows")
        initial_path = OUT / "translation_reviews_roundtrip_qwen" / f"{lang}.jsonl"
        if (detail.get("initial_review_sha256") != sha(initial_path)
                or initial_detail.get("review_sha256") != sha(initial_path)):
            raise ValueError(f"{lang}: adjudication does not bind the first-stage review")
        initial_by_key = {(r.get("prompt_id"), r.get("kind")): r for r in read_jsonl(initial_path)}
        source_rows, _ = collect_rows(lang)
        source_hashes = current_sources
        source_by_key = {(r["prompt_id"], r["kind"]): r for r in source_rows}
        by_key = {}
        for row in read_jsonl(rows_path):
            key = (row.get("prompt_id"), row.get("kind"))
            if key in by_key:
                raise ValueError(f"{lang}: duplicate review record {key}")
            source = source_by_key.get(key)
            initial_row = initial_by_key.get(key)
            if (not source or row.get("source_hashes") != source_hashes
                    or row.get("adjudication_protocol") != expected_protocol
                    or row.get("adjudicator_model") != report.get("adjudicator_model")
                    or row.get("human_translation_reviewed") is not False
                    or row.get("answers_sent_to_reviewer") is not False
                    or row.get("initial_review_sha256") != detail.get("initial_review_sha256")):
                raise ValueError(f"{lang}/{key}: unexpected review provenance or source row")
            if not initial_row or any(row.get(field) != value for field, value in initial_row.items() if field != "source_hashes"):
                raise ValueError(f"{lang}/{key}: adjudication changed fields from the hash-bound primary screen")
            if row.get("text_en") != source.get("text_en") or row.get("text_translation") != source.get("text_translation"):
                raise ValueError(f"{lang}/{key}: reviewed question differs from the current source translation")
            validate_adjudication_record(lang, key, row)
            by_key[key] = row
        if len(by_key) != EXPECTED_CANDIDATE_QUESTIONS_PER_LANG:
            raise ValueError(f"{lang}: expected {EXPECTED_CANDIDATE_QUESTIONS_PER_LANG} screened questions, found {len(by_key)}")
        if set(by_key) != set(source_by_key):
            raise ValueError(f"{lang}: source and adjudication review keys differ")
        if sum(detail.get("final_status_counts", {}).values()) != EXPECTED_CANDIDATE_QUESTIONS_PER_LANG:
            raise ValueError(f"{lang}: report counts do not cover the complete source set")
        if detail.get("qwen14b_reviewed_count") != EXPECTED_CANDIDATE_QUESTIONS_PER_LANG:
            raise ValueError(f"{lang}: report does not attest second-stage review of all 488 candidates")
        for (prompt_id, kind), row in by_key.items():
            translations[(lang, prompt_id, kind)] = row
    return translations


def make_row(base: dict, lang: str, question: str, translation_model: str, status: str) -> dict:
    answer = base["answer_en"]
    if not question.strip() or not answer.strip():
        raise ValueError(f"Empty question/answer in {base.get('sft_id')}/{lang}")
    if norm(answer) in norm(question):
        raise ValueError(f"Answer text leaks into question {base.get('sft_id')}/{lang}")
    split = base.get("split")
    if split not in SPLITS:
        raise ValueError(f"Forbidden or missing split {split!r} for {base.get('sft_id')}")
    return {
        "sft_id": base["sft_id"], "source_qa_id": base["source_qa_id"], "split": split,
        "hop": base["hop"], "fact_id": base.get("fact_id"), "fact_ids": base.get("fact_ids", []),
        "language": lang, "question": question, "answer": answer,
        "question_en": base["question_en"], "answer_en": answer,
        "translation_model": translation_model, "translation_status": status,
        "human_translation_reviewed": False,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
    }


def build_corpus(translations: dict[tuple[str, str, str], dict]) -> tuple[list[dict], dict]:
    atomic_reuse = read_jsonl(OUT / "atomic_reuse.jsonl")
    atomic_requests = read_jsonl(OUT / "atomic_translation_requests.jsonl")
    atomic_drafts = {lang: read_jsonl(OUT / f"atomic_translation_drafts/{lang}.jsonl") for lang in TARGET_LANGS}
    atomic_repairs = {}
    for lang in TARGET_LANGS:
        path = OUT / f"atomic_translation_repairs/{lang}.jsonl"
        atomic_repairs[lang] = {r["sft_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    comp_rows = read_jsonl(OUT / "composition_reuse.jsonl")
    comp_queue = read_jsonl(OUT / "composition_translation_review_queue.jsonl")

    reuse_by_id_lang = {(r["sft_id"], r["language"]): r for r in atomic_reuse}
    request_by_id_lang = {(r["sft_id"], r["language"]): r for r in atomic_requests}
    draft_by_id_lang = {}
    for lang, rows in atomic_drafts.items():
        for row in rows:
            draft_by_id_lang[(row["sft_id"], lang)] = {**row, **atomic_repairs[lang].get(row["sft_id"], {})}
    comp_by_id_lang = {(r["sft_id"], r["language"]): r for r in comp_rows}
    comp_queue_by_id_lang = {(r["prompt_id"], r["language"]): r for r in comp_queue}
    if len(reuse_by_id_lang) != len(atomic_reuse) or len(request_by_id_lang) != len(atomic_requests) or len(comp_by_id_lang) != len(comp_rows) or len(comp_queue_by_id_lang) != len(comp_queue):
        raise ValueError("Duplicate atomic/composition source rows")

    atomic_ids_by_case: dict[str, set[str]] = defaultdict(set)
    atomic_base_by_id = {}
    for row in atomic_reuse:
        atomic_ids_by_case[row["source_qa_id"]].add(row["sft_id"])
        if row["language"] == "en":
            atomic_base_by_id[row["sft_id"]] = row
    for row in atomic_requests:
        atomic_ids_by_case[row["source_qa_id"]].add(row["sft_id"])
        atomic_base_by_id.setdefault(row["sft_id"], row)
    eligible_atomic_cases = set()
    atomic_split_by_case = {}
    for case_id, ids in atomic_ids_by_case.items():
        if len(ids) != 2:
            raise ValueError(f"{case_id}: expected two atomic hops, found {len(ids)}")
        splits = {atomic_base_by_id[sft_id].get("split") for sft_id in ids if sft_id in atomic_base_by_id}
        if len(splits) != 1 or None in splits:
            raise ValueError(f"{case_id}: atomic hops have missing or inconsistent split labels: {splits}")
        atomic_split_by_case[case_id] = next(iter(splits))
        if all(translations.get((lang, sft_id, "atomic"), {}).get("final_review_status") in PASS_STATUSES
               for lang in TARGET_LANGS for sft_id in ids):
            eligible_atomic_cases.add(case_id)

    eligible_atomic_counts = Counter(atomic_split_by_case[case_id] for case_id in eligible_atomic_cases)
    if {split: eligible_atomic_counts.get(split, 0) for split in EXPECTED_ATOMIC_CASES_BY_SPLIT} != EXPECTED_ATOMIC_CASES_BY_SPLIT:
        raise ValueError(f"Translation screen excluded required atomic cases: {dict(eligible_atomic_counts)}; expected {EXPECTED_ATOMIC_CASES_BY_SPLIT}")
    if set(atomic_split_by_case.values()) != set(EXPECTED_ATOMIC_CASES_BY_SPLIT):
        raise ValueError(f"Unexpected atomic split membership: {dict(Counter(atomic_split_by_case.values()))}")

    composition_ids_by_chain: dict[str, set[str]] = defaultdict(set)
    for row in comp_rows:
        composition_ids_by_chain[row["source_qa_id"]].add(row["sft_id"])
    eligible_composition_chains = set()
    for source_qa_id, ids in composition_ids_by_chain.items():
        if len(ids) == 4 and all(translations.get((lang, sft_id, "composition"), {}).get("final_review_status") in PASS_STATUSES
                                 for lang in TARGET_LANGS for sft_id in ids):
            eligible_composition_chains.add(source_qa_id)
    if len(composition_ids_by_chain) != EXPECTED_COMPOSITION_CHAINS or len(eligible_composition_chains) != EXPECTED_COMPOSITION_CHAINS:
        raise ValueError(f"Translation screen excluded or altered training-only composition chains: {len(eligible_composition_chains)}/{len(composition_ids_by_chain)} eligible, expected {EXPECTED_COMPOSITION_CHAINS}")

    rows = []
    for (sft_id, lang), base in sorted(reuse_by_id_lang.items()):
        if base["source_qa_id"] not in eligible_atomic_cases:
            continue
        if lang == "en":
            question, model, status = base["question"], base["translation_model"], "SOURCE_ENGLISH"
        else:
            review = translations[(lang, sft_id, "atomic")]
            if review.get("text_en") != base["question_en"]:
                raise ValueError(f"{lang}/{sft_id}: reviewed source differs from reusable atomic question")
            question, model, status = review["text_translation"], review["translation_model"], review["final_review_status"]
        rows.append(make_row(base, lang, question, model, status))

    for (sft_id, lang), request in sorted(request_by_id_lang.items()):
        if request["source_qa_id"] not in eligible_atomic_cases:
            continue
        if lang == "en":
            raise ValueError("An English atomic translation request is invalid")
        review = translations[(lang, sft_id, "atomic")]
        if review.get("text_en") != request["question_en"]:
            raise ValueError(f"{lang}/{sft_id}: reviewed atomic translation source mismatch")
        base = {**request, "hop": request["hop"], "question_en": request["question_en"],
                "answer_en": request["answer_en"], "split": request["split"]}
        rows.append(make_row(base, lang, review["text_translation"], review["translation_model"], review["final_review_status"]))

    for (sft_id, lang), base in sorted(comp_by_id_lang.items()):
        if base["source_qa_id"] not in eligible_composition_chains:
            continue
        if lang == "en":
            question, model, status = base["question"], base["translation_model"], "SOURCE_ENGLISH"
        else:
            review = translations[(lang, sft_id, "composition")]
            source = comp_queue_by_id_lang.get((sft_id, lang))
            if not source or review.get("text_en") != base["question_en"] or source["text_en"] != base["question_en"]:
                raise ValueError(f"{lang}/{sft_id}: reviewed composition source provenance mismatch")
            question, model, status = review["text_translation"], review["translation_model"], review["final_review_status"]
        rows.append(make_row(base, lang, question, model, status))

    keys = [(r["language"], r["sft_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (language, sft_id) in assembled corpus")
    by_language = {lang: [r for r in rows if r["language"] == lang] for lang in LANGS}
    by_ids = {lang: {r["sft_id"] for r in values} for lang, values in by_language.items()}
    if any(not by_language[lang] for lang in LANGS):
        raise ValueError({lang: len(by_language[lang]) for lang in LANGS})
    if any(len(by_language[lang]) != EXPECTED_CANDIDATE_QUESTIONS_PER_LANG for lang in LANGS):
        raise ValueError(f"The complete five-language SFT corpus must have {EXPECTED_CANDIDATE_QUESTIONS_PER_LANG} rows per language: { {lang: len(by_language[lang]) for lang in LANGS} }")
    if any(by_ids[lang] != by_ids["en"] for lang in TARGET_LANGS):
        raise ValueError("Five languages do not cover exactly the same SFT IDs")
    for lang, language_rows in by_language.items():
        counts = Counter(r["split"] for r in language_rows)
        if set(counts) != set(SPLITS):
            raise ValueError(f"{lang}: a required split is missing: {dict(counts)}")
        answers = {r["sft_id"]: r["answer"] for r in language_rows}
        if answers != {r["sft_id"]: r["answer"] for r in by_language["en"]}:
            raise ValueError(f"{lang}: canonical answers differ from English")
    return sorted(rows, key=lambda r: (LANGS.index(r["language"]), r["split"], r["sft_id"])), {
        "rows_by_language": {lang: len(by_language[lang]) for lang in LANGS},
        "rows_by_split_per_language": {lang: dict(Counter(r["split"] for r in by_language[lang])) for lang in LANGS},
        "unique_sft_ids_per_language": len(by_ids["en"]),
        "eligible_atomic_cases": len(eligible_atomic_cases),
        "eligible_atomic_cases_by_split": {split: eligible_atomic_counts.get(split, 0) for split in EXPECTED_ATOMIC_CASES_BY_SPLIT},
        "excluded_atomic_cases": len(atomic_ids_by_case) - len(eligible_atomic_cases),
        "eligible_composition_chains": len(eligible_composition_chains),
        "excluded_composition_chains": len(composition_ids_by_chain) - len(eligible_composition_chains),
        "ai_screen_candidates_by_language": {lang: EXPECTED_CANDIDATE_QUESTIONS_PER_LANG for lang in TARGET_LANGS},
        "primary_screen_passed_candidates_by_language": {lang: sum(row.get("review_status") == "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS" for (r_lang, _, _), row in translations.items() if r_lang == lang) for lang in TARGET_LANGS},
        "final_review_passed_candidates_by_language": {lang: sum(row.get("final_review_status") in PASS_STATUSES for (r_lang, _, _), row in translations.items() if r_lang == lang) for lang in TARGET_LANGS},
        "total_rows": len(rows),
    }


def validate_split_and_evaluation(corpus: list[dict]) -> dict:
    cases_path = OUT / "evaluation/cases.jsonl"
    prompts_path = OUT / "evaluation/prompts.jsonl"
    labels_path = OUT / "evaluation/labels.jsonl"
    cases, prompts, labels = map(read_jsonl, (cases_path, prompts_path, labels_path))
    # Recompute the common evaluation cohort from the source requests and
    # hash-bound blinded translation review. This prevents changed, flagged,
    # failed, or same-model-self-checked target prompts from entering on row counts alone.
    script_dir = SCRIPT.parent
    sys.path.insert(0, str(script_dir))
    from prepare_release import approved_eval_cohort

    verified_cases, verified_eval, verified_summary = approved_eval_cohort()
    expected_case_count = len(verified_cases)
    expected_prompt_count = len(verified_eval["prompts"])
    if (not expected_case_count or expected_prompt_count != expected_case_count * len(LANGS) * 3
            or len(cases) != expected_case_count or len(prompts) != expected_prompt_count
            or len(labels) != expected_prompt_count):
        raise ValueError(f"Evaluation sizes unexpected: cases={len(cases)}, prompts={len(prompts)}, labels={len(labels)}; screened expectations={expected_case_count}/{expected_prompt_count}")
    if cases != verified_cases:
        raise ValueError("Evaluation cases differ from the source-verified, translation-screened cohort")
    if prompts != verified_eval["prompts"]:
        raise ValueError("Evaluation prompts differ from the hash-bound approved translations")
    if labels != verified_eval["labels"]:
        raise ValueError("Evaluation labels differ from the source-verified labels")
    if {(r["prompt_id"], r["language"]) for r in prompts} != {(r["prompt_id"], r["language"]) for r in labels}:
        raise ValueError("Evaluation prompt and label key sets differ")
    if any(p.get("prompt", "").strip() == "" for p in prompts):
        raise ValueError("An evaluation prompt is empty")
    case_ids = {r["source_qa_id"] for r in cases}
    candidate = read_jsonl(V51 / "candidate_cases.draft.jsonl")
    if len(candidate) != 195 or len({r["source_qa_id"] for r in candidate}) != 195:
        raise ValueError("Candidate split manifest must cover 195 unique cases")
    group_split = {}
    for row in candidate:
        group = row["group_id"]
        if group in group_split and group_split[group] != row["split"]:
            raise ValueError(f"Leakage group crosses split assignment: {group}")
        group_split[group] = row["split"]
    train_only_ids = {r["source_qa_id"] for r in corpus if r["split"] == "train_only"}
    if train_only_ids & case_ids:
        raise ValueError("A train-only composition chain is an evaluation candidate")
    if any(r["split"] == "holdout" for r in corpus):
        raise ValueError("Holdout data entered training corpus")
    en_multi = {norm(r["prompt"]) for r in prompts if r["language"] == "en" and r["probe"] == "multi_hop"}
    exact_overlap = [(r["sft_id"], r["language"]) for r in corpus if norm(r["question"]) in en_multi and r["language"] == "en"]
    if exact_overlap:
        raise ValueError(f"Exact English multi-hop evaluation prompt appears in SFT data: {exact_overlap[:5]}")
    review_root = V51 / "translation_reviews/translategemma_repair_v1_full"
    review_manifest = read_json(review_root / "report.json")
    return {"evaluation_cases": len(cases), "evaluation_prompt_rows": len(prompts), "evaluation_label_rows": len(labels),
            "evaluation_split_counts": dict(Counter(r["split"] for r in cases)),
            "indivisible_group_count": len(group_split), "train_only_source_chains": len(train_only_ids),
            "train_only_source_chains_disjoint_from_all_evaluation_candidates": True,
            "exact_english_multi_hop_prompt_overlap_count": 0,
            "evaluation_translation_screen": {
                "status": review_manifest.get("status"),
                "mode": "AI_ONLY_CROSS_MODEL_SCREEN",
                "independence_rule": "All three probes in each target language must have review_independence=CROSS_MODEL_SCREEN; same-model self-checks are excluded.",
                "human_translation_reviewed": False,
                "review_report_sha256": sha(review_root / "report.json"),
                "translation_sha256_by_language": {
                    lang: review_manifest["languages"][lang]["translation_sha256"] for lang in TARGET_LANGS
                },
                "review_sha256_by_language": {
                    lang: review_manifest["languages"][lang]["review_sha256"] for lang in TARGET_LANGS
                },
                "screened_cohort_counts": verified_summary,
            }}


def main() -> None:
    translations = checked_translation_reviews()
    corpus, counts = build_corpus(translations)
    eval_validation = validate_split_and_evaluation(corpus)

    # Audit source file integrity against the release-preparation report.
    prep = read_json(OUT / "PREPARATION_REPORT.json")
    if prep.get("do_not_train") is not True or prep.get("approval_status") != "NOT_APPROVED_FOR_TRAINING":
        raise ValueError("Expected a quarantined preparation report; refusing to overwrite an approved release")
    expected_case_counts = {"forget": 117, "retain": 39, "holdout": 39}
    cases_draft = read_jsonl(V51 / "candidate_cases.draft.jsonl")
    if dict(Counter(r["split"] for r in cases_draft)) != expected_case_counts:
        raise ValueError("Draft split counts differ from the validated 117/39/39 pilot design")
    if prep.get("composition_plan", {}).get("selected_chains") != 44:
        raise ValueError("Expected 44 source/path-verified composition chains")

    TRAINING.mkdir(parents=True, exist_ok=True)
    corpus_path = TRAINING / "corpus.jsonl"
    write_jsonl(corpus_path, corpus)
    review_report_path = REVIEW_DIR / "report.json"
    split_record_path = V51 / "split_approval.json"
    split_record = {
        "status": "PILOT_SPLIT_PIPELINE_VALIDATED",
        "reviewer_type": "AI_PIPELINE",
        "reviewer_name": "Automated fail-closed validation; no human translation reviewer",
        "review_date": date.today().isoformat(),
        "scope": "v5.1 feasibility pilot only; not the R1 main-study split",
        "split_counts": expected_case_counts,
        "entity_group_count": len({r["group_id"] for r in cases_draft}),
        "candidate_cases_sha256": sha(V51 / "candidate_cases.draft.jsonl"),
        "translation_review_report_sha256": sha(review_report_path),
        "validation_protocol": "assemble_training_release.py fail-closed validation",
        "validation_passed": True,
        "human_approval_claimed": False,
    }
    write_json(split_record_path, split_record)
    manifest = {
        "status": "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS",
        "approval_status": "PILOT_RELEASE_GATE_PASS_AI_PIPELINE_VALIDATED",
        "do_not_train": False,
        "pilot_only": True,
        "training_variant": "composition_sft",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "languages": list(LANGS),
        "translation_review_mode": "AI_ONLY",
        "translation_review_protocol": "TranslateGemma back-translation + Qwen2.5 7B primary semantic/fluency screen + Qwen2.5 14B second-pass semantic/fluency screen of every row; deterministic entity/number/script checks",
        "translation_review_authorization": "AI_ONLY_REVIEW_AUTHORIZED",
        "human_translation_reviewed": False,
        "human_fluency_review_performed": False,
        "review_limitations": ["AI-only translation review is not a fluent-human language audit.", "The Qwen2.5 7B and 14B reviewers share a model family, so their errors may be correlated.", "Automated translation screening can miss subtle semantic, pragmatic, or dialect errors.", "Pilot split validation is not a substitute for independent research supervisor approval."],
        "review_models": {"backtranslation": "ollama:translategemma:4b", "primary_semantic_and_fluency": "ollama:qwen2.5:7b-instruct", "flag_adjudicator": "ollama:qwen2.5:14b"},
        "ai_reviewed_rows": counts["unique_sft_ids_per_language"] * len(TARGET_LANGS),
        **counts,
        "holdout_rows": 0,
        "rows_by_split": {split: sum(row["split"] == split for row in corpus) for split in SPLITS},
        "sft_objective": "Causal-language modeling with user prompt loss masked; assistant answer tokens only.",
        "answer_policy": "Use the canonical fictional entity/year answer unchanged in every language; entity names are language-independent labels.",
        "split_approval": {**split_record, "record_sha256": sha(split_record_path)},
        "evaluation": {
            "cases_path": "../training_release_draft/evaluation/cases.jsonl",
            "prompts_path": "../training_release_draft/evaluation/prompts.jsonl",
            "labels_path": "../training_release_draft/evaluation/labels.jsonl",
            **eval_validation,
            "prompts_sha256": sha(OUT / "evaluation/prompts.jsonl"),
            "labels_sha256": sha(OUT / "evaluation/labels.jsonl"),
        },
        "translation_review_report_sha256": sha(REVIEW_DIR / "report.json"),
        "translation_review_files_sha256": {lang: sha(REVIEW_DIR / f"{lang}.jsonl") for lang in TARGET_LANGS},
        "adjudication_script_provenance": adjudication_script_provenance(
            read_json(REVIEW_DIR / "report.json").get("adjudication_script_sha256", "")
        ),
        "preparation_report_sha256": sha(OUT / "PREPARATION_REPORT.json"),
        "candidate_cases_sha256": sha(V51 / "candidate_cases.draft.jsonl"),
        "source_manifest_sha256": sha(V51 / "manifest.draft.json"),
        "corpus_sha256": sha(corpus_path),
        "corpus_path": "sft_training/corpus.jsonl",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(TRAINING / "manifest.json", manifest)
    report = {"status": "PILOT_TRAINING_RELEASE_READY", "do_not_train": False,
              "human_translation_reviewed": False, "corpus_sha256": manifest["corpus_sha256"],
              "manifest_sha256": sha(TRAINING / "manifest.json"), "split_record_sha256": sha(split_record_path),
              "corpus_counts": counts, "evaluation_validation": eval_validation,
              "note": "AI-only pilot validation; no human language audit and no claim of main-study approval."}
    write_json(TRAINING / "release_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
