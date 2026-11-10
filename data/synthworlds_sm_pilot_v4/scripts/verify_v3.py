#!/usr/bin/env python3
"""Read-only checks for the v3 proposal and its source/output manifests."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SOURCE = DATA / "synthworlds_sm"
OUT = DATA / "synthworlds_sm_pilot_v3"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify():
    report = json.loads((OUT / "audit_report.json").read_text(encoding="utf-8"))
    source = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    for key, name in (("qa", "qa-sm.jsonl"), ("documents", "qa-sm-docs.jsonl"), ("facts", "qa-sm-facts.jsonl")):
        path = SOURCE / name
        assert sha(path) == source[key]["sha256"]
        assert path.stat().st_size == source[key]["bytes"]

    assignments = jsonl(OUT / "assignments.jsonl")
    by_id = {r["source_qa_id"]: r for r in assignments}
    assert len(by_id) == len(assignments) == 129
    assert collections.Counter(r["split"] for r in assignments) == {"forget": 77, "retain": 26, "holdout": 26}
    groups = jsonl(OUT / "entity_fact_groups.jsonl")
    assert len(groups) == 101
    assert collections.Counter(r["size"] for r in groups) == {1: 84, 2: 12, 3: 3, 6: 2}
    assert all(all(by_id[i]["entity_fact_group_id"] == g["entity_fact_group_id"] and by_id[i]["split"] == g["split"] for i in g["instance_ids"]) for g in groups)

    pairs = jsonl(OUT / "overlap_pairs.jsonl")
    assert sum(bool(r["was_cross_split_before"]) for r in pairs) == report["previous_cross_split_overlap_pairs_resolved"] == 9
    for r in pairs:
        assert by_id[r["qa_1"]]["split"] == by_id[r["qa_2"]]["split"]
    assert report["remaining_cross_split_entity_pairs"] == 0
    assert report["remaining_cross_split_chain_fact_pairs"] == 0
    assert report["remaining_cross_split_exact_answer_pairs"] == 0
    assert {r["source_qa_id"] for r in assignments}.isdisjoint({"0080-qa-sm"})

    sft = jsonl(OUT / "sft_natural_en_candidate.jsonl")
    assert len(sft) == 205
    assert all(r["split"] in ("forget", "retain") for r in sft)
    assert collections.Counter(r["split"] for r in sft) == {"forget": 153, "retain": 52}
    source_examples = [example for r in sft for example in r["source_examples"]]
    assert len(source_examples) == 206
    assert collections.Counter(r["hop"] for r in source_examples) == {"r1": 103, "r2": 103}
    assert len({example["sft_id"] for example in source_examples}) == 206
    assert sum(len(r["source_examples"]) > 1 for r in sft) == 1
    assert all(len(set(r["source_qa_ids"])) == len(r["source_qa_ids"]) for r in sft)
    assert all(r["source_qa_id"] in by_id and r["split"] == by_id[r["source_qa_id"]]["split"] for r in sft)
    assert all(r["messages"] == [{"role": "user", "content": r["question_en"]}, {"role": "assistant", "content": r["answer"]}] for r in sft)
    source_qa = {r["instance_id"]: r for r in jsonl(SOURCE / "qa-sm.jsonl")}
    expected_sft_ids = {f"{iid}:{hop}" for iid, assignment in by_id.items() if assignment["split"] != "holdout" for hop in ("r1", "r2")}
    assert {example["sft_id"] for example in source_examples} == expected_sft_ids
    for r in sft:
        graph = json.loads(source_qa[r["source_qa_id"]]["mhop_qa"])
        graph_qa = graph["graph_qa"]
        node = graph_qa["n2" if r["hop"] == "r1" else "n3"]
        expected_answer = re.sub(r" \(Q\d+\)$", "", node)
        expected_question = graph["decomposed_questions"]["Q1" if r["hop"] == "r1" else "Q2"]
        if r["hop"] == "r2":
            expected_question = re.sub(r"</?bridge>", "", expected_question)
        chain = by_id[r["source_qa_id"]]["chain_fact_ids"]
        assert r["answer"] == expected_answer
        assert r["question_en"] == expected_question
        assert r["fact_id"] == chain[0 if r["hop"] == "r1" else 1]
        assert r["target_fact_id"] == chain[0]
        assert not ({"query", "gold_answers", "gold_docs", "mhop_qa"} & set(r))
        assert len(r["source_examples"]) == len(r["source_sft_ids"])
        for example in r["source_examples"]:
            source_assignment = by_id[example["source_qa_id"]]
            source_graph = json.loads(source_qa[example["source_qa_id"]]["mhop_qa"])
            expected_source_fact = source_assignment["chain_fact_ids"][0 if example["hop"] == "r1" else 1]
            assert source_assignment["split"] == r["split"]
            assert example["target_fact_id"] == source_assignment["target_fact_id"]
            assert r["fact_id"] == expected_source_fact
            source_question = source_graph["decomposed_questions"]["Q1" if example["hop"] == "r1" else "Q2"]
            if example["hop"] == "r2":
                source_question = re.sub(r"</?bridge>", "", source_question)
            source_answer = re.sub(r" \(Q\d+\)$", "", source_graph["graph_qa"]["n2" if example["hop"] == "r1" else "n3"])
            assert (r["question_en"], r["answer"]) == (source_question, source_answer)

    pilot = jsonl(OUT / "balanced_translation_pilot_selection.jsonl")
    assert len(pilot) == 3
    assert {r["split"] for r in pilot} == {"forget", "retain", "holdout"}
    assert all(by_id[r["source_qa_id"]]["split"] == r["split"] for r in pilot)

    translation_report = json.loads((OUT / "translation_pilot" / "report.json").read_text(encoding="utf-8"))
    assert translation_report["status"] == "DRAFT_TRANSLATIONS_PENDING_FLUENT_REVIEW_NOT_APPROVED"
    assert translation_report["human_review_status"] == "PENDING_FLUENT_REVIEW"
    assert report["pilot_translation_status"] == translation_report["status"]
    assert translation_report["holdout_sft_candidate_count"] == 0
    for lang in ("fr", "de", "hi", "th"):
        base = OUT / "translation_pilot" / lang
        qa_lang = jsonl(base / "qa.jsonl")
        prompts_lang = jsonl(base / "evaluation_prompts.jsonl")
        labels_lang = jsonl(base / "evaluation_labels.jsonl")
        docs_lang = jsonl(base / "documents_audit_only.jsonl")
        facts_lang = jsonl(base / "chain_facts_audit_only.jsonl")
        sft_lang = jsonl(base / "sft_questions_candidate.jsonl")
        review_lang = jsonl(base / "review_queue.jsonl")
        assert (len(qa_lang), len(prompts_lang), len(labels_lang), len(docs_lang), len(facts_lang), len(sft_lang), len(review_lang)) == (3, 3, 3, 6, 6, 4, 21)
        assert all(r["translation_status"] == "DRAFT_PENDING_FLUENT_REVIEW" and r["human_review_status"] == "PENDING" for r in qa_lang + prompts_lang + docs_lang + facts_lang)
        assert all(not ({"gold_answers", "gold_docs", "mhop_qa"} & set(r)) and r["do_not_train"] for r in qa_lang)
        assert all(set(r) <= {"instance_id", "split", "language", "question", "evaluation_condition", "translation_status", "human_review_status", "do_not_train"} for r in prompts_lang)
        assert all(r["label_file_only_never_send_to_model"] for r in labels_lang)
        assert all(r["audit_only"] and not r["training_eligible"] and not r["evaluation_context_eligible"] for r in docs_lang)
        assert all(r["audit_only"] and not r["training_eligible"] for r in facts_lang)
        assert all(r["do_not_train"] and r["split"] in ("forget", "retain") for r in sft_lang)
        assert all(r["split"] != "holdout" for r in sft_lang)
        assert all(r["review_status"] == "PENDING" and r["reviewer_name"] == "" for r in review_lang)
        assert sum(r["item_type"] == "question" for r in review_lang) == 9
        assert sum(r["item_type"] == "supporting_document_audit_only" for r in review_lang) == 6
        assert sum(r["item_type"] == "verified_chain_fact_audit_only" for r in review_lang) == 6

    passage_report = json.loads((OUT / "passage_leakage_audit" / "report.json").read_text(encoding="utf-8"))
    pilot_risks = jsonl(OUT / "passage_leakage_audit" / "pilot_document_risks.jsonl")
    all_doc_risks = jsonl(OUT / "passage_leakage_audit" / "all_split_document_risks.jsonl")
    assert passage_report["approved_training_documents"] == 0
    assert len(pilot_risks) == passage_report["pilot_documents"] == 6
    assert sum(bool(r["other_split_risks"]) for r in pilot_risks) == passage_report["pilot_documents_with_other_split_risk"] == 1
    assert len(all_doc_risks) == passage_report["all_split_gold_documents"] == 249
    assert all(r["training_disposition"] == "DO_NOT_TRAIN_ON_FULL_PASSAGE" for r in pilot_risks + all_doc_risks)
    assert report["passage_audit_status"] == passage_report["status"]

    candidate_data = {}
    pending_by_language = {"en": 0}
    for lang in ("fr", "de", "hi", "th"):
        rows = jsonl(OUT / "sft_translation_candidates" / f"{lang}.jsonl")
        candidate_data[lang] = rows
        pending_by_language[lang] = sum(
            r.get("translation_status") != "APPROVED_BY_AI_REVIEW"
            or r.get("review_status") != "APPROVED_BY_AI_REVIEW"
            or r.get("review_mode") != "AI_ONLY"
            or r.get("reviewer_type") != "AI"
            or r.get("human_review_status") != "NOT_PERFORMED"
            or not str(r.get("reviewer_name", "")).strip()
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.get("review_date", "")))
            for r in rows
        )
    approval_path = OUT / "split_approval.json"
    split_approval = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.is_file() else {}
    split_approved = (
        split_approval.get("status") == "APPROVED_PILOT_SPLIT"
        and bool(str(split_approval.get("reviewer_name", "")).strip())
        and split_approval.get("reviewer_type") == "AI"
        and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(split_approval.get("review_date", ""))))
    )
    corpus_path = OUT / "sft_training" / "corpus.jsonl"
    corpus_manifest_path = OUT / "sft_training" / "manifest.json"
    corpus_ready = False
    if corpus_path.is_file() and corpus_manifest_path.is_file():
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        corpus_ready = (
            corpus_manifest.get("status") == "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS"
            and corpus_manifest.get("do_not_train") is False
            and corpus_manifest.get("translation_review_mode") == "AI_ONLY"
            and corpus_manifest.get("human_translation_reviewed") is False
            and corpus_manifest.get("split_approval", {}).get("reviewer_type") == "AI"
        )

    sft_readiness = json.loads((OUT / "sft_stage_readiness.json").read_text(encoding="utf-8"))
    assert sft_readiness["status"] in {"SFT_PREFLIGHT_BLOCKED_NO_TRAINING_STARTED", "SFT_DATA_READY_KAGGLE_RUNTIME_CHECK_REQUIRED"}
    assert sft_readiness["training_started"] is False
    assert sft_readiness["sft_unique_english_messages"] == 205
    assert sft_readiness["sft_candidate_rows_by_language"] == {"en": 205, "fr": 205, "de": 205, "hi": 205, "th": 205}
    assert sft_readiness["non_english_rows_missing_per_language"] == {"fr": 0, "de": 0, "hi": 0, "th": 0}
    assert sft_readiness["translation_reviews_pending_by_language"] == pending_by_language
    assert sft_readiness["gates"]["complete_five_language_sft_coverage"] is True
    assert sft_readiness["gates"]["native_translation_review_complete"] is False
    assert sft_readiness["gates"]["split_structure_verified"] is True
    assert sft_readiness["gates"]["split_proposal_approved"] is split_approved
    assert sft_readiness["gates"]["approved_corpus_built"] is corpus_ready
    assert sft_readiness["gates"]["native_translation_review_complete"] is False
    assert sft_readiness["gates"]["ai_only_review_authorized"] is True
    assert sft_readiness["gates"]["ai_translation_review_complete"] is (all(v == 0 for k, v in pending_by_language.items() if k != "en"))
    assert sft_readiness["gates"]["translation_review_complete_for_selected_mode"] is (all(v == 0 for k, v in pending_by_language.items() if k != "en"))
    assert "Kaggle runtime" in sft_readiness["available_environment"]["note"]

    mt_report = json.loads((OUT / "sft_translation_machine_drafts" / "report.json").read_text(encoding="utf-8"))
    assert mt_report["status"] == "MACHINE_TRANSLATION_DRAFTS_PENDING_FLUENT_REVIEW"
    assert mt_report["approval_claimed"] is False
    assert mt_report["model_id"] == "facebook/nllb-200-distilled-600M"
    assert set(mt_report["languages"]) == {"fr", "de", "hi", "th"}
    for lang, meta in mt_report["languages"].items():
        assert meta["records"] == 205 and meta["batches"] == 9 and meta["status"] == "PENDING_FLUENT_REVIEW"
    english_sft = {r["sft_id"]: r for r in sft}
    for lang in ("fr", "de", "hi", "th"):
        candidate = candidate_data[lang]
        assert {r["sft_id"] for r in candidate} == set(english_sft)
        for r in candidate:
            status_pair = (r.get("translation_status"), r.get("review_status"))
            assert status_pair in {
                ("PENDING_FLUENT_REVIEW", "PENDING"),
                ("APPROVED_BY_AI_REVIEW", "APPROVED_BY_AI_REVIEW"),
            }
            if status_pair[0] == "APPROVED_BY_AI_REVIEW":
                assert str(r.get("reviewer_name", "")).strip()
                assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.get("review_date", "")))
                assert r.get("review_mode") == "AI_ONLY" and r.get("reviewer_type") == "AI"
                assert r.get("human_review_status") == "NOT_PERFORMED"
            assert r["do_not_train"]
        assert all(r["translation_method"] == "machine_translation_draft" and r["translation_model"] == "facebook/nllb-200-distilled-600M" for r in candidate)
        request_by_id = {}
        for request_file in sorted((OUT / "sft_translation_requests" / lang).glob("batch_*.jsonl")):
            request_by_id.update({x["sft_id"]: x for x in jsonl(request_file)})
        for r in candidate:
            target_text = r["question"] + " " + r["answer_translation"]
            request = request_by_id[r["sft_id"]]
            assert all(name in target_text for name in request["protected_entity_names"])
            numbers = re.findall(r"\d+(?:[-/.]\d+)*", request["question_en"] + " " + request["answer_en"])
            assert all(number in target_text for number in numbers)
            assert not re.search(r"<ENTITY_[A-Z]+>|ZXQENT", target_text)
            assert r["split"] in ("forget", "retain")

    request_manifest = json.loads((OUT / "sft_translation_requests" / "manifest.json").read_text(encoding="utf-8"))
    expected_sft_ids = {r["sft_id"] for r in sft}
    assert request_manifest["do_not_train_until_complete_and_reviewed"] is True
    assert request_manifest["source_question_set_sha256"]
    for lang in ("fr", "de", "hi", "th"):
        meta = request_manifest["languages"][lang]
        assert meta["translation_rows_needed_or_requiring_review"] == 205
        ids = []
        for batch in meta["batches"]:
            rows = jsonl(OUT / batch["path"])
            assert len(rows) == batch["rows"]
            assert all(r["translate_to"] == lang and r["question_translation"] == "" and r["answer_translation"] == "" for r in rows)
            ids.extend(r["sft_id"] for r in rows)
        assert len(ids) == len(set(ids)) == 205 and set(ids) == expected_sft_ids
    build_status = json.loads((OUT / "sft_training" / "build_status.json").read_text(encoding="utf-8"))
    assert build_status["status"] in {"SFT_CORPUS_BLOCKED", "SFT_CORPUS_READY"}
    assert build_status["do_not_train"] is (build_status["status"] != "SFT_CORPUS_READY")

    composition_path = OUT / "sft_training" / "composition_corpus.jsonl"
    composition_manifest_path = OUT / "sft_training" / "composition_manifest.json"
    if composition_path.is_file() or composition_manifest_path.is_file():
        assert composition_path.is_file() and composition_manifest_path.is_file()
        composition_manifest = json.loads(composition_manifest_path.read_text(encoding="utf-8"))
        composition_rows = jsonl(composition_path)
        composition_examples = [r for r in composition_rows if r.get("example_type") == "retained_multihop_composition"]
        assert composition_manifest["status"] == "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS"
        assert composition_manifest["do_not_train"] is False
        assert composition_manifest["corpus_sha256"] == sha(composition_path)
        assert composition_manifest["total_rows"] == len(composition_rows) == 1155
        assert composition_manifest["rows_by_language"] == {lang: 231 for lang in ("en", "fr", "de", "hi", "th")}
        assert len(composition_examples) == 130
        assert all(r.get("split") == "retain" and r.get("hop") == "composition_2hop" for r in composition_examples)
        assert composition_manifest["holdout_rows"] == composition_manifest["forget_composition_rows"] == 0
        assert composition_manifest["retain_composition_rows"] == 130
        assert composition_manifest["human_translation_reviewed"] is False
        assert len({(r["language"], r["sft_id"]) for r in composition_rows}) == len(composition_rows)

    manifest = json.loads((OUT / "output_manifest.json").read_text(encoding="utf-8"))
    listed = manifest["files"]
    for rel, meta in listed.items():
        path = OUT / rel
        assert path.is_file(), rel
        assert sha(path) == meta["sha256"], rel
    # output_manifest.json is a snapshot from the v3 proposal build. Later SFT,
    # evaluation, Kaggle-package, and output artifacts live beside that snapshot
    # and carry their own manifests. Verify every listed build artifact, but do
    # not misclassify those later, separately manifested files as tampering.
    sums = {}
    for line in (OUT / "sha256sums.txt").read_text(encoding="utf-8").splitlines():
        value, rel = line.split("  ", 1)
        sums[rel] = value
    assert sums == {rel: meta["sha256"] for rel, meta in listed.items()}
    return {
        "verification": "PASS",
        "source_hashes": "PASS",
        "split_counts": report["split_counts"],
        "group_count": len(groups),
        "cross_split_overlaps": 0,
        "sft_records": len(sft),
        "translation_pilot_cases": len(pilot),
        "translation_drafts_per_language": translation_report["counts_per_language"],
        "passage_audit": {
            "pilot_documents_with_detected_risk": passage_report["pilot_documents_with_other_split_risk"],
            "all_split_documents_with_detected_risk": passage_report["all_split_documents_with_detected_other_split_risk"],
            "training_documents_approved": passage_report["approved_training_documents"],
        },
        "training_approval": "AI_TECHNICALLY_APPROVED_PILOT_SPLIT_NOT_RESEARCHER_SIGNOFF" if split_approved else "PENDING_PILOT_SPLIT_REVIEW",
        "sft_readiness": sft_readiness["status"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
