#!/usr/bin/env python3
"""Validate and assemble the multilingual draft pilot; never marks text reviewed."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SOURCE = DATA / "synthworlds_sm"
OUT = DATA / "synthworlds_sm_pilot_v3"
PILOT = OUT / "translation_pilot"
LANGS = ("fr", "de", "hi", "th")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def numeric_tokens(text):
    return collections.Counter(re.findall(r"\d{4}-\d{2}-\d{2}|\d+", text))


def main():
    payload = read_json(OUT / "translation_payloads" / "pilot_draft_0001.json")
    selections = read_jsonl(OUT / "balanced_translation_pilot_selection.jsonl")
    assignment = {r["source_qa_id"]: r for r in read_jsonl(OUT / "assignments.jsonl")}
    qa = {r["instance_id"]: r for r in read_jsonl(SOURCE / "qa-sm.jsonl")}
    docs = {r["qid"]: r for r in read_jsonl(SOURCE / "qa-sm-docs.jsonl")}
    eligibility = {r["instance_id"]: r for r in read_jsonl(DATA / "synthworlds_sm_reaudit" / "type_a_target_eligibility.jsonl")}
    selected = {r["source_qa_id"]: r for r in selections}
    assert set(selected) == {"0077-qa-sm", "0055-qa-sm", "0110-qa-sm"}
    qids = sorted({qid for iid in selected for qid in qa[iid]["gold_qids"]})
    expected_facts = {f"{iid}:{hop}" for iid in selected for hop in ("r1", "r2")}
    assert set(payload["languages"]) == set(LANGS)
    output_counts = {}

    for lang in LANGS:
        data = payload["languages"][lang]
        assert set(data["qa"]) == set(selected)
        assert set(data["documents"]) == set(qids)
        assert set(data["chain_facts"]) == expected_facts
        qa_rows, prompt_rows, label_rows, doc_rows, fact_rows, sft_rows, review_rows = [], [], [], [], [], [], []

        for iid, tr in data["qa"].items():
            original = qa[iid]
            graph = json.loads(original["mhop_qa"])
            assert set(tr) == {"query", "Q1", "Q2"}
            for key, source_text in (("query", original["query"]), ("Q1", graph["decomposed_questions"]["Q1"]), ("Q2", graph["decomposed_questions"]["Q2"])):
                translated = tr[key]
                assert isinstance(translated, str) and translated.strip()
                assert numeric_tokens(source_text) == numeric_tokens(translated), f"numeric token mismatch: {lang}/{iid}/{key}"
                if "<bridge>" in source_text:
                    bridge = re.findall(r"<bridge>(.*?)</bridge>", source_text)
                    assert re.findall(r"<bridge>(.*?)</bridge>", translated) == bridge, f"bridge mismatch: {lang}/{iid}/{key}"
                for entry in graph.get("entity_rename_map", {}).values():
                    name = entry.get("new_name", "")
                    if name and name in source_text:
                        assert name in translated, f"protected name missing: {lang}/{iid}/{key}/{name}"
                graph_qa = graph["graph_qa"]
                expected_answer = original["gold_answers"] if key == "query" else [re.sub(r" \(Q\d+\)$", "", graph_qa["n2" if key == "Q1" else "n3"])]
                review_rows.append({
                    "item_id": f"{iid}:{key}",
                    "item_type": "question",
                    "source_qa_id": iid,
                    "split": assignment[iid]["split"],
                    "source_en": source_text,
                    "draft_translation": translated,
                    "expected_answer_reference_only": expected_answer,
                    "review_status": "PENDING",
                    "reviewer_name": "",
                    "review_date": "",
                    "semantic_equivalence": "",
                    "language_quality": "",
                    "entity_and_bridge_preservation": "",
                    "numbers_dates_qualifiers": "",
                    "correction": "",
                    "reviewer_notes": "",
                })
            split = assignment[iid]["split"]
            q1_en, q2_en = graph["decomposed_questions"]["Q1"], graph["decomposed_questions"]["Q2"]
            qa_rows.append({
                "instance_id": iid,
                "split": split,
                "query_en": original["query"],
                f"query_{lang}": tr["query"],
                "decomposed_questions_en": {"Q1": q1_en, "Q2": q2_en},
                f"decomposed_questions_{lang}": {"Q1": tr["Q1"], "Q2": tr["Q2"]},
                "gold_doc_qids": original["gold_qids"],
                "chain_fact_ids": assignment[iid]["chain_fact_ids"],
                "translation_status": "DRAFT_PENDING_FLUENT_REVIEW",
                "human_review_status": "PENDING",
                "do_not_train": True,
            })
            prompt_rows.append({
                "instance_id": iid,
                "split": split,
                "language": lang,
                "question": tr["query"],
                "evaluation_condition": "question_only_no_supporting_passage",
                "translation_status": "DRAFT_PENDING_FLUENT_REVIEW",
                "human_review_status": "PENDING",
                "do_not_train": True,
            })
            graph_qa = graph["graph_qa"]
            label_rows.append({
                "instance_id": iid,
                "split": split,
                "language": lang,
                "direct_intermediate_answer": re.sub(r" \(Q\d+\)$", "", graph_qa["n2"]),
                "final_answer": original["gold_answers"],
                "target_fact_id": assignment[iid]["target_fact_id"],
                "chain_fact_ids": assignment[iid]["chain_fact_ids"],
                "label_file_only_never_send_to_model": True,
            })
            for hop, key in (("r1", "Q1"), ("r2", "Q2")):
                if assignment[iid]["split"] == "holdout":
                    continue
                question = tr[key]
                if hop == "r2":
                    question = re.sub(r"</?bridge>", "", question)
                sft_rows.append({
                    "sft_id": f"{iid}:{hop}",
                    "source_qa_id": iid,
                    "split": assignment[iid]["split"],
                    "hop": hop,
                    "question": question,
                    "answer": json.loads(original["mhop_qa"])["graph_qa"]["n2" if hop == "r1" else "n3"].rsplit(" (Q", 1)[0],
                    "language": lang,
                    "translation_status": "DRAFT_PENDING_FLUENT_REVIEW",
                    "status": "CANDIDATE_NOT_APPROVED",
                    "do_not_train": True,
                })

        for qid in qids:
            translated_doc = data["documents"][qid]
            original_doc = docs[qid]["doc"]
            assert isinstance(translated_doc, str) and translated_doc.strip()
            assert numeric_tokens(original_doc) == numeric_tokens(translated_doc), f"document number/date mismatch: {lang}/{qid}"
            doc_rows.append({
                "qid": qid,
                "doc_en": original_doc,
                f"doc_{lang}": translated_doc,
                "translation_status": "DRAFT_PENDING_FLUENT_REVIEW",
                "human_review_status": "PENDING",
                "training_eligible": False,
                "evaluation_context_eligible": False,
                "audit_only": True,
            })
            review_rows.append({
                "item_id": f"document:{qid}",
                "item_type": "supporting_document_audit_only",
                "source_qa_ids": sorted(iid for iid in selected if qid in qa[iid]["gold_qids"]),
                "source_en": original_doc,
                "draft_translation": translated_doc,
                "review_status": "PENDING",
                "reviewer_name": "",
                "review_date": "",
                "semantic_equivalence": "",
                "language_quality": "",
                "entity_and_relation_preservation": "",
                "numbers_dates_qualifiers": "",
                "correction": "",
                "reviewer_notes": "",
                "do_not_use_as_model_input": True,
            })

        for iid_hop in sorted(expected_facts):
            iid, hop = iid_hop.rsplit(":", 1)
            evidence = eligibility[iid]
            source_fact = evidence["target_fact_text"] if hop == "r1" else evidence["r2_matched_fact_texts"][0]
            translated_fact = data["chain_facts"][iid_hop]
            src_parts = [p.strip() for p in source_fact.split("→")]
            tr_parts = [p.strip() for p in translated_fact.split("→")]
            assert len(src_parts) == len(tr_parts) == 3, f"fact structure mismatch: {lang}/{iid_hop}"
            assert src_parts[0] == tr_parts[0] and src_parts[2] == tr_parts[2], f"fact endpoint changed: {lang}/{iid_hop}"
            assert numeric_tokens(source_fact) == numeric_tokens(translated_fact), f"fact number mismatch: {lang}/{iid_hop}"
            fact_rows.append({
                "source_qa_id": iid,
                "split": assignment[iid]["split"],
                "hop": hop,
                "fact_id": evidence["chain_fact_ids"][0 if hop == "r1" else 1],
                "fact_en": source_fact,
                f"fact_{lang}": translated_fact,
                "translation_status": "DRAFT_PENDING_FLUENT_REVIEW",
                "human_review_status": "PENDING",
                "training_eligible": False,
                "audit_only": True,
            })
            review_rows.append({
                "item_id": iid_hop,
                "item_type": "verified_chain_fact_audit_only",
                "source_qa_id": iid,
                "split": assignment[iid]["split"],
                "source_en": source_fact,
                "draft_translation": translated_fact,
                "review_status": "PENDING",
                "reviewer_name": "",
                "review_date": "",
                "semantic_equivalence": "",
                "entity_qid_and_value_preservation": "",
                "relation_direction_and_qualifier": "",
                "numbers_dates_qualifiers": "",
                "correction": "",
                "reviewer_notes": "",
                "do_not_use_as_model_input": True,
            })

        out = PILOT / lang
        write_jsonl(out / "qa.jsonl", sorted(qa_rows, key=lambda r: r["instance_id"]))
        write_jsonl(out / "evaluation_prompts.jsonl", sorted(prompt_rows, key=lambda r: r["instance_id"]))
        write_jsonl(out / "evaluation_labels.jsonl", sorted(label_rows, key=lambda r: r["instance_id"]))
        write_jsonl(out / "documents_audit_only.jsonl", sorted(doc_rows, key=lambda r: r["qid"]))
        write_jsonl(out / "chain_facts_audit_only.jsonl", fact_rows)
        write_jsonl(out / "sft_questions_candidate.jsonl", sft_rows)
        write_jsonl(out / "review_queue.jsonl", review_rows)
        for stale in (out / "documents.jsonl", out / "verified_chain_facts.jsonl"):
            if stale.exists():
                stale.unlink()
        output_counts[lang] = {
            "qa_review_rows": len(qa_rows),
            "question_only_eval_prompts": len(prompt_rows),
            "evaluation_labels": len(label_rows),
            "documents_audit_only": len(doc_rows),
            "chain_facts_audit_only": len(fact_rows),
            "review_queue_items": len(review_rows),
            "sft_candidate": len(sft_rows),
        }

    report = {
        "status": "DRAFT_TRANSLATIONS_PENDING_FLUENT_REVIEW_NOT_APPROVED",
        "method": payload["method"],
        "human_review_status": "PENDING_FLUENT_REVIEW",
        "languages": list(LANGS),
        "selected_qa_ids_by_split": {r["split"]: r["source_qa_id"] for r in selections},
        "counts_per_language": output_counts,
        "holdout_sft_candidate_count": 0,
        "translated_canonical_passages": len(qids),
        "whole_fact_rows_translated": 0,
        "evaluation_condition": "Question-only; support passages and labels are segregated from model inputs.",
        "note": "Only the two verified chain facts per selected QA are translated. Full passages and chain facts are audit-only; full fact rows are not included because they may expose unrelated examples. Draft text is not human-certified.",
    }
    write_json(PILOT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
