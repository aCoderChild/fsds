#!/usr/bin/env python3
"""Prepare leakage-safe cross-lingual evaluation translation requests.

English questions and gold labels are derived from the v3 source assignments.
Gold labels are written separately from model-facing question prompts. This
script deliberately does not translate or approve any text.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = REPO / "data/synthworlds_sm"
OUT = ROOT / "evaluation_dataset"
LANGUAGES = ("en", "fr", "de", "hi", "th")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


assignments_path = ROOT / "assignments.jsonl"
target_candidates_path = ROOT / "sft_natural_en_candidate.jsonl"
qa_path = SOURCE / "qa-sm.jsonl"
assignments = read_jsonl(assignments_path)
candidates = read_jsonl(target_candidates_path)
qa = {row["instance_id"]: row for row in read_jsonl(qa_path)}
candidate_by_fact = {row["fact_id"]: row for row in candidates}
assert len(assignments) == 129
assert len({row["source_qa_id"] for row in assignments}) == 129
assert len(candidate_by_fact) == len(candidates)

prompts_en, labels, request_rows = [], [], {lang: [] for lang in LANGUAGES[1:]}
shortcut_counts = {"final_answer_in_multihop_question": 0, "intermediate_answer_in_multihop_question": 0}
for assignment in assignments:
    iid = assignment["source_qa_id"]
    source_qa = qa[iid]
    target = candidate_by_fact.get(assignment["target_fact_id"])
    mhop = json.loads(source_qa["mhop_qa"])
    graph = mhop["graph_qa"]
    decomposed = mhop["decomposed_questions"]
    protected_entities = sorted({
        value.rsplit(" (Q", 1)[0]
        for value in (graph.get("n1", ""), graph.get("n2", ""), graph.get("n3", ""))
        if value
    } | {
        value.get("new_name", "")
        for value in mhop.get("entity_rename_map", {}).values()
        if value.get("new_name")
    })
    target_question = target["question_en"] if target else decomposed["Q1"]
    target_answer = target["answer"] if target else graph["n2"].rsplit(" (Q", 1)[0]
    answer_values = assignment.get("gold_answers") or source_qa.get("gold_answers") or []
    assert answer_values and source_qa.get("query") and target_question and target_answer
    normalized_multi = normalize(source_qa["query"])
    if any(re.search(r"(?<!\w)" + re.escape(normalize(ans)) + r"(?!\w)", normalized_multi) for ans in answer_values if normalize(ans)):
        shortcut_counts["final_answer_in_multihop_question"] += 1
    normalized_intermediate = normalize(target_answer)
    if len(normalized_intermediate) > 2 and re.search(r"(?<!\w)" + re.escape(normalized_intermediate) + r"(?!\w)", normalized_multi):
        shortcut_counts["intermediate_answer_in_multihop_question"] += 1
    if target:
        assert normalize(target["answer"]) == normalize(graph["n2"].rsplit(" (Q", 1)[0]), f"Intermediate target mismatch for {iid}"
    for probe, question in (("direct", target_question), ("multi", source_qa["query"])):
        prompt = {
            "instance_id": iid,
            "probe": probe,
            "split": assignment["split"],
            "language": "en",
            "question": question,
            "evaluation_condition": "question_only_no_supporting_passage",
            "do_not_train": True,
        }
        prompts_en.append(prompt)
        for lang in request_rows:
            request_rows[lang].append({
                "translation_id": f"{iid}:{probe}:{lang}",
                "instance_id": iid,
                "probe": probe,
                "split": assignment["split"],
                "source_language": "en",
                "target_language": lang,
                "source_question": question,
                "protected_entity_names": protected_entities,
                "instructions": (
                    "Translate only the question into the target language. Preserve proper names, "
                    "invented entity names, dates, and quoted titles exactly. Do not answer the question, "
                    "add explanations, or include any gold answer. Return only the translated question."
                ),
                "do_not_train": True,
                "review_status": "PENDING_TRANSLATION_AND_AI_REVIEW",
                "human_review_status": "NOT_PERFORMED",
            })
    labels.append({
        "instance_id": iid,
        "split": assignment["split"],
        "bootstrap_cluster_id": assignment["entity_fact_group_id"],
        "target_fact_id": assignment["target_fact_id"],
        "chain_fact_ids": assignment["chain_fact_ids"],
        "direct_intermediate_answer": target_answer,
        "final_answers": answer_values,
        "label_file_only_never_send_to_model": True,
    })

write_jsonl(OUT / "prompts" / "en.jsonl", prompts_en)
write_jsonl(OUT / "labels" / "gold_labels.jsonl", labels)
for lang, rows in request_rows.items():
    write_jsonl(OUT / "translation_requests" / f"{lang}.jsonl", rows)

manifest = {
    "status": "ENGLISH_EVAL_SET_AND_TRANSLATION_REQUESTS_READY_TRANSLATIONS_PENDING",
    "source_assignments": len(assignments),
    "source_sha256": {
        "assignments": sha(assignments_path),
        "target_candidates": sha(target_candidates_path),
        "qa": sha(qa_path),
    },
    "evaluation_languages": list(LANGUAGES),
    "split_counts": {s: sum(row["split"] == s for row in assignments) for s in ("forget", "retain", "holdout")},
    "question_prompts_en": len(prompts_en),
    "gold_label_rows": len(labels),
    "translation_requests_by_language": {lang: len(rows) for lang, rows in request_rows.items()},
    "translation_status": "NOT_TRANSLATED",
    "ai_review_status": "NOT_STARTED",
    "human_review_status": "NOT_PERFORMED",
    "label_separation": "Gold labels are separate; translation requests contain no gold answers or supporting passages.",
    "exact_answer_shortcut_screen": shortcut_counts,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
assert len(prompts_en) == 258 and len(labels) == 129
assert all(len(rows) == 258 for rows in request_rows.values())
assert shortcut_counts == {"final_answer_in_multihop_question": 0, "intermediate_answer_in_multihop_question": 0}
print(json.dumps(manifest, indent=2, ensure_ascii=False))
