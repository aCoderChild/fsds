#!/usr/bin/env python3
"""Prepare a hash-bound, quarantined v5.1 training/evaluation release draft.

This script derives data only. It never changes the v5.1 draft or starts model
training. It excludes holdout facts from SFT, reuses old translations only when
the English question and answer match exactly, and separates evaluation labels
from prompts.
"""
from __future__ import annotations

import hashlib
import json
import re
import importlib.util
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
V51 = REPO / "experiments/synthworlds_r1"
V4 = REPO / "data/synthworlds_sm_pilot_v4"
SOURCE = REPO / "data/synthworlds_sm"
OUT = V51 / "training_release_draft"
LANGS = ("en", "fr", "de", "hi", "th")
TARGET_LANGS = LANGS[1:]
PROBES = ("direct_positive", "direct_negative_control", "multi_hop")
COMPOSITION_HOPS = {
    "q1": "composition_support_q1",
    "q2": "composition_support_q2",
    "q3": "composition_support_q3",
    "multi": "composition_3hop",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def row_text(row: dict) -> tuple[str, str]:
    messages = row.get("messages", [])
    if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"Malformed two-message SFT row: {row.get('sft_id')}")
    return messages[0].get("content", ""), messages[1].get("content", "")


def graph_data(qa_row: dict) -> tuple[dict, set[str], set[str]]:
    graph = json.loads(qa_row["mhop_qa"])
    graph_qa = graph.get("graph_qa", {})
    qids = set(graph.get("entity_rename_map", {}))
    names: set[str] = set()
    for key, value in graph_qa.items():
        if re.fullmatch(r"n\d+", key):
            match = re.search(r"\((Q\d+)\)$", str(value))
            if match:
                qids.add(match.group(1))
            names.add(re.sub(r"\s*\(Q\d+\)$", "", str(value)).strip())
        elif key.startswith("r") and isinstance(value, dict):
            qids.update(x for x in (value.get("qid"), value.get("relation_subject_qid")) if x)
    for item in graph.get("entity_rename_map", {}).values():
        names.update(x for x in (item.get("old_name"), item.get("new_name")) if x)
    return graph, {x for x in qids if x}, {x for x in names if x}


def expected_atomic(case: dict, qa_row: dict) -> list[dict]:
    graph, _, _ = graph_data(qa_row)
    decomposed = graph.get("decomposed_questions", {})
    q2_tagged = str(decomposed.get("Q2", ""))
    if q2_tagged.count("<bridge>") != 1 or q2_tagged.count("</bridge>") != 1:
        raise ValueError(f"{case['source_qa_id']}: Q2 does not contain exactly one bridge marker")
    bridge_in_q2 = re.search(r"<bridge>(.*?)</bridge>", q2_tagged, flags=re.DOTALL)
    if not bridge_in_q2 or bridge_in_q2.group(1).strip() != case["target_bridge_en"]:
        raise ValueError(f"{case['source_qa_id']}: Q2 bridge differs from the verified R1 target")
    q2 = re.sub(r"</?bridge>", "", q2_tagged).strip()
    rows = [
        {
            "sft_id": f"{case['source_qa_id']}:r1",
            "source_qa_id": case["source_qa_id"], "split": case["split"], "hop": "r1",
            "fact_id": case["target_r1_fact_id"], "fact_text_en": case["target_r1_fact_text"],
            "question_en": case["q1_question_en"], "answer_en": case["target_bridge_en"],
            "protected_entity_names": case["protected_entity_names"],
        },
        {
            "sft_id": f"{case['source_qa_id']}:r2",
            "source_qa_id": case["source_qa_id"], "split": case["split"], "hop": "r2",
            "fact_id": case["target_r2_fact_id"], "fact_text_en": case["target_r2_fact_text"],
            "question_en": q2, "answer_en": case["final_answer_source_supported_en"],
            "protected_entity_names": case["protected_entity_names"],
        },
    ]
    for row in rows:
        if not row["question_en"].strip() or not row["answer_en"].strip():
            raise ValueError(f"Incomplete atomic source row: {row['sft_id']}")
        if norm(row["answer_en"]) in norm(row["question_en"]):
            raise ValueError(f"Atomic prompt discloses its answer: {row['sft_id']}")
    return rows


def approved_eval_cohort() -> tuple[list[dict], dict, dict]:
    request_path = V51 / "translation_requests/en.draft.jsonl"
    draft_root = V51 / "translation_drafts/translategemma_repair_v1_full"
    review_root = V51 / "translation_reviews/translategemma_repair_v1_full"
    review_manifest = read_json(review_root / "report.json")
    if review_manifest.get("status") != "AI_ONLY_REVIEW_COMPLETE_WITH_BLOCKED_TRANSLATIONS":
        raise ValueError("Unexpected full translation-review status")
    if review_manifest.get("source_requests_sha256") != sha(request_path):
        raise ValueError("Evaluation translation review is not bound to the current requests")
    if review_manifest.get("unresolved_total") != 0:
        raise ValueError("Evaluation translation review has unresolved rows")
    requests = {r["prompt_id"]: r for r in read_jsonl(request_path)}
    labels_rows = read_jsonl(V51 / "labels/gold_labels.draft.jsonl")
    labels = {f"{r['source_qa_id']}:{r['probe']}": r for r in labels_rows}
    if len(labels) != len(labels_rows):
        raise ValueError("Duplicate evaluation gold label IDs")
    passed: dict[str, set[str]] = {}
    translations: dict[tuple[str, str], dict] = {}
    for lang in TARGET_LANGS:
        review_path = review_root / f"{lang}.jsonl"
        draft_path = draft_root / f"{lang}.jsonl"
        rows = read_jsonl(review_path)
        drafts = read_jsonl(draft_path)
        review_by_id = {r["prompt_id"]: r for r in rows}
        draft_by_id = {r["prompt_id"]: r for r in drafts}
        if len(review_by_id) != 585 or set(review_by_id) != set(requests) or set(draft_by_id) != set(requests):
            raise ValueError(f"{lang}: translation/review coverage mismatch")
        info = review_manifest.get("languages", {}).get(lang, {})
        if info.get("review_sha256") != sha(review_path) or info.get("translation_sha256") != sha(draft_path):
            raise ValueError(f"{lang}: translation/review hash mismatch")
        passed[lang] = set()
        for prompt_id, review in review_by_id.items():
            draft, request = draft_by_id[prompt_id], requests[prompt_id]
            # A same-model fallback review is not independent corroboration.
            if (review.get("review_status") != "AI_ONLY_REVIEW_PASS"
                    or review.get("review_independence") != "CROSS_MODEL_SCREEN"):
                continue
            if draft.get("translation_status") == "FAILED" or not draft.get("text_translation", "").strip():
                raise ValueError(f"{lang}/{prompt_id}: reviewer passed a failed/empty translation")
            for field in ("source_qa_id", "group_id", "split", "probe"):
                if draft.get(field) != request.get(field) or review.get(field) != request.get(field):
                    raise ValueError(f"{lang}/{prompt_id}: translation provenance mismatch for {field}")
            if not review.get("independent_model_checks", {}).get("blinded_translation_model_pass"):
                raise ValueError(f"{lang}/{prompt_id}: pass lacks independent blinded model check")
            passed[lang].add(prompt_id)
            translations[(lang, prompt_id)] = draft

    case_ids = sorted({r["source_qa_id"] for r in requests.values()})
    split_by_case = {r["source_qa_id"]: r["split"] for r in requests.values()}
    group_by_case = {r["source_qa_id"]: r["group_id"] for r in requests.values()}
    cohort: list[dict] = []
    eval_prompts: list[dict] = []
    eval_labels: list[dict] = []
    for case_id in case_ids:
        ids = [f"{case_id}:{probe}" for probe in PROBES]
        if not all(pid in requests for pid in ids):
            raise ValueError(f"Incomplete evaluation probes for {case_id}")
        if not all(all(pid in passed[lang] for pid in ids) for lang in TARGET_LANGS):
            continue
        cohort.append({"source_qa_id": case_id, "split": split_by_case[case_id], "group_id": group_by_case[case_id]})
        for lang in LANGS:
            for pid in ids:
                request = requests[pid]
                label_row = labels.get(pid)
                if not label_row or label_row.get("source_qa_id") != case_id or label_row.get("split") != request["split"]:
                    raise ValueError(f"Evaluation label mismatch for {pid}")
                text = request["text_en"] if lang == "en" else translations[(lang, pid)]["text_translation"]
                eval_prompts.append({
                    "prompt_id": pid, "source_qa_id": case_id, "split": request["split"],
                    "group_id": request["group_id"], "language": lang, "probe": request["probe"],
                    "prompt": text, "translation_status": "SOURCE_ENGLISH" if lang == "en" else "AI_ONLY_CROSS_MODEL_PASS",
                })
                gold = label_row.get("gold_answers") or [label_row.get("gold_label")]
                eval_labels.append({
                    "prompt_id": pid, "source_qa_id": case_id, "split": label_row["split"],
                    "group_id": label_row["group_id"], "language": lang, "probe": label_row["probe"],
                    "gold_answers": [x for x in gold if isinstance(x, str) and x],
                })
    if len({(r["prompt_id"], r["language"]) for r in eval_prompts}) != len(eval_prompts):
        raise ValueError("Duplicate evaluation prompt rows")
    summary = {}
    for split in ("forget", "retain", "holdout"):
        rows = [r for r in cohort if r["split"] == split]
        summary[split] = {"cases": len(rows), "entity_groups": len({r["group_id"] for r in rows})}
    return cohort, {"prompts": eval_prompts, "labels": eval_labels}, summary


def prepare_atomic(case_rows: list[dict], source_qa: dict) -> tuple[list[dict], list[dict], dict]:
    english_rows = {}
    for row in read_jsonl(V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/corpus.jsonl"):
        if row.get("language") == "en":
            english_rows[row["sft_id"]] = row
    translated_rows = {}
    source_candidate_hashes = {}
    for lang in TARGET_LANGS:
        path = V4 / f"sft_translation_candidates/{lang}.jsonl"
        source_candidate_hashes[lang] = sha(path)
        for row in read_jsonl(path):
            translated_rows[(row["sft_id"], lang)] = row
    reused: list[dict] = []
    requests: list[dict] = []
    counts = defaultdict(int)
    for case in case_rows:
        if case["split"] not in {"forget", "retain"}:
            continue
        source = source_qa[case["source_qa_id"]]
        for expected in expected_atomic(case, source):
            en = english_rows.get(expected["sft_id"])
            exact_en = bool(en and row_text(en) == (expected["question_en"], expected["answer_en"]))
            all_reuse = exact_en
            for lang in TARGET_LANGS:
                candidate = translated_rows.get((expected["sft_id"], lang))
                good = bool(
                    exact_en and candidate
                    and candidate.get("review_status") == "APPROVED_BY_AI_REVIEW"
                    and candidate.get("human_translation_reviewed") is False
                    and candidate.get("question_en") == expected["question_en"]
                    and candidate.get("answer_en") == expected["answer_en"]
                    and candidate.get("question", "").strip()
                    and candidate.get("answer_translation", "").strip()
                )
                all_reuse &= good
            if all_reuse:
                for lang in LANGS:
                    if lang == "en":
                        question, answer = row_text(en)
                        model, status = "source-English", "SOURCE_ENGLISH"
                    else:
                        candidate = translated_rows[(expected["sft_id"], lang)]
                        question, answer = candidate["question"], expected["answer_en"]
                        model, status = candidate.get("translation_model", "unknown"), "REUSED_EXACT_SOURCE_AI_REVIEWED"
                    if not question.strip() or not answer.strip():
                        raise ValueError(f"Empty reusable SFT text: {expected['sft_id']}/{lang}")
                    reused.append({
                        **expected, "language": lang, "question": question, "answer": answer,
                        "translation_model": model, "translation_status": status,
                        "human_translation_reviewed": False,
                    })
                counts["reused_fact_rows"] += 1
                continue

            if exact_en:
                for lang in LANGS:
                    if lang == "en":
                        question, answer = row_text(en)
                        reused.append({
                            **expected, "language": lang, "question": question, "answer": answer,
                            "translation_model": "source-English", "translation_status": "SOURCE_ENGLISH",
                            "human_translation_reviewed": False,
                        })
                    else:
                        candidate = translated_rows.get((expected["sft_id"], lang))
                        if candidate and candidate.get("review_status") == "APPROVED_BY_AI_REVIEW" and candidate.get("question_en") == expected["question_en"] and candidate.get("answer_en") == expected["answer_en"]:
                            reused.append({
                                **expected, "language": lang, "question": candidate["question"],
                                "answer": expected["answer_en"],
                                "translation_model": candidate.get("translation_model", "unknown"),
                                "translation_status": "REUSED_EXACT_SOURCE_AI_REVIEWED_PENDING_CROSS_MODEL_SCREEN",
                                "human_translation_reviewed": False,
                            })
                        else:
                            requests.append({
                                **expected, "prompt_id": expected["sft_id"], "language": lang,
                                "text_en": expected["question_en"], "answer_en": expected["answer_en"],
                                "translation_status": "NOT_STARTED",
                            })
                counts["partially_reused_fact_rows"] += 1
            else:
                for lang in LANGS:
                    if lang == "en":
                        if not en:
                            question, answer = expected["question_en"], expected["answer_en"]
                        else:
                            question, answer = expected["question_en"], expected["answer_en"]
                        reused.append({
                            **expected, "language": lang, "question": question, "answer": answer,
                            "translation_model": "source-English", "translation_status": "SOURCE_ENGLISH",
                            "human_translation_reviewed": False,
                        })
                    else:
                        requests.append({
                            **expected, "prompt_id": expected["sft_id"], "language": lang,
                            "text_en": expected["question_en"], "answer_en": expected["answer_en"],
                            "translation_status": "NOT_STARTED",
                        })
                counts["new_fact_rows"] += 1
    # Requests and reuse rows may repeat a prompt ID across languages by design.
    return reused, requests, {**dict(counts), "reuse_source_candidate_hashes": source_candidate_hashes}


def prepare_composition(case_rows: list[dict], source_qa: dict) -> tuple[list[dict], list[dict], dict]:
    comp = V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/disjoint_composition_candidates"
    manifest = read_json(comp / "manifest.json")
    verification = read_json(comp / "independent_verification_report.json")
    ai_report = read_json(comp / "ai_review_report.json")
    candidate_path = comp / "english_candidate.jsonl"
    if verification.get("status") != "PASS" or verification.get("error_count") != 0:
        raise ValueError("Original Type-B source/path verification is not a clean PASS")
    if manifest.get("do_not_train") is not True:
        raise ValueError("Type-B source candidate is not quarantined")
    if manifest.get("candidate_sha256") != sha(candidate_path) or verification.get("candidate_sha256") != sha(candidate_path):
        raise ValueError("Type-B English candidate hashes do not match their manifests")
    if ai_report.get("status") != "AI_ONLY_REVIEWED_DISJOINT_TRAIN_ONLY_TRANSLATIONS" or ai_report.get("human_fluency_review") is not False:
        raise ValueError("Type-B translation audit provenance is incomplete")
    en_rows = read_jsonl(candidate_path)
    en_by_id = {r["sft_id"]: r for r in en_rows}
    target_translations: dict[tuple[str, str], dict] = {}
    translation_hashes = {}
    for lang in TARGET_LANGS:
        path = comp / f"translations/{lang}.jsonl"
        translation_hashes[lang] = sha(path)
        info = ai_report.get("languages", {}).get(lang, {})
        if info.get("translation_sha256") != sha(path) or info.get("status") != "APPROVED_BY_AI_REVIEW":
            raise ValueError(f"Type-B {lang} translation hash/status is invalid")
        for r in read_jsonl(path):
            target_translations[(r["sft_id"], lang)] = r

    eval_ids = {r["source_qa_id"] for r in case_rows}
    eval_qids, eval_surfaces, eval_answers, eval_questions, eval_facts = set(), set(), set(), set(), set()
    for case in case_rows:
        qrow = source_qa[case["source_qa_id"]]
        _, qids, names = graph_data(qrow)
        eval_qids.update(qids)
        eval_surfaces.update(norm(x) for x in names | set(case.get("protected_entity_names", [])) if norm(x))
        eval_answers.update(norm(x) for x in (case["target_bridge_en"], case["final_answer_source_supported_en"], case["negative_control_en"]) if norm(x))
        eval_questions.update(norm(x) for x in (case["q1_question_en"], case["multi_hop_question_en"]) if norm(x))
        eval_facts.update((case["target_r1_fact_id"], case["target_r2_fact_id"]))

    by_chain: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in en_rows:
        match = re.fullmatch(r"disjoint-b:(\d{4}-qa-sm):(q1|q2|q3|multi)", row.get("sft_id", ""))
        if not match:
            raise ValueError(f"Malformed Type-B SFT ID: {row.get('sft_id')}")
        source_id, suffix = match.groups()
        if row.get("source_qa_id") != source_id or row.get("language") != "en" or row.get("do_not_train") is not True:
            raise ValueError(f"Type-B row provenance/quarantine mismatch: {row['sft_id']}")
        if suffix in by_chain[source_id]:
            raise ValueError(f"Duplicate Type-B prompt: {row['sft_id']}")
        by_chain[source_id][suffix] = row

    selected_ids, reason_counts = [], defaultdict(int)
    for source_id, rows in sorted(by_chain.items()):
        if set(rows) != set(COMPOSITION_HOPS):
            reason_counts["incomplete_prompt_set"] += 1
            continue
        if source_id in eval_ids:
            reason_counts["evaluation_source_id_overlap"] += 1
            continue
        qrow = source_qa.get(source_id)
        if not qrow or qrow.get("question_graph_type") != "B":
            reason_counts["not_type_b_source"] += 1
            continue
        _, qids, names = graph_data(qrow)
        chain_qids = set().union(*(set(rows[s].get("entity_qids", [])) for s in rows)) | qids
        chain_surfaces = {norm(x) for x in names if norm(x)}
        for s in rows:
            chain_surfaces.update(norm(x) for x in rows[s].get("protected_entity_names", []) if norm(x))
        chain_answers = {norm(row_text(r)[1]) for r in rows.values() if norm(row_text(r)[1])}
        chain_questions = {norm(row_text(r)[0]) for r in rows.values() if norm(row_text(r)[0])}
        chain_fact_ids = set().union(*(set(rows[s].get("chain_fact_ids", [])) for s in rows))
        if chain_qids & eval_qids:
            reason_counts["evaluation_qid_overlap"] += 1
            continue
        if chain_surfaces & (eval_surfaces | eval_answers):
            reason_counts["evaluation_surface_overlap"] += 1
            continue
        if chain_answers & eval_answers:
            reason_counts["evaluation_answer_overlap"] += 1
            continue
        if chain_questions & eval_questions:
            reason_counts["evaluation_question_overlap"] += 1
            continue
        if chain_fact_ids & eval_facts:
            reason_counts["evaluation_fact_overlap"] += 1
            continue
        if any(norm(row_text(r)[1]) and norm(row_text(r)[1]) in norm(row_text(r)[0]) for r in rows.values()):
            reason_counts["answer_disclosed_in_prompt"] += 1
            continue
        # Require full five-language coverage and the previous hash-bound
        # translation audit before asking for an independent Qwen screen.
        valid = True
        for suffix, row in rows.items():
            q_en, a_en = row_text(row)
            for lang in TARGET_LANGS:
                translated = target_translations.get((row["sft_id"], lang))
                if not translated or translated.get("review_status") != "APPROVED_BY_AI_REVIEW" or translated.get("human_translation_reviewed") is not False:
                    valid = False
                    break
                if translated.get("question_en") != q_en or translated.get("answer_en") != a_en:
                    valid = False
                    break
                if row_text(translated) != (translated.get("question"), translated.get("answer_translation")) or translated.get("answer_translation") != a_en:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            reason_counts["incomplete_or_mismatched_translation"] += 1
            continue
        selected_ids.append(source_id)

    training_rows: list[dict] = []
    review_queue: list[dict] = []
    for source_id in selected_ids:
        for suffix, hop in COMPOSITION_HOPS.items():
            source_row = by_chain[source_id][suffix]
            q_en, a_en = row_text(source_row)
            for lang in LANGS:
                if lang == "en":
                    q, a, model, status = q_en, a_en, "source-English", "SOURCE_ENGLISH"
                else:
                    tr = target_translations[(source_row["sft_id"], lang)]
                    q, a = tr["question"], tr["answer_translation"]
                    model, status = "facebook/nllb-200-distilled-600M", "REUSED_AI_ONLY_PENDING_CROSS_MODEL_SCREEN"
                    review_queue.append({
                        "prompt_id": source_row["sft_id"], "source_qa_id": source_id,
                        "language": lang, "probe": "multi_hop_question" if suffix == "multi" else "atomic_support_question",
                        "text_en": q_en, "text_translation": q,
                        "protected_entity_names": source_row.get("protected_entity_names", []),
                        "translation_model": model,
                    })
                training_rows.append({
                    "sft_id": source_row["sft_id"], "source_qa_id": source_id,
                    "split": "train_only", "hop": hop, "fact_ids": source_row.get("chain_fact_ids", []),
                    "language": lang, "question": q, "answer": a,
                    "question_en": q_en, "answer_en": a_en,
                    "translation_model": model, "translation_status": status,
                    "human_translation_reviewed": False,
                })
    return training_rows, review_queue, {
        "candidate_chains": len(by_chain), "selected_chains": len(selected_ids),
        "selected_source_qa_ids": selected_ids, "exclusion_counts": dict(reason_counts),
        "original_verification_status": verification.get("status"),
        "original_ai_review_status": ai_report.get("status"),
        "english_candidate_sha256": sha(candidate_path), "original_manifest_sha256": sha(comp / "manifest.json"),
        "original_independent_verification_sha256": sha(comp / "independent_verification_report.json"),
        "original_translation_review_report_sha256": sha(comp / "ai_review_report.json"),
        "translation_sha256_by_language": translation_hashes,
    }


def main() -> int:
    # Re-run the independent v4 Type-B verifier against the immutable source
    # using this workspace's source-data location. Its default path assumes a
    # Kaggle-mounted copy of the original dataset.
    verifier = V4 / "kaggle/r1_experiment_dataset_disjoint/scripts/verify_type_b_disjoint_candidates.py"
    spec = importlib.util.spec_from_file_location("type_b_verifier", verifier)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the independent Type-B verifier")
    verifier_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier_module)
    verifier_module.ROOT = V4
    verifier_module.SOURCE = SOURCE
    verifier_module.CANDIDATES = V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/disjoint_composition_candidates"
    verifier_module.main()

    source_qa_rows = read_jsonl(SOURCE / "qa-sm.jsonl")
    source_qa = {r["instance_id"]: r for r in source_qa_rows}
    cases = read_jsonl(V51 / "candidate_cases.draft.jsonl")
    validator = read_json(V51 / "LOCAL_VALIDATION.json")
    if validator.get("status") != "PASS_DRAFT_SOURCE_AND_SPLIT_VALIDATION" or validator.get("do_not_train") is not True or validator.get("do_not_evaluate") is not True:
        raise ValueError("v5.1 source draft is not in the expected validated quarantine state")
    for relative, expected in validator.get("artifact_hashes", {}).items():
        path = V51 / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Source draft artifact checksum mismatch: {relative}")
    expected_split = {"forget": 117, "retain": 39, "holdout": 39}
    actual_split = {split: sum(c["split"] == split for c in cases) for split in expected_split}
    if actual_split != expected_split or len({c["source_qa_id"] for c in cases}) != 195:
        raise ValueError(f"Unexpected v5.1 case split/count: {actual_split}")
    if any(c.get("status") != "DRAFT_NOT_APPROVED" or c.get("do_not_train") is not True for c in cases):
        raise ValueError("Unexpected source draft approval/training flags")
    if any(c["source_qa_id"] not in source_qa for c in cases):
        raise ValueError("Candidate references a missing source QA")

    cohort, eval_data, cohort_summary = approved_eval_cohort()
    expected_cohort_summary = {
        "forget": {"cases": 43, "entity_groups": 24},
        "retain": {"cases": 20, "entity_groups": 12},
        "holdout": {"cases": 17, "entity_groups": 11},
    }
    if cohort_summary != expected_cohort_summary:
        raise ValueError(
            "Translation-screened evaluation cohort differs from the frozen, "
            f"cross-model-only expectation: {cohort_summary}"
        )
    write_jsonl(OUT / "evaluation/cases.jsonl", cohort)
    write_jsonl(OUT / "evaluation/prompts.jsonl", eval_data["prompts"])
    write_jsonl(OUT / "evaluation/labels.jsonl", eval_data["labels"])

    atomic_rows, atomic_requests, atomic_summary = prepare_atomic(cases, source_qa)
    write_jsonl(OUT / "atomic_reuse.jsonl", atomic_rows)
    write_jsonl(OUT / "atomic_translation_requests.jsonl", atomic_requests)
    composition_rows, composition_review_queue, composition_summary = prepare_composition(cases, source_qa)
    write_jsonl(OUT / "composition_reuse.jsonl", composition_rows)
    write_jsonl(OUT / "composition_translation_review_queue.jsonl", composition_review_queue)

    source_hashes = {
        "candidate_cases": sha(V51 / "candidate_cases.draft.jsonl"),
        "labels": sha(V51 / "labels/gold_labels.draft.jsonl"),
        "requests": sha(V51 / "translation_requests/en.draft.jsonl"),
        "source_qa": sha(SOURCE / "qa-sm.jsonl"),
        "source_facts": sha(SOURCE / "qa-sm-facts.jsonl"),
        "v4_atomic_corpus": sha(V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/corpus.jsonl"),
    }
    report = {
        "status": "PREPARED_QUARANTINED_RELEASE_PENDING_TRANSLATION_AND_QWEN_SCREEN",
        "approval_status": "NOT_APPROVED_FOR_TRAINING",
        "do_not_train": True, "do_not_evaluate": False,
        "human_translation_reviewed": False,
        "translation_review_mode": "AI_ONLY_WITH_INDEPENDENT_QWEN_SCREEN_PENDING",
        "source_hashes": source_hashes,
        "split_counts": actual_split,
        "evaluation_cohort": cohort_summary,
        "evaluation_prompt_rows": len(eval_data["prompts"]),
        "evaluation_label_rows": len(eval_data["labels"]),
        "atomic_fact_rows_expected": 312,
        "atomic_reuse_rows": len(atomic_rows),
        "atomic_translation_requests": len(atomic_requests),
        "atomic_plan": atomic_summary,
        "composition_training_rows": len(composition_rows),
        "composition_review_rows": len(composition_review_queue),
        "composition_plan": composition_summary,
        "training_design": {
            "full_sft": "forget+retain atomic R1/R2 facts in EN/FR/DE/HI/TH plus selected source/path-verified Type-B train-only composition chains; no holdout rows or evaluation prompts/labels",
            "retain_only": "retain atomic facts plus the same disjoint train-only composition chains",
            "unlearning_input": "English R1 facts for the 117 forget cases; retain facts only as the method's retain regularizer; no holdout data",
            "evaluation": "translation-screened common cohort, all three probes in EN/FR/DE/HI/TH; prompts and gold labels are separate files",
            "design_limit": "the 15-case/8-group floor is a feasibility gate, not a power calculation; do not claim confirmatory power from passing it",
        },
    }
    write_json(OUT / "PREPARATION_REPORT.json", report)
    print(json.dumps({k: report[k] for k in (
        "status", "evaluation_cohort", "evaluation_prompt_rows", "atomic_fact_rows_expected",
        "atomic_reuse_rows", "atomic_translation_requests", "composition_training_rows",
        "composition_review_rows", "composition_plan",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
