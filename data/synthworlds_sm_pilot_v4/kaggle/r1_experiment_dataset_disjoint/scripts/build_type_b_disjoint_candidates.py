#!/usr/bin/env python3
"""Build source-verified, evaluation-disjoint Type-B three-edge SFT candidates.

Candidates remain quarantined until their four prompt types have complete
five-language translations and the declared AI-only checks finish.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "synthworlds_sm"
PILOT = ROOT
OUT = PILOT / "sft_training/disjoint_composition_candidates"
LANGUAGES = ("en", "fr", "de", "hi", "th")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qid(text):
    match = re.search(r"\((Q\d+)\)$", str(text))
    return match.group(1) if match else None


def surface(text):
    return re.sub(r"\s*\(Q\d+\)$", "", str(text)).strip().rstrip(".")


def normalize(text):
    text = unicodedata.normalize("NFKC", surface(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def fact_id(subject_qid, statement):
    payload = (subject_qid + "\0" + statement).encode("utf-8")
    return "derived-sha256:" + hashlib.sha256(payload).hexdigest()


def graph_qids(row, graph):
    result = set(row.get("gold_qids", []))
    graph_qa = graph.get("graph_qa", {})
    for key, value in graph_qa.items():
        if re.fullmatch(r"n\d+", key):
            entity_qid = qid(value)
            if entity_qid:
                result.add(entity_qid)
        if key.startswith("r") and isinstance(value, dict):
            if value.get("qid"):
                result.add(value["qid"])
            if value.get("relation_subject_qid"):
                result.add(value["relation_subject_qid"])
    result.update(graph.get("entity_rename_map", {}))
    return result


def graph_names(graph):
    names = set()
    graph_qa = graph.get("graph_qa", {})
    for key, value in graph_qa.items():
        if re.fullmatch(r"n\d+", key):
            names.add(normalize(value))
    for entry in graph.get("entity_rename_map", {}).values():
        for key in ("new_name", "old_name"):
            if entry.get(key):
                names.add(normalize(entry[key]))
    return {name for name in names if name}


def main():
    qa_path = SOURCE / "qa-sm.jsonl"
    facts_path = SOURCE / "qa-sm-facts.jsonl"
    assignments_path = PILOT / "assignments.jsonl"
    labels_path = PILOT / "evaluation_dataset/labels/gold_labels.jsonl"

    qa_rows = read_jsonl(qa_path)
    facts_rows = read_jsonl(facts_path)
    assignments = read_jsonl(assignments_path)
    labels = read_jsonl(labels_path)
    qa_by_id = {row["instance_id"]: row for row in qa_rows}
    facts = {row["qid"]: row["facts"] for row in facts_rows}
    labels_by_id = {row["instance_id"]: row for row in labels}
    eval_ids = {row["source_qa_id"] for row in assignments}
    if len(qa_by_id) != len(qa_rows) or len(labels_by_id) != len(labels) or len(eval_ids) != len(assignments):
        raise ValueError("Duplicate source QA, label, or assignment IDs")
    if eval_ids != set(labels_by_id):
        raise ValueError("Evaluation assignments and labels do not have exact ID coverage")

    eval_qids, eval_names, eval_answers, eval_questions, eval_fact_ids = set(), set(), set(), set(), set()
    for instance_id in eval_ids:
        row = qa_by_id[instance_id]
        graph = json.loads(row["mhop_qa"])
        label = labels_by_id[instance_id]
        eval_qids.update(graph_qids(row, graph))
        eval_names.update(graph_names(graph))
        eval_answers.update(normalize(value) for value in label["final_answers"])
        eval_answers.add(normalize(label["direct_intermediate_answer"]))
        eval_questions.add(normalize(row["query"]))
        eval_fact_ids.update(label["chain_fact_ids"])

    candidates, excluded = [], Counter()
    selected_chains = []
    prompt_specs = (
        ("q1", "composition_support_q1", 1, 2),
        ("q2", "composition_support_q2", 2, 3),
        ("q3", "composition_support_q3", 3, 4),
        ("multi", "composition_3hop", None, 4),
    )
    source_qa_sha = sha(qa_path)
    source_facts_sha = sha(facts_path)

    for row in qa_rows:
        instance_id = row["instance_id"]
        if row.get("question_graph_type") != "B" or instance_id in eval_ids:
            continue
        graph = json.loads(row["mhop_qa"])
        graph_qa = graph.get("graph_qa", {})
        nodes = [graph_qa.get(f"n{i}") for i in range(1, 5)]
        relations = [graph_qa.get(f"r{i}", {}) for i in range(1, 4)]
        if not all(nodes) or any(not relation.get("qid") or not relation.get("relation_str") for relation in relations):
            excluded["incomplete_graph"] += 1
            continue

        path_statements = []
        path_fact_ids = []
        edge_valid = True
        for index, (subject, obj, relation) in enumerate(zip(nodes, nodes[1:], relations), start=1):
            subject_qid = qid(subject)
            if not subject_qid or relation["qid"] != subject_qid:
                excluded["relation_subject_not_ordered_node"] += 1
                edge_valid = False
                break
            statement = f"{subject} → {relation['relation_str']} → {obj}"
            source_statements = facts.get(subject_qid, [])
            if source_statements.count(statement) != 1:
                excluded["ordered_source_fact_missing_or_nonunique"] += 1
                edge_valid = False
                break
            path_statements.append(statement)
            path_fact_ids.append(fact_id(subject_qid, statement))
        if not edge_valid:
            continue

        final_answer = surface(nodes[3])
        if len(row.get("gold_answers", [])) != 1 or normalize(row["gold_answers"][0]) != normalize(final_answer):
            excluded["gold_answer_final_node_mismatch"] += 1
            continue
        if any(qid(node) is None for node in nodes) or len({qid(node) for node in nodes}) != 4:
            excluded["missing_or_repeated_node_qid"] += 1
            continue
        if graph.get("composed_question") != row.get("query"):
            excluded["composed_question_mismatch"] += 1
            continue

        entity_qids = graph_qids(row, graph)
        entity_names = graph_names(graph)
        answers = {normalize(surface(node)) for node in nodes}
        if entity_qids & eval_qids:
            excluded["entity_qid_overlap_with_eval"] += 1
            continue
        if entity_names & (eval_names | eval_answers) or answers & eval_answers:
            excluded["entity_or_answer_surface_overlap_with_eval"] += 1
            continue
        if normalize(row["query"]) in eval_questions:
            excluded["question_overlap_with_eval"] += 1
            continue
        if set(path_fact_ids) & eval_fact_ids:
            excluded["path_fact_overlap_with_eval"] += 1
            continue

        decomposed = graph.get("decomposed_questions", {})
        chain_rows = []
        for suffix, hop, question_index, answer_index in prompt_specs:
            question = row["query"].strip() if question_index is None else re.sub(
                r"</?bridge>", "", str(decomposed.get(f"Q{question_index}", ""))
            ).strip()
            answer = surface(nodes[answer_index - 1])
            if not question:
                excluded["decomposition_missing"] += 1
                chain_rows = []
                break
            if normalize(answer) and normalize(answer) in normalize(question):
                excluded["answer_disclosed_in_prompt"] += 1
                chain_rows = []
                break
            chain_rows.append({
                "sft_id": f"disjoint-b:{instance_id}:{suffix}",
                "source_qa_id": instance_id,
                "source_qa_ids": [instance_id],
                "split": "train_only",
                "hop": hop,
                "language": "en",
                "example_type": "disjoint_type_b_composition_support",
                "chain_fact_ids": path_fact_ids,
                "entity_qids": sorted(entity_qids),
                "gold_answers_source": row["gold_answers"],
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "question_en": question,
                "answer_en": answer,
                "protected_entity_names": sorted(
                    {surface(node) for node in nodes if surface(node)}
                    | {entry.get("new_name", "") for entry in graph.get("entity_rename_map", {}).values() if entry.get("new_name")},
                    key=lambda value: (value.casefold(), value),
                ),
                "translation_review_mode": "SOURCE_ENGLISH",
                "human_translation_reviewed": False,
                "do_not_train": True,
                "source_qa_sha256": source_qa_sha,
                "source_facts_sha256": source_facts_sha,
                "source_fact_texts": path_statements,
            })
        if not chain_rows:
            continue
        candidates.extend(chain_rows)
        selected_chains.append(instance_id)

    if not selected_chains or len(candidates) != 4 * len(selected_chains):
        raise RuntimeError(f"Incomplete Type-B candidates: {len(selected_chains)} chains, {len(candidates)} prompt rows")
    if len({row["sft_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("Duplicate Type-B candidate SFT IDs")

    OUT.mkdir(parents=True, exist_ok=True)
    candidate_path = OUT / "english_candidate.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    manifest = {
        "status": "TYPE_B_DISJOINT_COMPOSITION_CANDIDATES_PENDING_TRANSLATION_REVIEW",
        "do_not_train": True,
        "source_qa_sha256": source_qa_sha,
        "source_facts_sha256": source_facts_sha,
        "evaluation_assignments_sha256": sha(assignments_path),
        "evaluation_labels_sha256": sha(labels_path),
        "evaluation_entity_qids": len(eval_qids),
        "candidate_chain_count": len(selected_chains),
        "candidate_source_qa_ids": selected_chains,
        "candidate_sft_rows_en": len(candidates),
        "rows_by_hop": dict(sorted(Counter(row["hop"] for row in candidates).items())),
        "prompt_rows_per_chain": 4,
        "languages_required": list(LANGUAGES),
        "human_translation_reviewed": False,
        "exclusion_counts": dict(sorted(excluded.items())),
        "selection_policy": (
            "Type-B source QA rows outside the frozen evaluation IDs; three ordered relation-subject QIDs must match n1/n2/n3; "
            "each of the three exact subject-relation-object statements must occur once in source facts; single gold answer must "
            "match n4; no evaluation entity/QID, answer, question, or selected path-fact overlap; each support and composed "
            "prompt must not disclose its answer. Four English prompt-answer rows per accepted chain (Q1, Q2, Q3, and full composition)."
        ),
        "source_fact_ids": {row["sft_id"]: row["chain_fact_ids"] for row in candidates},
        "candidate_sha256": sha(candidate_path),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "candidate_chains": len(selected_chains),
        "candidate_rows_en": len(candidates),
        "rows_by_hop": manifest["rows_by_hop"],
        "excluded": manifest["exclusion_counts"],
        "candidate_sha256": manifest["candidate_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
