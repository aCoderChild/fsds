#!/usr/bin/env python3
"""Independently verify the Type-B train-only candidate artifact."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "synthworlds_sm"
CANDIDATES = ROOT / "sft_training/disjoint_composition_candidates"
SUFFIXES = ("q1", "q2", "q3", "multi")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    return "derived-sha256:" + hashlib.sha256((subject_qid + "\0" + statement).encode()).hexdigest()


def graph_qids(row, graph):
    values = set(row.get("gold_qids", [])) | set(graph.get("entity_rename_map", {}))
    for key, value in graph.get("graph_qa", {}).items():
        if re.fullmatch(r"n\d+", key) and qid(value):
            values.add(qid(value))
        if key.startswith("r") and isinstance(value, dict):
            values.update(x for x in (value.get("qid"), value.get("relation_subject_qid")) if x)
    return values


def graph_names(graph):
    result = set()
    for key, value in graph.get("graph_qa", {}).items():
        if re.fullmatch(r"n\d+", key):
            result.add(normalize(value))
    for entry in graph.get("entity_rename_map", {}).values():
        for field in ("new_name", "old_name"):
            if entry.get(field):
                result.add(normalize(entry[field]))
    return {value for value in result if value}


def main():
    qa_path = SOURCE / "qa-sm.jsonl"
    facts_path = SOURCE / "qa-sm-facts.jsonl"
    assignments_path = ROOT / "assignments.jsonl"
    labels_path = ROOT / "evaluation_dataset/labels/gold_labels.jsonl"
    candidate_path = CANDIDATES / "english_candidate.jsonl"
    manifest_path = CANDIDATES / "manifest.json"

    manifest = read_json(manifest_path)
    rows = read_jsonl(candidate_path)
    qa_rows = read_jsonl(qa_path)
    assignments = read_jsonl(assignments_path)
    labels = read_jsonl(labels_path)
    qa = {row["instance_id"]: row for row in qa_rows}
    facts = {row["qid"]: row["facts"] for row in read_jsonl(facts_path)}
    labels_by_id = {row["instance_id"]: row for row in labels}
    eval_ids = {row["source_qa_id"] for row in assignments}
    errors = []

    expected_hashes = {
        "candidate_sha256": (manifest.get("candidate_sha256"), sha(candidate_path)),
        "source_qa_sha256": (manifest.get("source_qa_sha256"), sha(qa_path)),
        "source_facts_sha256": (manifest.get("source_facts_sha256"), sha(facts_path)),
        "evaluation_assignments_sha256": (manifest.get("evaluation_assignments_sha256"), sha(assignments_path)),
        "evaluation_labels_sha256": (manifest.get("evaluation_labels_sha256"), sha(labels_path)),
    }
    for name, (recorded, actual) in expected_hashes.items():
        if recorded != actual:
            errors.append(f"{name} mismatch")
    if manifest.get("status") != "TYPE_B_DISJOINT_COMPOSITION_CANDIDATES_PENDING_TRANSLATION_REVIEW":
        errors.append("Candidate status is not the expected quarantined Type-B state")
    if manifest.get("do_not_train") is not True:
        errors.append("Candidate set is not quarantined from training")
    if len({row.get("sft_id") for row in rows}) != len(rows):
        errors.append("Duplicate SFT IDs")
    if any(row.get("split") != "train_only" or row.get("do_not_train") is not True for row in rows):
        errors.append("A candidate row is not marked train_only/do_not_train")

    eval_qids, eval_names, eval_answers, eval_questions, eval_fact_ids = set(), set(), set(), set(), set()
    for instance_id in eval_ids:
        row = qa[instance_id]
        graph = json.loads(row["mhop_qa"])
        label = labels_by_id[instance_id]
        eval_qids.update(graph_qids(row, graph))
        eval_names.update(graph_names(graph))
        eval_answers.update(normalize(value) for value in label["final_answers"])
        eval_answers.add(normalize(label["direct_intermediate_answer"]))
        eval_questions.add(normalize(row["query"]))
        eval_fact_ids.update(label["chain_fact_ids"])

    by_chain = defaultdict(dict)
    for row in rows:
        match = re.fullmatch(r"disjoint-b:(\d{4}-qa-sm):(q1|q2|q3|multi)", row.get("sft_id", ""))
        if not match:
            errors.append(f"Malformed candidate ID: {row.get('sft_id')}")
            continue
        instance_id, suffix = match.groups()
        if row.get("source_qa_id") != instance_id or row.get("source_qa_ids") != [instance_id]:
            errors.append(f"Source QA provenance mismatch: {row['sft_id']}")
        if suffix in by_chain[instance_id]:
            errors.append(f"Duplicate prompt type for chain {instance_id}/{suffix}")
        by_chain[instance_id][suffix] = row

        source = qa.get(instance_id)
        if not source or source.get("question_graph_type") != "B" or instance_id in eval_ids:
            errors.append(f"Candidate is not a non-evaluation Type-B source row: {instance_id}")
            continue
        graph = json.loads(source["mhop_qa"])
        graph_qa = graph["graph_qa"]
        nodes = [graph_qa.get(f"n{i}") for i in range(1, 5)]
        relations = [graph_qa.get(f"r{i}", {}) for i in range(1, 4)]
        answer_index = {"q1": 1, "q2": 2, "q3": 3, "multi": 3}[suffix]
        decomposed = graph.get("decomposed_questions", {})
        expected_question = source["query"] if suffix == "multi" else re.sub(
            r"</?bridge>", "", str(decomposed.get(f"Q{int(suffix[1])}", ""))
        ).strip()
        expected_answer = surface(nodes[answer_index])
        if row.get("question_en") != expected_question or row.get("answer_en") != expected_answer:
            errors.append(f"English prompt/answer differs from source graph: {row['sft_id']}")
        if row.get("language") != "en" or row.get("hop") not in {
            "composition_support_q1", "composition_support_q2", "composition_support_q3", "composition_3hop"
        }:
            errors.append(f"Language or hop label mismatch: {row['sft_id']}")
        if row.get("messages") != [
            {"role": "user", "content": expected_question},
            {"role": "assistant", "content": expected_answer},
        ]:
            errors.append(f"Chat message content mismatch: {row['sft_id']}")
        if normalize(expected_answer) and normalize(expected_answer) in normalize(expected_question):
            errors.append(f"Prompt discloses its answer: {row['sft_id']}")
        if graph_qids(source, graph) & eval_qids:
            errors.append(f"Entity QID overlap with evaluation: {row['sft_id']}")
        if graph_names(graph) & (eval_names | eval_answers):
            errors.append(f"Entity/answer surface overlap with evaluation: {row['sft_id']}")
        if normalize(source["query"]) in eval_questions:
            errors.append(f"Question overlap with evaluation: {row['sft_id']}")

        expected_facts = []
        for subject, obj, relation in zip(nodes, nodes[1:], relations):
            subject_qid = qid(subject)
            statement = f"{subject} → {relation['relation_str']} → {obj}"
            if relation.get("qid") != subject_qid or facts.get(subject_qid, []).count(statement) != 1:
                errors.append(f"Ordered source fact missing or nonunique: {row['sft_id']}")
            expected_facts.append(fact_id(subject_qid, statement))
        if row.get("chain_fact_ids") != expected_facts:
            errors.append(f"Chain fact IDs differ from recomputed source facts: {row['sft_id']}")
        if set(expected_facts) & eval_fact_ids:
            errors.append(f"Path fact overlap with evaluation: {row['sft_id']}")

    if len(by_chain) != manifest.get("candidate_chain_count"):
        errors.append("Manifest chain count differs from candidate rows")
    for instance_id, prompts in by_chain.items():
        if set(prompts) != set(SUFFIXES):
            errors.append(f"Incomplete prompt coverage for {instance_id}: {sorted(prompts)}")
    if len(rows) != manifest.get("candidate_sft_rows_en"):
        errors.append("Manifest row count differs from candidate rows")

    result = {
        "status": "FAILED" if errors else "PASS",
        "candidate_sha256": sha(candidate_path),
        "source_qa_sha256": sha(qa_path),
        "source_facts_sha256": sha(facts_path),
        "evaluation_assignments_sha256": sha(assignments_path),
        "evaluation_labels_sha256": sha(labels_path),
        "candidate_chains": len(by_chain),
        "candidate_rows_en": len(rows),
        "prompt_types_per_chain": {"q1+q2+q3+multi": len(by_chain)},
        "errors": errors[:80],
        "error_count": len(errors),
    }
    (CANDIDATES / "independent_verification_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
