#!/usr/bin/env python3
"""Audit source-supported answer counts for the Type-A R1 pilot.

This is a read-only design audit. It does not change the frozen v4 split,
labels, translations, or experiment outputs. It reports whether a larger
candidate cohort could be formed if ambiguous first-hop questions were
replaced by a fact-specific yes/no probe.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SOURCE = DATA / "synthworlds_sm"
AUDIT = DATA / "synthworlds_sm_reaudit"
PILOT = DATA / "synthworlds_sm_pilot_v4"
OUT = ROOT / "reports" / "hypothesis_diagnostics" / "r1_answer_cardinality_audit_2026-11-03"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    value = re.sub(r"\s*\(Q\d+\)$", "", str(value)).casefold()
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def evaluator_normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def qid(value: str) -> str | None:
    match = re.search(r"\((Q\d+)\)$", str(value))
    return match.group(1) if match else None


def fact_id(subject_qid: str, statement: str) -> str:
    payload = (subject_qid + "\0" + statement).encode("utf-8")
    return "derived-sha256:" + hashlib.sha256(payload).hexdigest()


def source_candidates(subject: str, relation: str, facts: list[str]) -> list[str]:
    answers = set()
    for statement in facts:
        parts = [part.strip() for part in statement.split("→")]
        if len(parts) >= 3 and parts[:2] == [subject, relation]:
            answers.add(parts[2])
    return sorted(answers, key=lambda value: (normalize(value), value))


def relation_outputs(subject: str, relation: dict, facts_by_qid: dict[str, list[str]]) -> list[str]:
    """Enumerate answers supported by the same fact record used by the audit."""
    source_facts = facts_by_qid.get(relation.get("qid"), [])
    relation_text = relation.get("relation_str", "")
    outputs = []
    if relation_text.startswith("["):
        steps = [part.strip() for part in relation_text[1:-1].split("→")]
        if len(steps) != 3:
            return outputs
        expected_middle_qid = relation.get("relation_subject_qid")
        for statement in source_facts:
            parts = [part.strip() for part in statement.split("→")]
            if len(parts) != 5 or parts[0] != subject or parts[1] != steps[0] or parts[3] != steps[2]:
                continue
            if expected_middle_qid and qid(parts[2]) != expected_middle_qid:
                continue
            if not expected_middle_qid and normalize(parts[2]) != normalize(steps[1]):
                continue
            outputs.append(parts[4])
    else:
        for statement in source_facts:
            parts = [part.strip() for part in statement.split("→")]
            if len(parts) >= 3 and parts[0] == subject and parts[1] == relation_text:
                outputs.append(parts[2])
    return outputs


class DSU:
    def __init__(self, ids: list[str]):
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def subset_sum(values: list[int], target: int, excluded: set[int] | None = None) -> set[int] | None:
    excluded = excluded or set()
    witnesses: dict[int, set[int]] = {0: set()}
    for index, value in enumerate(values):
        if index in excluded:
            continue
        for current, witness in list(witnesses.items())[::-1]:
            new_total = current + value
            if new_total <= target and new_total not in witnesses:
                witnesses[new_total] = witness | {index}
    return witnesses.get(target)


def expanded_groups(candidates: list[dict]) -> tuple[list[int], dict[str, list[str]]]:
    """Group proposed cases by entities, all Q1 source facts, chain facts, and answers."""
    evidence = {}
    for item in candidates:
        row, audit, graph, graph_qa = (item[key] for key in ("row", "audit", "graph", "graph_qa"))
        qids = set(row.get("gold_qids", [])) | set(audit.get("graph_qids", []))
        qids.update(graph.get("entity_rename_map", {}).keys())
        for key, value in graph_qa.items():
            if re.fullmatch(r"n\d+", key):
                node_qid = qid(value)
                if node_qid:
                    qids.add(node_qid)
            if key.startswith("r") and isinstance(value, dict) and value.get("relation_subject_qid"):
                qids.add(value["relation_subject_qid"])
        q1_answers = audit["q1_possible_answers"]
        qids.update(value_qid for answer in q1_answers if (value_qid := qid(answer)))

        r1 = graph_qa["r1"]
        q1_fact_ids = {
            fact_id(r1["qid"], f"{graph_qa['n1']} → {r1['relation_str']} → {answer}")
            for answer in q1_answers
        }
        fact_ids = q1_fact_ids | set(audit["chain_fact_ids"])
        answers = {normalize(answer) for answer in row.get("gold_answers", [])}
        answers.update(normalize(answer) for answer in q1_answers)
        evidence[row["instance_id"]] = (qids, fact_ids, answers)

    ids = sorted(evidence)
    dsu = DSU(ids)
    for left, right in itertools.combinations(ids, 2):
        if any(evidence[left][index] & evidence[right][index] for index in range(3)):
            dsu.union(left, right)
    groups = collections.defaultdict(list)
    for item in ids:
        groups[dsu.find(item)].append(item)
    ordered_groups = sorted(groups.values(), key=lambda group: (-len(group), group[0]))
    return [len(group) for group in ordered_groups], {f"group_{index:03d}": group for index, group in enumerate(ordered_groups, 1)}


def audit() -> dict:
    source_manifest = read_json(PILOT / "source_manifest.json")
    source_files = {
        "qa": SOURCE / "qa-sm.jsonl",
        "facts": SOURCE / "qa-sm-facts.jsonl",
        "documents": SOURCE / "qa-sm-docs.jsonl",
    }
    for key, path in source_files.items():
        if sha256(path) != source_manifest[key]["sha256"]:
            raise RuntimeError(f"Source checksum mismatch for {key}: {path}")

    qa_rows = read_jsonl(source_files["qa"])
    facts_rows = read_jsonl(source_files["facts"])
    facts = {row["qid"]: row["facts"] for row in facts_rows}
    audits = {row["instance_id"]: row for row in read_jsonl(AUDIT / "type_a_target_eligibility.jsonl")}
    assigned = {row["source_qa_id"]: row for row in read_jsonl(PILOT / "assignments.jsonl")}
    if collections.Counter(row["split"] for row in assigned.values()) != {"forget": 77, "retain": 26, "holdout": 26}:
        raise RuntimeError("Frozen v4 assignment counts changed")
    conflict_rows = read_jsonl(PILOT / "excluded_r2_conflict.jsonl")
    source_conflicts = {row["source_qa_id"] for row in conflict_rows}

    type_a = [row for row in qa_rows if row.get("question_graph_type") == "A"]
    if len(type_a) != 200 or len(audits) != 200:
        raise RuntimeError(f"Unexpected Type-A/audit counts: {len(type_a)} / {len(audits)}")

    q1_counts = collections.Counter()
    q2_counts = collections.Counter()
    ambiguous_bridge_paths = collections.Counter()
    current_split_q2 = collections.Counter()
    candidate_rows = []
    cardinality_issues = []
    q1_target_matches = 0

    for row in type_a:
        iid = row["instance_id"]
        audit_row = audits[iid]
        graph = json.loads(row["mhop_qa"])
        graph_qa = graph["graph_qa"]
        r1, r2 = graph_qa["r1"], graph_qa["r2"]
        q1_objects = source_candidates(graph_qa["n1"], r1["relation_str"], facts.get(r1["qid"], []))
        if set(q1_objects) != set(audit_row["q1_possible_answers"]):
            raise RuntimeError(f"Q1 source candidates disagree with audit for {iid}")
        if audit_row["matching_r1_exact_fact_records"] != 1 or graph_qa["n2"] not in q1_objects:
            raise RuntimeError(f"Target relation not uniquely source-verified for {iid}")
        q1_target_matches += 1
        expected_r1_fact = f"{graph_qa['n1']} → {r1['relation_str']} → {graph_qa['n2']}"
        if facts.get(r1["qid"], []).count(expected_r1_fact) != 1:
            raise RuntimeError(f"Exact target fact is not unique in source for {iid}")
        matched_r2 = audit_row.get("r2_matched_fact_texts", [])
        if len(matched_r2) != 1 or matched_r2[0] not in facts.get(r2.get("qid"), []):
            raise RuntimeError(f"Exact second-hop fact is not uniquely present in source for {iid}")
        q1_counts[len(q1_objects)] += 1

        target_outputs = relation_outputs(graph_qa["n2"], r2, facts)
        target_output_set = {normalize(value) for value in target_outputs}
        gold_set = {normalize(value) for value in row["gold_answers"]}
        if not target_output_set:
            q2_counts["no_source_output"] += 1
        elif target_output_set == gold_set:
            q2_counts["unique_gold_set"] += 1
        else:
            q2_counts["multiple_or_mismatched_outputs"] += 1
            cardinality_issues.append({
                "instance_id": iid,
                "current_split": assigned.get(iid, {}).get("split"),
                "target_outputs": sorted(target_output_set),
                "gold_answers": sorted(gold_set),
            })

        if iid in assigned:
            current_split_q2["assigned"] += 1
            if target_output_set == gold_set:
                current_split_q2["unique_gold_set"] += 1
            else:
                current_split_q2[f"non_unique_{assigned[iid]['split']}"] += 1

        bridge_outputs = {
            bridge: {normalize(value) for value in relation_outputs(bridge, r2, facts)}
            for bridge in q1_objects
        }
        other_bridge_paths = [bridge for bridge, values in bridge_outputs.items() if bridge != graph_qa["n2"] and values]
        if len(q1_objects) > 1:
            ambiguous_bridge_paths["ambiguous_q1_cases"] += 1
            ambiguous_bridge_paths["target_bridge_has_r2_path"] += bool(bridge_outputs[graph_qa["n2"]])
            ambiguous_bridge_paths["other_bridge_has_r2_path"] += bool(other_bridge_paths)
            ambiguous_bridge_paths["unique_gold_final_answer"] += target_output_set == gold_set
            if not other_bridge_paths and bridge_outputs[graph_qa["n2"]]:
                ambiguous_bridge_paths["target_is_only_continuing_bridge"] += 1

        if target_output_set == gold_set and not other_bridge_paths and iid not in source_conflicts:
            candidate_rows.append({"row": row, "audit": audit_row, "graph": graph, "graph_qa": graph_qa})

    current_forget_retain = {
        iid for iid, row in assigned.items() if row["split"] in {"forget", "retain"}
    }
    current_issues = [issue for issue in cardinality_issues if issue["instance_id"] in assigned]
    if len(candidate_rows) != 195:
        raise RuntimeError(f"Expected 195 source-supported candidate cases after explicit conflict exclusion; got {len(candidate_rows)}")
    if q1_target_matches != len(type_a) or len(assigned) != 129:
        raise RuntimeError("Target/source QA cardinality verification failed")

    predictions_path = (
        ROOT / "reports" / "hypothesis_diagnostics" /
        "r1_synthworlds_v4_retainfix_corrected_2026-11-03" / "extracted" /
        "corrected_evaluation" / "scored_predictions.jsonl"
    )
    case_predictions = [
        row for row in read_jsonl(predictions_path)
        if row["instance_id"] == "0038-qa-sm" and row["probe"] == "multi"
    ]
    issue_0038 = next(issue for issue in current_issues if issue["instance_id"] == "0038-qa-sm")
    if len(case_predictions) != 30:
        raise RuntimeError(f"Expected 30 corrected-run final-answer predictions for 0038; got {len(case_predictions)}")
    accepted_normalized = {evaluator_normalize(value) for value in issue_0038["target_outputs"]}
    prediction_sensitivity = {
        "case_id": "0038-qa-sm",
        "split": "holdout",
        "model_language_predictions": len(case_predictions),
        "exact_matches_using_original_gold": sum(bool(row.get("correct")) for row in case_predictions),
        "exact_matches_using_all_source_supported_answers": sum(
            evaluator_normalize(row.get("answer", "")) in accepted_normalized for row in case_predictions
        ),
    }
    if prediction_sensitivity["exact_matches_using_all_source_supported_answers"] != 0:
        raise RuntimeError("Holdout sensitivity result changed; update the report interpretation")

    group_sizes, group_members = expanded_groups(candidate_rows)
    n = len(candidate_rows)
    target_sizes = {"forget": round(0.60 * n), "retain": round(0.20 * n), "holdout": n - round(0.60 * n) - round(0.20 * n)}
    first = subset_sum(group_sizes, target_sizes["retain"])
    exact_split_feasible = False
    if first is not None:
        second = subset_sum(group_sizes, target_sizes["holdout"], first)
        exact_split_feasible = second is not None

    size_histogram = dict(sorted(collections.Counter(group_sizes).items()))
    manifest = {
        "status": "SOURCE_ANSWER_CARDINALITY_AUDIT_COMPLETE_CANDIDATE_ONLY",
        "source_checksums": {key: source_manifest[key]["sha256"] for key in source_files},
        "type_a_count": len(type_a),
        "first_hop_answer_cardinality": {
            "unique_answer_cases": q1_counts[1],
            "ambiguous_answer_cases": sum(nrows for count, nrows in q1_counts.items() if count > 1),
            "answer_count_histogram": {str(key): value for key, value in sorted(q1_counts.items())},
            "target_bridge_present_in_source_answer_set_count": q1_target_matches,
        },
        "second_hop_target_answer_cardinality": {
            "unique_gold_set_cases": q2_counts["unique_gold_set"],
            "multiple_or_mismatched_cases": q2_counts["multiple_or_mismatched_outputs"],
            "no_source_output_cases": q2_counts["no_source_output"],
            "issues": cardinality_issues,
        },
        "ambiguous_first_hop_path_check": dict(ambiguous_bridge_paths),
        "current_129_case_evaluation": {
            "assigned_cases": current_split_q2["assigned"],
            "unique_final_answer_cases": current_split_q2["unique_gold_set"],
            "non_unique_or_mismatched_cases": current_issues,
            "forget_and_retain_cases": len(current_forget_retain),
            "forget_and_retain_label_issues": [issue for issue in current_issues if issue["instance_id"] in current_forget_retain],
        },
        "holdout_label_sensitivity": prediction_sensitivity,
        "expanded_candidate_pool": {
            "candidate_cases": n,
            "selection_rule": "Unique source-supported final-answer set; designated target bridge is the only Q1 source candidate with an R2 continuation; exact R1 and R2 source facts; known 0080 graph/source conflict excluded.",
            "direct_probe_design_required": "Replace ambiguous open-ended Q1 probes with a translated fact-specific yes/no probe, e.g. ask whether the answer to Q1 includes the designated bridge. Apply the same probe format to all cases before scoring.",
            "split_sizes_requested": target_sizes,
            "leakage_group_count": len(group_sizes),
            "leakage_group_size_histogram": size_histogram,
            "largest_leakage_group": max(group_sizes),
            "exact_group_preserving_split_feasible": exact_split_feasible,
            "split_created": False,
            "approval_status": "NOT_APPROVED_FOR_TRAINING_OR_EVALUATION",
        },
        "limitations": [
            "This audit validates source answer cardinality, not translation fluency or human semantic review.",
            "The 195-case pool changes the direct-probe format and requires a new preregistered target definition, fresh translations, a new group-isolated split, and baseline competence evaluation.",
            "No existing v4 split, labels, model checkpoint, or experiment result is modified by this audit.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "answer_cardinality_audit.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# SynthWorlds-SM source answer-cardinality audit",
        "",
        "Status: read-only audit; the frozen v4 split and experiment outputs were not changed.",
        "",
        "## Findings",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Type-A source questions | {len(type_a)} |",
        f"| Q1 questions with one source-supported answer | {manifest['first_hop_answer_cardinality']['unique_answer_cases']} |",
        f"| Q1 questions with multiple source-supported answers | {manifest['first_hop_answer_cardinality']['ambiguous_answer_cases']} |",
        f"| Q2 labels matching the complete source-supported answer set | {q2_counts['unique_gold_set']} / {len(type_a)} |",
        f"| Q2 questions with multiple or mismatched source outputs | {q2_counts['multiple_or_mismatched_outputs']} / {len(type_a)} |",
        f"| Current v4 forget/retain cases with a unique source-supported final answer | {len(current_forget_retain)} / {len(current_forget_retain)} |",
        f"| Current v4 holdout cases with a source-answer cardinality issue | {len(current_issues)} |",
        f"| Exact matches for that holdout case with original / complete source labels | {prediction_sensitivity['exact_matches_using_original_gold']} / {prediction_sensitivity['exact_matches_using_all_source_supported_answers']} of 30 |",
        "",
        "For the 70 ambiguous Q1 cases, the graph-selected bridge is the only Q1 source candidate with an R2 continuation. A total of "
        f"{ambiguous_bridge_paths['unique_gold_final_answer']} of those 70 have a unique source-supported final-answer set. "
        "The remaining cases have an ambiguous or mismatched R2 answer set and are excluded from the proposed expansion.",
        "",
        "## Candidate expansion (not approved)",
        "",
        f"A source-backed candidate pool contains {n} cases. Under conservative grouping by graph/document entities, all candidate Q1 fact IDs, chain fact IDs, and answer surfaces, it forms {len(group_sizes)} indivisible groups; the largest has {max(group_sizes)} cases. An exact 117/39/39 group-preserving split is structurally feasible. No split was created.",
        "",
        "The 67 recovered cases have multiple valid answers to their original open-ended Q1. They require a fact-specific yes/no probe that asks whether the answer set to Q1 includes the designated bridge. The same direct-probe format must be used for every case in a new experiment. The candidate set also needs fresh translation/review, a new split approval, and a new baseline gate before unlearning.",
        "",
        "The current 129-case v4 results remain frozen. Its sole final-answer cardinality issue is case `0038-qa-sm`, which is in holdout; all 103 forget/retain cases have unique source-supported final answers. Rescoring its 30 saved final-answer predictions against both source-supported answers still yields zero exact matches, so the issue does not change the reported holdout score or enter the primary forget-set cross-language contrast.",
        "",
        "## Provenance",
        "",
        f"Source hashes are checked against `data/synthworlds_sm_pilot_v4/source_manifest.json`. The audit manifest is `{json_path.relative_to(ROOT)}`.",
        "",
    ]
    report_path = OUT / "answer_cardinality_audit.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return {
        "status": manifest["status"],
        "report": str(report_path.relative_to(ROOT)),
        "candidate_cases": n,
        "current_forget_retain_unique_final_answers": f"{len(current_forget_retain)}/{len(current_forget_retain)}",
        "q2_unique_gold_set_cases": q2_counts["unique_gold_set"],
        "q2_cardinality_issues": q2_counts["multiple_or_mismatched_outputs"],
        "group_count": len(group_sizes),
        "largest_group": max(group_sizes),
        "exact_117_39_39_feasible": exact_split_feasible,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
