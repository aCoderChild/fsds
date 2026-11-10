#!/usr/bin/env python3
"""Independently validate the quarantined v5 source-backed split draft."""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "data/synthworlds_sm_pilot_v4"
SOURCE = ROOT / "data/synthworlds_sm"
AUDIT = ROOT / "data/synthworlds_sm_reaudit"
OUT = ROOT / "data/synthworlds_sm_pilot_v5"
SPLITS = ("forget", "retain", "holdout")
EXPECTED_COUNTS = {"forget": 117, "retain": 39, "holdout": 39}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helpers():
    path = PILOT / "scripts/audit_source_answer_cardinality.py"
    spec = importlib.util.spec_from_file_location("r1_v5_validation_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source-audit helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate():
    helpers = load_helpers()
    manifest = read_json(OUT / "manifest.draft.json")
    paths = {
        "candidate_cases.draft.jsonl": OUT / "candidate_cases.draft.jsonl",
        "assignments.draft.jsonl": OUT / "assignments.draft.jsonl",
        "labels/gold_labels.draft.jsonl": OUT / "labels/gold_labels.draft.jsonl",
        "translation_requests/en.draft.jsonl": OUT / "translation_requests/en.draft.jsonl",
        "leakage_groups.draft.jsonl": OUT / "leakage_groups.draft.jsonl",
    }
    for rel, path in paths.items():
        if sha256(path) != manifest["artifact_sha256"][rel]:
            raise RuntimeError(f"Draft artifact hash mismatch: {rel}")
    if manifest.get("status") != "DRAFT_NOT_APPROVED" or manifest.get("do_not_train") is not True:
        raise RuntimeError("Draft safety gate is missing")
    if manifest.get("do_not_evaluate") is not True:
        raise RuntimeError("Draft must not be evaluated before approval and translation review")

    source_manifest = read_json(PILOT / "source_manifest.json")
    for key, path in {
        "qa": SOURCE / "qa-sm.jsonl",
        "facts": SOURCE / "qa-sm-facts.jsonl",
        "documents": SOURCE / "qa-sm-docs.jsonl",
    }.items():
        if sha256(path) != source_manifest[key]["sha256"] or sha256(path) != manifest["source_hashes"][key]:
            raise RuntimeError(f"Source provenance mismatch: {key}")

    cases = read_jsonl(paths["candidate_cases.draft.jsonl"])
    assignments = read_jsonl(paths["assignments.draft.jsonl"])
    labels = read_jsonl(paths["labels/gold_labels.draft.jsonl"])
    translation = read_jsonl(paths["translation_requests/en.draft.jsonl"])
    groups = read_jsonl(paths["leakage_groups.draft.jsonl"])
    qa_rows = read_jsonl(SOURCE / "qa-sm.jsonl")
    qa_by_id = {row["instance_id"]: row for row in qa_rows}
    facts_by_qid = {row["qid"]: row["facts"] for row in read_jsonl(SOURCE / "qa-sm-facts.jsonl")}
    eligibility = {row["instance_id"]: row for row in read_jsonl(AUDIT / "type_a_target_eligibility.jsonl")}
    legacy_corpus = read_jsonl(PILOT / "sft_training/composition_corpus_disjoint.jsonl")
    legacy_train_ids = {row["source_qa_id"] for row in legacy_corpus if row.get("split") == "train_only"}
    legacy_qids, legacy_surfaces = set(), set()
    for source_id in legacy_train_ids:
        legacy_source = qa_by_id[source_id]
        legacy_graph = json.loads(legacy_source["mhop_qa"])
        legacy_graph_qa = legacy_graph.get("graph_qa", {})
        legacy_qids.update(legacy_source.get("gold_qids", []))
        legacy_qids.update(legacy_graph.get("entity_rename_map", {}))
        for key, value in legacy_graph_qa.items():
            if re.fullmatch(r"n\d+", key):
                legacy_surfaces.add(helpers.normalize(value))
                node_qid = helpers.qid(value)
                if node_qid:
                    legacy_qids.add(node_qid)
            if key.startswith("r") and isinstance(value, dict):
                legacy_qids.update(
                    qid for qid in (value.get("qid"), value.get("relation_subject_qid")) if qid
                )
    for train_row in legacy_corpus:
        if train_row.get("split") == "train_only" and train_row.get("language") == "en":
            legacy_surfaces.add(helpers.normalize(train_row["messages"][-1]["content"]))
    assignment_by_id = {row["source_qa_id"]: row for row in assignments}
    case_by_id = {row["source_qa_id"]: row for row in cases}
    if len(cases) != 195 or len(case_by_id) != len(cases) or set(case_by_id) != set(assignment_by_id):
        raise RuntimeError("Candidate and assignment ID coverage mismatch")
    if len(assignments) != 195 or len(labels) != 585 or len(translation) != 585:
        raise RuntimeError("Case, label, or prompt row counts do not match the 3-probe design")
    if collections.Counter(row["split"] for row in assignments) != EXPECTED_COUNTS:
        raise RuntimeError("The draft split does not have the requested exact 60/20/20 row totals")
    if any(row.get("do_not_train") is not True for row in cases + assignments + labels + translation):
        raise RuntimeError("A draft artifact escaped the training quarantine")
    if any("gold_label" in row or "gold_answers" in row for row in translation):
        raise RuntimeError("Translation requests include answer labels")
    if len({row["probe_id"] for row in labels}) != len(labels):
        raise RuntimeError("Gold-label probe IDs are not unique")
    if len({row["prompt_id"] for row in translation}) != len(translation):
        raise RuntimeError("Translation prompt IDs are not unique")

    # Recheck source eligibility, both target facts, unique final answers,
    # positive/negative probe labels, and the source for each false control.
    label_by_probe = {row["probe_id"]: row for row in labels}
    source_checks = 0
    group_by_id = {row["source_qa_id"]: row["group_id"] for row in assignments}
    evidence_group = {"qids": {}, "facts": {}, "surfaces": {}}
    evidence_by_split = {
        split: {"qids": set(), "facts": set(), "surfaces": set()}
        for split in SPLITS
    }
    candidate_qids, candidate_surfaces = set(), set()
    for instance_id in case_by_id:
        source = qa_by_id[instance_id]
        graph = json.loads(source["mhop_qa"])
        graph_qa = graph["graph_qa"]
        candidate_qids.update(source.get("gold_qids", []))
        candidate_qids.update(graph.get("entity_rename_map", {}))
        for key, value in graph_qa.items():
            if re.fullmatch(r"n\d+", key):
                candidate_surfaces.add(helpers.normalize(value))
                node_qid = helpers.qid(value)
                if node_qid:
                    candidate_qids.add(node_qid)
            if key.startswith("r") and isinstance(value, dict):
                candidate_qids.update(
                    qid for qid in (value.get("qid"), value.get("relation_subject_qid")) if qid
                )
        for alias in graph.get("entity_rename_map", {}).values():
            for key in ("new_name", "old_name"):
                if alias.get(key):
                    candidate_surfaces.add(helpers.normalize(alias[key]))
    for case in cases:
        source = qa_by_id[case["source_qa_id"]]
        candidate_surfaces.update(helpers.normalize(value) for value in source.get("gold_answers", []))
        candidate_surfaces.update(
            helpers.normalize(value) for value in eligibility[case["source_qa_id"]].get("q1_possible_answers", [])
        )
    for instance_id, case in case_by_id.items():
        source = qa_by_id[instance_id]
        graph = json.loads(source["mhop_qa"])
        graph_qa = graph["graph_qa"]
        r1, r2 = graph_qa["r1"], graph_qa["r2"]
        audit_row = eligibility[instance_id]
        source_q1 = helpers.source_candidates(graph_qa["n1"], r1["relation_str"], facts_by_qid[r1["qid"]])
        if set(source_q1) != set(audit_row["q1_possible_answers"]):
            raise RuntimeError(f"Q1 source answer set mismatch: {instance_id}")
        expected_r1 = f"{graph_qa['n1']} → {r1['relation_str']} → {graph_qa['n2']}"
        if facts_by_qid[r1["qid"]].count(expected_r1) != 1 or case["target_r1_fact_text"] != expected_r1:
            raise RuntimeError(f"Target R1 fact is absent, duplicated, or mismatched: {instance_id}")
        r2_matches = audit_row.get("r2_matched_fact_texts", [])
        if (
            len(r2_matches) != 1
            or facts_by_qid[r2["qid"]].count(r2_matches[0]) != 1
            or case["target_r2_fact_text"] != r2_matches[0]
        ):
            raise RuntimeError(f"Target R2 fact is not unique in source: {instance_id}")
        target_outputs = {
            helpers.normalize(value)
            for value in helpers.relation_outputs(graph_qa["n2"], r2, facts_by_qid)
        }
        gold_outputs = {helpers.normalize(value) for value in source["gold_answers"]}
        if target_outputs != gold_outputs or len(gold_outputs) != 1:
            raise RuntimeError(f"Final answer does not match one unique source-supported value: {instance_id}")
        if set(case["q1_source_supported_answers_en"]) != set(source_q1):
            raise RuntimeError(f"Case metadata Q1 answer set mismatch: {instance_id}")

        decoy = case["negative_control_en"]
        decoy_norm = helpers.normalize(decoy)
        if decoy_norm in {helpers.normalize(value) for value in source_q1}:
            raise RuntimeError(f"Negative control is actually a valid Q1 answer: {instance_id}")
        decoy_qid = case.get("negative_control_source_subject_qid")
        source_fact = case["negative_control_source_fact"]
        if not decoy_qid or facts_by_qid.get(decoy_qid, []).count(source_fact) != 1:
            raise RuntimeError(f"Negative control source fact is not unique: {instance_id}")
        decoy_entity_qid = helpers.qid(decoy)
        if decoy_norm in candidate_surfaces or (decoy_entity_qid and decoy_entity_qid in candidate_qids):
            raise RuntimeError(f"Negative control overlaps a candidate entity or answer: {instance_id}")
        if decoy_norm in legacy_surfaces or (decoy_entity_qid and decoy_entity_qid in legacy_qids):
            raise RuntimeError(f"Negative control entity overlaps old train-only composition rows: {instance_id}")

        for probe, expected in (("direct-positive", "YES"), ("direct-negative", "NO")):
            label = label_by_probe[f"{instance_id}:{probe}"]
            if label.get("gold_label") != expected or label.get("split") != case["split"]:
                raise RuntimeError(f"Direct probe gold/split mismatch: {instance_id}/{probe}")
        mh_label = label_by_probe[f"{instance_id}:multi-hop"]
        if mh_label.get("gold_answers") != source["gold_answers"]:
            raise RuntimeError(f"Multi-hop label mismatch: {instance_id}")

        split = case["split"]
        evidence = evidence_by_split[split]
        qids = set(source.get("gold_qids", [])) | set(audit_row.get("graph_qids", []))
        qids.update(graph.get("entity_rename_map", {}))
        case_evidence = {
            "qids": qids,
            "facts": set(),
            "surfaces": set(),
        }
        for key, value in graph_qa.items():
            if re.fullmatch(r"n\d+", key):
                node_qid = helpers.qid(value)
                if node_qid:
                    case_evidence["qids"].add(node_qid)
            if key.startswith("r") and isinstance(value, dict):
                case_evidence["qids"].update(
                    x for x in (value.get("qid"), value.get("relation_subject_qid")) if x
                )
        case_evidence["surfaces"].update(
            helpers.normalize(value) for value in source["gold_answers"]
        )
        case_evidence["surfaces"].update(helpers.normalize(value) for value in source_q1)
        for key, value in graph_qa.items():
            if re.fullmatch(r"n\d+", key):
                case_evidence["surfaces"].add(helpers.normalize(value))
        for alias in graph.get("entity_rename_map", {}).values():
            for alias_key in ("new_name", "old_name"):
                if alias.get(alias_key):
                    case_evidence["surfaces"].add(helpers.normalize(alias[alias_key]))
        r1_fact_ids = {
            helpers.fact_id(r1["qid"], f"{graph_qa['n1']} → {r1['relation_str']} → {answer}")
            for answer in source_q1
        }
        case_evidence["facts"].update(r1_fact_ids)
        case_evidence["facts"].add(helpers.fact_id(r2["qid"], r2_matches[0]))
        for field in ("qids", "facts", "surfaces"):
            evidence[field].update(case_evidence[field])
            for value in case_evidence[field]:
                prior_group = evidence_group[field].get(value)
                if prior_group is not None and prior_group != case["group_id"]:
                    raise RuntimeError(f"Shared {field} evidence was split across groups: {instance_id}")
                evidence_group[field][value] = case["group_id"]
        source_checks += 1

    # Independently enforce entity, source-fact, answer-surface, and group
    # isolation across the three splits.
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            for field in ("qids", "facts", "surfaces"):
                overlap = evidence_by_split[left][field] & evidence_by_split[right][field]
                if overlap:
                    raise RuntimeError(f"{field} leakage across {left}/{right}: {len(overlap)} overlaps")
    split_groups = collections.defaultdict(set)
    decoys_by_split = {split: {"qids": set(), "surfaces": set()} for split in SPLITS}
    for row in assignments:
        split_groups[row["split"]].add(row["group_id"])
    for case in cases:
        split = case["split"]
        decoys_by_split[split]["surfaces"].add(helpers.normalize(case["negative_control_en"]))
        qid = helpers.qid(case["negative_control_en"])
        if qid:
            decoys_by_split[split]["qids"].add(qid)
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if split_groups[left] & split_groups[right]:
                raise RuntimeError(f"Leakage groups cross {left}/{right}")
            for field in ("qids", "surfaces"):
                if decoys_by_split[left][field] & decoys_by_split[right][field]:
                    raise RuntimeError(f"Negative-control entities cross {left}/{right}: {field}")
    group_rows = {row["group_id"]: row for row in groups}
    if set(row["group_id"] for row in assignments) != set(group_rows):
        raise RuntimeError("Leakage-group manifest and assignments have different group IDs")
    for group_id, group in group_rows.items():
        member_ids = set(group["source_qa_ids"])
        if group.get("case_count") != len(member_ids):
            raise RuntimeError(f"Group case count is inconsistent: {group_id}")
        if any(group_by_id.get(instance_id) != group_id for instance_id in member_ids):
            raise RuntimeError(f"Group members disagree with assignments: {group_id}")
        if any(case_by_id.get(instance_id, {}).get("split") != group.get("split") for instance_id in member_ids):
            raise RuntimeError(f"Group members cross splits: {group_id}")
    if set().union(*(set(row["source_qa_ids"]) for row in group_rows.values())) != set(case_by_id):
        raise RuntimeError("Leakage groups do not cover every candidate exactly once")

    result = {
        "status": "PASS_DRAFT_SOURCE_AND_SPLIT_VALIDATION",
        "approval_status": "NOT_APPROVED_FOR_TRAINING_OR_EVALUATION",
        "do_not_train": True,
        "do_not_evaluate": True,
        "candidate_cases": len(cases),
        "source_cases_rechecked": source_checks,
        "split_counts": dict(collections.Counter(row["split"] for row in assignments)),
        "leakage_groups": len(groups),
        "groups_by_split": {split: len(split_groups[split]) for split in SPLITS},
        "negative_control_source_counts": dict(collections.Counter(row["negative_control_source_kind"] for row in cases)),
        "source_fact_checks": "PASS: unique target R1/R2 facts and exact final-answer support for every case",
        "negative_control_checks": "PASS: source-derived controls are outside all Q1 answer sets, candidate entities, old train-only composition entities, and controls in other splits",
        "cross_split_leakage_checks": {field: "PASS" for field in ("qids", "facts", "surfaces", "groups")},
        "translation_requests_contain_gold_labels": False,
        "artifact_hashes": {rel: sha256(path) for rel, path in paths.items()},
        "training_or_model_run_started": False,
    }
    (OUT / "LOCAL_VALIDATION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    validate()
