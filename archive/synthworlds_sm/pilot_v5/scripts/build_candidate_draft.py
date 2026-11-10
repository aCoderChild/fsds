#!/usr/bin/env python3
"""Build a source-checked, leakage-grouped v5 candidate split draft.

All outputs remain quarantined. This script does not translate data, approve
the split, prepare a training corpus, or start model training.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
PILOT = DATA / "synthworlds_sm_pilot_v4"
SOURCE = DATA / "synthworlds_sm"
AUDIT = DATA / "synthworlds_sm_reaudit"
OUT = DATA / "synthworlds_sm_pilot_v5"
SPLITS = ("forget", "retain", "holdout")
LANGUAGES = ("en", "fr", "de", "hi", "th")
TARGET_ROWS = {"forget": 117, "retain": 39, "holdout": 39}
DATE_ANSWER = re.compile(
    r"^\d{1,4}(?:[-/.]\d{1,2}(?:[-/.]\d{1,4})?)?"
    r"(?:\s*(?:\((?:year|date|month|day)\)|CE|BCE|BC|AD))?$",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_audit_helpers():
    path = PILOT / "scripts" / "audit_source_answer_cardinality.py"
    spec = importlib.util.spec_from_file_location("r1_v5_source_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source-audit helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_candidates(helpers, qa_rows, facts_by_qid, eligibility, conflicts):
    """Reproduce the 195-case source-backed Type-A selection rule."""
    candidates = []
    for row in qa_rows:
        if row.get("question_graph_type") != "A":
            continue
        instance_id = row["instance_id"]
        audit_row = eligibility[instance_id]
        graph = json.loads(row["mhop_qa"])
        graph_qa = graph["graph_qa"]
        r1, r2 = graph_qa["r1"], graph_qa["r2"]
        q1_answers = helpers.source_candidates(
            graph_qa["n1"], r1["relation_str"], facts_by_qid.get(r1["qid"], [])
        )
        if set(q1_answers) != set(audit_row["q1_possible_answers"]):
            raise RuntimeError(f"Source Q1 answer inventory mismatch for {instance_id}")
        target_r1 = f"{graph_qa['n1']} → {r1['relation_str']} → {graph_qa['n2']}"
        if (
            audit_row.get("matching_r1_exact_fact_records") != 1
            or facts_by_qid.get(r1["qid"], []).count(target_r1) != 1
        ):
            raise RuntimeError(f"Target first-hop fact is not unique in source: {instance_id}")
        matched_r2 = audit_row.get("r2_matched_fact_texts", [])
        if len(matched_r2) != 1 or facts_by_qid.get(r2["qid"], []).count(matched_r2[0]) != 1:
            raise RuntimeError(f"Target second-hop fact is not unique in source: {instance_id}")

        target_outputs = {
            helpers.normalize(value)
            for value in helpers.relation_outputs(graph_qa["n2"], r2, facts_by_qid)
        }
        gold = {helpers.normalize(value) for value in row.get("gold_answers", [])}
        bridge_outputs = {
            bridge: {
                helpers.normalize(value)
                for value in helpers.relation_outputs(bridge, r2, facts_by_qid)
            }
            for bridge in q1_answers
        }
        other_continuing_bridges = [
            bridge for bridge, values in bridge_outputs.items()
            if bridge != graph_qa["n2"] and values
        ]
        if target_outputs == gold and not other_continuing_bridges and instance_id not in conflicts:
            candidates.append({
                "row": row,
                "audit": audit_row,
                "graph": graph,
                "graph_qa": graph_qa,
                "q1_answers": q1_answers,
                "target_r1_fact": target_r1,
                "target_r2_fact": matched_r2[0],
            })
    if len(candidates) != 195:
        raise RuntimeError(f"Expected 195 source-backed candidates; found {len(candidates)}")
    return candidates


def graph_evidence(item, helpers):
    row, audit, graph, graph_qa = (item[key] for key in ("row", "audit", "graph", "graph_qa"))
    qids = set(row.get("gold_qids", [])) | set(audit.get("graph_qids", []))
    qids.update(graph.get("entity_rename_map", {}))
    surfaces = set(row.get("gold_answers", [])) | set(audit.get("q1_possible_answers", []))
    for alias in graph.get("entity_rename_map", {}).values():
        for key in ("new_name", "old_name"):
            if alias.get(key):
                surfaces.add(alias[key])
    for key, value in graph_qa.items():
        if re.fullmatch(r"n\d+", key):
            surfaces.add(value)
            node_qid = helpers.qid(value)
            if node_qid:
                qids.add(node_qid)
        if key.startswith("r") and isinstance(value, dict):
            if value.get("qid"):
                qids.add(value["qid"])
            if value.get("relation_subject_qid"):
                qids.add(value["relation_subject_qid"])
    normalized_surfaces = {helpers.normalize(value) for value in surfaces if helpers.normalize(value)}
    r1, r2 = graph_qa["r1"], graph_qa["r2"]
    q1_fact_ids = {
        helpers.fact_id(r1["qid"], f"{graph_qa['n1']} → {r1['relation_str']} → {answer}")
        for answer in item["q1_answers"]
    }
    fact_ids = q1_fact_ids | {helpers.fact_id(r2["qid"], item["target_r2_fact"])}
    return qids, fact_ids, normalized_surfaces


def candidate_groups(candidates, helpers):
    """Group on QIDs, source facts, answer strings, and every graph-node alias."""
    ids = sorted(item["row"]["instance_id"] for item in candidates)
    by_id = {item["row"]["instance_id"]: item for item in candidates}
    parent = {instance_id: instance_id for instance_id in ids}

    def find(instance_id):
        while parent[instance_id] != instance_id:
            parent[instance_id] = parent[parent[instance_id]]
            instance_id = parent[instance_id]
        return instance_id

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen = {"qids": {}, "facts": {}, "surfaces": {}}
    evidence = {}
    for instance_id in ids:
        item = by_id[instance_id]
        qids, fact_ids, surfaces = graph_evidence(item, helpers)
        evidence[instance_id] = {"qids": qids, "facts": fact_ids, "surfaces": surfaces}
        for kind, values in evidence[instance_id].items():
            for value in values:
                prior = seen[kind].get(value)
                if prior is None:
                    seen[kind][value] = instance_id
                else:
                    union(instance_id, prior)
    grouped = collections.defaultdict(list)
    for instance_id in ids:
        grouped[find(instance_id)].append(instance_id)
    members = sorted((sorted(values) for values in grouped.values()), key=lambda values: (-len(values), values[0]))
    group_rows = []
    for values in members:
        group_id = "g-" + hashlib.sha256("\n".join(values).encode()).hexdigest()[:12]
        group_items = [by_id[instance_id] for instance_id in values]
        group_rows.append({
            "group_id": group_id,
            "source_qa_ids": values,
            "feature_counts": {
                "group_count": 1,
                "ambiguous_q1_cases": sum(len(item["q1_answers"]) > 1 for item in group_items),
                "date_like_final_answers": sum(
                    bool(DATE_ANSWER.fullmatch(str(item["row"]["gold_answers"][0]).strip()))
                    for item in group_items
                ),
            },
        })
    return group_rows


def assign_groups(candidates, helpers):
    group_rows = candidate_groups(candidates, helpers)
    by_id = {item["row"]["instance_id"]: item for item in candidates}

    # Stratify the indivisible groups on group count, multi-answer Q1 cases,
    # and date/year-shaped final answers. These are balancing fields, not
    # inclusion or exclusion criteria.
    features = ("group_count", "ambiguous_q1_cases", "date_like_final_answers")
    group_rows.sort(key=lambda group: group["group_id"])
    split_for_group = solve_group_assignment(group_rows, features)
    assignments = []
    for group in group_rows:
        split = split_for_group[group["group_id"]]
        for instance_id in group["source_qa_ids"]:
            assignments.append({
                "source_qa_id": instance_id,
                "group_id": group["group_id"],
                "split": split,
                "status": "DRAFT_NOT_APPROVED",
                "do_not_train": True,
            })
    assignments.sort(key=lambda row: row["source_qa_id"])
    return group_rows, assignments, by_id


def solve_group_assignment(groups, features):
    """Minimize proportional stratification error subject to exact row counts."""
    n_groups, n_splits = len(groups), len(SPLITS)
    n_x = n_groups * n_splits
    n_z = len(features) * n_splits
    n_vars = n_x + n_z
    rows, lower, upper = [], [], []

    def add_constraint(coeffs, lo=-float("inf"), hi=float("inf")):
        rows.append(coeffs)
        lower.append(lo)
        upper.append(hi)

    for gi in range(n_groups):
        add_constraint({gi * n_splits + si: 1.0 for si in range(n_splits)}, 1.0, 1.0)
    for si, split in enumerate(SPLITS):
        add_constraint(
            {gi * n_splits + si: len(groups[gi]["source_qa_ids"]) for gi in range(n_groups)},
            TARGET_ROWS[split], TARGET_ROWS[split],
        )

    feature_totals = {
        feature: sum(group["feature_counts"][feature] for group in groups)
        for feature in features
    }
    for fi, feature in enumerate(features):
        for si, split in enumerate(SPLITS):
            target = feature_totals[feature] * TARGET_ROWS[split] / sum(TARGET_ROWS.values())
            z_index = n_x + fi * n_splits + si
            count_coeffs = {
                gi * n_splits + si: groups[gi]["feature_counts"][feature]
                for gi in range(n_groups)
                if groups[gi]["feature_counts"][feature]
            }
            add_constraint({**count_coeffs, z_index: -1.0}, hi=target)
            add_constraint({**{index: -value for index, value in count_coeffs.items()}, z_index: -1.0}, hi=-target)

    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    for ri, coeffs in enumerate(rows):
        for ci, value in coeffs.items():
            matrix[ri, ci] = value
    objective = [0.0] * n_vars
    for fi, feature in enumerate(features):
        scale = 1.0 / max(1, feature_totals[feature])
        for si in range(n_splits):
            objective[n_x + fi * n_splits + si] = scale
    integrality = [1] * n_x + [0] * n_z
    bounds = Bounds([0] * n_x + [0] * n_z, [1] * n_x + [float("inf")] * n_z)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Could not find exact group allocation: {result.message}")
    allocation = {}
    for gi, group in enumerate(groups):
        chosen = [si for si in range(n_splits) if result.x[gi * n_splits + si] > 0.5]
        if len(chosen) != 1:
            raise RuntimeError(f"Solver did not assign exactly one split to {group['group_id']}")
        allocation[group["group_id"]] = SPLITS[chosen[0]]
    return allocation


def entity_qids_and_surfaces(candidates, helpers):
    all_qids, all_surfaces = set(), set()
    for item in candidates:
        qids, _, surfaces = graph_evidence(item, helpers)
        all_qids.update(qids)
        all_surfaces.update(surfaces)
    return all_qids, all_surfaces


def choose_false_entity(item, facts_by_qid, forbidden_qids, forbidden_surfaces, helpers, other_split_decoys):
    graph_qa = item["graph_qa"]
    r1 = graph_qa["r1"]
    q1_answers = {helpers.normalize(value) for value in item["q1_answers"]}
    subject_qid, target_relation = r1["qid"], r1["relation_str"]
    same_relation, same_subject = [], []
    for source_qid in sorted(facts_by_qid):
        if source_qid == subject_qid:
            continue
        for statement in facts_by_qid[source_qid]:
            parts = [part.strip() for part in statement.split("→")]
            if len(parts) < 3 or parts[1] != target_relation:
                continue
            object_text = parts[2]
            object_norm = helpers.normalize(object_text)
            object_qid = helpers.qid(object_text)
            if (
                object_norm not in forbidden_surfaces
                and object_norm not in q1_answers
                and object_norm not in other_split_decoys["surfaces"]
                and (not object_qid or object_qid not in forbidden_qids | other_split_decoys["qids"])
            ):
                same_relation.append((object_norm, object_text, source_qid, statement))
    if same_relation:
        source_kind = "same_relation_other_subject"
        options = same_relation
    else:
        for statement in facts_by_qid.get(subject_qid, []):
            parts = [part.strip() for part in statement.split("→")]
            if len(parts) < 3 or parts[1] == target_relation:
                continue
            object_text = parts[2]
            object_norm = helpers.normalize(object_text)
            object_qid = helpers.qid(object_text)
            if (
                object_norm not in forbidden_surfaces
                and object_norm not in q1_answers
                and object_norm not in other_split_decoys["surfaces"]
                and (not object_qid or object_qid not in forbidden_qids | other_split_decoys["qids"])
            ):
                same_subject.append((object_norm, object_text, subject_qid, statement))
        source_kind = "same_subject_other_relation"
        options = same_subject
    if not options:
        raise RuntimeError(f"No source-backed false-answer control for {item['row']['instance_id']}")
    options.sort(key=lambda value: (value[0], value[2], value[3]))
    index = int(hashlib.sha256(item["row"]["instance_id"].encode()).hexdigest()[:8], 16) % len(options)
    _, entity, source_qid, source_fact = options[index]
    if helpers.normalize(entity) in q1_answers:
        raise RuntimeError("Selected negative control is a source-supported Q1 answer")
    return {
        "entity": re.sub(r"\s*\(Q\d+\)$", "", entity).strip(),
        "entity_qid": helpers.qid(entity),
        "source_kind": source_kind,
        "source_subject_qid": source_qid,
        "source_fact_text": source_fact,
    }


def make_probe(candidate, decoy):
    row, graph_qa = candidate["row"], candidate["graph_qa"]
    q1 = re.sub(r"</?bridge>", "", str(candidate["graph"].get("decomposed_questions", {}).get("Q1", ""))).strip()
    if not q1:
        raise RuntimeError(f"Missing Q1 text for {row['instance_id']}")
    target = re.sub(r"\s*\(Q\d+\)$", "", graph_qa["n2"]).strip()
    if target.casefold() in q1.casefold():
        raise RuntimeError(f"Target bridge appears in direct question for {row['instance_id']}")
    return q1, target, decoy["entity"], (
        f'Would "{target}" be a valid answer to this question: "{q1}"? '
        "Reply with exactly YES or NO."
    ), (
        f'Would "{decoy["entity"]}" be a valid answer to this question: "{q1}"? '
        "Reply with exactly YES or NO."
    )


def main():
    helpers = load_audit_helpers()
    source_manifest = read_json(PILOT / "source_manifest.json")
    source_paths = {
        "qa": SOURCE / "qa-sm.jsonl",
        "facts": SOURCE / "qa-sm-facts.jsonl",
        "documents": SOURCE / "qa-sm-docs.jsonl",
    }
    source_hashes = {key: sha256(path) for key, path in source_paths.items()}
    for key, path in source_paths.items():
        if source_hashes[key] != source_manifest[key]["sha256"]:
            raise RuntimeError(f"Source hash changed for {key}; aborting")

    qa_rows = read_jsonl(source_paths["qa"])
    facts_rows = read_jsonl(source_paths["facts"])
    facts_by_qid = {row["qid"]: row["facts"] for row in facts_rows}
    eligibility = {
        row["instance_id"]: row
        for row in read_jsonl(AUDIT / "type_a_target_eligibility.jsonl")
    }
    conflicts = {
        row["source_qa_id"]
        for row in read_jsonl(PILOT / "excluded_r2_conflict.jsonl")
    }
    candidates = source_candidates(helpers, qa_rows, facts_by_qid, eligibility, conflicts)
    groups, assignments, candidate_by_id = assign_groups(candidates, helpers)
    assignment_by_id = {row["source_qa_id"]: row for row in assignments}
    all_candidate_qids, all_candidate_surfaces = entity_qids_and_surfaces(candidates, helpers)
    legacy_corpus = read_jsonl(PILOT / "sft_training/composition_corpus_disjoint.jsonl")
    legacy_train_ids = {row["source_qa_id"] for row in legacy_corpus if row.get("split") == "train_only"}
    qa_by_id = {row["instance_id"]: row for row in qa_rows}
    legacy_train_qids, legacy_train_surfaces = set(), set()
    for source_id in legacy_train_ids:
        train_source = qa_by_id[source_id]
        train_graph = json.loads(train_source["mhop_qa"])
        train_graph_qa = train_graph.get("graph_qa", {})
        legacy_train_qids.update(train_source.get("gold_qids", []))
        legacy_train_qids.update(train_graph.get("entity_rename_map", {}))
        for key, value in train_graph_qa.items():
            if re.fullmatch(r"n\d+", key):
                legacy_train_surfaces.add(helpers.normalize(value))
                node_qid = helpers.qid(value)
                if node_qid:
                    legacy_train_qids.add(node_qid)
            if key.startswith("r") and isinstance(value, dict):
                legacy_train_qids.update(
                    qid for qid in (value.get("qid"), value.get("relation_subject_qid")) if qid
                )
    for train_row in legacy_corpus:
        if train_row.get("split") == "train_only" and train_row.get("language") == "en":
            legacy_train_surfaces.add(helpers.normalize(train_row["messages"][-1]["content"]))
    all_candidate_qids.update(legacy_train_qids)
    all_candidate_surfaces.update(value for value in legacy_train_surfaces if value)
    if len(assignment_by_id) != len(candidates):
        raise RuntimeError("Candidate IDs are not unique in draft assignments")

    case_rows, label_rows, translation_rows = [], [], []
    decoy_kinds = collections.Counter()
    decoys_by_split = {split: {"qids": set(), "surfaces": set()} for split in SPLITS}
    for instance_id in sorted(candidate_by_id):
        item = candidate_by_id[instance_id]
        row, graph_qa = item["row"], item["graph_qa"]
        split = assignment_by_id[instance_id]["split"]
        other_split_decoys = {
            "qids": set().union(*(decoys_by_split[other]["qids"] for other in SPLITS if other != split)),
            "surfaces": set().union(*(decoys_by_split[other]["surfaces"] for other in SPLITS if other != split)),
        }
        decoy = choose_false_entity(
            item, facts_by_qid, all_candidate_qids, all_candidate_surfaces, helpers, other_split_decoys
        )
        decoy_kinds[decoy["source_kind"]] += 1
        if decoy["entity_qid"]:
            decoys_by_split[split]["qids"].add(decoy["entity_qid"])
        decoys_by_split[split]["surfaces"].add(helpers.normalize(decoy["entity"]))
        q1, target, false_entity, positive_prompt, negative_prompt = make_probe(item, decoy)
        gold_final = str(row["gold_answers"][0]).strip()
        multi_question = str(row["query"]).strip()
        group_id = assignment_by_id[instance_id]["group_id"]
        protected = sorted({
            re.sub(r"\s*\(Q\d+\)$", "", str(value)).strip()
            for key, value in graph_qa.items()
            if re.fullmatch(r"n\d+", key)
        } | {false_entity})
        case_rows.append({
            "source_qa_id": instance_id,
            "group_id": group_id,
            "split": split,
            "question_graph_type": "A",
            "q1_question_en": q1,
            "target_bridge_en": target,
            "direct_positive_probe_en": positive_prompt,
            "direct_negative_probe_en": negative_prompt,
            "multi_hop_question_en": multi_question,
            "target_r1_fact_text": item["target_r1_fact"],
            "target_r2_fact_text": item["target_r2_fact"],
            "target_r1_fact_id": helpers.fact_id(graph_qa["r1"]["qid"], item["target_r1_fact"]),
            "target_r2_fact_id": helpers.fact_id(graph_qa["r2"]["qid"], item["target_r2_fact"]),
            "q1_source_supported_answers_en": item["q1_answers"],
            "final_answer_source_supported_en": gold_final,
            "negative_control_en": false_entity,
            "negative_control_source_kind": decoy["source_kind"],
            "negative_control_source_subject_qid": decoy["source_subject_qid"],
            "negative_control_source_fact": decoy["source_fact_text"],
            "protected_entity_names": protected,
            "do_not_train": True,
            "status": "DRAFT_NOT_APPROVED",
        })
        label_rows.extend([
            {"source_qa_id": instance_id, "group_id": group_id, "split": split,
             "probe_id": f"{instance_id}:direct-positive", "probe": "direct_positive",
             "gold_label": "YES", "target_fact_id": helpers.fact_id(graph_qa["r1"]["qid"], item["target_r1_fact"]),
             "do_not_train": True},
            {"source_qa_id": instance_id, "group_id": group_id, "split": split,
             "probe_id": f"{instance_id}:direct-negative", "probe": "direct_negative_control",
             "gold_label": "NO", "target_fact_id": None, "do_not_train": True},
            {"source_qa_id": instance_id, "group_id": group_id, "split": split,
             "probe_id": f"{instance_id}:multi-hop", "probe": "multi_hop",
             "gold_answers": [gold_final], "source_supported_answers": [gold_final], "do_not_train": True},
        ])
        for probe_type, prompt in (
            ("direct_positive", positive_prompt),
            ("direct_negative_control", negative_prompt),
            ("multi_hop", multi_question),
        ):
            translation_rows.append({
                "prompt_id": f"{instance_id}:{probe_type}",
                "source_qa_id": instance_id,
                "group_id": group_id,
                "split": split,
                "probe": probe_type,
                "text_en": prompt,
                "protected_entity_names": protected,
                "translation_status": "NOT_STARTED",
                "do_not_train": True,
            })

    # Check the optimized split and leakage graph before persisting anything.
    counts = collections.Counter(row["split"] for row in assignments)
    if dict(counts) != TARGET_ROWS:
        raise RuntimeError(f"Unexpected split row counts: {dict(counts)}")
    groups_by_split = collections.defaultdict(set)
    for assignment in assignments:
        groups_by_split[assignment["split"]].add(assignment["group_id"])
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            if groups_by_split[left] & groups_by_split[right]:
                raise RuntimeError(f"Leakage group crosses {left}/{right}")
    case_ids = {row["source_qa_id"] for row in case_rows}
    if len(case_ids) != 195 or len(label_rows) != 3 * len(case_rows) or len(translation_rows) != 3 * len(case_rows):
        raise RuntimeError("Draft case, label, or prompt coverage is incomplete")
    if any(row["do_not_train"] is not True for row in case_rows + label_rows + translation_rows):
        raise RuntimeError("Draft artifacts must remain quarantined")

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "candidate_cases.draft.jsonl", case_rows)
    write_jsonl(OUT / "assignments.draft.jsonl", assignments)
    write_jsonl(OUT / "labels/gold_labels.draft.jsonl", label_rows)
    write_jsonl(OUT / "translation_requests/en.draft.jsonl", translation_rows)
    group_manifest_rows = [
        {"group_id": group["group_id"], "source_qa_ids": group["source_qa_ids"],
         "split": assignment_by_id[group["source_qa_ids"][0]]["split"],
         "case_count": len(group["source_qa_ids"]), "feature_counts": group["feature_counts"]}
        for group in groups
    ]
    write_jsonl(OUT / "leakage_groups.draft.jsonl", group_manifest_rows)

    run_manifest = {
        "status": "DRAFT_NOT_APPROVED",
        "do_not_train": True,
        "do_not_evaluate": True,
        "approval_status": "NOT_APPROVED_FOR_TRAINING_OR_EVALUATION",
        "source_hashes": source_hashes,
        "source_candidate_rule": "Type-A row with exact unique R1/R2 source facts, a unique source-supported final-answer set equal to the gold set, designated bridge as only Q1 source candidate with an R2 continuation, and not in the known 0080 source/graph conflict list.",
        "candidate_cases": len(case_rows),
        "requested_split_proportions": {"forget": 0.60, "retain": 0.20, "holdout": 0.20},
        "split_counts": dict(counts),
        "leakage_groups": len(groups),
        "largest_group": max(len(group["source_qa_ids"]) for group in groups),
        "group_assignment_method": "SciPy HiGHS mixed-integer allocation; exact case totals, minimize normalized absolute deviation from proportional allocation for group count, multi-answer-Q1 count, and date/year-shaped final answers.",
        "balancing_feature_counts": {
            "ambiguous_q1_cases": sum(len(item["q1_answers"]) > 1 for item in candidates),
            "date_like_final_answers": sum(bool(DATE_ANSWER.fullmatch(str(item["row"]["gold_answers"][0]).strip())) for item in candidates),
        },
        "probe_design": {
            "direct_positive": "Would [designated bridge] be a valid answer to [Q1]? Reply exactly YES or NO.",
            "direct_negative_control": "Same prompt with a source-derived entity that is not in the complete source-supported Q1 answer set.",
            "negative_control_source_counts": dict(decoy_kinds),
            "final_answer_probe": "Original English composed multi-hop question; final label source-verified unique.",
            "output_tokens": ["YES", "NO"],
            "translation_status": "NOT_STARTED",
            "translation_review_policy": "AI_ONLY_AUTHORIZED; not human-fluency certification",
        },
        "artifact_counts": {
            "case_rows": len(case_rows),
            "assignment_rows": len(assignments),
            "gold_label_rows": len(label_rows),
            "translation_request_rows": len(translation_rows),
            "prompts_per_case": 3,
        },
        "gates": {
            "translation_and_ai_review": "PENDING",
            "candidate_split_approval": "PENDING",
            "fresh_sft_corpus": "NOT_BUILT",
            "baseline_competence_gate": "NOT_RUN",
            "unlearning": "NOT_STARTED",
        },
    }
    for rel in (
        "candidate_cases.draft.jsonl", "assignments.draft.jsonl", "labels/gold_labels.draft.jsonl",
        "translation_requests/en.draft.jsonl", "leakage_groups.draft.jsonl",
    ):
        run_manifest.setdefault("artifact_sha256", {})[rel] = sha256(OUT / rel)
    (OUT / "manifest.draft.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": run_manifest["status"],
        "candidate_cases": run_manifest["candidate_cases"],
        "leakage_groups": run_manifest["leakage_groups"],
        "split_counts": run_manifest["split_counts"],
        "negative_control_source_counts": dict(decoy_kinds),
        "artifacts": run_manifest["artifact_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
