#!/usr/bin/env python3
"""Audit whether the proposed expanded Type-A pool is disjoint from current SFT.

This read-only audit preserves the existing v4 split, checkpoint, and results.
It compares the 195 source-backed candidate questions with the train-only rows
in the SFT corpus used by the corrected exploratory run. Entity and answer
surface overlaps are reported as leakage signals, not as semantic proof.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "data" / "synthworlds_sm_pilot_v4"
SOURCE = ROOT / "data" / "synthworlds_sm"
AUDIT = ROOT / "data" / "synthworlds_sm_reaudit"
RUN = ROOT / "reports" / "hypothesis_diagnostics" / "r1_synthworlds_v4_retainfix_corrected_2026-11-03"
CORPUS = RUN / "extracted" / "sft_training" / "composition_corpus_disjoint.jsonl"
OUT = ROOT / "reports" / "hypothesis_diagnostics" / "r1_candidate_sft_overlap_audit_2026-11-03"


def load_audit_module():
    path = PILOT / "scripts" / "audit_source_answer_cardinality.py"
    spec = importlib.util.spec_from_file_location("r1_source_cardinality_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source cardinality audit helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def qids_for(row: dict, graph: dict, audit_row: dict | None = None) -> set[str]:
    """Use graph/entity QID evidence fields from the candidate-group audit."""
    qids = set(row.get("gold_qids", []))
    if audit_row:
        qids.update(audit_row.get("graph_qids", []))
    qids.update(graph.get("entity_rename_map", {}).keys())
    graph_qa = graph.get("graph_qa", {})
    for key, value in graph_qa.items():
        if re.fullmatch(r"n\d+", key):
            match = re.search(r"\((Q\d+)\)$", str(value))
            if match:
                qids.add(match.group(1))
        if key.startswith("r") and isinstance(value, dict) and value.get("relation_subject_qid"):
            qids.add(value["relation_subject_qid"])
    return qids


def entity_surfaces(row: dict, graph: dict, audit_row: dict | None, normalize) -> set[str]:
    """Collect exact normalized answer/entity surfaces for conservative overlap screening."""
    values = list(row.get("gold_answers", []))
    if audit_row:
        values.extend(audit_row.get("q1_possible_answers", []))
    graph_qa = graph.get("graph_qa", {})
    values.extend(
        value for key, value in graph_qa.items()
        if re.fullmatch(r"n\d+", key) and isinstance(value, str)
    )
    return {normalized for value in values if (normalized := normalize(value))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    helpers = load_audit_module()
    read_jsonl = helpers.read_jsonl
    read_json = helpers.read_json

    source_manifest = read_json(PILOT / "source_manifest.json")
    source_paths = {
        "qa": SOURCE / "qa-sm.jsonl",
        "facts": SOURCE / "qa-sm-facts.jsonl",
        "documents": SOURCE / "qa-sm-docs.jsonl",
    }
    for key, path in source_paths.items():
        actual = sha256(path)
        expected = source_manifest[key]["sha256"]
        if actual != expected:
            raise RuntimeError(f"Source checksum mismatch for {key}: {actual} != {expected}")

    qa_rows = read_jsonl(source_paths["qa"])
    qa_by_id = {row["instance_id"]: row for row in qa_rows}
    facts_rows = read_jsonl(source_paths["facts"])
    facts_by_qid = {row["qid"]: row["facts"] for row in facts_rows}
    eligibility = {
        row["instance_id"]: row
        for row in read_jsonl(AUDIT / "type_a_target_eligibility.jsonl")
    }
    known_conflicts = {
        row["source_qa_id"]
        for row in read_jsonl(PILOT / "excluded_r2_conflict.jsonl")
    }

    # Reproduce the explicitly source-backed 195-case candidate selection.
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
            raise RuntimeError(f"Source Q1 candidates disagree with audit for {instance_id}")
        expected_r1 = f"{graph_qa['n1']} → {r1['relation_str']} → {graph_qa['n2']}"
        if (
            audit_row.get("matching_r1_exact_fact_records") != 1
            or facts_by_qid.get(r1["qid"], []).count(expected_r1) != 1
        ):
            raise RuntimeError(f"Target first-hop fact is not exactly source-verified for {instance_id}")
        matched_r2 = audit_row.get("r2_matched_fact_texts", [])
        if len(matched_r2) != 1 or matched_r2[0] not in facts_by_qid.get(r2["qid"], []):
            raise RuntimeError(f"Target second-hop fact is not exactly source-verified for {instance_id}")
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
        other_bridges = [
            bridge
            for bridge, outputs in bridge_outputs.items()
            if bridge != graph_qa["n2"] and outputs
        ]
        if (
            target_outputs == gold
            and not other_bridges
            and instance_id not in known_conflicts
        ):
            candidates.append({"row": row, "audit": audit_row, "graph": graph, "graph_qa": graph_qa})

    if len(candidates) != 195:
        raise RuntimeError(f"Expected 195 candidate rows; found {len(candidates)}")

    corpus_rows = read_jsonl(CORPUS)
    sft_run = read_json(RUN / "extracted" / "sft_baseline" / "run_manifest.json")
    if sha256(CORPUS) != sft_run["corpus_sha256"]:
        raise RuntimeError("Current SFT corpus hash does not match the completed SFT run manifest")
    if len(corpus_rows) != sft_run["rows"]:
        raise RuntimeError("Current SFT corpus row count does not match the completed run manifest")

    candidate_ids = {item["row"]["instance_id"] for item in candidates}
    candidate_qids = {
        item["row"]["instance_id"]: qids_for(item["row"], item["graph"], item["audit"])
        for item in candidates
    }
    candidate_answer_sources: dict[str, set[str]] = collections.defaultdict(set)
    candidate_surface_sources: dict[str, set[str]] = collections.defaultdict(set)
    candidate_query_norms = set()
    for item in candidates:
        instance_id = item["row"]["instance_id"]
        candidate_query_norms.add(helpers.normalize(item["row"].get("query", "")))
        for question in item["graph"].get("decomposed_questions", {}).values():
            candidate_query_norms.add(
                helpers.normalize(re.sub(r"</?bridge>", "", str(question)))
            )
        answers = list(item["row"].get("gold_answers", []))
        answers.extend(item["audit"].get("q1_possible_answers", []))
        for answer in answers:
            candidate_answer_sources[helpers.normalize(answer)].add(instance_id)
        for candidate_surface in entity_surfaces(
            item["row"], item["graph"], item["audit"], helpers.normalize
        ):
            candidate_surface_sources[candidate_surface].add(instance_id)

    train_only_ids = {
        row["source_qa_id"] for row in corpus_rows if row.get("split") == "train_only"
    }
    train_only_types = collections.Counter(
        qa_by_id[instance_id]["question_graph_type"] for instance_id in train_only_ids
    )
    if set(train_only_types) - {"A", "B"}:
        raise RuntimeError(f"Unexpected train-only graph types: {sorted(set(train_only_types) - {'A', 'B'})}")

    train_only_candidate_id_overlap = candidate_ids & train_only_ids
    train_b_ids = {
        instance_id
        for instance_id in train_only_ids
        if qa_by_id[instance_id]["question_graph_type"] == "B"
    }
    train_b_qids = {
        instance_id: qids_for(qa_by_id[instance_id], json.loads(qa_by_id[instance_id]["mhop_qa"]))
        for instance_id in train_b_ids
    }
    all_candidate_qids = set().union(*candidate_qids.values())
    train_b_qid_overlap_ids = {
        instance_id
        for instance_id, qids in train_b_qids.items()
        if qids & all_candidate_qids
    }
    candidate_qid_overlap_ids = {
        instance_id
        for instance_id, qids in candidate_qids.items()
        if qids & set().union(*train_b_qids.values())
    }

    # Exact answer-surface collisions are conservative signals, not proof of a
    # repeated target relation. Count English outputs so language variants do
    # not inflate the source-QA count.
    train_b_english_outputs = [
        row
        for row in corpus_rows
        if row.get("split") == "train_only"
        and row.get("language") == "en"
        and row.get("source_qa_id") in train_b_ids
    ]
    answer_surface_hits = []
    answer_surface_candidate_ids = set()
    for row in train_b_english_outputs:
        answer = row["messages"][-1]["content"]
        matched_candidates = candidate_answer_sources.get(helpers.normalize(answer), set())
        if matched_candidates:
            answer_surface_hits.append({
                "source_qa_id": row["source_qa_id"],
                "sft_id": row["sft_id"],
                "answer": answer,
                "candidate_ids": sorted(matched_candidates),
            })
            answer_surface_candidate_ids.update(matched_candidates)
    answer_surface_train_b_ids = {item["source_qa_id"] for item in answer_surface_hits}

    # Estimate whether other source chains could supply independent composition
    # practice for a fresh SFT corpus. This is a conservative row-level screen
    # against every candidate case, not yet a fact/passage-verified training set.
    candidate_surface_set = set(candidate_surface_sources)
    source_pool_screen = collections.defaultdict(collections.Counter)
    touched_candidate_by_pool_row = collections.defaultdict(set)
    for row in qa_rows:
        instance_id = row["instance_id"]
        if instance_id in candidate_ids:
            continue
        graph = json.loads(row["mhop_qa"])
        audit_row = eligibility.get(instance_id)
        row_qids = qids_for(row, graph, audit_row)
        row_surfaces = entity_surfaces(row, graph, audit_row, helpers.normalize)
        qid_hit = bool(row_qids & all_candidate_qids)
        surface_hit = bool(row_surfaces & candidate_surface_set)
        graph_type = str(row.get("question_graph_type", "unknown"))
        bucket = (
            "both" if qid_hit and surface_hit else
            "qid_overlap_only" if qid_hit else
            "surface_overlap_only" if surface_hit else
            "screen_disjoint"
        )
        source_pool_screen[graph_type][bucket] += 1
        if qid_hit or surface_hit:
            touched_candidate_by_pool_row[graph_type].update(
                instance_id_candidate
                for surface in row_surfaces
                for instance_id_candidate in candidate_surface_sources.get(surface, set())
            )
            for candidate_id, qids in candidate_qids.items():
                if qids & row_qids:
                    touched_candidate_by_pool_row[graph_type].add(candidate_id)
    source_pool_summary = {
        "selection_screen": "Rows outside the 195 candidate IDs with no exact candidate graph/entity QID overlap and no exact normalized graph-node, Q1-answer, or final-answer surface overlap.",
        "counts_by_graph_type": {
            graph_type: {
                **dict(sorted(counts.items())),
                "rows_total": sum(counts.values()),
                "rows_screen_disjoint": counts["screen_disjoint"],
            }
            for graph_type, counts in sorted(source_pool_screen.items())
        },
        "screen_disjoint_rows_total": sum(
            counts["screen_disjoint"] for counts in source_pool_screen.values()
        ),
        "candidate_rows_touched_by_screened_source_rows_by_graph_type": {
            graph_type: len(candidate_ids_for_type)
            for graph_type, candidate_ids_for_type in sorted(touched_candidate_by_pool_row.items())
        },
        "status": "SCREEN_ONLY_NOT_APPROVED_FOR_TRAINING",
        "limitations": [
            "Exact QID and surface disjointness does not verify source fact paths, prompt correctness, document-passage non-disclosure, or translation quality.",
            "Counts are QA rows, not independent groups or approved SFT examples.",
            "A new candidate split must be frozen before building and auditing any replacement training corpus.",
        ],
    }

    # The existing B-chain extension is a more actionable rebuild candidate
    # than the broader row-level screen above. Require exact source facts,
    # candidate-disjoint graph/entity QIDs and surfaces, and complete reviewed
    # five-language rows already present in the corpus.
    type_b_rows_by_id: dict[str, list[dict]] = collections.defaultdict(list)
    for row in corpus_rows:
        if row.get("example_type") == "disjoint_type_b_composition_support":
            type_b_rows_by_id[row["source_qa_id"]].append(row)
    type_b_verified_ids = []
    type_b_exclusion_counts = collections.Counter()
    expected_type_b_hops = {
        "composition_support_q1",
        "composition_support_q2",
        "composition_support_q3",
        "composition_3hop",
    }
    for instance_id in sorted(train_b_ids):
        row = qa_by_id[instance_id]
        graph = json.loads(row["mhop_qa"])
        graph_qa = graph.get("graph_qa", {})
        nodes = [graph_qa.get(f"n{index}") for index in range(1, 5)]
        relations = [graph_qa.get(f"r{index}", {}) for index in range(1, 4)]
        if not all(nodes) or any(not rel.get("qid") or not rel.get("relation_str") for rel in relations):
            type_b_exclusion_counts["incomplete_graph"] += 1
            continue
        node_qids = [helpers.qid(node) for node in nodes]
        if any(not node_qid for node_qid in node_qids) or len(set(node_qids)) != 4:
            type_b_exclusion_counts["invalid_node_qids"] += 1
            continue
        chain_statements = []
        chain_fact_ids = []
        source_path_valid = True
        for subject, obj, relation in zip(nodes, nodes[1:], relations):
            subject_qid = helpers.qid(subject)
            statement = f"{subject} → {relation['relation_str']} → {obj}"
            if relation["qid"] != subject_qid or facts_by_qid.get(subject_qid, []).count(statement) != 1:
                source_path_valid = False
                break
            chain_statements.append(statement)
            chain_fact_ids.append(helpers.fact_id(subject_qid, statement))
        if not source_path_valid:
            type_b_exclusion_counts["source_path_missing_or_nonunique"] += 1
            continue
        if len(row.get("gold_answers", [])) != 1 or helpers.normalize(row["gold_answers"][0]) != helpers.normalize(nodes[3]):
            type_b_exclusion_counts["gold_final_node_mismatch"] += 1
            continue
        if graph.get("composed_question") != row.get("query"):
            type_b_exclusion_counts["composition_query_mismatch"] += 1
            continue
        if helpers.normalize(nodes[3]) in helpers.normalize(row.get("query", "")):
            type_b_exclusion_counts["final_answer_disclosed"] += 1
            continue
        if qids_for(row, graph) & all_candidate_qids:
            type_b_exclusion_counts["candidate_entity_qid_overlap"] += 1
            continue
        if entity_surfaces(row, graph, None, helpers.normalize) & candidate_surface_set:
            type_b_exclusion_counts["candidate_entity_or_answer_surface_overlap"] += 1
            continue
        if helpers.normalize(row.get("query", "")) in candidate_query_norms:
            type_b_exclusion_counts["candidate_question_overlap"] += 1
            continue
        translated_rows = type_b_rows_by_id.get(instance_id, [])
        by_language_hop = {
            (item.get("language"), item.get("hop")): item
            for item in translated_rows
        }
        expected_pairs = {
            (language, hop)
            for language in ("en", "fr", "de", "hi", "th")
            for hop in expected_type_b_hops
        }
        if len(translated_rows) != 20 or set(by_language_hop) != expected_pairs:
            type_b_exclusion_counts["incomplete_five_language_translation_rows"] += 1
            continue
        if any(
            item.get("split") != "train_only"
            or item.get("do_not_train") is not False
            or item.get("translation_review_mode") != "AI_ONLY"
            or item.get("human_translation_reviewed") is not False
            or item.get("review_mode") != "AI_ONLY"
            or item.get("human_review_status") != "NOT_PERFORMED"
            for item in translated_rows
        ):
            type_b_exclusion_counts["translation_review_provenance_incomplete"] += 1
            continue
        if any(
            set(item.get("source_fact_texts", [])) != set(chain_statements)
            or set(item.get("chain_fact_ids", [])) != set(chain_fact_ids)
            for item in translated_rows
        ):
            type_b_exclusion_counts["translated_training_source_path_mismatch"] += 1
            continue
        type_b_verified_ids.append(instance_id)
    type_b_rebuild_pool = {
        "status": "CANDIDATE_ONLY_NOT_APPROVED",
        "source_qa_ids": type_b_verified_ids,
        "verified_source_chains": len(type_b_verified_ids),
        "train_rows_per_language": len(type_b_verified_ids) * 4,
        "all_language_rows": len(type_b_verified_ids) * 4 * 5,
        "complete_reviewed_five_language_coverage": True,
        "translation_review_mode": "AI_ONLY",
        "human_translation_reviewed": False,
        "excluded_current_type_b_train_only_ids": dict(sorted(type_b_exclusion_counts.items())),
        "limitations": [
            "These source-disjoint chains may provide composition practice for a rebuilt SFT corpus; they are not an approved corpus by themselves.",
            "Candidate selection still needs review with the finalized new split and passage-leakage policy.",
            "Translation checks are AI-only and do not establish human fluency.",
        ],
    }

    group_sizes, group_members = helpers.expanded_groups(candidates)
    full_target = {"forget": 117, "retain": 39, "holdout": 39}
    full_retain_subset = helpers.subset_sum(group_sizes, full_target["retain"])
    full_holdout_subset = (
        helpers.subset_sum(group_sizes, full_target["holdout"], full_retain_subset)
        if full_retain_subset is not None else None
    )
    full_exact_split = full_retain_subset is not None and full_holdout_subset is not None
    touched_candidate_ids = (
        train_only_candidate_id_overlap
        | candidate_qid_overlap_ids
        | answer_surface_candidate_ids
    )
    touched_group_ids = {
        group_id
        for group_id, members in group_members.items()
        if touched_candidate_ids & set(members)
    }
    touched_group_members = {
        instance_id
        for group_id in touched_group_ids
        for instance_id in group_members[group_id]
    }
    remaining_candidates = candidate_ids - touched_group_members
    remaining_group_sizes, _ = helpers.expanded_groups([
        item for item in candidates if item["row"]["instance_id"] in remaining_candidates
    ])
    remaining_n = len(remaining_candidates)
    remaining_targets = {
        "forget": round(0.60 * remaining_n),
        "retain": round(0.20 * remaining_n),
        "holdout": remaining_n - round(0.60 * remaining_n) - round(0.20 * remaining_n),
    }
    retain_subset = helpers.subset_sum(remaining_group_sizes, remaining_targets["retain"])
    holdout_subset = (
        helpers.subset_sum(remaining_group_sizes, remaining_targets["holdout"], retain_subset)
        if retain_subset is not None else None
    )
    remaining_exact_split = retain_subset is not None and holdout_subset is not None

    result = {
        "status": "CURRENT_SFT_OVERLAP_AUDIT_COMPLETE_CANDIDATE_ONLY",
        "source_hashes": {key: sha256(path) for key, path in source_paths.items()},
        "sft_corpus_sha256": sha256(CORPUS),
        "current_sft_run_manifest_sha256": sha256(RUN / "extracted" / "sft_baseline" / "run_manifest.json"),
        "candidate_pool": {
            "candidate_count": len(candidates),
            "candidate_internal_group_count": len(group_sizes),
            "candidate_internal_largest_group": max(group_sizes),
            "candidate_60_20_20_sizes": full_target,
            "candidate_60_20_20_group_preserving_split_feasible": full_exact_split,
        },
        "current_sft_train_only": {
            "unique_source_qa_ids": len(train_only_ids),
            "source_qa_ids_by_graph_type": dict(sorted(train_only_types.items())),
            "type_a_train_only_ids_that_are_in_candidate_pool": len(train_only_candidate_id_overlap),
            "type_a_candidate_ids_directly_trained_as_train_only": sorted(train_only_candidate_id_overlap),
            "type_b_train_only_qas_with_candidate_entity_qid_overlap": len(train_b_qid_overlap_ids),
            "candidate_rows_with_type_b_train_only_entity_qid_overlap": len(candidate_qid_overlap_ids),
            "candidate_qas_with_type_b_answer_surface_overlap": len(answer_surface_candidate_ids),
            "type_b_train_only_qas_with_exact_candidate_answer_surface_overlap": len(answer_surface_train_b_ids),
            "english_type_b_train_only_output_rows_with_answer_surface_overlap": len(answer_surface_hits),
            "candidate_groups_touched_by_current_train_only_overlap_signals": len(touched_group_ids),
            "candidate_rows_removed_if_touched_groups_are_excluded": len(touched_group_members),
            "rows_remaining_after_conservative_group_exclusion": remaining_n,
            "groups_remaining_after_conservative_group_exclusion": len(remaining_group_sizes),
            "remaining_split_sizes": remaining_targets,
            "remaining_group_preserving_split_feasible": remaining_exact_split,
            "answer_surface_overlap_is_semantic_proof": False,
        },
        "replacement_train_only_source_pool_screen": source_pool_summary,
        "verified_type_b_rebuild_pool": type_b_rebuild_pool,
        "conclusion": (
            "Do not evaluate the 195-case expansion with the current v4 SFT checkpoint. "
            "Fifty candidate questions are present as train_only Type-A examples. "
            "Additional Type-B entity and answer-surface overlaps need a training-corpus "
            "rebuild and a fresh baseline. The exact 117/39/39 split feasibility applies "
            "only within the candidate pool; it does not certify separation from SFT data."
        ),
        "limitations": [
            "QID overlap is an entity-level warning and does not alone prove exposure of the exact target relation.",
            "Exact answer-surface overlap may be a shared answer entity and is a conservative screen, not semantic adjudication.",
            "The audit does not approve a split, translated prompts, or any training/evaluation use.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_overlap_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Candidate-pool overlap with the completed v4 SFT corpus",
        "",
        "Status: read-only audit. The existing candidate split, SFT checkpoints, translations, and experiment outputs were not changed.",
        "",
        "## Findings",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Source-backed Type-A candidate pool | {len(candidates)} cases in {len(group_sizes)} internal entity/fact/answer groups |",
        f"| Candidate-only exact 60/20/20 allocation | 117 / 39 / 39; feasible: {str(full_exact_split).lower()} |",
        f"| Current SFT train-only source QA IDs | {len(train_only_ids)} ({train_only_types['A']} Type A, {train_only_types['B']} Type B) |",
        f"| Candidate cases already present as train-only Type A | {len(train_only_candidate_id_overlap)} |",
        f"| Type-B train-only QA IDs sharing a candidate graph/entity QID | {len(train_b_qid_overlap_ids)} / {len(train_b_ids)} |",
        f"| Candidate cases with a Type-B train-only QID overlap | {len(candidate_qid_overlap_ids)} |",
        f"| Type-B train-only QA IDs with an exact candidate answer-surface overlap | {len(answer_surface_train_b_ids)} / {len(train_b_ids)} |",
        f"| English Type-B SFT output rows with an exact candidate answer-surface overlap | {len(answer_surface_hits)} |",
        f"| Candidate groups touched by these train-only overlap screens | {len(touched_group_ids)} |",
        f"| Candidate rows remaining after excluding those whole groups | {remaining_n} in {len(remaining_group_sizes)} groups |",
        "",
        "## Candidate source pool for a rebuilt SFT corpus",
        "",
        "This first-pass screen checks non-candidate QA rows for exact graph/entity QID and normalized entity/answer-surface overlap with the full 195-case pool. A row passing it is only a training-pool candidate; source-path and passage audits are still required.",
        "",
        "| Graph type | Non-candidate rows | Entity-QID and surface disjoint rows | Entity-QID-only overlap | Surface-only overlap | Both overlaps |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {graph_type} | {counts['rows_total']} | {counts['screen_disjoint']} | {counts.get('qid_overlap_only', 0)} | {counts.get('surface_overlap_only', 0)} | {counts.get('both', 0)} |"
            for graph_type, counts in sorted(source_pool_summary["counts_by_graph_type"].items())
        ],
        f"| **Total disjoint rows** | -- | **{source_pool_summary['screen_disjoint_rows_total']}** | -- | -- | -- |",
        "",
        "The disjoint-row total is not a verified replacement corpus. Each selected chain still needs exact fact-path validation, composition-question/answer checks, passage-disclosure screening, and translation review.",
        "",
        f"A stronger check within the already translated Type-B training-only rows validates exact three-edge source paths, final-node labels, no candidate QID/entity/answer-surface overlap, and all 20 prompt rows per chain (four prompt types in five languages). It leaves **{len(type_b_verified_ids)} chains** ({len(type_b_verified_ids) * 4} rows per language). These are useful replacement-corpus candidates, not an approved SFT corpus; they remain AI-only reviewed.",
        "",
        "## Interpretation",
        "",
        "The candidate-only split calculation is not enough to make this expansion safe with the existing checkpoint. Fifty of the 195 candidate questions were explicitly included as Type-A `train_only` SFT examples. The Type-B train-only set also shares graph/entity QIDs and exact answer strings with some candidates. QID overlap and answer-surface overlap are conservative exposure warnings, not proof that a specific target triple was taught, but they require a fresh corpus design before this set can be evaluated.",
        "",
        "The 117/39/39 feasibility result means only that the candidate set can be partitioned internally by its connected groups. It does not account for SFT exposure. The current SFT checkpoint must not be reused for this 195-case expansion. A new version should reserve candidate forget/retain/holdout groups first, remove or replace train-only rows that overlap those groups, build the direct-probe and multilingual evaluation data, train a new baseline, and re-run the competence gate before unlearning.",
        "",
        "The answer-surface screen matches candidate Q1 possible-answer and final-answer strings against English Type-B SFT outputs after the audit's normalized exact-answer transform. It is a conservative screen; no claim-level human adjudication was performed.",
        "",
        "## Provenance",
        "",
        f"The source JSONL hashes are rechecked against `{(PILOT / 'source_manifest.json').relative_to(ROOT)}`. The SFT corpus hash is rechecked against its completed run manifest. Candidate membership is reproduced using `{(PILOT / 'scripts' / 'audit_source_answer_cardinality.py').relative_to(ROOT)}` helpers and the same exact source-answer/path conditions. Machine-readable results: `{(OUT / 'training_overlap_audit.json').relative_to(ROOT)}`.",
        "",
    ]
    (OUT / "training_overlap_audit.md").write_text("\n".join(md), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
