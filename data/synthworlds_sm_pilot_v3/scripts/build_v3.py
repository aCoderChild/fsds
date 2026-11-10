#!/usr/bin/env python3
"""Build an entity/fact/answer-isolated, review-only SynthWorlds-SM pilot."""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SOURCE = DATA / "synthworlds_sm"
AUDIT = DATA / "synthworlds_sm_reaudit"
OUT = DATA / "synthworlds_sm_pilot_v3"
SEED = "20261102-entity-fact-answer-v1"
SOURCE_FILES = {
    "qa": "qa-sm.jsonl",
    "documents": "qa-sm-docs.jsonl",
    "facts": "qa-sm-facts.jsonl",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fact_id(qid: str, statement: str) -> str:
    return "derived-sha256:" + sha_bytes((qid + "\0" + statement).encode("utf-8"))


def qids_for(row, graph):
    qids = set(row.get("gold_qids", []))
    graph_qa = graph.get("graph_qa", {})
    for key, value in graph_qa.items():
        if re.fullmatch(r"n\d+", key):
            match = re.search(r"\((Q\d+)\)$", value)
            if match:
                qids.add(match.group(1))
    rename_map = graph.get("entity_rename_map", {})
    qids.update(rename_map)
    for key, rel in graph_qa.items():
        if key.startswith("r") and isinstance(rel, dict) and rel.get("relation_subject_qid"):
            qids.add(rel["relation_subject_qid"])
    return qids


class DSU:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def validate_and_build():
    qa_path = SOURCE / SOURCE_FILES["qa"]
    docs_path = SOURCE / SOURCE_FILES["documents"]
    facts_path = SOURCE / SOURCE_FILES["facts"]
    audit_path = AUDIT / "type_a_target_eligibility.jsonl"
    old_assignment_path = DATA / "synthworlds_sm_pilot_v2" / "verified_pilot_assignments.jsonl"
    for path in (qa_path, docs_path, facts_path, audit_path, old_assignment_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")

    qa_rows, docs_rows, facts_rows = map(read_jsonl, (qa_path, docs_path, facts_path))
    qa = {r["instance_id"]: r for r in qa_rows}
    docs = {r["qid"]: r for r in docs_rows}
    facts = {r["qid"]: r for r in facts_rows}
    if (len(qa), len(docs), len(facts)) != (1200, 6290, 6290):
        raise RuntimeError("Unexpected source cardinalities; stop without writing outputs")
    if any(len(m) != len(rows) for m, rows in ((qa, qa_rows), (docs, docs_rows), (facts, facts_rows))):
        raise RuntimeError("Duplicate source identifiers detected")

    source_checksums = {}
    for category, name in SOURCE_FILES.items():
        path = SOURCE / name
        source_checksums[category] = {
            "filename": name,
            "rows": sum(1 for line in path.open("rb") if line.strip()),
            "bytes": path.stat().st_size,
            "sha256": sha_file(path),
        }
    candidate_manifest = read_json(AUDIT / "candidate_split_manifest.json")
    for category, actual in source_checksums.items():
        expected = candidate_manifest["source_checksums"][category]
        if any(actual[k] != expected[k] for k in ("filename", "rows", "bytes", "sha256")):
            raise RuntimeError(f"Source checksum differs from prior audit for {category}")

    eligibility = {r["instance_id"]: r for r in read_jsonl(audit_path)}
    old_assignments = {r["source_qa_id"]: r for r in read_jsonl(old_assignment_path)}
    ids = sorted(i for i, r in eligibility.items() if r["eligible"] and i != "0080-qa-sm")
    if len(ids) != 129:
        raise RuntimeError(f"Expected 129 usable eligible cases after quarantine; got {len(ids)}")
    if set(ids) - set(qa):
        raise RuntimeError("Eligibility audit includes missing source QA IDs")

    evidence = {}
    for iid in ids:
        e, row = eligibility[iid], qa[iid]
        graph = json.loads(row["mhop_qa"])
        if graph["graph_qa"]["graph_type"] != "A" or row["question_graph_type"] != "A":
            raise RuntimeError(f"Non-Type-A case marked eligible: {iid}")
        n1, n2, n3 = (graph["graph_qa"][f"n{k}"] for k in (1, 2, 3))
        r1 = graph["graph_qa"]["r1"]
        if e["matching_r1_exact_fact_records"] != 1 or e["q1_possible_answer_count"] != 1:
            raise RuntimeError(f"First-hop target not uniquely selected: {iid}")
        if len(e["r2_matched_fact_texts"]) != 1:
            raise RuntimeError(f"Second-hop source fact not unique: {iid}")
        n3_surface = re.sub(r" \(Q\d+\)$", "", n3)
        if n3_surface not in row["gold_answers"]:
            raise RuntimeError(f"Gold answer does not match graph endpoint: {iid}")
        r1_text = e["target_fact_text"]
        r2_text = e["r2_matched_fact_texts"][0]
        r1_id = fact_id(r1["qid"], r1_text)
        r2_id = fact_id(e["r2_source_qid"], r2_text)
        if [r1_id, r2_id] != e["chain_fact_ids"]:
            raise RuntimeError(f"Derived chain fact IDs do not match audit: {iid}")
        evidence[iid] = {
            "graph": graph,
            "qids": qids_for(row, graph),
            "chain_fact_ids": [r1_id, r2_id],
            "gold_answers": set(row["gold_answers"]),
            "target_fact_id": r1_id,
            "r2_fact_id": r2_id,
            "target_fact_text": r1_text,
            "r2_fact_text": r2_text,
            "old_split": old_assignments[iid]["proposed_split"],
        }

    dsu = DSU(ids)
    overlap_pairs = []
    for a, b in itertools.combinations(ids, 2):
        x, y = evidence[a], evidence[b]
        shared_qids = sorted(x["qids"] & y["qids"])
        shared_chain = sorted(set(x["chain_fact_ids"]) & set(y["chain_fact_ids"]))
        shared_answers = sorted(x["gold_answers"] & y["gold_answers"])
        if shared_qids or shared_chain or shared_answers:
            dsu.union(a, b)
            overlap_pairs.append({
                "qa_1": a,
                "qa_2": b,
                "old_split_1": x["old_split"],
                "old_split_2": y["old_split"],
                "shared_entity_qids": shared_qids,
                "shared_exact_chain_fact_ids": shared_chain,
                "shared_exact_gold_answers": shared_answers,
                "was_cross_split_before": x["old_split"] != y["old_split"],
            })
    components = collections.defaultdict(list)
    for iid in ids:
        components[dsu.find(iid)].append(iid)
    groups = sorted(
        [sorted(members) for members in components.values()],
        key=lambda g: (-len(g), sha_bytes((SEED + "|" + ",".join(g)).encode("utf-8"))),
    )

    targets = {"forget": 77, "retain": 26, "holdout": 26}
    counts = {k: 0 for k in targets}
    assignments = {}
    for members in groups:
        size = len(members)
        choices = [s for s, target in targets.items() if counts[s] + size <= target]
        if not choices:
            choices = list(targets)
        split = max(
            choices,
            key=lambda s: (
                (targets[s] - counts[s]) / targets[s],
                targets[s] - counts[s],
                sha_bytes((SEED + "|" + ",".join(members) + "|" + s).encode("utf-8")),
            ),
        )
        counts[split] += size
        group_id = "eg-" + sha_bytes(",".join(members).encode("utf-8"))[:16]
        for iid in members:
            assignments[iid] = {"split": split, "group_id": group_id, "group_size": size}
    if counts != targets:
        raise RuntimeError(f"Deterministic group allocation failed requested near-ratio counts: {counts}")

    # Verify every leakage relation is contained within a single split.
    for pair in overlap_pairs:
        if assignments[pair["qa_1"]]["split"] != assignments[pair["qa_2"]]["split"]:
            raise RuntimeError("Cross-split entity/fact/answer relation remains")

    # Build English natural-language hop SFT. Holdout and excluded examples never enter SFT.
    sft = []
    for iid in ids:
        assignment, ev = assignments[iid], evidence[iid]
        if assignment["split"] == "holdout":
            continue
        row, graph = qa[iid], ev["graph"]["graph_qa"]
        nodes = [graph[f"n{k}"].rsplit(" (Q", 1)[0] for k in (1, 2, 3)]
        node_qids = [
            (match.group(1) if (match := re.search(r"\((Q\d+)\)$", graph[f"n{k}"])) else None)
            for k in (1, 2, 3)
        ]
        decomposed = ev["graph"]["decomposed_questions"]
        for hop, question_key, answer, fact, fid, relation_id, subject_qid, object_qid, fact_record_qid in (
            ("r1", "Q1", nodes[1], ev["target_fact_text"], ev["target_fact_id"], graph["r1"]["pid"], node_qids[0], node_qids[1], graph["r1"]["qid"]),
            ("r2", "Q2", nodes[2], ev["r2_fact_text"], ev["r2_fact_id"], graph["r2"]["pid"], node_qids[1], node_qids[2], graph["r2"]["qid"]),
        ):
            question = decomposed[question_key]
            if hop == "r2":
                question = re.sub(r"</?bridge>", "", question)
            sft.append({
                "sft_id": f"{iid}:{hop}",
                "source_qa_id": iid,
                "split": assignment["split"],
                "hop": hop,
                "entity_fact_group_id": assignment["group_id"],
                "target_fact_id": ev["target_fact_id"],
                "fact_id": fid,
                "fact_text_en": fact,
                "relation_id": relation_id,
                "source_fact_record_qid": fact_record_qid,
                "subject_entity_qid": subject_qid,
                "object_entity_qid": object_qid,
                "question_en": question,
                "answer": answer,
                "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                "status": "CANDIDATE_NOT_APPROVED",
            })

    raw_sft_count = len(sft)
    deduplicated = {}
    for record in sft:
        key = (record["fact_id"], record["question_en"], record["answer"])
        if key not in deduplicated:
            record["source_examples"] = [{
                "sft_id": record["sft_id"],
                "source_qa_id": record["source_qa_id"],
                "hop": record["hop"],
                "target_fact_id": record["target_fact_id"],
            }]
            record["source_sft_ids"] = [record["sft_id"]]
            record["source_qa_ids"] = [record["source_qa_id"]]
            record["hops"] = [record["hop"]]
            deduplicated[key] = record
        else:
            prior = deduplicated[key]
            prior["source_examples"].append({
                "sft_id": record["sft_id"],
                "source_qa_id": record["source_qa_id"],
                "hop": record["hop"],
                "target_fact_id": record["target_fact_id"],
            })
            prior["source_sft_ids"].append(record["sft_id"])
            prior["source_qa_ids"].append(record["source_qa_id"])
            prior["hops"].append(record["hop"])
    sft = list(deduplicated.values())
    for record in sft:
        record["source_sft_ids"] = sorted(set(record["source_sft_ids"]))
        record["source_qa_ids"] = sorted(set(record["source_qa_ids"]))
        record["hops"] = sorted(set(record["hops"]))
        record["source_examples"] = sorted(record["source_examples"], key=lambda x: x["sft_id"])
    if raw_sft_count != 206 or len(sft) != 205 or any(r["split"] == "holdout" for r in sft):
        raise RuntimeError("SFT count or holdout exclusion check failed")
    pilot_ids = {}
    for split in targets:
        options = []
        for iid in ids:
            if assignments[iid]["split"] != split:
                continue
            chars = sum(len(docs[qid]["doc"]) for qid in qa[iid]["gold_qids"])
            tie = sha_bytes((SEED + "|translation-pilot|" + split + "|" + iid).encode("utf-8"))
            options.append((chars, tie, iid))
        pilot_ids[split] = min(options)[2]
    pilot = []
    for split, iid in pilot_ids.items():
        row = qa[iid]
        pilot.append({
            "source_qa_id": iid,
            "split": split,
            "gold_qids": row["gold_qids"],
            "source_docs": [{"qid": qid, "source_characters": len(docs[qid]["doc"])} for qid in row["gold_qids"]],
            "translation_status": "NOT_STARTED",
            "human_review_status": "PENDING",
            "note": "Translate full QA and canonical gold passages for all target languages; holdout is evaluation-only.",
        })

    excluded = {
        "source_qa_id": "0080-qa-sm",
        "old_proposed_split": old_assignments["0080-qa-sm"]["proposed_split"],
        "reason": "R2_GRAPH_RELATION_VS_SOURCE_FACT_INCONSISTENT",
        "graph_relation": "[participant → Irvin Oathen → end time]",
        "verified_source_r2_fact": "Tempest of Nations (Q33143) → participant → Irvin Thorne (Q131981) → end time → 1799 (year)",
        "action": "Excluded from the experimental split pending source resolution; not used for SFT.",
    }
    old_cross = [p for p in overlap_pairs if p["was_cross_split_before"]]
    translation_report_path = OUT / "translation_pilot" / "report.json"
    translation_status = (
        read_json(translation_report_path)["status"]
        if translation_report_path.is_file()
        else "NOT_STARTED"
    )
    passage_report_path = OUT / "passage_leakage_audit" / "report.json"
    passage_report = read_json(passage_report_path) if passage_report_path.is_file() else None
    summary = {
        "status": "ENTITY_FACT_ANSWER_ISOLATED_PROPOSAL_NOT_APPROVED",
        "do_not_train": True,
        "source_checksums": source_checksums,
        "source_counts": {"qa": len(qa_rows), "documents": len(docs_rows), "facts": len(facts_rows)},
        "strict_type_a_eligible_before_conflict_exclusion": 130,
        "r2_conflict_excluded": 1,
        "eligible_for_this_split": len(ids),
        "excluded_other_type_a_ambiguous": 70,
        "excluded_non_type_a": 1000,
        "split_counts": counts,
        "split_ratio_actual": {k: round(v / len(ids), 6) for k, v in counts.items()},
        "grouping_basis": ["shared graph/document entity QID", "shared exact chain fact ID", "shared exact final answer"],
        "indivisible_group_count": len(groups),
        "group_size_histogram": dict(sorted(collections.Counter(len(g) for g in groups).items())),
        "largest_group_size": max(map(len, groups)),
        "overlap_pair_count": len(overlap_pairs),
        "previous_cross_split_overlap_pairs_resolved": len(old_cross),
        "remaining_cross_split_entity_pairs": 0,
        "remaining_cross_split_chain_fact_pairs": 0,
        "remaining_cross_split_exact_answer_pairs": 0,
        "sft_source_records_before_exact_deduplication": raw_sft_count,
        "sft_unique_message_records": len(sft),
        "sft_exact_duplicates_merged": raw_sft_count - len(sft),
        "sft_unique_message_counts_by_split": dict(collections.Counter(r["split"] for r in sft)),
        "holdout_sft_records": 0,
        "pilot_selection": pilot_ids,
        "pilot_translation_status": translation_status,
        "passage_audit_status": passage_report["status"] if passage_report else "NOT_STARTED",
        "pilot_documents_with_detected_other_split_risk": passage_report["pilot_documents_with_other_split_risk"] if passage_report else None,
        "all_split_documents_with_detected_other_split_risk": passage_report["all_split_documents_with_detected_other_split_risk"] if passage_report else None,
        "approved_training_documents": passage_report["approved_training_documents"] if passage_report else None,
        "human_translation_review_status": "PENDING",
        "limitations": [
            "The 77/26/26 assignment is a deterministic proposal, not a statistically powered final dataset.",
            "Translation and native-speaker review are not complete.",
            "A conservative passage-risk screen is complete; all full documents and whole fact rows remain excluded from training.",
            "Relation PIDs are graph metadata and are not independently encoded in the free-text source facts.",
        ],
    }

    # All validation is complete before writing the new versioned bundle.
    OUT.mkdir(parents=True, exist_ok=True)
    assignment_rows = []
    component_rows = []
    for group in groups:
        gid = assignments[group[0]]["group_id"]
        split = assignments[group[0]]["split"]
        qids = sorted(set().union(*(evidence[i]["qids"] for i in group)))
        component_rows.append({"entity_fact_group_id": gid, "split": split, "size": len(group), "instance_ids": group, "entity_qids": qids})
        for iid in group:
            e = evidence[iid]
            assignment_rows.append({
                "source_qa_id": iid,
                "split": split,
                "entity_fact_group_id": gid,
                "group_size": len(group),
                "target_fact_id": e["target_fact_id"],
                "r2_fact_id": e["r2_fact_id"],
                "chain_fact_ids": e["chain_fact_ids"],
                "gold_answers": sorted(e["gold_answers"]),
                "source_gold_qids": qa[iid]["gold_qids"],
            })
    write_jsonl(OUT / "assignments.jsonl", sorted(assignment_rows, key=lambda r: r["source_qa_id"]))
    write_jsonl(OUT / "entity_fact_groups.jsonl", component_rows)
    write_jsonl(OUT / "overlap_pairs.jsonl", overlap_pairs)
    write_jsonl(OUT / "sft_natural_en_candidate.jsonl", sft)
    write_jsonl(OUT / "balanced_translation_pilot_selection.jsonl", pilot)
    write_jsonl(OUT / "excluded_r2_conflict.jsonl", [excluded])
    write_json(OUT / "source_manifest.json", source_checksums)
    write_json(OUT / "audit_report.json", summary)

    output_files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"output_manifest.json", "sha256sums.txt", "README.md"}:
            output_files[path.relative_to(OUT).as_posix()] = {
                "sha256": sha_file(path),
                "bytes": path.stat().st_size,
                "rows": sum(1 for line in path.open("rb") if line.strip()) if path.suffix == ".jsonl" else None,
            }
    write_json(OUT / "output_manifest.json", {"status": summary["status"], "do_not_train": True, "files": output_files})
    with (OUT / "sha256sums.txt").open("w", encoding="utf-8") as f:
        for name, metadata in output_files.items():
            f.write(f"{metadata['sha256']}  {name}\n")
    print(json.dumps({"status": summary["status"], "split_counts": counts, "groups": len(groups), "old_cross_split_pairs_resolved": len(old_cross), "pilot": pilot_ids}, indent=2))


if __name__ == "__main__":
    validate_and_build()
