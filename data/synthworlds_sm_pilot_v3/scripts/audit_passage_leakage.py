#!/usr/bin/env python3
"""Conservatively screen selected full passages against the v3 split's other-split facts."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SOURCE = DATA / "synthworlds_sm"
AUDIT = DATA / "synthworlds_sm_reaudit"
OUT = DATA / "synthworlds_sm_pilot_v3"


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def label(value):
    value = re.sub(r"\s*\(Q\d+\)$", "", value)
    return re.sub(r"\s*\(year\)$", "", value).strip().rstrip(".")


def fact_edges(iid, eligible):
    row = eligible[iid]
    graph = json.loads(row["mhop_qa"])["graph_qa"]
    audit = row["audit"]
    n1, n2 = graph["n1"], graph["n2"]
    edges = [{
        "qa_id": iid,
        "kind": "target_r1",
        "fact_id": audit["target_fact_id"],
        "subject_qid": graph["r1"]["qid"],
        "subject": label(n1),
        "object": label(n2),
    }]
    for text, fid in zip(audit["r2_matched_fact_texts"], audit["r2_matched_fact_ids"]):
        parts = [p.strip() for p in text.split("→")]
        if len(parts) < 3:
            continue
        edges.append({
            "qa_id": iid,
            "kind": "second_hop",
            "fact_id": fid,
            "subject_qid": audit["r2_source_qid"],
            "subject": label(parts[0]),
            "object": label(parts[2] if len(parts) == 3 else parts[-1]),
        })
    return edges


def main():
    assignments = {r["source_qa_id"]: r for r in jsonl(OUT / "assignments.jsonl")}
    eligible_qa = {r["instance_id"]: r for r in jsonl(AUDIT / "type_a_target_eligibility.jsonl") if r["eligible"]}
    docs = {r["qid"]: r for r in jsonl(SOURCE / "qa-sm-docs.jsonl")}
    source_qa = {r["instance_id"]: r for r in jsonl(SOURCE / "qa-sm.jsonl")}
    selection = jsonl(OUT / "balanced_translation_pilot_selection.jsonl")
    all_edges = {split: [] for split in ("forget", "retain", "holdout")}
    for iid in assignments:
        row = dict(eligible_qa[iid])
        row["audit"] = eligible_qa[iid]
        row["mhop_qa"] = source_qa[iid]["mhop_qa"]
        all_edges[assignments[iid]["split"]].extend(fact_edges(iid, {iid: row}))

    # Every selected gold passage gets all other-split fact guards. This is a screen,
    # not a semantic proof that a passage is safe when no co-mention is found.
    pilot_rows = []
    for qa_rec in selection:
        iid, owner = qa_rec["source_qa_id"], qa_rec["split"]
        for qid in qa_rec["gold_qids"]:
            body = docs[qid]["doc"].casefold()
            risks = []
            for other_split, edges in all_edges.items():
                if other_split == owner:
                    continue
                for edge in edges:
                    by_qid = qid == edge["subject_qid"]
                    subject, obj = edge["subject"].casefold(), edge["object"].casefold()
                    co_mention = bool(subject and obj and subject in body and obj in body)
                    if by_qid or co_mention:
                        risks.append({
                            "other_split": other_split,
                            "other_qa_id": edge["qa_id"],
                            "fact_id": edge["fact_id"],
                            "fact_kind": edge["kind"],
                            "risk_type": "source_subject_doc" if by_qid else "subject_object_co_mention",
                            "confirmation": "possible_disclosure_not_semantically_certified",
                        })
            pilot_rows.append({
                "pilot_qa_id": iid,
                "owner_split": owner,
                "document_qid": qid,
                "source_document_text_sha256": text_sha(docs[qid]["doc"]),
                "other_split_risk_count": len(risks),
                "other_split_risks": risks,
                "training_disposition": "DO_NOT_TRAIN_ON_FULL_PASSAGE",
                "evaluation_disposition": "USE_ONLY_IF_PROTOCOL_INTENTIONALLY_PROVIDES_SUPPORTING_CONTEXT",
                "if_no_detected_risk": "STILL_UNCERTIFIED_FOR_TRAINING",
            })

    # Check all split-specific gold passages, not just the three pilot examples.
    split_passages = collections.defaultdict(set)
    for iid, assignment in assignments.items():
        for qid in source_qa[iid]["gold_qids"]:
            split_passages[assignment["split"]].add(qid)
    split_rows = []
    for owner, qids in split_passages.items():
        for qid in sorted(qids):
            text = docs[qid]["doc"].casefold()
            risks = []
            for other_split, edges in all_edges.items():
                if other_split == owner:
                    continue
                for edge in edges:
                    by_qid = qid == edge["subject_qid"]
                    subject, obj = edge["subject"].casefold(), edge["object"].casefold()
                    co_mention = bool(subject and obj and subject in text and obj in text)
                    if by_qid or co_mention:
                        risks.append({
                            "other_split": other_split,
                            "other_qa_id": edge["qa_id"],
                            "fact_id": edge["fact_id"],
                            "risk_type": "source_subject_doc" if by_qid else "subject_object_co_mention",
                        })
            split_rows.append({
                "owner_split": owner,
                "document_qid": qid,
                "other_split_risk_count": len(risks),
                "other_split_risks": risks,
                "training_disposition": "DO_NOT_TRAIN_ON_FULL_PASSAGE",
                "risk_screen_limit": "No detected co-mention does not certify absence of paraphrase, alias, or indirect disclosure.",
            })

    # Whole fact rows are blocked if their QID is the subject record for a protected
    # other-split chain fact; rows are never used as SFT input in this bundle.
    fact_rows = []
    for owner, qids in split_passages.items():
        other_subjects = {
            edge["subject_qid"]: edge
            for split, edges in all_edges.items() if split != owner
            for edge in edges
        }
        for qid in sorted(qids & set(other_subjects)):
            fact_rows.append({
                "owner_split": owner,
                "fact_record_qid": qid,
                "exposed_other_split_fact_id": other_subjects[qid]["fact_id"],
                "training_disposition": "DO_NOT_TRAIN_ON_WHOLE_FACT_ROW",
            })

    summary = {
        "status": "CONSERVATIVE_PASSAGE_SCREEN_COMPLETE_NO_DOCUMENTS_APPROVED_FOR_TRAINING",
        "method": "Source-subject-QID and subject/object co-mention screen against all other-split target and second-hop facts.",
        "screen_scope": "All canonical gold passages referenced by the 129 v3 QA assignments.",
        "pilot_documents": len(pilot_rows),
        "pilot_documents_with_other_split_risk": sum(bool(r["other_split_risks"]) for r in pilot_rows),
        "pilot_documents_without_detected_risk_but_still_uncertified": sum(not r["other_split_risks"] for r in pilot_rows),
        "all_split_gold_documents": len(split_rows),
        "all_split_documents_with_detected_other_split_risk": sum(bool(r["other_split_risks"]) for r in split_rows),
        "whole_fact_row_qid_exposures": len(fact_rows),
        "approved_training_documents": 0,
        "limitations": [
            "This conservative string/QID screen identifies possible leakage; it does not adjudicate claim meaning.",
            "A clean screen does not certify safety because aliases, paraphrases, and indirect disclosure can be missed.",
            "All full documents and full fact rows remain withheld from training; only verified single-hop examples are SFT candidates.",
            "Evaluation documents may be supplied only if the experiment is explicitly a grounded/context-provided QA condition.",
        ],
    }
    out = OUT / "passage_leakage_audit"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pilot_document_risks.jsonl").open("w", encoding="utf-8") as f:
        for row in pilot_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "all_split_document_risks.jsonl").open("w", encoding="utf-8") as f:
        for row in split_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "whole_fact_row_risks.jsonl").open("w", encoding="utf-8") as f:
        for row in fact_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
