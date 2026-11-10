#!/usr/bin/env python3
"""Recompute source-verified Type-B chains under the frozen v5 evaluation set.

This is a structural replacement-pool audit only. It does not inspect or approve
translations, alter the frozen split, or create training data.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SOURCE = REPO / "data/synthworlds_sm"
V4_BUILDER = REPO / "data/synthworlds_sm_pilot_v4/kaggle/r1_experiment_dataset_disjoint/scripts/build_type_b_disjoint_candidates.py"
V5_PREP = REPO / "experiments/synthworlds_r1/training_release_draft/scripts/prepare_release.py"
CASES = REPO / "experiments/synthworlds_r1/candidate_cases.draft.jsonl"
PREPARATION = REPO / "experiments/synthworlds_r1/training_release_draft/PREPARATION_REPORT.json"
OUT = Path(__file__).with_name("type_b_candidate_exhaustion.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import audit dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    v5 = load_module("r1_v5_prepare_release", V5_PREP)
    qa_rows = read_jsonl(SOURCE / "qa-sm.jsonl")
    qa_by_id = {row["instance_id"]: row for row in qa_rows}
    cases = read_jsonl(CASES)
    if len(qa_by_id) != len(qa_rows) or len({c["source_qa_id"] for c in cases}) != len(cases):
        raise ValueError("Source QA or frozen evaluation case IDs are not unique")

    # Reuse the reviewed source builder in a temporary directory, substituting
    # the actual v5 evaluation cases/labels. No repository candidate is changed.
    builder = load_module("r1_type_b_builder", V4_BUILDER)
    with tempfile.TemporaryDirectory(prefix="r1-v5-typeb-audit-") as tmp:
        pilot = Path(tmp)
        labels_dir = pilot / "evaluation_dataset/labels"
        labels_dir.mkdir(parents=True)
        assignments = [{"source_qa_id": c["source_qa_id"], "split": c["split"]} for c in cases]
        labels = [{
            "instance_id": c["source_qa_id"],
            "final_answers": [c["final_answer_source_supported_en"], c["negative_control_en"]],
            "direct_intermediate_answer": c["target_bridge_en"],
            "chain_fact_ids": [c["target_r1_fact_id"], c["target_r2_fact_id"]],
        } for c in cases]
        (pilot / "assignments.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assignments), encoding="utf-8")
        (labels_dir / "gold_labels.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
        builder.SOURCE, builder.PILOT, builder.OUT = SOURCE, pilot, pilot / "candidate_output"
        with contextlib.redirect_stdout(io.StringIO()):
            builder.main()
        candidate_rows = read_jsonl(builder.OUT / "english_candidate.jsonl")

    # Apply the stricter v5 leakage screen to every source-built chain, including
    # candidates absent from the older v4 candidate artifact.
    eval_ids = {c["source_qa_id"] for c in cases}
    eval_qids: set[str] = set()
    eval_surfaces: set[str] = set()
    eval_answers: set[str] = set()
    eval_questions: set[str] = set()
    eval_facts: set[str] = set()
    for case in cases:
        _, qids, names = v5.graph_data(qa_by_id[case["source_qa_id"]])
        eval_qids.update(qids)
        eval_surfaces.update(v5.norm(x) for x in names | set(case.get("protected_entity_names", [])) if v5.norm(x))
        eval_answers.update(v5.norm(x) for x in (
            case["target_bridge_en"], case["final_answer_source_supported_en"], case["negative_control_en"]
        ) if v5.norm(x))
        eval_questions.update(v5.norm(x) for x in (case["q1_question_en"], case["multi_hop_question_en"]) if v5.norm(x))
        eval_facts.update((case["target_r1_fact_id"], case["target_r2_fact_id"]))

    by_chain: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in candidate_rows:
        match = re.fullmatch(r"disjoint-b:(\d{4}-qa-sm):(q1|q2|q3|multi)", row.get("sft_id", ""))
        if not match:
            raise ValueError(f"Malformed source-built candidate ID: {row.get('sft_id')}")
        source_id, suffix = match.groups()
        if suffix in by_chain[source_id]:
            raise ValueError(f"Duplicate candidate prompt: {source_id}/{suffix}")
        by_chain[source_id][suffix] = row

    reasons: Counter[str] = Counter()
    eligible: list[str] = []
    for source_id, rows in sorted(by_chain.items()):
        reason = None
        if source_id in eval_ids:
            reason = "evaluation_source_id_overlap"
        elif set(rows) != {"q1", "q2", "q3", "multi"}:
            reason = "incomplete_prompt_set"
        else:
            qrow = qa_by_id[source_id]
            graph, qids, names = v5.graph_data(qrow)
            chain_qids = qids | set().union(*(set(r.get("entity_qids", [])) for r in rows.values()))
            chain_surfaces = {v5.norm(x) for x in names | set().union(*(
                set(r.get("protected_entity_names", [])) for r in rows.values()
            )) if v5.norm(x)}
            chain_answers = {v5.norm(r["answer_en"]) for r in rows.values() if v5.norm(r["answer_en"])}
            chain_questions = {v5.norm(r["question_en"]) for r in rows.values() if v5.norm(r["question_en"])}
            chain_facts = set().union(*(set(r.get("chain_fact_ids", [])) for r in rows.values()))
            if chain_qids & eval_qids:
                reason = "evaluation_qid_overlap"
            elif chain_surfaces & (eval_surfaces | eval_answers) or chain_answers & eval_answers:
                reason = "evaluation_surface_or_answer_overlap"
            elif chain_questions & eval_questions:
                reason = "evaluation_question_overlap"
            elif chain_facts & eval_facts:
                reason = "evaluation_fact_overlap"
            elif any(v5.norm(r["answer_en"]) in v5.norm(r["question_en"]) for r in rows.values()):
                reason = "answer_disclosed_in_prompt"
            elif graph.get("composed_question") != qrow.get("query"):
                reason = "source_composed_question_mismatch"
        if reason:
            reasons[reason] += 1
        else:
            eligible.append(source_id)

    frozen = read_json(PREPARATION)["composition_plan"]["selected_source_qa_ids"]
    if len(eligible) != len(set(eligible)) or set(eligible) != set(frozen):
        raise ValueError("Recomputed strict candidate IDs differ from the frozen v5 composition plan")
    result = {
        "status": "PASS_TYPE_B_CANDIDATE_EXHAUSTION_AUDIT",
        "scope": "source/path and v5 leakage eligibility only; no translation review or training approval",
        "source_qa_sha256": sha(SOURCE / "qa-sm.jsonl"),
        "source_facts_sha256": sha(SOURCE / "qa-sm-facts.jsonl"),
        "frozen_cases_sha256": sha(CASES),
        "v5_preparation_report_sha256": sha(PREPARATION),
        "source_type_b_rows": sum(row.get("question_graph_type") == "B" for row in qa_rows),
        "source_builder_candidates_under_v5_cases": len(by_chain),
        "strict_v5_eligible_chains": len(eligible),
        "frozen_composition_chains": len(frozen),
        "strict_ids_match_frozen_plan": True,
        "replacement_chains_available_from_existing_type_b_qa": len(set(eligible) - set(frozen)),
        "excluded_candidate_counts": dict(sorted(reasons.items())),
        "eligible_source_qa_ids": eligible,
        "limitations": [
            "This establishes exhaustion only for existing Type-B QA rows under the frozen v5 leakage rules.",
            "Newly derived chains from source facts were not generated or validated.",
            "Translation quality and the four-language composition release gate remain separate and unresolved.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
