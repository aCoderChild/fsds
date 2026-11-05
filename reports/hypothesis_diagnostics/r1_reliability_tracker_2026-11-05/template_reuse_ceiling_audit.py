#!/usr/bin/env python3
"""Audit an optimistic ceiling for reusing AI-screened translation templates.

This is a read-only diagnostic. It does not create or approve a training corpus.
It joins the frozen source prompts to their hash-bound AI review inputs, groups
prompts by source entity slots and relation signatures, and asks how many rows
could be covered if any passing translation for a matching template were
reusable. That assumption is deliberately optimistic and is not a translation
approval rule.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
RELEASE = REPO / "experiments/synthworlds_r1/training_release_draft"
SOURCE = REPO / "data/synthworlds_sm"
LANGUAGES = ("fr", "de", "hi", "th")
BASE_PASS = {
    "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS",
    "AI_ONLY_14B_ADJUDICATED_PASS",
}
REPAIR_PASS = "AI_ONLY_REPAIR_REVIEW_PASS"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def source_fact_id(qid: str, statement: str) -> str:
    payload = (qid + "\0" + statement).encode("utf-8")
    return "derived-sha256:" + hashlib.sha256(payload).hexdigest()


def fact_entities(fact_ids: list[str], facts_by_id: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for fact_id in fact_ids:
        qid, statement = facts_by_id[fact_id]
        for surface, entity_qid in re.findall(r"([^→]+?)\s+\((Q\d+)\)", statement):
            aliases.append((surface.strip(), entity_qid))
    return aliases


def source_template(question: str, aliases: list[tuple[str, str]], protected_names: list[str]) -> tuple[str, bool]:
    """Mask source-grounded entities in first-mention order; keep repetitions.

    Names that map to multiple QIDs make a row non-reusable. Unknown protected
    names are masked only as their own exact surface, never merged by fuzzy match.
    """
    identities: dict[str, set[str]] = collections.defaultdict(set)
    display: dict[str, str] = {}
    for surface, qid in aliases:
        key = norm(surface)
        if key:
            identities[key].add("qid:" + qid)
            display[key] = surface
    for surface in protected_names:
        key = norm(surface)
        if key and key not in identities:
            identities[key].add("surface:" + key)
            display[key] = surface

    matches: list[tuple[int, int, str, str]] = []
    ambiguous = False
    for key, ids in identities.items():
        surface = display[key]
        found = list(re.finditer(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", question, flags=re.IGNORECASE))
        if len(ids) > 1 and found:
            ambiguous = True
        identity = next(iter(ids)) if len(ids) == 1 else "ambiguous:" + key
        for match in found:
            matches.append((match.start(), match.end(), identity, key))

    matches.sort(key=lambda row: (row[0], -(row[1] - row[0]), row[3]))
    selected: list[tuple[int, int, str, str]] = []
    end = -1
    for match in matches:
        if match[0] >= end:
            selected.append(match)
            end = match[1]

    slot_by_identity: dict[str, str] = {}
    output: list[str] = []
    cursor = 0
    for start, finish, identity, _key in selected:
        if identity not in slot_by_identity:
            slot_by_identity[identity] = f"<ENTITY_{len(slot_by_identity)}>"
        output.extend((question[cursor:start], slot_by_identity[identity]))
        cursor = finish
    output.append(question[cursor:])
    return norm("".join(output)), ambiguous


def verify_original_review_inputs(
    source_meta: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], set[tuple[str, str, str]], dict[str, Any]]:
    review_report = read_json(RELEASE / "translation_reviews_roundtrip_qwen14b/report.json")
    if review_report.get("do_not_train") is not True or review_report.get("human_translation_reviewed") is not False:
        raise RuntimeError("Expected quarantined AI-only original review")

    reuse_rows = read_jsonl(RELEASE / "atomic_reuse.jsonl")
    reuse_target = {
        (row["sft_id"], row["language"]): row.get("question")
        for row in reuse_rows
        if row["language"] != "en" and row.get("question")
    }
    composition_queue = read_jsonl(RELEASE / "composition_translation_review_queue.jsonl")
    composition_target = {
        (row["prompt_id"], row["language"]): row
        for row in composition_queue
    }
    if len(composition_target) != len(composition_queue):
        raise RuntimeError("Duplicate composition translation source rows")

    reviews: dict[tuple[str, str, str], dict[str, Any]] = {}
    passes: set[tuple[str, str, str]] = set()
    source_counts: dict[str, dict[str, Any]] = {}
    input_paths = [
        RELEASE / "translation_reviews_roundtrip_qwen14b/report.json",
        RELEASE / "atomic_reuse.jsonl",
        RELEASE / "composition_translation_review_queue.jsonl",
    ]

    for language in LANGUAGES:
        lang_report = review_report["languages"][language]
        source_hashes = lang_report["source_hashes"]
        source_files = {
            "atomic_reuse": RELEASE / "atomic_reuse.jsonl",
            "composition_review_queue": RELEASE / "composition_translation_review_queue.jsonl",
            f"atomic_drafts_{language}": RELEASE / "atomic_translation_drafts" / f"{language}.jsonl",
            f"atomic_repairs_{language}": RELEASE / "atomic_translation_repairs" / f"{language}.jsonl",
        }
        for key, path in source_files.items():
            if not path.exists() or sha_file(path) != source_hashes.get(key):
                raise RuntimeError(f"Original-review source hash mismatch: {language}/{key}")
            input_paths.append(path)

        draft_rows = read_jsonl(source_files[f"atomic_drafts_{language}"])
        repair_rows = read_jsonl(source_files[f"atomic_repairs_{language}"])
        draft_text = {row["prompt_id"]: row.get("question_translation") for row in draft_rows}
        repair_text = {row["sft_id"]: row.get("question_translation") for row in repair_rows}
        if len(draft_text) != len(draft_rows) or len(repair_text) != len(repair_rows):
            raise RuntimeError(f"Duplicate atomic translation source IDs: {language}")

        review_path = RELEASE / "translation_reviews_roundtrip_qwen14b" / f"{language}.jsonl"
        if sha_file(review_path) != lang_report.get("adjudication_sha256"):
            raise RuntimeError(f"Original-review output hash mismatch: {language}")
        input_paths.append(review_path)
        rows = read_jsonl(review_path)
        expected_counts = lang_report.get("final_status_counts", {})
        actual_counts = collections.Counter(row["final_review_status"] for row in rows)
        if dict(actual_counts) != expected_counts or len(rows) != 488:
            raise RuntimeError(f"Original-review status/count mismatch: {language}")
        if len({(row["kind"], row["prompt_id"]) for row in rows}) != 488:
            raise RuntimeError(f"Duplicate original-review prompt: {language}")

        source_counts[language] = {"atomic_reuse": 0, "atomic_draft": 0, "atomic_repair": 0, "composition": 0}
        for row in rows:
            kind, prompt_id = row["kind"], row["prompt_id"]
            if row.get("human_translation_reviewed") is not False:
                raise RuntimeError(f"Original review is not explicitly AI-only: {language}/{prompt_id}")
            source = source_meta.get((kind, prompt_id))
            if source is None:
                raise RuntimeError(f"Review prompt is absent from source: {language}/{kind}/{prompt_id}")
            source_row = source[0]
            if row.get("text_en") != source_row.get("question_en"):
                raise RuntimeError(f"Reviewed English source mismatch: {language}/{prompt_id}")

            if kind == "atomic":
                candidates = [
                    ("atomic_reuse", reuse_target.get((prompt_id, language))),
                    ("atomic_draft", draft_text.get(prompt_id)),
                    ("atomic_repair", repair_text.get(prompt_id)),
                ]
                matched_sources = [name for name, text in candidates if text and text == row.get("text_translation")]
                if not matched_sources:
                    raise RuntimeError(f"Reviewed atomic translation is not in a hash-bound source: {language}/{prompt_id}")
                source_counts[language][matched_sources[0]] += 1
            elif kind == "composition":
                candidate = composition_target.get((prompt_id, language))
                if not candidate or candidate.get("text_en") != row.get("text_en") or candidate.get("text_translation") != row.get("text_translation"):
                    raise RuntimeError(f"Reviewed composition translation is not source-bound: {language}/{prompt_id}")
                source_counts[language]["composition"] += 1
            else:
                raise RuntimeError(f"Unknown reviewed prompt kind: {kind}")

            key = (kind, prompt_id, language)
            reviews[key] = row
            if row.get("final_review_status") in BASE_PASS:
                passes.add(key)

    return reviews, passes, {"source_pool_rows_by_language": source_counts, "input_paths": input_paths}


def verify_repair_reviews(
    source_meta: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[set[tuple[str, str, str]], dict[str, Any]]:
    repair_root = RELEASE / "translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04"
    review_path = repair_root / "remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl"
    candidate_path = repair_root / "parsed_repair_candidates.jsonl"
    private_map_path = repair_root / "private_row_map.jsonl"
    merge_path = repair_root / "remote_output/merged_full799/version_1/merge_verification.json"
    reviews = read_jsonl(review_path)
    candidates = read_jsonl(candidate_path)
    private_rows = read_jsonl(private_map_path)
    merge = read_json(merge_path)

    def index_unique(rows: list[dict[str, Any]], key_fn, label: str) -> dict[Any, dict[str, Any]]:
        indexed: dict[Any, dict[str, Any]] = {}
        for row in rows:
            key = key_fn(row)
            if key in indexed:
                raise RuntimeError(f"Duplicate {label} key: {key}")
            indexed[key] = row
        return indexed

    candidate_by_key = index_unique(candidates, lambda row: (row["language"], row["prompt_id"]), "candidate")
    private_by_key = index_unique(private_rows, lambda row: (row["language"], row["prompt_id"]), "private map")
    review_by_key = index_unique(reviews, lambda row: (row["language"], row["prompt_id"]), "repair review")
    if len(reviews) != 799 or set(candidate_by_key) != set(review_by_key) or set(private_by_key) != set(review_by_key):
        raise RuntimeError("Full flagged repair candidate/review/map coverage is incomplete")
    status_counts = collections.Counter(row["review_status"] for row in reviews)
    if (
        dict(status_counts) != merge.get("counts")
        or merge.get("exact_source_coverage") is not True
        or merge.get("status") != "VERIFIED_799_ROW_AI_ONLY_DIAGNOSTIC_MERGE_NOT_RELEASE_APPROVAL"
        or merge.get("release_approved") is not False
        or merge.get("training_eligible") is not False
        or merge.get("human_translation_reviewed") is not False
        or merge.get("answers_or_splits_sent") is not False
    ):
        raise RuntimeError("Repair merge receipt does not match its rows")

    passed: set[tuple[str, str, str]] = set()
    for key, review in review_by_key.items():
        language, prompt_id = key
        candidate = candidate_by_key[key]
        private = private_by_key[key]
        kind = review["kind"]
        source = source_meta.get((kind, prompt_id))
        if source is None:
            raise RuntimeError(f"Repair review prompt is absent from source: {language}/{prompt_id}")
        question = source[0]["question_en"]
        if (
            candidate.get("kind") != kind
            or private.get("kind") != kind
            or candidate.get("text_en") != question
            or private.get("source_question_en") != question
            or private.get("source_question_en_sha256") != sha_text(question)
            or private.get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED"
            or review.get("candidate_translation_sha256") != sha_text(candidate.get("text_translation", ""))
            or candidate.get("human_translation_reviewed") is not False
            or review.get("human_translation_reviewed") is not False
        ):
            raise RuntimeError(f"Repair candidate failed source/hash binding: {language}/{prompt_id}")
        if review.get("review_status") == REPAIR_PASS:
            passed.add((kind, prompt_id, language))

    paths = (review_path, candidate_path, private_map_path, merge_path)
    return passed, {
        "review_rows": len(reviews),
        "candidate_rows": len(candidates),
        "private_map_rows": len(private_rows),
        "hash_and_source_binding_failures": 0,
        "status_counts": dict(status_counts),
        "passing_rows": len(passed),
        "input_paths": list(paths),
    }


def build_source_meta() -> tuple[dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    qa_by_id = {}
    for row in read_jsonl(SOURCE / "qa-sm.jsonl"):
        graph = json.loads(row["mhop_qa"])
        qa_by_id[row["instance_id"]] = {"graph": graph, "qa": graph.get("graph_qa", {})}

    facts_by_id: dict[str, tuple[str, str]] = {}
    for record in read_jsonl(SOURCE / "qa-sm-facts.jsonl"):
        for statement in record["facts"]:
            facts_by_id[source_fact_id(record["qid"], statement)] = (record["qid"], statement)

    atomic_en = {
        row["sft_id"]: row
        for row in read_jsonl(RELEASE / "atomic_reuse.jsonl")
        if row["language"] == "en"
    }
    composition_en = {
        row["sft_id"]: row
        for row in read_jsonl(RELEASE / "composition_reuse.jsonl")
        if row["language"] == "en"
    }
    composition_queue = {
        (row["language"], row["prompt_id"]): row
        for row in read_jsonl(RELEASE / "composition_translation_review_queue.jsonl")
    }
    if len(atomic_en) != 312 or len(composition_en) != 176:
        raise RuntimeError("Frozen source prompt counts differ from 312 atomic and 176 composition rows")

    source_meta: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    ambiguous_alias_rows: set[str] = set()
    text_only_relations: dict[tuple[str, str, str], set[str]] = collections.defaultdict(set)
    required_composition_fact_ids: set[str] = set()

    for prompt_id, row in atomic_en.items():
        source = qa_by_id[row["source_qa_id"]]
        relation = source["qa"].get(row["hop"])
        if not relation or not relation.get("pid"):
            raise RuntimeError(f"Missing atomic graph relation signature: {prompt_id}")
        aliases = []
        for qid, item in source["graph"].get("entity_rename_map", {}).items():
            aliases.extend((name, qid) for name in (item.get("new_name"), item.get("old_name")) if name)
        for name, value in source["qa"].items():
            if re.fullmatch(r"n\d+", name):
                match = re.search(r"\((Q\d+)\)$", str(value))
                if match:
                    aliases.append((re.sub(r"\s*\(Q\d+\)$", "", str(value)).strip(), match.group(1)))
        aliases.extend(fact_entities([row["fact_id"]], facts_by_id))
        masked, ambiguous = source_template(row["question_en"], aliases, row.get("protected_entity_names", []))
        if ambiguous:
            ambiguous_alias_rows.add(prompt_id)
            masked += "|UNREUSABLE:" + prompt_id
        relation_signature = f"{relation['pid']}|{norm(relation.get('relation_str', ''))}"
        template = ("atomic", row["hop"], relation_signature, masked)
        meta = {"template": template, "relation_signature": relation_signature, "masked_question": masked}
        source_meta[("atomic", prompt_id)] = (row, meta)
        text_only_relations[("atomic", row["hop"], masked)].add(relation_signature)

    for prompt_id, row in composition_en.items():
        required_composition_fact_ids.update(row.get("fact_ids", []))
        relation_labels = []
        for fact_id in row.get("fact_ids", []):
            if fact_id not in facts_by_id:
                raise RuntimeError(f"Composition fact ID is absent from source facts: {prompt_id}")
            _qid, statement = facts_by_id[fact_id]
            pieces = [piece.strip() for piece in statement.split("→")]
            if len(pieces) < 3 or len(pieces) % 2 == 0:
                raise RuntimeError(f"Malformed source fact statement: {prompt_id}")
            relation_labels.extend(norm(piece) for piece in pieces[1::2])
        aliases = fact_entities(row.get("fact_ids", []), facts_by_id)
        source = qa_by_id.get(row["source_qa_id"], {})
        for qid, item in source.get("graph", {}).get("entity_rename_map", {}).items():
            aliases.extend((name, qid) for name in (item.get("new_name"), item.get("old_name")) if name)
        protected = composition_queue[("fr", prompt_id)].get("protected_entity_names", [])
        masked, ambiguous = source_template(row["question_en"], aliases, protected)
        if ambiguous:
            ambiguous_alias_rows.add(prompt_id)
            masked += "|UNREUSABLE:" + prompt_id
        relation_signature = "|".join(relation_labels)
        template = ("composition", row["hop"], relation_signature, masked)
        meta = {"template": template, "relation_signature": relation_signature, "masked_question": masked}
        source_meta[("composition", prompt_id)] = (row, meta)
        text_only_relations[("composition", row["hop"], masked)].add(relation_signature)

    text_only_collisions = sum(len(signatures) > 1 for signatures in text_only_relations.values())
    return source_meta, {
        "atomic_prompt_rows": len(atomic_en),
        "composition_prompt_rows": len(composition_en),
        "unique_atomic_relation_aware_templates": len({source_meta[("atomic", pid)][1]["template"] for pid in atomic_en}),
        "unique_composition_relation_aware_templates": len({source_meta[("composition", pid)][1]["template"] for pid in composition_en}),
        "rows_with_ambiguous_entity_aliases_excluded_from_reuse": len(ambiguous_alias_rows),
        "text_only_template_groups_with_relation_collisions": text_only_collisions,
        "composition_fact_ids_required": len(required_composition_fact_ids),
        "composition_fact_ids_resolved": sum(fact_id in facts_by_id for fact_id in required_composition_fact_ids),
        "input_paths": [SOURCE / "qa-sm.jsonl", SOURCE / "qa-sm-facts.jsonl", RELEASE / "atomic_reuse.jsonl", RELEASE / "composition_reuse.jsonl", RELEASE / "composition_translation_review_queue.jsonl"],
    }


def main() -> None:
    source_meta, structure = build_source_meta()
    base_reviews, base_passes, original_integrity = verify_original_review_inputs(source_meta)
    repair_passes, repair_integrity = verify_repair_reviews(source_meta)
    accepted_rows = base_passes | repair_passes

    pass_groups_by_mode: dict[str, dict[tuple[tuple[Any, ...], str], set[str]]] = {
        "source_review_only": collections.defaultdict(set),
        "verified_candidate_union": collections.defaultdict(set),
    }
    for mode, rows in (("source_review_only", base_passes), ("verified_candidate_union", accepted_rows)):
        for kind, prompt_id, language in rows:
            template = source_meta[(kind, prompt_id)][1]["template"]
            pass_groups_by_mode[mode][(template, language)].add(prompt_id)
    pass_groups = pass_groups_by_mode["verified_candidate_union"]

    row_coverage: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        row_coverage[language] = {}
        for kind in ("atomic", "composition"):
            prompt_ids = [prompt_id for row_kind, prompt_id in source_meta if row_kind == kind]
            base_direct = sum((kind, prompt_id, language) in base_passes for prompt_id in prompt_ids)
            combined_direct = sum((kind, prompt_id, language) in accepted_rows for prompt_id in prompt_ids)
            ceiling = sum(
                bool(pass_groups.get((source_meta[(kind, prompt_id)][1]["template"], language)))
                for prompt_id in prompt_ids
            )
            row_coverage[language][kind] = {
                "prompt_rows": len(prompt_ids),
                "source_review_pass_rows": base_direct,
                "verified_candidate_union_rows": combined_direct,
                "relation_aware_template_reuse_ceiling": ceiling,
                "optimistic_rows_added_by_reuse": ceiling - combined_direct,
                "unique_templates": len({source_meta[(kind, prompt_id)][1]["template"] for prompt_id in prompt_ids}),
            }

    atomic_rows = {prompt_id: row for (kind, prompt_id), (row, _meta) in source_meta.items() if kind == "atomic"}
    composition_rows = {prompt_id: row for (kind, prompt_id), (row, _meta) in source_meta.items() if kind == "composition"}
    atomic_by_case: dict[str, list[str]] = collections.defaultdict(list)
    for prompt_id, row in atomic_rows.items():
        atomic_by_case[row["source_qa_id"]].append(prompt_id)
    comp_by_chain: dict[str, list[str]] = collections.defaultdict(list)
    for prompt_id, row in composition_rows.items():
        comp_by_chain[row["source_qa_id"]].append(prompt_id)

    case_split = {row["source_qa_id"]: row["split"] for row in atomic_rows.values()}
    complete_cases: dict[str, dict[str, list[str]]] = {
        "source_review_only": {},
        "verified_candidate_union": {},
        "template_reuse_ceiling": {},
    }
    for mode in complete_cases:
        allowed = {
            "source_review_only": base_passes,
            "verified_candidate_union": accepted_rows,
        }.get(mode)
        for split in ("forget", "retain"):
            complete: list[str] = []
            for case_id, prompt_ids in atomic_by_case.items():
                if case_split[case_id] != split or len(prompt_ids) != 2:
                    continue
                if mode in {"source_review_only", "verified_candidate_union"}:
                    passed = all(("atomic", prompt_id, language) in allowed for prompt_id in prompt_ids for language in LANGUAGES)
                else:
                    passed = all(
                        bool(pass_groups_by_mode["verified_candidate_union"].get((source_meta[("atomic", prompt_id)][1]["template"], language)))
                        for prompt_id in prompt_ids for language in LANGUAGES
                    )
                if passed:
                    complete.append(case_id)
            complete_cases[mode][split] = sorted(complete)

    complete_chains: dict[str, list[str]] = {}
    for mode in ("source_review_only", "verified_candidate_union", "template_reuse_ceiling"):
        complete: list[str] = []
        for chain_id, prompt_ids in comp_by_chain.items():
            if len(prompt_ids) != 4:
                continue
            if mode in {"source_review_only", "verified_candidate_union"}:
                allowed = base_passes if mode == "source_review_only" else accepted_rows
                passed = all(("composition", prompt_id, language) in allowed for prompt_id in prompt_ids for language in LANGUAGES)
            else:
                passed = all(
                    bool(pass_groups_by_mode["verified_candidate_union"].get((source_meta[("composition", prompt_id)][1]["template"], language)))
                    for prompt_id in prompt_ids for language in LANGUAGES
                )
            if passed:
                complete.append(chain_id)
        complete_chains[mode] = sorted(complete)

    all_input_paths = set(original_integrity["input_paths"]) | set(repair_integrity["input_paths"]) | set(structure["input_paths"])
    result = {
        "audit": "AI-only relation-aware translation-template reuse ceiling; diagnostic only",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat(),
        "human_translation_reviewed": False,
        "release_approved": False,
        "corpus_written": False,
        "template_method": {
            "entity_slots": "Exact source entity mentions, mapped to QIDs where available, assigned first-mention slots; repeated mentions share a slot. Ambiguous aliases are excluded from reuse.",
            "atomic_relation_signature": "Graph relation PID plus source relation label and atomic hop.",
            "composition_relation_signature": "Ordered relation labels reconstructed from hash-matched source facts plus composition prompt hop.",
            "optimistic_assumption": "A passing translation for any member of a relation-aware template is treated as reusable for every source row in that template, without retranslation or independent review of each rendering.",
        },
        "integrity": {
            "original_review_rows": len(base_reviews),
            "original_review_pass_rows": len(base_passes),
            "original_review_input_checks": "Source hashes, review output hashes, English prompts, and exact translation text were verified against atomic reuse/draft/repair files and the composition review queue.",
            "original_review_source_pool_rows": original_integrity["source_pool_rows_by_language"],
            "repair_review": {key: value for key, value in repair_integrity.items() if key != "input_paths"},
            "verified_repair_pass_rows": len(repair_passes),
            "repair_review_overlap_with_original_pass": len(repair_passes & base_passes),
            "verified_candidate_union_pass_rows": len(accepted_rows),
            "source_files_sha256": {str(path.relative_to(REPO)): sha_file(path) for path in sorted(all_input_paths)},
        },
        "source_structure": {key: value for key, value in structure.items() if key != "input_paths"},
        "row_coverage_by_language_and_kind": row_coverage,
        "complete_atomic_cases_all_four_target_languages": {
            "required": {"forget": 117, "retain": 39},
            "source_review_only": {split: len(ids) for split, ids in complete_cases["source_review_only"].items()},
            "verified_candidate_union": {split: len(ids) for split, ids in complete_cases["verified_candidate_union"].items()},
            "template_reuse_ceiling": {split: len(ids) for split, ids in complete_cases["template_reuse_ceiling"].items()},
            "template_ceiling_case_ids": complete_cases["template_reuse_ceiling"],
        },
        "complete_composition_chains_all_four_target_languages": {
            "required": 44,
            "source_review_only": len(complete_chains["source_review_only"]),
            "verified_candidate_union": len(complete_chains["verified_candidate_union"]),
            "template_reuse_ceiling": len(complete_chains["template_reuse_ceiling"]),
            "template_ceiling_chain_ids": complete_chains["template_reuse_ceiling"],
        },
        "interpretation": "The estimate remains far below the frozen full-release requirements. Template reuse cannot repair the current coverage gap, especially for composition: relation-aware grouping yields 176 unique templates for 176 prompts. This estimate is not training data and does not certify human fluency.",
    }
    output = Path(__file__).resolve().parent / "template_reuse_ceiling.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(REPO)),
        "original_review_rows": len(base_reviews),
        "verified_repair_rows": repair_integrity["review_rows"],
        "repair_hash_source_failures": repair_integrity["hash_and_source_binding_failures"],
        "atomic_case_ceiling": result["complete_atomic_cases_all_four_target_languages"]["template_reuse_ceiling"],
        "composition_chain_ceiling": result["complete_composition_chains_all_four_target_languages"]["template_reuse_ceiling"],
    }, indent=2))


if __name__ == "__main__":
    main()
