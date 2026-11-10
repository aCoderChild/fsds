#!/usr/bin/env python3
"""Build source-verified, evaluation-disjoint Type-A composition SFT candidates.

The output is a review candidate only. It must not be merged into an approved
training corpus until all five language versions pass the authorized review.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/synthworlds_sm"
PILOT = ROOT / "data/synthworlds_sm_pilot_v3"
OUT = PILOT / "sft_training/disjoint_composition_candidates"
LANGS = ("en", "fr", "de", "hi", "th")
PROMPT_CORRECTIONS = {
    "0001-qa-sm": {
        "composition_support_q2": "In what year was The Artistic Exchange founded by Merith Watts?",
        "composition_2hop": "In what year was the institution where Jorith Luque studied founded?",
    },
    "0011-qa-sm": {
        "composition_support_q2": "Which language is listed as one of Pyrathen's components?",
        "composition_2hop": "Which language is listed as a component of Pyrathen, which is among Quindra Pandya's fields of work?",
    },
    "0032-qa-sm": {
        "composition_2hop": "Who participated in the Battle of Zolara, the event distinct from Echoing Vale?",
    },
    "0046-qa-sm": {
        "composition_support_q2": "Which literary work includes the city of Lighthaven?",
        "composition_2hop": "Which literary work includes the capital city of Lumeria?",
    },
    "0041-qa-sm": {
        "composition_2hop": "The language used to write \"Flames of Destiny\" is a dialect of which language?",
    },
    "0050-qa-sm": {
        "composition_support_q2": "To which country or other power is Elian Calyssar loyal?",
        "composition_2hop": "To which country or other power is the person whose godparent is Calix Silverwood loyal?",
    },
    "0076-qa-sm": {
        "composition_2hop": "What does a creator featured in The Ephemeral Showcase own?",
    },
    "0087-qa-sm": {
        "composition_support_q2": "On what date was Sapphire Bay formally incorporated as a municipality?",
        "composition_2hop": "On what date was the place where Chyra Sampson lives formally incorporated as a municipality?",
    },
    "0092-qa-sm": {
        "composition_support_q2": "In what area of activity does the Nexus Learning Society operate?",
        "composition_2hop": "In what area of activity does the organization of which Menara Carrizales is a member operate?",
    },
    "0101-qa-sm": {
        "composition_support_q2": "Which literary work is the source of Celestial Rework, an edition or translation?",
        "composition_2hop": "Which literary work is an edition or translation of the literary work that describes Draal Minaya?",
    },
    "0122-qa-sm": {
        "composition_2hop": "What did the place that shares a border with Zephyron replace?",
    },
    "0125-qa-sm": {
        "composition_support_q2": "Which type of electrical plug is standard in Merithia?",
        "composition_2hop": "Which type of electrical plug is standard in the country that has diplomatic relations with the Caelum Archipelago?",
    },
    "0129-qa-sm": {
        "composition_2hop": "In what year did one of Wryten Nate's siblings become a citizen of Lumeria?",
    },
    "0104-qa-sm": {
        "composition_2hop": "Who influenced the field of work in which Vail Thorne is active?",
    },
    "0109-qa-sm": {
        "composition_2hop": "In which administrative region is the source that describes Therion Tostado located?",
    },
    "0112-qa-sm": {
        "composition_support_q2": "On what date did the fire at the Radiant Pavilion occur?",
        "composition_2hop": "On what date did a fire occur at a structure listed among Delwyn Oleary's notable works?",
    },
    "0124-qa-sm": {
        "composition_support_q2": "Which administrative division has Nethoria as its capital?",
        "composition_2hop": "Which administrative division has as its capital the town twinned with Uldarion?",
    },
    "0126-qa-sm": {
        "composition_2hop": "In what year did Brenyth Joan's tenure as architect end at the palace associated with Avelyn Tostado's works?",
    },
    "0140-qa-sm": {
        "composition_2hop": "In what year did Tarynth Braga's term as chairperson of the organization to which Nytheris Meister belongs end?",
    },
    "0175-qa-sm": {
        "composition_2hop": "In what year did Adeline Everhart's tenure in the office held by Irvin Everhart end?",
    },
    "0183-qa-sm": {
        "composition_support_q2": "In what year was Wyrlyn Lord recorded as active?",
        "composition_2hop": "In what year was the holder of the Directorate of Scientific Archives and Research position recorded as active?",
    },
    "0196-qa-sm": {
        "composition_support_q2": "What is Zyvra Gerber's first name?",
        "composition_2hop": "What is the first name of a doctoral student of Tamsin Rivers?",
    },
    "0146-qa-sm": {
        "composition_support_q2": "In what month and year did Braxton Goldenvale become the occupant of Opulent Haven Palace?",
        "composition_2hop": "In what month and year did Braxton Goldenvale become the occupant of the palace listed among Delwyn Oleary's notable works?",
    },
    "0194-qa-sm": {
        "composition_support_q2": "Who was a doctoral student of Aelynd Shakir?",
        "composition_2hop": "Who was a doctoral student of a person associated with Wyrden Hays?",
    },
    "0197-qa-sm": {
        "composition_2hop": "Which military branch did the godparent of Sylphina Marigold Everwyn belong to?",
    },
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qid(text):
    match = re.search(r"\((Q\d+)\)$", str(text))
    return match.group(1) if match else None


def surface(text):
    return re.sub(r"\s*\(Q\d+\)$", "", str(text)).strip().rstrip(".")


def is_numeric_label(text):
    return bool(re.fullmatch(r"\s*\d+(?:[-/.]\d+)*(?:\s*\([A-Za-z][A-Za-z -]*\))?\s*", str(text)))


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
        if key.startswith("r") and isinstance(value, dict) and value.get("relation_subject_qid"):
            result.add(value["relation_subject_qid"])
    result.update(graph.get("entity_rename_map", {}))
    return result


def r2_outputs(bridge, relation, facts):
    bridge_qid = qid(bridge)
    if not bridge_qid or bridge_qid not in facts:
        return []
    outputs = []
    relation_text = relation["relation_str"]
    if relation_text.startswith("["):
        steps = [part.strip() for part in relation_text[1:-1].split("→")]
        if len(steps) != 3:
            return []
        for statement in facts[bridge_qid]:
            parts = [part.strip() for part in statement.split("→")]
            if len(parts) != 5 or parts[0] != bridge or parts[1] != steps[0] or parts[3] != steps[2]:
                continue
            subject_qid = relation.get("relation_subject_qid")
            if subject_qid and qid(parts[2]) != subject_qid:
                continue
            if not subject_qid and normalize(parts[2]) != normalize(steps[1]):
                continue
            outputs.append(parts[4])
    else:
        for statement in facts[bridge_qid]:
            parts = [part.strip() for part in statement.split("→")]
            if len(parts) >= 3 and parts[0] == bridge and parts[1] == relation_text:
                outputs.append(parts[2])
    return outputs


def r2_matches(subject, answer, relation, facts):
    subject_qid = relation.get("qid")
    if not subject_qid or subject_qid not in facts:
        return []
    relation_text = relation["relation_str"]
    if not relation_text.startswith("["):
        expected = f"{subject} → {relation_text} → {answer}"
        return [statement for statement in facts[subject_qid] if statement == expected]
    steps = [part.strip() for part in relation_text[1:-1].split("→")]
    if len(steps) != 3:
        return []
    matches = []
    for statement in facts[subject_qid]:
        parts = [part.strip() for part in statement.split("→")]
        if len(parts) != 5 or parts[0] != subject or parts[1] != steps[0] or parts[3] != steps[2]:
            continue
        if normalize(parts[4]) != normalize(answer):
            continue
        middle_qid = relation.get("relation_subject_qid")
        if middle_qid and qid(parts[2]) != middle_qid:
            continue
        if not middle_qid and normalize(parts[2]) != normalize(steps[1]):
            continue
        matches.append(statement)
    return matches


def q1_candidates(subject, relation, facts):
    subject_qid = qid(subject)
    if not subject_qid or subject_qid not in facts:
        return []
    values = set()
    for statement in facts[subject_qid]:
        parts = [part.strip() for part in statement.split("→")]
        if len(parts) >= 3 and parts[:2] == [subject, relation]:
            values.add(parts[2])
    return sorted(values, key=lambda value: (normalize(value), value))


def main():
    qa_rows = read_jsonl(SOURCE / "qa-sm.jsonl")
    facts_rows = read_jsonl(SOURCE / "qa-sm-facts.jsonl")
    assignment_rows = read_jsonl(PILOT / "assignments.jsonl")
    labels = read_jsonl(PILOT / "evaluation_dataset/labels/gold_labels.jsonl")
    audits = read_jsonl(ROOT / "data/synthworlds_sm_reaudit/type_a_target_eligibility.jsonl")
    qa = {row["instance_id"]: row for row in qa_rows}
    facts = {row["qid"]: row["facts"] for row in facts_rows}
    audit = {row["instance_id"]: row for row in audits}
    assignments = {row["source_qa_id"]: row for row in assignment_rows}
    labels_by_id = {row["instance_id"]: row for row in labels}
    qa_sha = sha(SOURCE / "qa-sm.jsonl")
    facts_sha = sha(SOURCE / "qa-sm-facts.jsonl")
    eval_ids = set(assignments)

    eval_qids, eval_names, eval_answers, eval_queries, eval_fact_ids = set(), set(), set(), set(), set()
    for iid in eval_ids:
        row = qa[iid]
        graph = json.loads(row["mhop_qa"])
        eval_qids |= graph_qids(row, graph)
        eval_names.update(normalize(entry.get("new_name", "")) for entry in graph.get("entity_rename_map", {}).values())
        for key, value in graph["graph_qa"].items():
            if re.fullmatch(r"n\d+", key):
                eval_names.add(normalize(value))
        label = labels_by_id[iid]
        eval_answers.update(normalize(value) for value in label["final_answers"])
        eval_answers.add(normalize(label["direct_intermediate_answer"]))
        eval_queries.add(normalize(row["query"]))
        eval_fact_ids.update(label["chain_fact_ids"])

    candidates, excluded = [], Counter()
    for row in qa_rows:
        iid = row["instance_id"]
        if iid in eval_ids or row.get("question_graph_type") != "A":
            continue
        graph = json.loads(row["mhop_qa"])
        graph_qa = graph["graph_qa"]
        n1, n2, n3 = (graph_qa.get(f"n{index}") for index in (1, 2, 3))
        r1, r2 = graph_qa.get("r1", {}), graph_qa.get("r2", {})
        if not all((n1, n2, n3, r1.get("qid"), r2.get("qid"))):
            excluded["incomplete_graph"] += 1
            continue
        if r1["qid"] != qid(n1):
            excluded["r1_subject_mismatch"] += 1
            continue
        direct_fact = f"{n1} → {r1['relation_str']} → {n2}"
        source_r1 = facts.get(r1["qid"], [])
        if source_r1.count(direct_fact) != 1:
            excluded["r1_fact_not_unique"] += 1
            continue
        bridges = q1_candidates(n1, r1["relation_str"], facts)
        if n2 not in bridges:
            excluded["target_bridge_missing"] += 1
            continue
        target_r2_facts = r2_matches(n2, n3, r2, facts)
        if len(target_r2_facts) != 1:
            excluded["target_r2_not_unique"] += 1
            continue
        independent_audit = audit.get(iid)
        if (not independent_audit
                or independent_audit.get("target_fact_text") != f"{n1} → {r1['relation_str']} → {n2}"
                or set(independent_audit.get("q1_possible_answers", [])) != set(bridges)
                or target_r2_facts[0] not in independent_audit.get("r2_matched_fact_texts", [])):
            excluded["independent_source_fact_audit_mismatch"] += 1
            continue
        all_outputs = [value for bridge in bridges for value in r2_outputs(bridge, r2, facts)]
        output_labels = {normalize(value) for value in all_outputs}
        if output_labels != {normalize(n3)}:
            excluded["final_answer_not_unique_across_bridges"] += 1
            continue
        if len(row.get("gold_answers", [])) != 1 or normalize(row["gold_answers"][0]) != normalize(n3):
            excluded["gold_answer_graph_mismatch"] += 1
            continue
        if normalize(row["query"]) != normalize(graph.get("composed_question", row["query"])):
            excluded["composed_question_mismatch"] += 1
            continue

        chain_qids = graph_qids(row, graph)
        answers = {normalize(value) for value in row["gold_answers"]}
        if chain_qids & eval_qids:
            excluded["entity_qid_overlap_with_eval"] += 1
            continue
        query_names = {normalize(entry.get("new_name", "")) for entry in graph.get("entity_rename_map", {}).values()}
        query_names.update(normalize(value) for key, value in graph_qa.items() if re.fullmatch(r"n\d+", key))
        if query_names & (eval_names | eval_answers) or answers & eval_answers:
            excluded["entity_or_answer_surface_overlap_with_eval"] += 1
            continue
        if normalize(row["query"]) in eval_queries:
            excluded["question_overlap_with_eval"] += 1
            continue

        path_fact_ids = [fact_id(r1["qid"], direct_fact), fact_id(r2["qid"], target_r2_facts[0])]
        if set(path_fact_ids) & eval_fact_ids:
            excluded["fact_id_overlap_with_eval"] += 1
            continue

        q2 = re.sub(r"</?bridge>", "", graph.get("decomposed_questions", {}).get("Q2", "")).strip()
        if not q2:
            excluded["decomposition_missing"] += 1
            continue
        answer = row["gold_answers"][0].strip()
        corrected = PROMPT_CORRECTIONS.get(iid, {})
        q2 = corrected.get("composition_support_q2", q2)
        multi_question = corrected.get("composition_2hop", row["query"].strip())
        prompts = [("q2", "composition_support_q2", q2, answer),
                   ("multi", "composition_2hop", multi_question, answer)]
        chain = path_fact_ids
        entity_names = {surface(value) for value in (n1, n2, n3) if value and not is_numeric_label(surface(value))}
        entity_names.update(entry.get("new_name", "") for entry in graph.get("entity_rename_map", {}).values()
                            if entry.get("new_name"))
        for suffix, hop, question, target in prompts:
            row_qids = chain_qids
            row_facts = chain
            protected_names = sorted(
                {
                    name for name in entity_names
                    if name and (
                        re.search(re.escape(name), question, flags=re.IGNORECASE)
                        or re.search(re.escape(name), target, flags=re.IGNORECASE)
                    )
                },
                key=lambda value: (value.casefold(), value),
            )
            candidates.append({
                "sft_id": f"disjoint:{iid}:{suffix}", "source_qa_id": iid, "source_qa_ids": [iid],
                "split": "train_only", "hop": hop, "language": "en", "example_type": "disjoint_composition_support",
                "chain_fact_ids": row_facts, "entity_qids": sorted(row_qids), "gold_answers_source": row["gold_answers"],
                "q1_possible_answer_count": len(bridges), "messages": [
                    {"role": "user", "content": question}, {"role": "assistant", "content": target}],
                "question_en": question, "answer_en": target, "protected_entity_names": protected_names,
                "translation_review_mode": "SOURCE_ENGLISH", "human_translation_reviewed": False,
                "do_not_train": True, "source_qa_sha256": qa_sha,
                "source_facts_sha256": facts_sha,
            })

    chain_count = len({row["source_qa_id"] for row in candidates})
    row_count = len(candidates)
    if chain_count < 1 or row_count < chain_count * 2:
        raise RuntimeError(f"Candidate output is incomplete: {chain_count} chains, {row_count} rows")
    if len({row["sft_id"] for row in candidates}) != row_count:
        raise RuntimeError("Duplicate candidate SFT identifiers")
    by_language = {language: row_count if language == "en" else 0 for language in LANGS}
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT / "english_candidate.jsonl"
    corpus_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates), encoding="utf-8")
    manifest = {
        "status": "DISJOINT_COMPOSITION_CANDIDATES_PENDING_TRANSLATION_REVIEW",
        "do_not_train": True, "source_qa_sha256": qa_sha,
        "source_facts_sha256": facts_sha,
        "evaluation_assignments_sha256": sha(PILOT / "assignments.jsonl"),
        "evaluation_labels_sha256": sha(PILOT / "evaluation_dataset/labels/gold_labels.jsonl"),
        "evaluation_entity_qids": len(eval_qids), "candidate_chain_count": chain_count,
        "candidate_sft_rows_en": row_count,
        "rows_by_hop": dict(sorted(Counter(row["hop"] for row in candidates).items())),
        "chains_with_q1_all_support": 0,
        "languages_required": list(LANGS), "human_translation_reviewed": False,
        "candidate_rows_by_language": by_language,
        "exclusion_counts": dict(sorted(excluded.items())),
        "selection_policy": "Type-A source QA rows outside fixed evaluation IDs; exact first-hop source fact; exactly one second-hop source path; every source-supported Q1 bridge resolves to the same final answer; composition and Q2 rows have no chain entity/QID, answer, question, or selected chain-fact overlap with evaluation.",
        "q1_policy": "All source-supported Q1 bridge values are audited to ensure they lead to the same final answer, but multi-answer Q1 list rows are omitted from SFT. The train_only extension uses only Q2 support and full composed two-hop questions.",
        "prompt_corrections": PROMPT_CORRECTIONS,
        "candidate_sha256": sha(corpus_path),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "candidate_chains": chain_count,
                      "candidate_rows_en": row_count, "excluded": manifest["exclusion_counts"],
                      "candidate_sha256": manifest["candidate_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
