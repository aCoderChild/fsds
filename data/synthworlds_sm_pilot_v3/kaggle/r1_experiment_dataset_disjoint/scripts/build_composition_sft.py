#!/usr/bin/env python3
"""Add retained-chain multi-hop practice to the approved atomic SFT corpus.

Only the 26 retain-split final-answer questions are added, in each of the five
languages. Forget and holdout prompts/labels are never copied to the training
corpus. The retain questions deliberately overlap the retain evaluation set;
only forget-split multi-hop accuracy is used to assess transfer/generalization.
"""
import argparse
import hashlib
import json
from pathlib import Path

LANGS = ("en", "fr", "de", "hi", "th")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text):
    import re
    import unicodedata
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def build(root, output_path=None, manifest_path=None):
    root = Path(root)
    eval_root = root / "evaluation_dataset"
    atomic_path = root / "sft_training/corpus.jsonl"
    atomic_manifest_path = root / "sft_training/manifest.json"
    eval_manifest_path = eval_root / "manifest.json"
    prompts_path = eval_root / "prompts/en.jsonl"
    labels_path = eval_root / "labels/gold_labels.jsonl"
    required = [atomic_path, atomic_manifest_path, eval_manifest_path, prompts_path, labels_path]
    if any(not p.is_file() for p in required):
        raise ValueError("Atomic SFT corpus and complete evaluation source files are required")

    atomic_manifest = json.loads(atomic_manifest_path.read_text(encoding="utf-8"))
    if atomic_manifest.get("status") != "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS" or atomic_manifest.get("do_not_train") is not False:
        raise ValueError("The source atomic SFT corpus is not approved for this AI-only pilot")
    if atomic_manifest.get("human_translation_reviewed") is not False:
        raise ValueError("Source corpus human-review disclosure is missing or inconsistent")
    if atomic_manifest.get("corpus_sha256") != sha(atomic_path):
        raise ValueError("Atomic SFT corpus checksum mismatch")

    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if (eval_manifest.get("evaluation_ready") is not True
            or eval_manifest.get("ai_review_status") != "AI_ONLY_REVIEWED_COMPLETE"
            or eval_manifest.get("human_review_status") != "NOT_PERFORMED"):
        raise ValueError("Evaluation prompts are not complete AI-only reviewed data")
    if eval_manifest.get("split_counts") != {"forget": 77, "retain": 26, "holdout": 26}:
        raise ValueError("Unexpected source split counts")

    atomic_rows = read_jsonl(atomic_path)
    if any(row.get("split") == "holdout" for row in atomic_rows):
        raise ValueError("Atomic source corpus includes holdout rows")
    if any(row.get("language") not in LANGS for row in atomic_rows):
        raise ValueError("Atomic corpus contains an unsupported language")

    english_prompts = read_jsonl(prompts_path)
    prompt_map = {(r["instance_id"], r["probe"]): r for r in english_prompts}
    if len(prompt_map) != 258 or any(r.get("do_not_train") is not True for r in english_prompts):
        raise ValueError("English prompt file must contain 258 evaluation-only prompts")
    labels = read_jsonl(labels_path)
    label_map = {r["instance_id"]: r for r in labels}
    if len(labels) != 129 or len(label_map) != 129 or any(r.get("label_file_only_never_send_to_model") is not True for r in labels):
        raise ValueError("Gold label file is incomplete, duplicated, or lacks its separation marker")
    retain_ids = {iid for iid, row in label_map.items() if row.get("split") == "retain"}
    if len(retain_ids) != 26:
        raise ValueError("Expected exactly 26 retain chains")
    for iid in retain_ids:
        question = prompt_map.get((iid, "multi"))
        if not question or question.get("split") != "retain" or question.get("do_not_train") is not True:
            raise ValueError(f"Missing or inconsistent retained English multi-hop prompt: {iid}")

    translated = {"en": {iid: prompt_map[(iid, "multi")] for iid in retain_ids}}
    translation_hashes = {}
    for lang in LANGS[1:]:
        path = eval_root / "translations/ai_reviewed" / f"{lang}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing AI-reviewed evaluation prompts: {lang}")
        rows = read_jsonl(path)
        if len(rows) != 258 or any(r.get("review_status") != "AI_REVIEWED" for r in rows):
            raise ValueError(f"Incomplete or mislabeled AI-reviewed {lang} prompts")
        by_id = {r["translation_id"]: r for r in rows}
        if len(by_id) != 258:
            raise ValueError(f"Duplicate translated evaluation prompt in {lang}")
        translation_hashes[lang] = sha(path)
        translated[lang] = {}
        for iid in retain_ids:
            tid = f"{iid}:multi:{lang}"
            row = by_id.get(tid)
            if not row or not str(row.get("question", "")).strip():
                raise ValueError(f"Missing translated retain multi-hop prompt: {lang}/{iid}")
            translated[lang][iid] = row

    composition_rows = []
    for lang in LANGS:
        for iid in sorted(retain_ids):
            question_row = translated[lang][iid]
            label = label_map[iid]
            question = question_row.get("question", "")
            answers = label.get("final_answers")
            if not isinstance(answers, list) or len(answers) != 1 or not str(answers[0]).strip():
                raise ValueError(f"Expected one canonical final answer for retained chain {iid}")
            answer = str(answers[0]).strip()
            if normalize(answer) and normalize(answer) in normalize(question):
                raise ValueError(f"Final-answer shortcut occurs in retained multi-hop prompt: {lang}/{iid}")
            composition_rows.append({
                "sft_id": f"composition:{iid}",
                "source_qa_id": iid,
                "source_qa_ids": [iid],
                "hop": "composition_2hop",
                "split": "retain",
                "language": lang,
                "example_type": "retained_multihop_composition",
                "bootstrap_cluster_id": label["bootstrap_cluster_id"],
                "chain_fact_ids": label["chain_fact_ids"],
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "training_overlap_disclosure": "This retained multi-hop prompt is also in the retain evaluation set; do not use retain scores as unseen-chain generalization.",
                "translation_review_mode": "AI_ONLY" if lang != "en" else "SOURCE_ENGLISH",
                "human_translation_reviewed": False,
            })

    if len(composition_rows) != 26 * len(LANGS):
        raise ValueError("Composition row coverage is not 26 retained chains in all five languages")
    if any(row["split"] != "retain" for row in composition_rows):
        raise ValueError("Only retained composition examples may enter this corpus")
    if {row["source_qa_id"] for row in composition_rows} != retain_ids:
        raise ValueError("Composition rows do not match the retained source chain IDs")

    output_path = Path(output_path or root / "sft_training/composition_corpus.jsonl")
    manifest_path = Path(manifest_path or root / "sft_training/composition_manifest.json")
    corpus = atomic_rows + composition_rows
    if len({(r.get("language"), r.get("sft_id")) for r in corpus}) != len(corpus):
        raise ValueError("Duplicate language/SFT IDs in augmented corpus")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in corpus), encoding="utf-8")
    rows_by_language = {lang: sum(row.get("language") == lang for row in corpus) for lang in LANGS}
    manifest = {
        "status": "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS",
        "do_not_train": False,
        "training_variant": "atomic_five_language_sft_plus_retained_chain_composition",
        "source_corpus_sha256": sha(atomic_path),
        "source_manifest_sha256": sha(atomic_manifest_path),
        "evaluation_manifest_sha256": sha(eval_manifest_path),
        "translation_review_mode": "AI_ONLY",
        "human_translation_reviewed": False,
        "translation_review_authorization": atomic_manifest["translation_review_authorization"],
        "translation_review_policy_sha256": atomic_manifest["translation_review_policy_sha256"],
        "translation_review_report_sha256": atomic_manifest["translation_review_report_sha256"],
        "ai_reviewed_rows": atomic_manifest["ai_reviewed_rows"] + 26 * len(LANGS[1:]),
        "languages": list(LANGS),
        "rows_by_language": rows_by_language,
        "atomic_rows_by_language": atomic_manifest["rows_by_language"],
        "composition_rows_by_language": {lang: 26 for lang in LANGS},
        "total_rows": len(corpus),
        "unique_sft_ids_per_language": atomic_manifest["rows_by_language"]["en"] + 26,
        "holdout_rows": 0,
        "forget_composition_rows": 0,
        "retain_composition_rows": len(composition_rows),
        "composition_source_ids_sha256": hashlib.sha256("\n".join(sorted(retain_ids)).encode()).hexdigest(),
        "composition_prompt_sha256": {"en": sha(prompts_path), **translation_hashes},
        "composition_label_sha256": sha(labels_path),
        "composition_policy": {
            "source_split": "retain",
            "examples_per_language": 26,
            "total_examples": len(composition_rows),
            "uses_forget_or_holdout_labels": False,
            "retain_prompt_eval_overlap": True,
            "primary_generalization_split": "forget",
            "human_translation_reviewed": False,
        },
        "split_approval": atomic_manifest["split_approval"],
        "corpus_sha256": sha(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "COMPOSITION_SFT_CORPUS_READY", "corpus": str(output_path), "manifest": str(manifest_path),
            "rows_by_language": rows_by_language, "total_rows": len(corpus), "composition_rows": len(composition_rows),
            "forget_or_holdout_composition_rows": 0, "retain_prompt_eval_overlap": True,
            "corpus_sha256": manifest["corpus_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        result = build(args.root, args.output, args.manifest)
    except (KeyError, ValueError) as exc:
        print(json.dumps({"status": "COMPOSITION_SFT_CORPUS_BLOCKED", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
