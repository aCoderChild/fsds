#!/usr/bin/env python3
"""Small deterministic checks for the SFT handoff; never downloads a model."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from build_sft_corpus import APPROVED, LANGS, assemble
from import_sft_translations import LANGS as TARGET_LANGS, import_batches
from train_sft_lora import encode_assistant_only, preflight


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        tokens = []
        for message in messages:
            tokens.extend([10 if message["role"] == "user" else 20, *[ord(c) for c in message["content"]], 30])
        if add_generation_prompt:
            tokens.extend([20])
        return tokens


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class SFTPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.approval_path = self.root / "split_approval.json"
        self.approval_path.write_text(json.dumps({"status": "APPROVED_PILOT_SPLIT", "reviewer_name": "Fixture AI Reviewer", "reviewer_type": "AI", "review_date": "2026-11-02"}), encoding="utf-8")
        self.policy_path = self.root / "translation_review_policy.json"
        self.policy_path.write_text(json.dumps({"status": "AI_ONLY_REVIEW_AUTHORIZED", "review_mode": "AI_ONLY", "human_fluency_review": False}), encoding="utf-8")
        review_dir = self.root / "sft_translation_ai_review"
        review_dir.mkdir()
        self.en = [
            {"sft_id": "a:r1", "source_qa_id": "a", "hop": "r1", "split": "forget", "question_en": "Who founded A?", "answer": "A Person", "messages": [{"role": "user", "content": "Who founded A?"}, {"role": "assistant", "content": "A Person"}]},
            {"sft_id": "b:r2", "source_qa_id": "b", "hop": "r2", "split": "retain", "question_en": "When did B open?", "answer": "1942 (year)", "messages": [{"role": "user", "content": "When did B open?"}, {"role": "assistant", "content": "1942 (year)"}]},
        ]
        self.en_path = self.root / "en.jsonl"
        write_rows(self.en_path, self.en)
        self.lang_paths = {}
        self.request_paths = {}
        for lang in LANGS[1:]:
            path = self.root / f"{lang}.jsonl"
            req_dir = self.root / "sft_translation_requests" / lang
            req_dir.mkdir(parents=True)
            request_rows = [{"sft_id": r["sft_id"], "source_qa_id": r["source_qa_id"], "source_qa_ids": [r["source_qa_id"]], "hop": r["hop"], "hops": [r["hop"]], "split": r["split"], "question_en": r["question_en"], "answer_en": r["answer"], "protected_entity_names": [], "translate_to": lang} for r in self.en]
            write_rows(req_dir / "batch_001.jsonl", request_rows)
            self.request_paths[lang] = req_dir
            write_rows(path, [{"sft_id": r["sft_id"], "source_qa_id": r["source_qa_id"], "source_qa_ids": [r["source_qa_id"]], "hop": r["hop"], "hops": [r["hop"]], "split": r["split"], "question_en": r["question_en"], "answer_en": r["answer"], "language": lang, "question": f"{lang}: {r['question_en']}", "answer_translation": f"{lang}: {r['answer']}", "protected_entity_names": [], "translation_status": APPROVED, "review_status": APPROVED, "review_mode": "AI_ONLY", "reviewer_type": "AI", "human_review_status": "NOT_PERFORMED", "reviewer_name": "Fixture AI Reviewer", "review_date": "2026-11-02"} for r in self.en])
            self.lang_paths[lang] = path
        languages = {}
        for lang, path in self.lang_paths.items():
            languages[lang] = {"candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (review_dir / "review_report.json").write_text(json.dumps({"status": "AI_ONLY_TRANSLATION_REVIEW_COMPLETE_HUMAN_REVIEW_NOT_PERFORMED", "ai_approved_total": 8, "human_reviewed_total": 0, "languages": languages}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_complete_approved_five_language_corpus(self):
        out, manifest = self.root / "corpus.jsonl", self.root / "manifest.json"
        result = assemble(self.en_path, self.lang_paths, out, manifest, self.request_paths)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["total_rows"], 10)
        self.assertEqual({r["language"] for r in rows}, set(LANGS))
        self.assertEqual(sum(r["split"] == "holdout" for r in rows), 0)
        self.assertEqual(result["split_approval"]["status"], "APPROVED_PILOT_SPLIT")
        self.assertEqual(result["split_approval"]["reviewer_name"], "Fixture AI Reviewer")
        self.assertEqual(result["split_approval"]["reviewer_type"], "AI")
        self.assertEqual(result["translation_review_mode"], "AI_ONLY")
        self.assertFalse(result["human_translation_reviewed"])
        fr = next(r for r in rows if r["language"] == "fr" and r["sft_id"] == "a:r1")
        self.assertEqual(fr["messages"][1]["content"], "fr: A Person")

    def test_rejects_pending_translation_and_holdout(self):
        rows = [json.loads(x) for x in self.lang_paths["fr"].read_text(encoding="utf-8").splitlines()]
        rows[0]["review_status"] = "PENDING"
        write_rows(self.lang_paths["fr"], rows)
        with self.assertRaisesRegex(ValueError, "not AI-reviewed and approved"):
            assemble(self.en_path, self.lang_paths, self.root / "x.jsonl", self.root / "x.json", self.request_paths)
        self.en[0]["split"] = "holdout"
        write_rows(self.en_path, self.en)
        with self.assertRaisesRegex(ValueError, "Holdout"):
            assemble(self.en_path, self.lang_paths, self.root / "x.jsonl", self.root / "x.json", self.request_paths)

    def test_requires_split_approval_record(self):
        self.approval_path.unlink()
        with self.assertRaisesRegex(ValueError, "split_approval.json is required"):
            assemble(self.en_path, self.lang_paths, self.root / "x.jsonl", self.root / "x.json", self.request_paths)

    def test_rejects_changed_protected_entity(self):
        rows = [json.loads(x) for x in self.lang_paths["fr"].read_text(encoding="utf-8").splitlines()]
        rows[0]["protected_entity_names"] = ["A Person"]
        rows[0]["answer_translation"] = "Une personne"
        write_rows(self.lang_paths["fr"], rows)
        with self.assertRaisesRegex(ValueError, "Protected entity name"):
            assemble(self.en_path, self.lang_paths, self.root / "x.jsonl", self.root / "x.json", self.request_paths)

    def test_assistant_only_mask_and_padding_contract(self):
        ids, labels = encode_assistant_only(FakeTokenizer(), [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
        self.assertEqual(len(ids), len(labels))
        self.assertTrue(all(x == -100 for x in labels[:len(ids) - 3]))
        self.assertTrue(any(x != -100 for x in labels))

    def test_preflight_checks_approved_manifest_checksum(self):
        out, manifest = self.root / "corpus.jsonl", self.root / "manifest.json"
        assemble(self.en_path, self.lang_paths, out, manifest, self.request_paths)
        result = preflight(out, manifest, self.root / "run", train=False)
        self.assertEqual(result["status"], "READY_TO_TRAIN")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["do_not_train"] = True
        manifest.write_text(json.dumps(data), encoding="utf-8")
        result = preflight(out, manifest, self.root / "run", train=True)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["training_started"])

    def test_translation_import_is_complete_and_never_auto_approves(self):
        root = self.root / "dataset"
        (root / "sft_translation_candidates").mkdir(parents=True)
        write_rows(root / "sft_natural_en_candidate.jsonl", self.en)
        batches = self.root / "batches"
        for lang in TARGET_LANGS:
            completed_path = batches / lang / "batch_001.jsonl"
            completed_path.parent.mkdir(parents=True)
            request_rows = [{"sft_id": r["sft_id"], "source_qa_ids": [r["source_qa_id"]], "hops": [r["hop"]], "split": r["split"], "question_en": r["question_en"], "answer_en": r["answer"], "protected_entity_names": [], "translate_to": lang, "question_translation": "", "answer_translation": ""} for r in self.en]
            req_path = root / "sft_translation_requests" / lang / "batch_001.jsonl"
            req_path.parent.mkdir(parents=True, exist_ok=True)
            write_rows(req_path, request_rows)
            completed = [dict(row, question_translation=f"{lang} question", answer_translation=f"{lang} answer") for row in request_rows]
            write_rows(completed_path, completed)
        counts = import_batches(root, batches)
        self.assertEqual(counts, {lang: 2 for lang in TARGET_LANGS})
        imported = json.loads((root / "sft_translation_candidates" / "fr.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(imported["review_status"], "PENDING")
        self.assertEqual(imported["translation_status"], "PENDING_FLUENT_REVIEW")
        self.assertTrue(imported["do_not_train"])


if __name__ == "__main__":
    unittest.main()
