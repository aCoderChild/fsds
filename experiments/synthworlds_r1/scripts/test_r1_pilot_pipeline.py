#!/usr/bin/env python3
"""Offline fixture checks for the v5.1 experiment gates and analysis path."""
from __future__ import annotations

import hashlib
import io
import json
import ast
import importlib.util
import sys
import subprocess
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "training_release_draft" / "scripts"))
import evaluate_pilot_three_probe as evaluation
import unlearn_lora
import unlearning_objectives
import train_sft_lora
import repair_all_flagged_training_translations as repair_translations
import repair_rejected_training_candidates_feedback as feedback_repairs
import review_training_translations_roundtrip as roundtrip_reviews
from kaggle_result_snapshot import restore_snapshot, write_snapshot
from validate_8b_runtime_smoke_receipt import validate_receipt as validate_8b_smoke_receipt


EVAL = ROOT / "training_release_draft" / "evaluation"
PROMPTS = EVAL / "prompts.jsonl"
LABELS = EVAL / "labels.jsonl"
REVISION = "a" * 40
sys.path.insert(0, str(ROOT / "training_release_draft" / "scripts"))
import assemble_training_release as release
import assemble_sft_source_verified as verified_release
import validate_sft_source_integrity as source_integrity


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checkpoint_fingerprint(adapter_sha: str | None) -> str:
    profile = evaluation.get_model_profile(evaluation.PILOT_PROFILE)
    payload = {"model_id": profile.model_id, "model_revision": REVISION,
               "adapter_sha256": adapter_sha, "model_profile": profile.manifest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class FixtureTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        prompt = "<user>q<assistant>"
        full = prompt + "answer<eos>"
        if not tokenize:
            return prompt if add_generation_prompt else full
        # Deliberately model a subword boundary change: independently tokenized
        # prompt IDs are not a prefix of the full-conversation token IDs.
        return [11, 12, 99] if add_generation_prompt else [11, 12, 21, 22]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if text != "<user>q<assistant>answer<eos>" or add_special_tokens:
            raise AssertionError("Expected one complete rendered conversation without extra special tokens")
        return {
            "input_ids": [11, 12, 21, 22],
            "offset_mapping": [(0, 6), (6, 18), (18, 24), (24, len(text))],
        }


class PipelineFixtureTests(unittest.TestCase):
    def test_roundtrip_blocks_backtranslation_when_opaque_marker_is_lost(self):
        row = {
            "prompt_id": "fixture-marker-loss",
            "kind": "atomic",
            "text_en": "Who is Aster Vale's editor?",
            "text_translation": "Qui est l'éditeur d'Aster Vale ?",
            "protected_entity_names": ["Aster Vale"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            original_out = roundtrip_reviews.OUT
            roundtrip_reviews.OUT = Path(tmp)
            try:
                with patch.object(roundtrip_reviews, "translate_batch", return_value={"ROW_A": "Who is the editor?"}):
                    result = roundtrip_reviews.backtranslate_language(
                        "fr", [row], {"source_fixture": "fixture-hash"}, batch_size=1, timeout=1
                    )
            finally:
                roundtrip_reviews.OUT = original_out

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["backtranslation_status"], "BLOCKED_BACKTRANSLATION_FAILURE")
        self.assertIn("Protected marker counts changed", result[0]["backtranslation_error"])
        self.assertEqual(result[0]["backtranslation_en"], "")

    def test_main_notebook_requires_config_access_and_matching_8b_smoke_receipt(self):
        notebook_path = ROOT / "kaggle/r1_multiseed_experiment.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        setup_cell = "".join(notebook["cells"][2]["source"])
        access_cell = "".join(notebook["cells"][4]["source"])
        baseline_cell = "".join(notebook["cells"][8]["source"])
        post_cell = "".join(notebook["cells"][10]["source"])
        export_cell = "".join(notebook["cells"][11]["source"])
        sft_cell = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                        if "def validate_sft_run" in "".join(cell.get("source", [])))
        unlearning_cell = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                               if "def validate_unlearning_run" in "".join(cell.get("source", [])))
        self.assertIn('INPUT.rglob("r1_8b_runtime_smoke.json")', setup_cell)
        self.assertIn('filename="config.json"', access_cell)
        self.assertIn("validate_model_config(MODEL_PROFILE, config)", access_cell)
        self.assertIn("get_pinned_model_revision(MODEL_PROFILE)", setup_cell)
        self.assertIn("revision=MODEL_REVISION_PIN", access_cell)
        self.assertIn("resolved_revision != MODEL_REVISION_PIN", access_cell)
        self.assertNotIn("MODEL_REVISION = model_info.sha", access_cell)
        self.assertIn("Selected-profile config access verified", access_cell)
        self.assertIn('STAGE_ENABLED and PROFILE.role == "main_study"', access_cell)
        self.assertIn('STAGE_ENABLED and PROFILE.role == "feasibility_pilot"', access_cell)
        self.assertIn("validate_receipt(smoke_receipt", access_cell)
        self.assertIn('if expected != sha256_file(LABELS):', setup_cell)
        self.assertIn('"evaluation_labels_sha256": EVALUATION_LABELS_SHA256', export_cell)
        self.assertLess(baseline_cell.index("EVALUATION_LABELS_SHA256 = verify_evaluation_labels_sha256()"), baseline_cell.index("score_command ="))
        self.assertLess(post_cell.index("EVALUATION_LABELS_SHA256 = verify_evaluation_labels_sha256()"), post_cell.index('"score"'))
        verify_node = next(node for node in ast.parse(setup_cell).body
                           if isinstance(node, ast.FunctionDef) and node.name == "verify_evaluation_labels_sha256")
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "labels.jsonl"
            label_path.write_text('{"fixture": true}\n', encoding="utf-8")
            namespace = {
                "manifest": {"evaluation": {"labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()}},
                "LABELS": label_path,
                "sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            exec(compile(ast.Module(body=[verify_node], type_ignores=[]), "notebook-label-hash-test", "exec"), namespace)
            self.assertEqual(namespace["verify_evaluation_labels_sha256"](), namespace["manifest"]["evaluation"]["labels_sha256"])
            namespace["manifest"]["evaluation"]["labels_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "Evaluation labels do not match"):
                namespace["verify_evaluation_labels_sha256"]()
        self.assertIn("runtime_smoke_receipt_sha256", export_cell)
        self.assertIn('"scripts/assistant_masking.py"', setup_cell)
        self.assertIn('"assistant_masking_script_sha256": sha256_file(PROJECT_ROOT / "scripts/assistant_masking.py")', sft_cell)
        self.assertIn('"assistant_masking_script_sha256": sha256_file(PROJECT_ROOT / "scripts/assistant_masking.py")', unlearning_cell)
        self.assertIn('PROJECT_ROOT / "scripts/assistant_masking.py"', export_cell)
        self.assertIn("UNLEARNING_OBJECTIVE_SCRIPT", setup_cell)
        stage_c = "".join(notebook["cells"][9]["source"])
        self.assertIn('"objective_script_sha256"', stage_c)

        from model_profiles import MAIN_PROFILE, PINNED_MODEL_REVISIONS, PILOT_PROFILE
        self.assertRegex(PINNED_MODEL_REVISIONS[MAIN_PROFILE], r"^[0-9a-f]{40}$")
        self.assertIsNone(PINNED_MODEL_REVISIONS[PILOT_PROFILE])

    def test_model_config_validation_matches_selected_1b_or_8b_profile(self):
        from model_profiles import MAIN_PROFILE, PILOT_PROFILE, validate_model_config

        validate_model_config(PILOT_PROFILE, {
            "model_type": "llama", "num_hidden_layers": 16, "hidden_size": 2048,
        })
        validate_model_config(MAIN_PROFILE, {
            "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
        })
        with self.assertRaisesRegex(ValueError, "does not match profile"):
            validate_model_config(PILOT_PROFILE, {
                "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
            })
        with self.assertRaisesRegex(ValueError, "does not match profile"):
            validate_model_config(MAIN_PROFILE, {
                "model_type": "llama", "num_hidden_layers": 16, "hidden_size": 2048,
            })

    def test_runtime_smoke_notebook_binds_shared_dual_gpu_npo_objective(self):
        smoke_root = ROOT / "kaggle_8b_runtime_smoke"
        notebook = json.loads((smoke_root / "llama31_8b_runtime_smoke.ipynb").read_text(encoding="utf-8"))
        manifest = json.loads((smoke_root / "notebook_manifest.json").read_text(encoding="utf-8"))
        embedded = "".join(notebook["cells"][1]["source"])
        objective_source = (ROOT / "scripts/unlearning_objectives.py").read_text(encoding="utf-8")
        runner_source = (ROOT / "scripts/kaggle_8b_runtime_smoke.py").read_text(encoding="utf-8")
        mask_preflight_source = (ROOT / "scripts/kaggle_8b_mask_preflight.py").read_text(encoding="utf-8")
        self.assertEqual(digest(objective_source), manifest["npo_objective_sha256"])
        self.assertIn(manifest["npo_objective_sha256"], embedded)
        self.assertEqual(digest(runner_source), manifest["runner_sha256"])
        self.assertIn(manifest["runner_sha256"], embedded)
        self.assertEqual(digest(mask_preflight_source), manifest["mask_preflight_sha256"])
        self.assertIn(manifest["mask_preflight_sha256"], embedded)
        self.assertIn("from unlearning_objectives import npo_loss", embedded)
        self.assertIn("two visible CUDA GPUs", embedded)
        self.assertIn("npo_optimizer_steps_completed", embedded)
        self.assertIn("RUN_MASK_PREFLIGHT = False", embedded)
        self.assertIn("RUN_SMOKE = False", embedded)
        masking_source = (ROOT / "scripts/assistant_masking.py").read_text(encoding="utf-8")
        self.assertEqual(digest(masking_source), manifest["assistant_masking_sha256"])
        self.assertIn(manifest["assistant_masking_sha256"], embedded)
        self.assertIn("single_render_fast_tokenizer_offsets", (ROOT / "scripts/kaggle_8b_runtime_smoke.py").read_text(encoding="utf-8"))
        self.assertLess(
            runner_source.index("encode_assistant_only(tokenizer, messages)"),
            runner_source.index("AutoModelForCausalLM.from_pretrained("),
            "Tokenizer/mask validation must fail before the large checkpoint download.",
        )
        from model_profiles import MAIN_PROFILE, PINNED_MODEL_REVISIONS
        revision_pin = PINNED_MODEL_REVISIONS[MAIN_PROFILE]
        self.assertIn(revision_pin, manifest["model_revision_pin"])
        self.assertIn(revision_pin, embedded)
        self.assertIn("revision=PINNED_REVISION", embedded)

    def test_runtime_smoke_receipt_is_bound_to_exact_8b_config_and_stack(self):
        from model_profiles import MAIN_PROFILE, get_pinned_model_revision

        pinned_revision = get_pinned_model_revision(MAIN_PROFILE)
        receipt = {
            "status": "PASS_RUNTIME_ONLY",
            "runner_sha256": digest((ROOT / "scripts/kaggle_8b_runtime_smoke.py").read_text(encoding="utf-8")),
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "scope": "SYNTHETIC_RUNTIME_SMOKE_ONLY",
            "model_revision": pinned_revision,
            "config_sha256": "b" * 64,
            "config_model_type": "llama",
            "package_versions": {
                "transformers": "5.22.1", "peft": "0.21.2",
                "accelerate": "1.15.0", "bitsandbytes": "0.50.2",
            },
            "model_loaded_4bit": True,
            "optimizer_steps_requested": 1,
            "trainer_global_step": 1,
            "loss_mask_method": "single_render_fast_tokenizer_offsets",
            "assistant_supervised_tokens": 3,
            "research_data_loaded": False,
            "gpus": [{"name": "fixture GPU 0"}, {"name": "fixture GPU 1"}],
            "trainer_train_loss": 1.0,
            "reference_model_loaded_4bit": True,
            "npo_reference_cuda_device": 1,
            "npo_optimizer_steps_requested": 1,
            "npo_optimizer_steps_completed": 1,
            "npo_beta": 0.1,
            "npo_loss": 1.0,
        }
        kwargs = {"model_id": receipt["model_id"], "model_revision": pinned_revision,
                  "config_sha256": receipt["config_sha256"]}
        validate_8b_smoke_receipt(receipt, **kwargs)
        for field, value in (("runner_sha256", "e" * 64), ("model_revision", "c" * 40), ("config_sha256", "d" * 64),
                             ("model_loaded_4bit", False), ("reference_model_loaded_4bit", False),
                             ("npo_optimizer_steps_completed", 0), ("npo_loss", float("nan")),
                             ("research_data_loaded", True), ("gpus", [{"name": "one GPU"}])):
            invalid = dict(receipt, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_8b_smoke_receipt(invalid, **kwargs)

        verifier = ROOT / "kaggle_8b_runtime_smoke/verify_receipt.py"
        with tempfile.TemporaryDirectory() as td:
            receipt_path = Path(td) / "r1_8b_runtime_smoke.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            passed = subprocess.run([sys.executable, str(verifier), str(receipt_path)], capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            wrong_revision = dict(receipt, model_revision="a" * 40)
            receipt_path.write_text(json.dumps(wrong_revision), encoding="utf-8")
            rejected = subprocess.run([sys.executable, str(verifier), str(receipt_path)], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("revision", rejected.stdout)

    def test_shared_npo_objective_matches_reference_formula(self):
        import torch

        beta = 0.1
        policy_logp = torch.tensor([-2.0], requires_grad=True)
        reference_logp = torch.tensor([-2.0])
        actual = unlearning_objectives.npo_loss(policy_logp, reference_logp, beta)
        expected = -(2.0 / beta) * torch.nn.functional.logsigmoid(
            -beta * (policy_logp - reference_logp)
        )
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(policy_logp.grad).all())
        for invalid_beta in (0, -0.1, float("nan"), float("inf")):
            with self.subTest(beta=invalid_beta), self.assertRaises(ValueError):
                unlearning_objectives.npo_loss(policy_logp, reference_logp, invalid_beta)

    def test_stage_script_defaults_match_the_r1_main_model(self):
        from model_profiles import MAIN_PROFILE, get_model_profile

        main_profile = get_model_profile(MAIN_PROFILE)
        self.assertEqual(train_sft_lora.MODEL_ID, main_profile.model_id)
        self.assertEqual(unlearn_lora.MODEL_ID, main_profile.model_id)
        self.assertEqual(evaluation.MODEL_ID, main_profile.model_id)
        self.assertEqual(train_sft_lora.preflight.__defaults__[-1], MAIN_PROFILE)
        self.assertEqual(train_sft_lora.run_training.__defaults__[-1], MAIN_PROFILE)
        self.assertEqual(unlearn_lora.preflight.__defaults__[-1], MAIN_PROFILE)
        self.assertEqual(unlearn_lora.train.__defaults__[-1], MAIN_PROFILE)
        self.assertEqual(evaluation.validate_resume_rows.__defaults__[-1], MAIN_PROFILE)

    def test_tg_complement_compact_hash_ignores_jsonl_key_sorting(self):
        verifier_path = (ROOT / "training_release_draft/translation_repair_diagnostics/"
                         "chatgpt_fullflagged_queue_2026-11-04/verify_tg_complement_review.py")
        spec = importlib.util.spec_from_file_location("tg_complement_verifier_fixture", verifier_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = {
            "translated_question": "Question cible",
            "source_question_en": "Source question",
            "kind": "atomic",
            "language": "fr",
            "review_key": "T0001",
        }
        expected = json.dumps([{
            "review_key": "T0001", "language": "fr", "kind": "atomic",
            "source_question_en": "Source question", "translated_question": "Question cible",
        }], ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(module.compact_input_payload([row]), expected)

    def test_full799_repair_input_hash_contract(self):
        audit_path = (ROOT / "training_release_draft/translation_repair_diagnostics/"
                      "chatgpt_fullflagged_queue_2026-11-04/analyze_chatgpt_repair_coverage.py")
        spec = importlib.util.spec_from_file_location("repair_coverage_audit", audit_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        input_dir = audit_path.parent / "independent_review_inputs_2026-11-04"
        rows = module.read_jsonl(input_dir / "full799.jsonl")
        launch = json.loads((input_dir / "full799.kaggle_launch_manifest.json").read_text(encoding="utf-8"))
        file_hash = hashlib.sha256((input_dir / "full799.jsonl").read_bytes()).hexdigest()
        self.assertEqual(len(rows), 799)
        self.assertEqual(module.compact_json_array_sha256(rows), launch["input_compact_array_sha256"])
        self.assertEqual(file_hash, launch["input_jsonl_sha256"])
        self.assertNotEqual(file_hash, launch["input_compact_array_sha256"])

    def test_full799_coverage_uses_the_exact_frozen_case_universe(self):
        audit_path = (ROOT / "training_release_draft/translation_repair_diagnostics/"
                      "chatgpt_fullflagged_queue_2026-11-04/analyze_chatgpt_repair_coverage.py")
        spec = importlib.util.spec_from_file_location("repair_coverage_audit", audit_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        atomic_rows = module.read_jsonl(ROOT / "training_release_draft/atomic_reuse.jsonl")
        composition_rows = module.read_jsonl(ROOT / "training_release_draft/composition_translation_review_queue.jsonl")
        result = module.compute_coverage({lang: set() for lang in module.LANGS}, atomic_rows, composition_rows)
        self.assertEqual(result["atomic_cases_total"], {"forget": 117, "retain": 39})
        self.assertEqual(result["composition_chains_total"], 44)

    def test_full799_verified_rows_require_hash_and_count_binding(self):
        audit_path = (ROOT / "training_release_draft/translation_repair_diagnostics/"
                      "chatgpt_fullflagged_queue_2026-11-04/analyze_chatgpt_repair_coverage.py")
        spec = importlib.util.spec_from_file_location("repair_coverage_audit", audit_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verified.jsonl"
            row = {"review_status": "AI_ONLY_REPAIR_REVIEW_PASS", "prompt_id": "P1"}
            path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            receipt = {
                "verified_rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "counts": {"AI_ONLY_REPAIR_REVIEW_PASS": 1}, "rows": 1,
            }
            self.assertEqual(module.read_bound_verified_rows(receipt, path, 1), [row])
            path.write_text(json.dumps({**row, "review_status": "AI_ONLY_REPAIR_REVIEW_FLAGGED"}, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                module.read_bound_verified_rows(receipt, path, 1)
            receipt["verified_rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt["counts"] = {"AI_ONLY_REPAIR_REVIEW_PASS": 1}
            with self.assertRaisesRegex(ValueError, "counts differ"):
                module.read_bound_verified_rows(receipt, path, 1)

    def test_full799_candidate_override_handoff_is_quarantined_and_source_bound(self):
        script_path = (ROOT / "training_release_draft/translation_repair_diagnostics/"
                       "chatgpt_fullflagged_queue_2026-11-04/stage_verified_repair_overrides.py")
        spec = importlib.util.spec_from_file_location("stage_verified_repair_overrides", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        queue = script_path.parent
        candidates = module.read_jsonl(queue / "parsed_repair_candidates.jsonl")
        sources = {lang: module.collect_rows(lang)[0] for lang in module.LANGS}
        reviews = []
        for index, candidate in enumerate(candidates):
            reviews.append({
                "language": candidate["language"], "prompt_id": candidate["prompt_id"],
                "kind": candidate["kind"],
                "candidate_translation_sha256": hashlib.sha256(candidate["text_translation"].encode()).hexdigest(),
                "review_status": ("AI_ONLY_REPAIR_REVIEW_PASS" if index == 0 else "AI_ONLY_REPAIR_REVIEW_FLAGGED"),
            })
        overrides = module.build_overrides(candidates, reviews, sources)
        self.assertEqual(sum(map(len, overrides.values())), 1)
        staged = next(row for values in overrides.values() for row in values)
        self.assertEqual(staged["translation_status"], "MACHINE_DRAFT_PENDING_QWEN_RERUN")
        self.assertIs(staged["human_translation_reviewed"], False)

        broken_reviews = [dict(row) for row in reviews]
        broken_reviews[0]["candidate_translation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not bind"):
            module.build_overrides(candidates, broken_reviews, sources)

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "translation_flagged_repairs"
            manifest = module.write_bundle(overrides, {"input_receipts": {"fixture": "bound"}}, target)
            self.assertEqual(manifest["status"], "CANDIDATE_OVERRIDES_STAGED_FOR_FULL_PRODUCTION_REREVIEW_NOT_APPROVED")
            self.assertIs(manifest["release_approved"], False)
            payload = (target / "fr.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["override_sha256_by_language"]["fr"])
            with self.assertRaises(FileExistsError):
                module.write_bundle(overrides, {"input_receipts": {"fixture": "bound"}}, target)

    def test_translation_reviewers_write_to_authoritative_tree(self):
        scripts = (
            ROOT / "training_release_draft/scripts/review_training_translations_roundtrip.py",
            ROOT / "training_release_draft/scripts/adjudicate_training_translations_qwen14b.py",
        )
        for path in scripts:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            assignments = {
                target.id: ast.unparse(node.value)
                for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)
            }
            self.assertEqual(assignments.get("V51"), "REPO / 'experiments/synthworlds_r1'")
            self.assertEqual(assignments.get("OUT"), "V51 / 'training_release_draft'")

    def test_legacy_adjudication_receipt_is_accepted_only_for_exact_root_alias(self):
        report = release.read_json(release.REVIEW_DIR / "report.json")
        receipt_sha = report["adjudication_script_sha256"]
        provenance = release.adjudication_script_provenance(receipt_sha)
        self.assertEqual(provenance["status"], "PASS_EXACT_LEGACY_ROOT_PATH_ALIAS")
        self.assertEqual(provenance["reconstructed_legacy_script_sha256"], receipt_sha)
        self.assertTrue(provenance["legacy_root_resolves_to_canonical_project"])
        # Verify all saved decisions against current source rows and all four
        # complete 488-row files; only the root-path assignment may differ.
        self.assertEqual(len(release.checked_translation_reviews()), 4 * 488)
        with self.assertRaisesRegex(ValueError, "different script version"):
            release.adjudication_script_provenance("0" * 64)

    def test_verified_assembler_stops_before_release_on_source_mismatch(self):
        with patch.object(verified_release.source, "validate_prepared_inputs", side_effect=ValueError("source mismatch")), \
                patch.object(verified_release.legacy, "main") as legacy_main:
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                verified_release.main()
            legacy_main.assert_not_called()

    def test_prepared_source_reconstruction_rejects_changed_composition_answer(self):
        if (not (source_integrity.prepare.SOURCE / "qa-sm.jsonl").is_file()
                or not (source_integrity.prepare.V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/corpus.jsonl").is_file()):
            self.skipTest("full original source snapshots are not attached")
        canonical, proof = source_integrity.validate_prepared_inputs()
        self.assertEqual(len(canonical), 488)
        self.assertEqual(proof["canonical_sft_ids"], 488)
        original = source_integrity.read_jsonl
        source_path = source_integrity.TRAINING / "composition_reuse.jsonl"
        victim = original(source_path)[0]["sft_id"]

        def tampered(path):
            rows = original(path)
            if Path(path) == source_path:
                return [{**row, "answer": "IncorrectFixtureAnswer", "answer_en": "IncorrectFixtureAnswer"}
                        if row["sft_id"] == victim else row for row in rows]
            return rows

        with patch.object(source_integrity, "read_jsonl", side_effect=tampered):
            with self.assertRaisesRegex(ValueError, "composition_reuse.jsonl"):
                source_integrity.validate_prepared_inputs()

    def test_release_source_proof_rejects_self_consistent_wrong_answers(self):
        if (not (source_integrity.prepare.SOURCE / "qa-sm.jsonl").is_file()
                or not (source_integrity.prepare.V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/corpus.jsonl").is_file()):
            self.skipTest("full original source snapshots are not attached")
        # This source-integrity test should use the current review rows as
        # synthetic inputs without depending on a live approval receipt or
        # the current adjudicator-script digest. Those provenance gates have
        # their own validation path and may correctly be stale between runs.
        reviews = {}
        for lang in release.TARGET_LANGS:
            rows_path = release.REVIEW_DIR / f"{lang}.jsonl"
            for row in release.read_jsonl(rows_path):
                reviews[(lang, row["prompt_id"], row["kind"])] = row
        for row in reviews.values():
            row["final_review_status"] = "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS"  # synthetic fixture only
        corpus, _ = release.build_corpus(reviews)
        with tempfile.TemporaryDirectory(prefix="r1-source-release-") as td:
            root = Path(td)
            corpus_path, manifest_path = root / "corpus.jsonl", root / "manifest.json"
            release_path = root / "release_validation.json"

            def write_fixture(rows):
                release.write_jsonl(corpus_path, rows)
                manifest_path.write_text(json.dumps({"corpus_sha256": train_sft_lora.sha(corpus_path)}), encoding="utf-8")
                release_path.write_text(json.dumps({"status": "PILOT_TRAINING_RELEASE_READY", "do_not_train": False,
                    "corpus_sha256": train_sft_lora.sha(corpus_path), "manifest_sha256": train_sft_lora.sha(manifest_path)}), encoding="utf-8")

            write_fixture(corpus)
            with patch.object(source_integrity.legacy, "checked_translation_reviews", return_value=reviews):
                proof = source_integrity.validate_release(corpus_path, manifest_path)
                self.assertEqual(proof["status"], "PASS_SOURCE_BOUND_SFT_VALIDATION")
                victim = next(row["sft_id"] for row in corpus if row["split"] == "train_only")
                altered = [{**row, "answer": "IncorrectFixtureAnswer", "answer_en": "IncorrectFixtureAnswer",
                            "messages": [row["messages"][0], {"role": "assistant", "content": "IncorrectFixtureAnswer"}]}
                           if row["sft_id"] == victim else row for row in corpus]
                write_fixture(altered)  # matching corpus/manifest hashes must still be insufficient
                with self.assertRaisesRegex(ValueError, "differs from verified source"):
                    source_integrity.validate_release(corpus_path, manifest_path)

    def test_corpus_source_alignment_rejects_consistent_wrong_answer(self):
        source = {"source_qa_id": "case", "split": "train_only", "hop": "composition_3hop",
                  "question_en": "Who is the subject?", "answer_en": "CorrectEntity", "fact_ids": ["fact"]}
        canonical = {"id": source}
        rows = [{**source, "sft_id": "id", "language": lang, "question": source["question_en"],
                 "answer": source["answer_en"]} for lang in ("en", "fr", "de", "hi", "th")]
        source_integrity.validate_corpus_source_rows(rows, canonical)
        wrong = [{**row, "answer": "WrongEntity", "answer_en": "WrongEntity"} for row in rows]
        with self.assertRaisesRegex(ValueError, "answer_en differs from verified source"):
            source_integrity.validate_corpus_source_rows(wrong, canonical)

    def test_sft_source_proof_is_required_and_bound_to_exact_files(self):
        with tempfile.TemporaryDirectory(prefix="r1-source-proof-") as td:
            root = Path(td)
            (root / "scripts").mkdir()
            (root / "sft_training").mkdir()
            verifier = root / "scripts/validate_sft_source_integrity.py"
            corpus = root / "sft_training/corpus.jsonl"
            manifest_path = root / "sft_training/manifest.json"
            release_path = root / "sft_training/release_validation.json"
            verifier.write_text("fixture verifier\n", encoding="utf-8")
            corpus.write_text("fixture corpus\n", encoding="utf-8")
            release_path.write_text("{}\n", encoding="utf-8")
            manifest = {"total_rows": 2440, "unique_sft_ids_per_language": 488,
                        "source_manifest_sha256": "a" * 64, "preparation_report_sha256": "b" * 64,
                        "candidate_cases_sha256": "c" * 64}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("proof", train_sft_lora.source_integrity_error(corpus, manifest_path, manifest))
            preflight = train_sft_lora.preflight(corpus, manifest_path, root / "output")
            self.assertTrue(any("Source-bound SFT validation proof" in error for error in preflight["errors"]))
            proof = {"status": "PASS_SOURCE_BOUND_SFT_VALIDATION", "human_translation_reviewed": False,
                     "corpus_sha256": train_sft_lora.sha(corpus), "manifest_sha256": train_sft_lora.sha(manifest_path),
                     "release_validation_sha256": train_sft_lora.sha(release_path), "verifier_sha256": train_sft_lora.sha(verifier),
                     "corpus_rows": 2440, "canonical_sft_ids": 488, "languages": ["en", "fr", "de", "hi", "th"],
                     "source_checks": {"canonical_sft_ids": 488, "source_manifest_sha256": "a" * 64,
                                       "preparation_report_sha256": "b" * 64, "source_inputs_sha256": {"candidate_cases": "c" * 64}}}
            (corpus.parent / "source_integrity_validation.json").write_text(json.dumps(proof), encoding="utf-8")
            self.assertIsNone(train_sft_lora.source_integrity_error(corpus, manifest_path, manifest))
            corpus.write_text("changed corpus\n", encoding="utf-8")
            self.assertIn("does not match", train_sft_lora.source_integrity_error(corpus, manifest_path, manifest))

    def test_qwen14_downgrade_is_excluded_without_invalidating_the_review_file(self):
        valid_checks = {"meaning": True, "script": True}
        downgraded = {
            "review_status": "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS",
            "final_review_status": "AI_ONLY_REVIEW_FLAGGED",
            "qwen_semantic_pass": True,
            "qwen_fluency_pass": True,
            "qwen14b_semantic_pass": False,
            "qwen14b_fluency_pass": True,
            "target_deterministic_checks": valid_checks,
            "backtranslation_checks": valid_checks,
        }
        release.validate_adjudication_record("fr", ("fact-1", "atomic"), downgraded)

        adjudicated_pass = {
            **downgraded,
            "review_status": "AI_ONLY_REVIEW_FLAGGED",
            "final_review_status": "AI_ONLY_14B_ADJUDICATED_PASS",
            "qwen_semantic_pass": False,
            "qwen_fluency_pass": False,
            "qwen14b_semantic_pass": True,
            "qwen14b_fluency_pass": True,
        }
        release.validate_adjudication_record("fr", ("fact-2", "atomic"), adjudicated_pass)

        incorrectly_flagged = {**adjudicated_pass, "final_review_status": "AI_ONLY_REVIEW_FLAGGED"}
        with self.assertRaisesRegex(ValueError, "satisfies every pass condition"):
            release.validate_adjudication_record("fr", ("fact-3", "atomic"), incorrectly_flagged)

    def test_full_flagged_repair_selector_is_complete_stable_and_answer_blind(self):
        rows_by_lang = {}
        for lang in repair_translations.LANGS:
            rows_by_lang[lang] = [
                {
                    "prompt_id": f"{lang}-{i:04d}",
                    "kind": "atomic" if i % 2 == 0 else "composition",
                    "review_status": "AI_ONLY_REVIEW_FLAGGED" if i < 2 else "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS",
                    "text_en": f"Question {i}?",
                    "text_translation": f"Ancienne question {i}?",
                }
                for i in range(488)
            ]
        selected = repair_translations.select_flagged_rows(rows_by_lang)
        selected_keys = [(x["language"], x["row"]["prompt_id"], x["row"]["kind"]) for x in selected]
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(selected_keys), len(set(selected_keys)))
        self.assertEqual(selected_keys, [
            (x["language"], x["row"]["prompt_id"], x["row"]["kind"])
            for x in repair_translations.select_flagged_rows(rows_by_lang)
        ])
        self.assertEqual({key[0] for key in selected_keys}, set(repair_translations.LANGS))
        rows_by_lang["fr"][0]["gold_answers"] = ["answer leak"]
        with self.assertRaisesRegex(ValueError, "answer-bearing data"):
            repair_translations.select_flagged_rows(rows_by_lang)

    def test_repair_resume_uses_hash_bound_ordered_checkpoint(self):
        selected = [
            {"language": "fr", "row": {
                "prompt_id": f"p{i}", "kind": "atomic", "text_en": f"Question {i}?",
                "text_translation": f"Ancienne question {i}?",
            }}
            for i in range(2)
        ]
        decisions = []
        for item in selected:
            row = item["row"]
            translation = f"Nouvelle question {row['prompt_id']}?"
            decisions.append({
                "language": "fr", "prompt_id": row["prompt_id"], "kind": row["kind"],
                "source_en_sha256": repair_translations.sha_text(row["text_en"]),
                "old_translation_sha256": repair_translations.sha_text(row["text_translation"]),
                "new_translation": translation,
                "new_translation_sha256": repair_translations.sha_text(translation),
            })
        with tempfile.TemporaryDirectory(prefix="r1-repair-checkpoint-") as td:
            path = Path(td) / "decisions.jsonl"
            first = (json.dumps(decisions[0], sort_keys=True) + "\n").encode()
            uncommitted = (json.dumps(decisions[1], sort_keys=True) + "\n").encode()
            path.write_bytes(first + uncommitted)
            report = {
                "processed_rows": 1,
                "decisions_checkpoint_sha256": hashlib.sha256(first).hexdigest(),
            }
            loaded, committed = repair_translations.load_checkpoint(
                path, report, selected, allow_uncommitted_tail=True,
            )
            self.assertEqual(loaded, [decisions[0]])
            self.assertEqual(committed, first)
            self.assertEqual(path.read_bytes(), first)

            tampered = first.replace(b"p0", b"px")
            path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                repair_translations.load_checkpoint(
                    path, report, selected, allow_uncommitted_tail=True,
                )

    def test_feedback_repair_selector_is_stable_stratified_and_blind(self):
        rows = []
        for lang in feedback_repairs.LANGS:
            for kind in feedback_repairs.KINDS:
                for i in range(3):
                    rows.append({
                        "language": lang, "kind": kind, "prompt_id": f"{lang}-{kind}-{i}",
                        "status": "DIAGNOSTIC_QWEN_FLAGGED", "qwen_pass": False,
                        "qwen_reason": "The translated question reverses relation direction.",
                        "new_translation": f"Question candidate {i}?",
                    })
        selected = feedback_repairs.select_feedback_rows(rows, per_cell=1)
        keys = [(r["language"], r["kind"], r["prompt_id"]) for r in selected]
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(keys)), 8)
        self.assertEqual(keys, [
            (r["language"], r["kind"], r["prompt_id"])
            for r in feedback_repairs.select_feedback_rows(rows, per_cell=1)
        ])
        self.assertEqual({(r["language"], r["kind"]) for r in selected},
                         {(lang, kind) for lang in feedback_repairs.LANGS for kind in feedback_repairs.KINDS})
        self.assertEqual(
            feedback_repairs.protect_old_translation(
                "Did Aster Vale arrive in 2010?",
                {"ZXQENTAAQXZ": "Aster Vale", "ZXQNUMBAQXZ": "2010"},
            ),
            "Did ZXQENTAAQXZ arrive in ZXQNUMBAQXZ?",
        )
        requests = []
        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(json.dumps({"message": {"content": "Eine korrigierte Frage?"}}).encode())
        with patch.object(feedback_repairs.urllib.request, "urlopen", side_effect=fake_urlopen):
            translated = feedback_repairs.translate_feedback_once(
                "Who is ZXQENTAAQXZ named after?",
                "Who is ZXQENTAAQXZ named?",
                "The earlier candidate changes the relation direction.",
                "de", 7,
            )
        self.assertEqual(translated, "Eine korrigierte Frage?")
        request_payload = json.loads(requests[0][0].data)
        prompt = request_payload["messages"][0]["content"]
        self.assertIn("critique may be wrong", prompt)
        self.assertIn("English question is the source of truth", prompt)
        self.assertNotIn("gold_answers", prompt)
        self.assertEqual(requests[0][1], 7)
        rows[0]["gold_answers"] = ["secret answer"]
        with self.assertRaisesRegex(ValueError, "answer-bearing"):
            feedback_repairs.select_feedback_rows(rows, per_cell=1)

    def test_repair_runner_completes_and_revalidates_before_resume(self):
        rows_by_lang = {}
        for lang in repair_translations.LANGS:
            rows_by_lang[lang] = [
                {
                    "prompt_id": f"{lang}-{i:04d}",
                    "kind": "atomic",
                    "review_status": (
                        "AI_ONLY_REVIEW_FLAGGED" if lang == "fr" and i == 0
                        else "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS"
                    ),
                    "text_en": f"English question {lang}-{i}?",
                    "text_translation": f"Previous question {lang}-{i}?",
                    "protected_entity_names": [],
                }
                for i in range(488)
            ]
        checks = {name: True for name in repair_translations.REQUIRED_CHECKS}
        with tempfile.TemporaryDirectory(prefix="r1-repair-run-") as td:
            output = Path(td) / "diagnostic"
            with (
                patch.object(repair_translations, "source_manifest", return_value=({}, rows_by_lang)),
                patch.object(repair_translations, "protect", return_value=("English question fr-0000?", {}, {})),
                patch.object(repair_translations, "restore", side_effect=lambda text, _mapping, _expected: text),
                patch.object(repair_translations, "translate_with_retries", return_value="Une question traduite?") as translator,
                patch.object(repair_translations, "deterministic_checks", return_value=checks),
                patch.object(repair_translations, "masked_pair", return_value=(None, {"text": "masked"})),
                patch.object(repair_translations, "ask_qwen", return_value=[{"decision": True, "reason": "consistent"}]) as reviewer,
            ):
                completed = repair_translations.run(output, timeout=1, resume=False)
                self.assertEqual(completed["status"], "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW")
                self.assertEqual(completed["processed_rows"], 1)
                self.assertFalse(completed["human_translation_reviewed"])
                self.assertFalse(completed["answers_sent_to_reviewer"])
                self.assertEqual(translator.call_count, 1)
                self.assertEqual(reviewer.call_count, 1)

                resumed = repair_translations.run(output, timeout=1, resume=True)
                self.assertEqual(resumed["status"], "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW")
                self.assertEqual(translator.call_count, 1)
                self.assertEqual(reviewer.call_count, 1)

                with (output / "repair_candidates.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write("{}\n")
                with self.assertRaisesRegex(ValueError, "uncommitted decision row"):
                    repair_translations.run(output, timeout=1, resume=True)

    def test_assistant_only_mask_and_retain_schedule(self):
        ids, labels = train_sft_lora.encode_assistant_only(
            FixtureTokenizer(), [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        self.assertEqual(ids, [11, 12, 21, 22])
        self.assertEqual(labels, [-100, -100, 21, 22])
        unlearning_ids, unlearning_labels = unlearn_lora.encode_unlearning_example(
            FixtureTokenizer(), [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], 8
        )
        self.assertEqual(unlearning_ids, ids)
        self.assertEqual(unlearning_labels, labels)
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            unlearn_lora.encode_unlearning_example(
                FixtureTokenizer(), [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], 3
            )
        unlearning_source = (ROOT / "scripts/unlearn_lora.py").read_text(encoding="utf-8")
        self.assertIn("encode_assistant_only(tokenizer, messages)", unlearning_source)
        self.assertNotIn("Unsafe chat-template mask", unlearning_source)

        class MissingOffsets(FixtureTokenizer):
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                return {"input_ids": [11, 12, 21, 22]}

        with self.assertRaisesRegex(ValueError, "fast tokenizer with offset_mapping"):
            train_sft_lora.encode_assistant_only(
                MissingOffsets(), [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
            )
        schedule = unlearn_lora.retain_schedule_indices(45, 26)
        self.assertEqual(schedule[:26], list(range(26)))
        self.assertEqual(schedule[26:], list(range(19)))
        self.assertEqual(unlearn_lora.retain_schedule_indices(0, 0), [])

    def test_notebook_rejects_sft_resume_from_different_training_script(self):
        notebook = json.loads((ROOT / "kaggle/r1_multiseed_experiment.ipynb").read_text(encoding="utf-8"))
        source = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                      if "def validate_sft_run" in "".join(cell.get("source", [])))
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "validate_sft_run")
        with tempfile.TemporaryDirectory(prefix="r1-sft-provenance-") as td:
            root = Path(td)
            manifest_path = root / "sft_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            source_proof = root / "source_integrity_validation.json"
            source_proof.write_text("{}\n", encoding="utf-8")
            output_dir = root / "run"
            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True)
            run_path = output_dir / "run_manifest.json"
            expected = {
                "status": "TRAINING_COMPLETE", "training_started": True,
                "model_id": train_sft_lora.get_model_profile(train_sft_lora.PILOT_PROFILE).model_id,
                "model_profile": train_sft_lora.get_model_profile(train_sft_lora.PILOT_PROFILE).manifest(),
                "model_revision": REVISION,
                "training_script_sha256": "current-script-sha",
                "assistant_masking_script_sha256": "assistant-mask-sha",
                "seed": 42, "corpus_sha256": "corpus-sha",
                "corpus_manifest_sha256": "manifest-sha",
                "source_integrity_validation_sha256": "proof-sha",
                "training_variant": "composition_sft",
                "included_splits": ["forget", "retain", "train_only"],
                "adapter_dir": str(adapter_dir), "adapter_sha256": "adapter-sha",
            }
            run_path.write_text(json.dumps(expected), encoding="utf-8")
            namespace = {
                "Path": Path, "json": json, "MODEL_REVISION": REVISION,
                "PROJECT_ROOT": ROOT,
                "MODEL_ID": train_sft_lora.get_model_profile(train_sft_lora.PILOT_PROFILE).model_id,
                "PROFILE": train_sft_lora.get_model_profile(train_sft_lora.PILOT_PROFILE),
                "manifest": {"corpus_sha256": "corpus-sha"},
                "CORPUS_MANIFEST": manifest_path,
                "SOURCE_PROOF": source_proof,
                "TRAIN_SCRIPT": ROOT / "scripts/train_sft_lora.py",
                "sha256_file": lambda path: {"train_sft_lora.py": "current-script-sha",
                                               "assistant_masking.py": "assistant-mask-sha",
                                               "source_integrity_validation.json": "proof-sha"}.get(Path(path).name, "manifest-sha"),
                "sha256_tree": lambda _path: "adapter-sha",
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_validate_sft_run", "exec"), namespace)
            namespace["validate_sft_run"](
                run_path, output_dir, 42, ("forget", "retain", "train_only"), "composition_sft",
            )
            expected["source_integrity_validation_sha256"] = "stale-proof-sha"
            run_path.write_text(json.dumps(expected), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source_integrity_validation_sha256"):
                namespace["validate_sft_run"](
                    run_path, output_dir, 42, ("forget", "retain", "train_only"), "composition_sft",
                )
            expected["source_integrity_validation_sha256"] = "proof-sha"
            expected["training_script_sha256"] = "stale-script-sha"
            run_path.write_text(json.dumps(expected), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "training_script_sha256"):
                namespace["validate_sft_run"](
                    run_path, output_dir, 42, ("forget", "retain", "train_only"), "composition_sft",
                )

    def test_unlearning_rejects_noop_or_invalid_training_parameters(self):
        self.assertIsNone(unlearn_lora.validate_training_parameters("npo", 3, 1e-5, 1.0, 0.1, 512))
        invalid = [
            ("npo", 0, 1e-5, 1.0, 0.1, 512),
            ("npo", 3, 0.0, 1.0, 0.1, 512),
            ("npo", 3, 1e-5, -1.0, 0.1, 512),
            ("npo", 3, 1e-5, 1.0, 0.0, 512),
            ("ga", 3, 1e-5, 1.0, 0.1, 0),
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                unlearn_lora.validate_training_parameters(*parameters)

    def test_invalid_output_guards(self):
        self.assertEqual(evaluation.invalid_output_reason("", False), "empty")
        self.assertEqual(evaluation.invalid_output_reason("<|eot_id|>", False), "template_artifact")
        self.assertEqual(evaluation.invalid_output_reason("blue blue blue blue", False), "repetitive_ngram")
        self.assertIsNone(evaluation.invalid_output_reason("Aster Vale", False))
        self.assertEqual(evaluation.invalid_output_reason("ok", True), "generation_failed")

    def test_single_entity_group_does_not_claim_a_confidence_interval(self):
        self.assertEqual(evaluation.bootstrap_mean({"only": [0.0, 1.0]}, 1, 20), (0.5, None))
        self.assertEqual(evaluation.bootstrap_ratio({"only": (1, 2)}, 1, 20), (0.5, None))

    def test_main_effects_cli_requires_minimum_bootstrap_draws(self):
        argv = ["evaluate_pilot_three_probe.py", "effects", "--case-metrics", "metrics.jsonl",
                "--target-manifest", "targets.json", "--model-pairs", "pairs.json",
                "--output-dir", "effects", "--bootstrap", "99999"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            evaluation.main()
        self.assertEqual(raised.exception.code, 2)

    def test_dap_interval_targets_unweighted_seed_mean(self):
        counts = {42: {"a": (1, 1), "b": (0, 9)}, 43: {"a": (0, 1), "b": (1, 1)}}
        point, interval = evaluation.bootstrap_seed_ratio(counts, 1, 100)
        self.assertAlmostEqual(point, 0.3)  # mean of 1/10 and 1/2, not pooled 2/12
        self.assertIsNotNone(interval)
        point, interval = evaluation.bootstrap_seed_ratio({42: {"a": (1, 1)}, 43: {"b": (1, 1)}}, 1, 100)
        self.assertEqual(point, 1.0)
        self.assertIsNone(interval)  # some cluster resamples have a zero denominator

    def test_multilingual_normalization_preserves_vowel_and_tone_marks(self):
        self.assertNotEqual(evaluation.normalize("का"), evaluation.normalize("कि"))
        self.assertNotEqual(evaluation.normalize("ป่า"), evaluation.normalize("ป้า"))
        self.assertEqual(evaluation.normalize("É"), evaluation.normalize("e\u0301"))
        self.assertEqual(evaluation.normalize("Aster-Vale"), "aster vale")

    def test_resume_rejects_changed_prompt_seed_and_configuration(self):
        prompt = {"prompt_id": "p", "language": "hi", "prompt": "Original question?",
                  "source_qa_id": "case", "split": "forget", "group_id": "group", "probe": "multi_hop"}
        config = {"do_sample": False, "batch_size": 2}
        row = {**{key: prompt[key] for key in ("prompt_id", "language", "source_qa_id", "split", "group_id", "probe")},
               "model": "sft_seed_42", "model_id": evaluation.get_model_profile(evaluation.PILOT_PROFILE).model_id,
               "model_profile": evaluation.get_model_profile(evaluation.PILOT_PROFILE).manifest(), "seed": 42,
               "checkpoint_sha256": "c" * 64, "model_revision": REVISION,
               "generation_config": config, "generation_config_sha256": "d" * 64,
               "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest()}
        args = ([row], [prompt], "sft_seed_42", 42, "c" * 64, REVISION, config, "d" * 64)
        self.assertEqual(evaluation.validate_resume_rows(*args, model_profile=evaluation.PILOT_PROFILE), {("p", "hi")})
        changed_prompt = {**prompt, "prompt": "Changed question?"}
        with self.assertRaisesRegex(ValueError, "current prompt, seed, or metadata"):
            evaluation.validate_resume_rows([row], [changed_prompt], *args[2:], model_profile=evaluation.PILOT_PROFILE)
        with self.assertRaisesRegex(ValueError, "current prompt, seed, or metadata"):
            evaluation.validate_resume_rows([row], [prompt], "sft_seed_42", 43, *args[4:], model_profile=evaluation.PILOT_PROFILE)
        changed_config = {**row, "generation_config": {"do_sample": True, "batch_size": 2}}
        with self.assertRaisesRegex(ValueError, "checkpoint/config"):
            evaluation.validate_resume_rows([changed_config], [prompt], *args[2:], model_profile=evaluation.PILOT_PROFILE)

    def test_kaggle_stage_snapshot_restore_and_checksum_guard(self):
        with tempfile.TemporaryDirectory(prefix="r1-snapshot-fixture-") as td:
            temp = Path(td)
            source_root = temp / "source" / "synthworlds_r1_results"
            source_file = source_root / "sft" / "seed_42" / "run_manifest.json"
            source_file.parent.mkdir(parents=True)
            source_file.write_text('{"status":"TRAINING_COMPLETE"}\n', encoding="utf-8")
            archive_path = temp / "stageA_sft_latest.zip"
            write_snapshot(source_root, archive_path, "stageA_sft")

            restored_working = temp / "restored"
            restored = restore_snapshot([archive_path], restored_working,
                restored_working / "synthworlds_r1_results")
            restored_file = restored_working / "synthworlds_r1_results/sft/seed_42/run_manifest.json"
            self.assertEqual(restored["stage"], "stageA_sft")
            self.assertEqual(restored_file.read_text(encoding="utf-8"), source_file.read_text(encoding="utf-8"))

            corrupt_path = temp / "corrupt_latest.zip"
            with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(corrupt_path, "w") as corrupt:
                for name in source.namelist():
                    payload = source.read(name)
                    if name.endswith("run_manifest.json"):
                        payload += b"tampered"
                    corrupt.writestr(name, payload)
            with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
                restore_snapshot([corrupt_path], temp / "corrupt_restore",
                    temp / "corrupt_restore/synthworlds_r1_results")

    def test_evaluation_release_matches_screened_sources(self):
        verified = release.validate_split_and_evaluation([])
        self.assertEqual(verified["evaluation_cases"], 80)
        self.assertEqual(verified["evaluation_translation_screen"]["screened_cohort_counts"]["forget"]["cases"], 43)

        original_read_jsonl = release.read_jsonl

        def tamper_prompts(path):
            rows = original_read_jsonl(path)
            if Path(path).resolve() == (EVAL / "prompts.jsonl").resolve():
                rows[0]["prompt"] += " changed after review"
            return rows

        with patch.object(release, "read_jsonl", side_effect=tamper_prompts):
            with self.assertRaisesRegex(ValueError, "hash-bound approved translations"):
                release.validate_split_and_evaluation([])

    def test_scoring_three_seed_gate_paired_effects_and_tamper_guard(self):
        prompts = read_jsonl(PROMPTS)
        labels = read_jsonl(LABELS)
        label_by_key = {(r["prompt_id"], r["language"]): r for r in labels}
        self.assertEqual(len(prompts), 1200)
        self.assertEqual(len(labels), 1200)

        full_models = ["sft_seed_42", "sft_seed_43", "sft_seed_44"]
        retain_models = ["retain_seed_42", "retain_seed_43", "retain_seed_44"]
        model_seeds = dict(zip(full_models + retain_models, [42, 43, 44, 42, 43, 44]))
        baseline = []
        for model in ["pretrained", *full_models, *retain_models]:
            seed = model_seeds.get(model)
            adapter_sha = None if model == "pretrained" else digest(model)
            for prompt in prompts:
                label = label_by_key[(prompt["prompt_id"], prompt["language"])]
                baseline.append({
                    "model": model, "model_id": evaluation.get_model_profile(evaluation.PILOT_PROFILE).model_id,
                    "model_profile": evaluation.get_model_profile(evaluation.PILOT_PROFILE).manifest(),
                    "model_revision": REVISION,
                    "checkpoint_sha256": checkpoint_fingerprint(adapter_sha), "adapter_sha256": adapter_sha,
                    "seed": seed, "prompt_id": prompt["prompt_id"], "source_qa_id": prompt["source_qa_id"],
                    "split": prompt["split"], "group_id": prompt["group_id"], "probe": prompt["probe"],
                    "language": prompt["language"],
                    "prompt_sha256": hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest(),
                    "answer": label["gold_answers"][0], "generation_failed": False, "generation_error": "",
                    "generation_config": {"do_sample": False},
                    "generation_config_sha256": digest("greedy-test-fixture"),
                })

        with tempfile.TemporaryDirectory(prefix="r1-v51-fixture-") as td:
            temp = Path(td)
            # Even a stale answer matching gold must not score as correct when generation failed.
            baseline[0]["generation_failed"] = True
            baseline[0]["generation_error"] = "fixture generation failure"
            baseline_predictions = temp / "baseline_predictions.jsonl"
            evaluation.write_jsonl(baseline_predictions, baseline)
            baseline_dir = temp / "baseline_scored"
            evaluation.score(Namespace(prompts=PROMPTS, labels=LABELS,
                predictions=[baseline_predictions], output_dir=baseline_dir))
            failed_row = read_jsonl(baseline_dir / "scored_predictions.jsonl")[0]
            self.assertFalse(failed_row["correct"])
            self.assertEqual(failed_row["invalid_output_reason"], "generation_failed")
            case_metrics = baseline_dir / "case_metrics.jsonl"
            target_path = temp / "targets.json"
            evaluation.baseline_gate(Namespace(case_metrics=case_metrics,
                full_sft_models=full_models, output=target_path, expected_cases=80,
                minimum_cases=15, minimum_groups=8))
            target = json.loads(target_path.read_text())
            self.assertEqual(target["baseline_gate_status"], "PASS")
            self.assertEqual(len(target["full_sft_artifacts"]), 3)
            self.assertEqual(target["expected_cases"], 80)
            evaluation.verify_baseline_target_manifest(target)

            bad_baseline = read_jsonl(case_metrics)
            next(row for row in bad_baseline if row["model"] == "sft_seed_43")["group_id"] = "wrong-group"
            bad_baseline_path = temp / "bad_baseline_metrics.jsonl"
            evaluation.write_jsonl(bad_baseline_path, bad_baseline)
            with self.assertRaisesRegex(ValueError, "disagree on split or leakage group"):
                evaluation.baseline_gate(Namespace(case_metrics=bad_baseline_path,
                    full_sft_models=full_models, output=temp / "bad_targets.json", expected_cases=80,
                    minimum_cases=15, minimum_groups=8))

            tampered = [dict(row) for row in baseline]
            tampered[0]["prompt_sha256"] = "0" * 64
            tampered_path = temp / "tampered_predictions.jsonl"
            evaluation.write_jsonl(tampered_path, tampered)
            with self.assertRaisesRegex(ValueError, "exact evaluation prompt"):
                evaluation.score(Namespace(prompts=PROMPTS, labels=LABELS,
                    predictions=[tampered_path], output_dir=temp / "tampered_scored"))

            selected_case = target["selected_source_qa_ids"][0]
            fr_effect_case = target["bilingual_precompetent_source_qa_ids_by_language"]["fr"][1]
            post_records = list(baseline)
            pairs = []
            for method in ("ga", "gd", "npo"):
                for full_model in full_models:
                    seed = model_seeds[full_model]
                    post_model = f"{method}_seed_{seed}"
                    adapter_sha = digest(post_model)
                    pairs.append({"method": method, "reference": full_model, "post": post_model})
                    for prompt in prompts:
                        label = label_by_key[(prompt["prompt_id"], prompt["language"])]
                        answer = label["gold_answers"][0]
                        if (prompt["source_qa_id"] == selected_case and prompt["language"] == "fr"
                                and prompt["probe"] == "direct_positive"):
                            answer = "fixture deliberately incorrect"
                        if (method == "ga" and prompt["source_qa_id"] == fr_effect_case
                                and prompt["language"] == "fr" and prompt["probe"] == "multi_hop"):
                            answer = "fixture target-language final answer change"
                        post_records.append({
                            "model": post_model, "model_id": evaluation.get_model_profile(evaluation.PILOT_PROFILE).model_id,
                            "model_profile": evaluation.get_model_profile(evaluation.PILOT_PROFILE).manifest(),
                            "model_revision": REVISION,
                            "checkpoint_sha256": checkpoint_fingerprint(adapter_sha), "adapter_sha256": adapter_sha,
                            "seed": seed, "prompt_id": prompt["prompt_id"], "source_qa_id": prompt["source_qa_id"],
                            "split": prompt["split"], "group_id": prompt["group_id"], "probe": prompt["probe"],
                            "language": prompt["language"],
                            "prompt_sha256": hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest(),
                            "answer": answer, "generation_failed": False, "generation_error": "",
                            "generation_config": {"do_sample": False},
                            "generation_config_sha256": digest("greedy-test-fixture"),
                        })
            pairs_path = temp / "model_pairs.json"
            pairs_path.write_text(json.dumps(pairs), encoding="utf-8")
            all_predictions = temp / "all_predictions.jsonl"
            evaluation.write_jsonl(all_predictions, post_records)
            post_dir = temp / "post_scored"
            evaluation.score(Namespace(prompts=PROMPTS, labels=LABELS,
                predictions=[all_predictions], output_dir=post_dir))
            effects_dir = temp / "effects"
            evaluation.effects(Namespace(case_metrics=post_dir / "case_metrics.jsonl",
                target_manifest=target_path, model_pairs=pairs_path,
                output_dir=effects_dir, bootstrap=100))
            summary = read_jsonl(effects_dir / "multi_seed_summary.jsonl")
            fr_direct = next(r for r in summary if r["method"] == "ga" and r["language"] == "fr"
                             and r["split"] == "forget" and r["probe"] == "direct_positive")
            self.assertEqual(fr_direct["n_seeds"], 3)
            self.assertAlmostEqual(fr_direct["mean_delta_accuracy_pp"], -100 / 43)
            self.assertIsNotNone(fr_direct["sd_delta_accuracy_across_seeds_pp"])
            fr_dap = next(r for r in summary if r["method"] == "ga" and r["language"] == "fr"
                          and r["probe"] == "downstream_answer_persistence")
            self.assertEqual(fr_dap["n_seeds_with_defined_dap"], 3)
            self.assertEqual(fr_dap["mean_dap_persistence_pct"], 100.0)
            self.assertIsNone(fr_dap["mean_delta_accuracy_pp"])
            self.assertIsNone(fr_dap["paired_entity_group_bootstrap_95ci_pp"])
            fr_did = next(r for r in summary if r["method"] == "ga" and r["language"] == "fr"
                          and r["split"] == "forget" and r["probe"] == "multi_hop_language_minus_english_DID")
            self.assertAlmostEqual(fr_did["matched_english_mean_delta_accuracy_pp"], 0.0)
            self.assertAlmostEqual(fr_did["matched_target_mean_delta_accuracy_pp"], -100 / 43)
            self.assertAlmostEqual(fr_did["mean_delta_accuracy_pp"], -100 / 43)
            self.assertEqual(fr_did["n_cases"], 43)
            self.assertEqual(fr_did["entity_groups"], 24)
            self.assertEqual(fr_did["confidence_interval_scope"], "secondary_12_cell_familywise_95_percent")
            self.assertEqual(fr_did["multiplicity_adjustment"], "Bonferroni over 3 methods x 4 target languages")
            fr_primary = next(r for r in summary if r["method"] == "ga" and r["language"] == "fr"
                              and r["split"] == "forget" and r["probe"] == "multi_hop")
            self.assertEqual(fr_primary["confidence_interval_scope"], "primary_12_cell_familywise_95_percent")
            self.assertAlmostEqual(fr_primary["bootstrap_tail_alpha"], 0.05 / 24)
            effect_manifest = json.loads((effects_dir / "effect_analysis_manifest.json").read_text())
            self.assertIn("Post-minus-pre multi-hop exact-match change", effect_manifest["primary_outcome"])
            self.assertIn("Bonferroni familywise 95%", effect_manifest["primary_confidence_intervals"])
            self.assertEqual(effect_manifest["bootstrap_replicates_minimum_for_kaggle_main_run"], 100000)
            seed_fr_did = next(r for r in read_jsonl(effects_dir / "paired_effects.jsonl")
                               if r["method"] == "ga" and r["language"] == "fr" and r["seed"] == 42
                               and r["probe"] == "multi_hop_language_minus_english_DID")
            self.assertAlmostEqual(seed_fr_did["matched_english_delta_accuracy_pp"], 0.0)
            self.assertAlmostEqual(seed_fr_did["matched_target_delta_accuracy_pp"], -100 / 43)
            self.assertEqual(len(read_jsonl(effects_dir / "paired_effects.jsonl")), 486)
            joint = read_jsonl(effects_dir / "intermediate_final_outcomes.jsonl")
            self.assertEqual(len(joint), 15)
            fr_joint = next(r for r in joint if r["method"] == "ga" and r["language"] == "fr")
            self.assertAlmostEqual(fr_joint["b_fail_c_pass_mean_pct"], 100 / 43)
            self.assertAlmostEqual(sum(fr_joint[f"{name}_mean_pct"] for name in (
                "b_fail_c_pass", "b_fail_c_fail", "b_pass_c_pass", "b_pass_c_fail")), 100.0)

            # Removing the same control case from every GA seed previously
            # allowed a smaller, apparently consistent paired case set.
            complete_metrics = read_jsonl(post_dir / "case_metrics.jsonl")
            victim = next(r for r in complete_metrics if r["model"] == "ga_seed_42" and r["split"] == "holdout" and r["language"] == "fr")
            incomplete = [r for r in complete_metrics if not (
                r["model"] in {"ga_seed_42", "ga_seed_43", "ga_seed_44"}
                and r["source_qa_id"] == victim["source_qa_id"] and r["language"] == "fr")]
            incomplete_path = temp / "incomplete_metrics.jsonl"
            evaluation.write_jsonl(incomplete_path, incomplete)
            with self.assertRaisesRegex(ValueError, "coverage differs from its baseline"):
                evaluation.effects(Namespace(case_metrics=incomplete_path, target_manifest=target_path,
                    model_pairs=pairs_path, output_dir=temp / "incomplete_effects", bootstrap=100))

            altered_metadata = [dict(row) for row in complete_metrics]
            next(r for r in altered_metadata if r["model"] == "ga_seed_42")["group_id"] = "wrong-group"
            altered_path = temp / "altered_metadata.jsonl"
            evaluation.write_jsonl(altered_path, altered_metadata)
            with self.assertRaisesRegex(ValueError, "case metadata differs"):
                evaluation.effects(Namespace(case_metrics=altered_path, target_manifest=target_path,
                    model_pairs=pairs_path, output_dir=temp / "altered_effects", bootstrap=100))

            reused_pairs = [dict(pair) for pair in pairs]
            reused_pairs[3]["post"] = reused_pairs[0]["post"]
            reused_path = temp / "reused_pairs.json"
            reused_path.write_text(json.dumps(reused_pairs), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct post-unlearning model"):
                evaluation.effects(Namespace(case_metrics=post_dir / "case_metrics.jsonl", target_manifest=target_path,
                    model_pairs=reused_path, output_dir=temp / "reused_effects", bootstrap=100))

            forged_target = dict(target)
            forged_target["selected_source_qa_ids"] = ["forged-case"]
            forged_target_path = temp / "forged_targets.json"
            forged_target_path.write_text(json.dumps(forged_target), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected_source_qa_ids.*recomputed"):
                evaluation.verify_baseline_target_manifest(forged_target)
            with self.assertRaisesRegex(ValueError, "selected_source_qa_ids.*recomputed"):
                evaluation.effects(Namespace(case_metrics=post_dir / "case_metrics.jsonl",
                    target_manifest=forged_target_path, model_pairs=pairs_path,
                    output_dir=temp / "forged_effects", bootstrap=100))
            corpus_path = temp / "corpus.jsonl"
            corpus_path.write_text("", encoding="utf-8")
            corpus_manifest_path = temp / "corpus_manifest.json"
            corpus_manifest_path.write_text("{}", encoding="utf-8")
            sft_run_path = temp / "sft_run_manifest.json"
            sft_run_path.write_text("{}", encoding="utf-8")
            unlearn_check = unlearn_lora.preflight(corpus_path, corpus_manifest_path, sft_run_path,
                forged_target_path, "ga")
            self.assertEqual(unlearn_check["status"], "BLOCKED")
            self.assertTrue(any("Baseline target manifest failed recomputation" in error
                                for error in unlearn_check["errors"]))
            self.assertTrue(any("source-integrity validation proof" in error for error in unlearn_check["errors"]))
            self.assertTrue(any("shared assistant-mask implementation" in error for error in unlearn_check["errors"]))

            short_seed_target = dict(target)
            short_seed_target["pilot_gate_thresholds"] = dict(target["pilot_gate_thresholds"])
            short_seed_target["pilot_gate_thresholds"]["common_across_full_sft_seeds"] = target["pilot_gate_thresholds"]["common_across_full_sft_seeds"][:2]
            short_seed_target_path = temp / "short_seed_targets.json"
            short_seed_target_path.write_text(json.dumps(short_seed_target), encoding="utf-8")
            short_seed_check = unlearn_lora.preflight(corpus_path, corpus_manifest_path, sft_run_path,
                short_seed_target_path, "ga")
            self.assertEqual(short_seed_check["status"], "BLOCKED")
            self.assertTrue(any("required three-seed bilingual competence gate thresholds" in error
                                for error in short_seed_check["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
