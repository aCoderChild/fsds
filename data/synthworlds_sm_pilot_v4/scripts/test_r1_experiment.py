import importlib.util
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R1ExperimentTests(unittest.TestCase):
    def test_source_bound_trainer_accepts_exact_upstream_and_already_patched_inputs(self):
        runner = load("run_audited_unlearning")
        patched_source = (ROOT / "scripts" / "unlearn_lora.py").read_text(encoding="utf-8")
        upstream_source = patched_source
        for old, new in reversed(runner.PATCHES):
            self.assertEqual(upstream_source.count(new), 1)
            upstream_source = upstream_source.replace(new, old, 1)
        self.assertEqual(hashlib.sha256(upstream_source.encode("utf-8")).hexdigest(), runner.UPSTREAM_SHA256)

        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            upstream = temp_root / "upstream.py"
            upstream.write_text(upstream_source, encoding="utf-8")
            from_upstream = runner.build_source_bound_trainer(upstream, temp_root / "built_from_upstream.py")
            self.assertEqual(hashlib.sha256(from_upstream.read_bytes()).hexdigest(), runner.PATCHED_SHA256)

            patched = ROOT / "scripts" / "unlearn_lora.py"
            from_patched = runner.build_source_bound_trainer(patched, temp_root / "copied_patched.py")
            self.assertEqual(from_patched.read_bytes(), patched.read_bytes())

            altered = temp_root / "altered.py"
            altered.write_text(patched_source + "\n# altered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum is unexpected"):
                runner.build_source_bound_trainer(altered, temp_root / "refused.py")

    def test_retain_regularizer_cycles_through_all_rows_before_reusing(self):
        module = load("unlearn_lora")
        schedule = module.retain_schedule_indices(num_steps=45, num_retain_rows=26)
        self.assertEqual(len(schedule), 45)
        self.assertEqual(set(schedule), set(range(26)))
        counts = [schedule.count(index) for index in range(26)]
        self.assertEqual(max(counts) - min(counts), 1)
        self.assertEqual(module.retain_schedule_indices(num_steps=0, num_retain_rows=0), [])
        with self.assertRaises(ValueError):
            module.retain_schedule_indices(num_steps=1, num_retain_rows=0)

    def test_exploratory_override_only_allows_the_exact_gate_failure(self):
        module = load("unlearn_lora")
        check = {
            "status": "BLOCKED",
            "errors": sorted(module.GATE_FAILURE_ERRORS),
            "baseline_gate_status": "FAIL",
            "english_forget_rows": 15,
        }
        self.assertTrue(module.allows_failed_gate_exploration(check, True))
        self.assertFalse(module.allows_failed_gate_exploration(check, False))
        with_extra_blocker = {**check, "errors": check["errors"] + ["No Hugging Face token found"]}
        self.assertFalse(module.allows_failed_gate_exploration(with_extra_blocker, True))
        no_targets = {**check, "english_forget_rows": 0}
        self.assertFalse(module.allows_failed_gate_exploration(no_targets, True))

    def make_target_manifest(self, module, corpus, run_path, run, root, target_count=76):
        """Create an intact fixture manifest satisfying the current baseline gate."""
        rows = module.first_hop_unlearning_rows(corpus)
        forget_ids = sorted({r["source_qa_id"] for r in rows if r["split"] == "forget"})
        selected = forget_ids[:target_count]
        case_metrics = root / "baseline_case_metrics.jsonl"
        case_metrics.write_text("fixture baseline metrics\n", encoding="utf-8")
        target_manifest = {
            "status": "SFT_PRECOMPETENT_ENGLISH_FORGET_TARGETS",
            "language": "en",
            "split": "forget",
            "selection_rule": "SFT direct and full multi-hop answers both correct in English",
            "baseline_gate_status": "PASS",
            "pilot_gate_thresholds": {
                "minimum_matched_cases_per_language": 15,
                "minimum_entity_groups_per_language": 8,
            },
            "bilingual_counts_by_language": {
                lang: {"bilingual_precompetent_cases": 15, "entity_groups": 8}
                for lang in ("fr", "de", "hi", "th")
            },
            "case_metrics_path": str(case_metrics),
            "case_metrics_sha256": hashlib.sha256(case_metrics.read_bytes()).hexdigest(),
            "selected_source_qa_ids": selected,
            "source_corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "sft_run_manifest_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            "sft_adapter_dir": run["adapter_dir"],
        }
        path = root / "baseline_target_ids.json"
        path.write_text(json.dumps(target_manifest), encoding="utf-8")
        return path

    def test_translation_importer_keeps_question_only_names(self):
        module = load("import_crosslingual_eval_translations")
        self.assertEqual(module.names_in_source("Where was Hesthor Titus born?"), ["Hesthor Titus"])
        self.assertEqual(module.names_in_source("Who founded The Starlight Gala?"), ["Starlight Gala"])
        self.assertIsNotNone(module.PLACEHOLDER.search("Wer ist ENTITY_A?"))

    def test_exact_match_does_not_accept_wrong_relation_answer(self):
        module = load("evaluate_crosslingual_unlearning")
        self.assertTrue(module.exact_answer("Braxton Ellsworth.", "Braxton Ellsworth"))
        self.assertFalse(module.exact_answer("Hesper Braxton Quarven", "Braxton Ellsworth"))
        self.assertFalse(module.exact_answer("Not Target B", "Target B"))

    def test_invalid_output_diagnostic_flags_repetition_but_not_two_part_name(self):
        module = load("evaluate_crosslingual_unlearning")
        self.assertEqual(module.invalid_output_reason("Eld Eld Eld Eld Eld"), "repetitive_ngram")
        self.assertEqual(module.invalid_output_reason("<|assistant|> Final C"), "template_artifact")
        self.assertEqual(module.invalid_output_reason("Baden Baden"), None)
        self.assertEqual(module.invalid_output_reason("The city of the region is the center of the region."), None)
        self.assertEqual(module.invalid_output_reason("2000-01-01"), None)
        self.assertEqual(module.invalid_output_reason("$1,000,000"), None)
        self.assertEqual(module.invalid_output_reason("", generation_failed=True), "generation_failed")

    def test_scoring_reports_dap_and_precompetent_paired_delta(self):
        module = load("evaluate_crosslingual_unlearning")
        labels = {"case-1": {"split":"forget", "bootstrap_cluster_id":"cluster-1",
                              "direct_intermediate_answer":"Target B", "final_answers":["Final C"]}}
        predictions = []
        for lang in module.LANGS:
            for model in ("sft", "npo"):
                predictions.extend([
                    {"model":model,"instance_id":"case-1","probe":"direct","split":"forget","language":lang,
                     "answer":"Target B" if model=="sft" else "Not Target","generation_failed":False},
                    {"model":model,"instance_id":"case-1","probe":"multi","split":"forget","language":lang,
                     "answer":"Not Final" if model=="npo" and lang=="fr" else "Final C","generation_failed":False},
                ])
        with tempfile.TemporaryDirectory() as temp:
            pred = Path(temp)/"predictions.jsonl"
            pred.write_text("".join(json.dumps(p)+"\n" for p in predictions), encoding="utf-8")
            module.score(pred, labels, temp, bootstrap=100)
            summaries = json.loads((Path(temp)/"summary.json").read_text())
            contrasts = json.loads((Path(temp)/"cross_language_contrasts.json").read_text())
        npo = next(r for r in summaries if r["model"]=="npo")
        self.assertEqual(npo["precompetent_n"],1)
        self.assertEqual(npo["dap"],1.0)
        self.assertEqual(npo["dap_denominator"],1)
        self.assertEqual(npo["eligible_multi_delta_vs_sft_pp"],0.0)
        self.assertEqual(npo["eligible_multi_delta_95ci_pp"],[0.0,0.0])
        fr = next(r for r in contrasts if r["model"]=="npo" and r["language"]=="fr" and r["split"]=="forget")
        self.assertEqual(fr["n_bilingual_precompetent_cases"],1)
        self.assertEqual(fr["language_minus_english_did_pp"],-100.0)
        self.assertEqual(fr["cluster_bootstrap_95ci_pp"],[-100.0,-100.0])

    def test_score_separates_invalid_format_from_wrong_answer(self):
        module = load("evaluate_crosslingual_unlearning")
        labels = {"case-1": {"split":"forget", "bootstrap_cluster_id":"cluster-1",
                              "direct_intermediate_answer":"Target B", "final_answers":["Final C"]}}
        predictions = []
        for lang in module.LANGS:
            predictions.extend([
                {"model":"sft","instance_id":"case-1","probe":"direct","split":"forget","language":lang,"answer":"Target B"},
                {"model":"sft","instance_id":"case-1","probe":"multi","split":"forget","language":lang,"answer":"Final C Final C Final C"},
            ])
        with tempfile.TemporaryDirectory() as temp:
            pred = Path(temp)/"predictions.jsonl"
            pred.write_text("".join(json.dumps(p)+"\n" for p in predictions), encoding="utf-8")
            module.score(pred, labels, temp, bootstrap=10)
            scored = [json.loads(line) for line in (Path(temp)/"scored_predictions.jsonl").read_text().splitlines()]
            summaries = json.loads((Path(temp)/"summary.json").read_text())
            report = (Path(temp)/"diagnostic_report.md").read_text()
        self.assertEqual(sum(r["output_invalid"] for r in scored), 5)
        self.assertFalse(any(r["correct"] is False and r["output_invalid"] is False for r in scored if r["probe"] == "multi"))
        en = next(r for r in summaries if r["language"] == "en")
        self.assertEqual(en["invalid_generation_n"], 1)
        self.assertIn("repetitive_ngram", report)

    def test_scoring_rejects_incomplete_checkpoint_coverage(self):
        module = load("evaluate_crosslingual_unlearning")
        labels = {"case-1": {"split":"forget", "bootstrap_cluster_id":"cluster-1",
                              "direct_intermediate_answer":"Target B", "final_answers":["Final C"]}}
        predictions = [
            {"model":"sft","instance_id":"case-1","probe":probe,"split":"forget","language":"en","answer":answer}
            for probe,answer in (("direct","Target B"),("multi","Final C"))
        ]
        with tempfile.TemporaryDirectory() as temp:
            pred = Path(temp)/"predictions.jsonl"
            pred.write_text("".join(json.dumps(p)+"\n" for p in predictions),encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"Incomplete evaluation predictions"):
                module.score(pred,labels,temp,bootstrap=10)

    def test_ai_translation_import_is_complete_and_answer_labels_are_not_needed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); requests=root/"requests"; reviewed=root/"reviewed"; manifest_path=root/"manifest.json"
            requests.mkdir();reviewed.mkdir()
            manifest={"translation_drafts":{},"evaluation_ready":False}
            manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
            for lang in ("fr","de","hi","th"):
                req=[{"translation_id":f"case-{i}:multi:{lang}","source_question_en":s,"nllb_draft":"draft"} for i,s in enumerate(("Who founded the Starlight Gala?","Where was Hesthor Titus born?"))]
                out=[{"translation_id":r["translation_id"],"question":q,"review_status":"AI_REVIEWED"} for r,q in zip(req,("Qui a fondé Starlight Gala ?","Wo wurde Hesthor Titus geboren?"))]
                (requests/f"{lang}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in req),encoding="utf-8")
                (reviewed/f"{lang}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out),encoding="utf-8")
            result=subprocess.run([sys.executable,str(ROOT/"scripts/import_crosslingual_eval_translations.py"),"--requests",str(requests),"--reviewed",str(reviewed),"--output",str(root/"ai_reviewed"),"--manifest",str(manifest_path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            updated=json.loads(manifest_path.read_text())
            self.assertTrue(updated["evaluation_ready"])
            self.assertEqual(updated["human_review_status"],"NOT_PERFORMED")
            imported=json.loads((root/"ai_reviewed/fr.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(imported),{"translation_id","question","review_status"})

    def test_holdout_never_allowed_for_sft_runner(self):
        module = load("train_sft_lora")
        result = module.preflight(ROOT/"sft_training/corpus.jsonl", ROOT/"sft_training/manifest.json", Path("/tmp/no-run"), train=False, include_splits=("holdout",))
        self.assertTrue(any("holdout is forbidden" in e for e in result["errors"]))

    def test_composition_builder_uses_only_retained_chains_and_all_languages(self):
        module = load("build_composition_sft")
        train_module = load("train_sft_lora")
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            (temp_root/"sft_training").mkdir()
            shutil.copy2(ROOT/"split_approval.json", temp_root/"split_approval.json")
            corpus = temp_root/"sft_training"/"composition_corpus.jsonl"
            manifest = temp_root/"sft_training"/"composition_manifest.json"
            result = module.build(ROOT, corpus, manifest)
            rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
            comp = [r for r in rows if r.get("example_type") == "retained_multihop_composition"]
            counts = {lang: sum(r.get("language") == lang for r in rows) for lang in module.LANGS}
            self.assertEqual(result["composition_rows"], 130)
            self.assertEqual(counts, {lang: 231 for lang in module.LANGS})
            self.assertEqual({r["split"] for r in comp}, {"retain"})
            self.assertEqual({r["language"] for r in comp}, set(module.LANGS))
            self.assertEqual({r["source_qa_id"] for r in comp}, {r["instance_id"] for r in module.read_jsonl(ROOT/"evaluation_dataset/labels/gold_labels.jsonl") if r["split"] == "retain"})
            report = train_module.preflight(corpus, manifest, Path(temp)/"unused", train=False)
            self.assertEqual(report["status"], "READY_TO_TRAIN", report["errors"])
            self.assertFalse(any(r.get("split") == "holdout" for r in rows))

    def test_unlearning_selects_only_english_first_hop_from_atomic_and_composition_corpora(self):
        module = load("unlearn_lora")
        for source_name, run_variant in (("corpus.jsonl", "full_sft"), ("composition_corpus.jsonl", "composition_sft")):
            with self.subTest(corpus=source_name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                training = root / "sft_training"
                training.mkdir()
                corpus = training / source_name
                manifest = training / ("composition_manifest.json" if "composition" in source_name else "manifest.json")
                shutil.copy2(ROOT / "sft_training" / source_name, corpus)
                shutil.copy2(ROOT / "sft_training" / manifest.name, manifest)
                shutil.copy2(ROOT / "split_approval.json", root / "split_approval.json")
                adapter = root / "adapter"
                adapter.mkdir()
                (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
                sft_run = root / "run_manifest.json"
                sft_run.write_text(json.dumps({
                    "status": "TRAINING_COMPLETE", "training_started": True,
                    "model_id": module.MODEL_ID, "training_variant": run_variant,
                    "adapter_dir": str(adapter),
                    "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                    "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                }), encoding="utf-8")
                target_ids = self.make_target_manifest(module, corpus, sft_run, json.loads(sft_run.read_text()), root)
                with patch.dict(os.environ, {"HF_TOKEN": "fixture-token"}), patch.object(module.importlib.util, "find_spec", return_value=object()):
                    result = module.preflight(corpus, manifest, sft_run, target_ids, "ga")
                self.assertEqual(result["status"], "READY_TO_UNLEARN", result["errors"])
                self.assertEqual(result["rows_by_split"], {"forget": 76, "retain": 26, "holdout": 0})
                self.assertEqual(result["first_hop_training_rows"], 102)
                self.assertEqual(result["holdout_rows_used"], 0)
                selected = module.first_hop_unlearning_rows(corpus)
                self.assertEqual(len(selected), 102)
                self.assertTrue(all(r["language"] == "en" and r["hop"] == "r1" for r in selected))
                self.assertTrue(all(r["split"] in {"forget", "retain"} for r in selected))

    def test_unlearning_preflight_rejects_mismatched_sft_corpus_manifest_and_split_approval(self):
        module = load("unlearn_lora")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = root / "sft_training"
            training.mkdir()
            corpus = training / "corpus.jsonl"
            manifest = training / "manifest.json"
            shutil.copy2(ROOT / "sft_training/corpus.jsonl", corpus)
            shutil.copy2(ROOT / "sft_training/manifest.json", manifest)
            shutil.copy2(ROOT / "split_approval.json", root / "split_approval.json")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            run = root / "run_manifest.json"
            run.write_text(json.dumps({
                "status": "TRAINING_COMPLETE", "training_started": True,
                "model_id": module.MODEL_ID, "training_variant": "composition_sft",
                "adapter_dir": str(adapter), "corpus_sha256": "wrong",
                "corpus_manifest_sha256": "wrong",
            }), encoding="utf-8")
            (root / "split_approval.json").write_text("{}", encoding="utf-8")
            target_ids = self.make_target_manifest(module, corpus, run, json.loads(run.read_text()), root)
            with patch.dict(os.environ, {"HF_TOKEN": "fixture-token"}), patch.object(module.importlib.util, "find_spec", return_value=object()):
                result = module.preflight(corpus, manifest, run, target_ids, "npo")
            self.assertEqual(result["status"], "BLOCKED")
            errors = " ".join(result["errors"])
            self.assertIn("exact corpus checksum", errors)
            self.assertIn("exact corpus manifest", errors)
            self.assertIn("pilot split approval", errors)

    def test_direct_train_call_cannot_bypass_failed_competence_gate(self):
        module = load("unlearn_lora")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = root / "sft_training"
            training.mkdir()
            corpus = training / "composition_corpus_disjoint.jsonl"
            manifest = training / "composition_disjoint_manifest.json"
            shutil.copy2(ROOT / "sft_training" / corpus.name, corpus)
            shutil.copy2(ROOT / "sft_training" / manifest.name, manifest)
            shutil.copy2(ROOT / "split_approval.json", root / "split_approval.json")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            sft_run = root / "run_manifest.json"
            sft_run.write_text(json.dumps({
                "status": "TRAINING_COMPLETE", "training_started": True,
                "model_id": module.MODEL_ID, "training_variant": "composition_sft",
                "adapter_dir": str(adapter),
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            target_path = self.make_target_manifest(module, corpus, sft_run,
                                                    json.loads(sft_run.read_text()), root, target_count=15)
            target = json.loads(target_path.read_text())
            target["baseline_gate_status"] = "FAIL"
            target["bilingual_counts_by_language"] = {
                lang: {"bilingual_precompetent_cases": 9, "entity_groups": 5}
                for lang in ("fr", "de", "hi", "th")
            }
            target_path.write_text(json.dumps(target), encoding="utf-8")
            with patch.dict(os.environ, {"HF_TOKEN": "fixture-token"}), \
                    patch.object(module.importlib.util, "find_spec", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "did not authorize training"):
                    module.train(corpus, manifest, sft_run, target_path, root / "out", "ga",
                                 epochs=1, lr=1e-5, retain_weight=1.0, beta=0.1,
                                 max_length=512, seed=42)


if __name__ == "__main__":
    unittest.main()
