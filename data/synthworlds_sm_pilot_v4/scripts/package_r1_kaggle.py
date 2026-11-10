#!/usr/bin/env python3
"""Package the complete, AI-only-reviewed R1 feasibility experiment for Kaggle."""
import hashlib
import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/"evaluation_dataset"
OUT=ROOT/"kaggle"/"r1_experiment_dataset_disjoint"
ZIP=ROOT/"kaggle"/"r1_experiment_dataset_disjoint.zip"
UPLOAD=ROOT/"kaggle"/"r1_dataset_version_upload_disjoint"
DATASET_METADATA=ROOT/"kaggle"/"r1_dataset_version_upload"/"dataset-metadata.json"
SCRIPT_FILES=("train_sft_lora.py","unlearn_lora.py","evaluate_crosslingual_unlearning.py","run_audited_unlearning.py","run_audited_evaluation.py","import_crosslingual_eval_translations.py","build_composition_sft.py","build_disjoint_composition_candidates.py","build_type_b_disjoint_candidates.py","verify_type_b_disjoint_candidates.py","translate_disjoint_composition_nllb.py","review_disjoint_composition_translations.py","build_disjoint_composition_corpus.py")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out=OUT, zip_path=ZIP, upload=UPLOAD):
    em=json.loads((EVAL/"manifest.json").read_text(encoding="utf-8"))
    if em.get("evaluation_ready") is not True or em.get("ai_review_status")!="AI_ONLY_REVIEWED_COMPLETE":
        raise SystemExit("Refusing to package: four-language evaluation review/import is not complete")
    if em.get("human_review_status")!="NOT_PERFORMED":
        raise SystemExit("Refusing to package: human-review disclosure is inconsistent")
    sft=json.loads((ROOT/"sft_training/manifest.json").read_text(encoding="utf-8"))
    if sft.get("status")!="AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS" or sft.get("do_not_train") is not False:
        raise SystemExit("Refusing to package: multilingual SFT corpus is not approved for this pilot")
    disjoint_path=ROOT/"sft_training/composition_corpus_disjoint.jsonl"
    disjoint_manifest_path=ROOT/"sft_training/composition_disjoint_manifest.json"
    if not disjoint_path.is_file() or not disjoint_manifest_path.is_file():
        raise SystemExit("Refusing to package: reviewed disjoint composition corpus is missing")
    disjoint=json.loads(disjoint_manifest_path.read_text(encoding="utf-8"))
    if (disjoint.get("status")!="AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS"
            or disjoint.get("do_not_train") is not False
            or disjoint.get("corpus_sha256")!=sha(disjoint_path)):
        raise SystemExit("Refusing to package: disjoint composition corpus is unreviewed or its checksum is invalid")
    stage=out.with_name(out.name+".tmp")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    def copy(rel):
        src=ROOT/rel
        if not src.is_file(): raise SystemExit(f"Missing required package input: {rel}")
        dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for rel in ("requirements-sft.txt","split_approval.json","translation_review_policy.json","sft_training/corpus.jsonl","sft_training/manifest.json","sft_training/composition_corpus.jsonl","sft_training/composition_manifest.json","sft_training/composition_corpus_disjoint.jsonl","sft_training/composition_disjoint_manifest.json","sft_training/disjoint_composition_candidates/english_candidate.jsonl","sft_training/disjoint_composition_candidates/manifest.json","sft_training/disjoint_composition_candidates/independent_verification_report.json","sft_training/disjoint_composition_candidates/translations/report.json","sft_training/disjoint_composition_candidates/ai_review_report.json","evaluation_dataset/manifest.json","evaluation_dataset/README.md","evaluation_dataset/prompts/en.jsonl","evaluation_dataset/labels/gold_labels.jsonl","evaluation_dataset/translations/review_process_log.json","evaluation_dataset/translations/ai_reviewed/review_import_report.json","kaggle/sft_lora_pilot.ipynb"):
        copy(rel)
    for lang in ("fr","de","hi","th"):
        copy(f"sft_training/disjoint_composition_candidates/translations/{lang}.jsonl")
        copy(f"sft_training/disjoint_composition_candidates/backtranslations/{lang}.jsonl")
    if not DATASET_METADATA.is_file():
        raise SystemExit(f"Missing Kaggle dataset metadata: {DATASET_METADATA}")
    shutil.copy2(DATASET_METADATA, stage/"dataset-metadata.json")
    for name in SCRIPT_FILES: copy(f"scripts/{name}")
    for lang in ("fr","de","hi","th"):
        copy(f"evaluation_dataset/translations/ai_reviewed/{lang}.jsonl")
        copy(f"evaluation_dataset/translation_requests/{lang}.jsonl")
    readme="""# SynthWorlds-SM R1 exploratory composition follow-up\n\nThis package preserves the 129-case evaluation split and the 3,355-row five-language composition corpus: 671 SFT examples per language. The corpus adds 85 source-verified Type-B three-edge practice chains to the original 50 Type-A training-only chains. New Type-B entities, path facts, answers, and exact questions are disjoint from the frozen evaluation cases; the packaged verifier and checksums record that validation.\n\nThe v4 SFT, retain-only reference, baseline evaluation, and exploratory GA/GD/NPO runs have completed in Kaggle. The predeclared bilingual competence gate failed (FR 10 cases/6 groups, DE 9/5, HI 9/6, TH 11/7; required 15/8). The saved GD/NPO exploratory run used an earlier retain sampler that reused only 15 of 26 available English retain rows. Its outputs are preserved as historical diagnostics, not as corrected-run evidence. The updated runner cycles through every retain row before reuse.\n\nTranslation review is AI-only consistency checking with NLLB-200 and same-family back-translation, protected-text, provenance, and format checks. No fluent human reviewed these translations; these checks do not establish human fluency or semantic equivalence.\n\nUse `kaggle/sft_lora_pilot.ipynb` in the Kaggle Notebook Editor with the `HF_TOKEN` secret checked. Kaggle Notebook Secrets are available to the interactive editor session; API/CLI-launched kernels do not receive that secret. Training switches default to false, so a completed API job may only mean preflight finished. Run SFT, retain-only, and baseline evaluation first. The opt-in source-bound exploratory runner permits only the exact failed competence gate, embeds the trainer hash into each run manifest, and hashes the saved adapters. The final audited evaluator verifies those hashes and writes `evaluation_provenance.json`. The source-bound rerun is a new exploratory run; it does not retroactively add provenance to the older archived run.\n\nHoldout facts are never used for training or unlearning. Retain-composition prompts overlap training by design. All results remain one-seed, 1B feasibility diagnostics; they do not answer the hypothesis conclusively or substitute for the planned powered study.\n"""
    (stage/"R1_KAGGLE_README.md").write_text(readme,encoding="utf-8")
    files={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    manifest={"status":"R1_AI_ONLY_FEASIBILITY_PACKAGE","model":"meta-llama/Llama-3.2-1B-Instruct","translation_review_mode":"AI_ONLY","human_translation_reviewed":False,"evaluation_ready":True,"sft_corpus_sha256":sft["corpus_sha256"],"disjoint_composition_corpus_sha256":sha(disjoint_path),"evaluation_manifest_sha256":sha(EVAL/"manifest.json"),"file_count":len(files),"files_sha256":files}
    (stage/"package_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    old=out.with_name(out.name+".old")
    if old.exists(): shutil.rmtree(old)
    if out.exists(): out.rename(old)
    stage.rename(out)
    if old.exists(): shutil.rmtree(old)
    temp=zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file(): z.write(f,f.relative_to(out).as_posix())
    temp.replace(zip_path)
    upload_stage=upload.with_name(upload.name+".tmp")
    if upload_stage.exists(): shutil.rmtree(upload_stage)
    shutil.copytree(out,upload_stage)
    upload_old=upload.with_name(upload.name+".old")
    if upload_old.exists(): shutil.rmtree(upload_old)
    if upload.exists(): upload.rename(upload_old)
    upload_stage.rename(upload)
    if upload_old.exists(): shutil.rmtree(upload_old)
    print(json.dumps({"status":"PACKAGE_READY","folder":str(out),"upload_folder":str(upload),"zip":str(zip_path),"files":len(files)+1,"sha256":sha(zip_path)},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--zip", type=Path, default=ZIP)
    parser.add_argument("--upload-dir", type=Path, default=UPLOAD)
    args=parser.parse_args()
    main(args.out, args.zip, args.upload_dir)
