#!/usr/bin/env python3
"""Package the complete, AI-only-reviewed R1 feasibility experiment for Kaggle."""
import hashlib
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
SCRIPT_FILES=("train_sft_lora.py","unlearn_lora.py","evaluate_crosslingual_unlearning.py","import_crosslingual_eval_translations.py","build_composition_sft.py","build_disjoint_composition_candidates.py","translate_disjoint_composition_nllb.py","review_disjoint_composition_translations.py","build_disjoint_composition_corpus.py")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
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
    stage=OUT.with_name(OUT.name+".tmp")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    def copy(rel):
        src=ROOT/rel
        if not src.is_file(): raise SystemExit(f"Missing required package input: {rel}")
        dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for rel in ("requirements-sft.txt","split_approval.json","translation_review_policy.json","sft_training/corpus.jsonl","sft_training/manifest.json","sft_training/composition_corpus.jsonl","sft_training/composition_manifest.json","sft_training/composition_corpus_disjoint.jsonl","sft_training/composition_disjoint_manifest.json","sft_training/disjoint_composition_candidates/english_candidate.jsonl","sft_training/disjoint_composition_candidates/manifest.json","sft_training/disjoint_composition_candidates/translations/report.json","sft_training/disjoint_composition_candidates/ai_review_report.json","evaluation_dataset/manifest.json","evaluation_dataset/README.md","evaluation_dataset/prompts/en.jsonl","evaluation_dataset/labels/gold_labels.jsonl","evaluation_dataset/translations/review_process_log.json","evaluation_dataset/translations/ai_reviewed/review_import_report.json","kaggle/sft_lora_pilot.ipynb"):
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
    readme="""# SynthWorlds-SM R1 feasibility experiment\n\nThis package contains the five-language atomic SFT corpus, the historical retained-chain composition corpus, and the new evaluation-disjoint composition corpus with 50 train-only chains (100 rows per language). The disjoint training rows were kept separate until the declared AI-only review passed. No fluent human reviewed the translations. Results are feasibility-pilot evidence, not human-validated linguistic findings.\n\nRun `kaggle/sft_lora_pilot.ipynb` from the Kaggle Notebook Editor with the `HF_TOKEN` secret attached. Notebook Secrets are available from the interactive editor session; Kaggle CLI/API `kernels push` jobs do not receive them. The notebook explicitly checks GPU, secret/model access, corpus integrity, and baseline bilingual competence. All expensive training switches default to false.\n\nThe disjoint extension excludes evaluation entities, answers, questions, and selected chain facts. Unlearning targets only English forget cases where the SFT checkpoint answered both direct and full multi-hop probes correctly; its manifest is checksum-bound to that baseline and its case metrics. Holdout is never used. Translation review is AI-only. Back-translation uses the same NLLB model family, so it is a consistency check rather than independent proof of fluency or semantic correctness.\n"""
    (stage/"R1_KAGGLE_README.md").write_text(readme,encoding="utf-8")
    files={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    manifest={"status":"R1_AI_ONLY_FEASIBILITY_PACKAGE","model":"meta-llama/Llama-3.2-1B-Instruct","translation_review_mode":"AI_ONLY","human_translation_reviewed":False,"evaluation_ready":True,"sft_corpus_sha256":sft["corpus_sha256"],"disjoint_composition_corpus_sha256":sha(disjoint_path),"evaluation_manifest_sha256":sha(EVAL/"manifest.json"),"file_count":len(files),"files_sha256":files}
    (stage/"package_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    old=OUT.with_name(OUT.name+".old")
    if old.exists(): shutil.rmtree(old)
    if OUT.exists(): OUT.rename(old)
    stage.rename(OUT)
    if old.exists(): shutil.rmtree(old)
    temp=ZIP.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file(): z.write(f,f.relative_to(OUT).as_posix())
    temp.replace(ZIP)
    upload_stage=UPLOAD.with_name(UPLOAD.name+".tmp")
    if upload_stage.exists(): shutil.rmtree(upload_stage)
    shutil.copytree(OUT,upload_stage)
    upload_old=UPLOAD.with_name(UPLOAD.name+".old")
    if upload_old.exists(): shutil.rmtree(upload_old)
    if UPLOAD.exists(): UPLOAD.rename(upload_old)
    upload_stage.rename(UPLOAD)
    if upload_old.exists(): shutil.rmtree(upload_old)
    print(json.dumps({"status":"PACKAGE_READY","folder":str(OUT),"upload_folder":str(UPLOAD),"zip":str(ZIP),"files":len(files)+1,"sha256":sha(ZIP)},indent=2))


if __name__=="__main__": main()
