"""Inference-only MUSE-News target baseline runner for Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REPO = "muse-bench/MUSE-news_target"
MODEL_REV = "a2f39769e9a0b98ec1cdd12f65e9962502208935"
TOKENIZER_REPO = "meta-llama/Llama-2-7b-hf"
TOKENIZER_REV = "01c7f73d771dfac7d292323805ebc428287df4f9"
DATASET_REPO = "muse-bench/MUSE-News"
DATASET_REV = "506bd5b150b92814d45e4404a82f120ab2d748bf"
EVALUATOR_REPO = "https://github.com/jaechan-repo/muse_bench.git"
EVALUATOR_COMMIT = "6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e"
PROTOCOL_SHA256 = "__PROTOCOL_SHA256__"
BOOTSTRAP_SEED = 20261109
N_BOOTSTRAP = 9999
ROOT = Path("/kaggle/working")
DATA_ROOT = ROOT / "muse_news_data"
RESULT_ROOT = ROOT / "muse_news_baseline_results"
MODEL_GPU_MAX_MEMORY = {0: "14GiB", 1: "14GiB"}
EXPECTED_ROWS = {
    "verbmem_forget": 100,
    "knowmem_forget_qa": 100,
    "knowmem_retain_qa": 100,
    "knowmem_forget_qa_icl": 10,
    "knowmem_retain_qa_icl": 10,
}
DATA_FILES = {
    "verbmem_forget": ("verbmem", "forget", "verbmem/forget-00000-of-00001.parquet"),
    "knowmem_forget_qa": ("knowmem", "forget_qa", "knowmem/forget_qa-00000-of-00001.parquet"),
    "knowmem_retain_qa": ("knowmem", "retain_qa", "knowmem/retain_qa-00000-of-00001.parquet"),
    "knowmem_forget_qa_icl": ("knowmem", "forget_qa_icl", "knowmem/forget_qa_icl-00000-of-00001.parquet"),
    "knowmem_retain_qa_icl": ("knowmem", "retain_qa_icl", "knowmem/retain_qa_icl-00000-of-00001.parquet"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question(text: str) -> str:
    return " ".join(text.casefold().split())


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_hf_token() -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError(
            "HF_TOKEN is unavailable. In this Kaggle notebook, attach the existing HF_TOKEN under Add-ons > Secrets."
        ) from exc
    if not token or not token.strip():
        raise RuntimeError("HF_TOKEN secret is empty; stop before downloading the tokenizer or model.")
    return token.strip()


def fetch_data() -> tuple[dict[str, list[dict]], dict[str, str]]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    for name, (config, split, filename) in DATA_FILES.items():
        local = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=filename,
            revision=DATASET_REV,
            local_dir=str(DATA_ROOT),
        )
        table = pq.read_table(local)
        if name == "verbmem_forget":
            required = {"prompt", "gt"}
        elif name.endswith("_icl"):
            required = {"question", "answer"}
        else:
            required = {"question", "answer"}
        if not required.issubset(table.column_names):
            raise ValueError(f"Unexpected schema for {name}: {table.column_names}")
        records = table.to_pylist()
        if len(records) != EXPECTED_ROWS[name]:
            raise ValueError(f"Unexpected row count for {name}: {len(records)}")
        for index, record in enumerate(records):
            for key in required:
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise ValueError(f"Empty/malformed {key} in {name} row {index}")
        rows[name] = records
        hashes[name] = sha256_file(Path(local))
        print(f"DATA_PASS {name}: {len(records)} rows; schema={table.column_names}; sha256={hashes[name]}")

    qa_task_names = (
        "knowmem_forget_qa",
        "knowmem_retain_qa",
        "knowmem_forget_qa_icl",
        "knowmem_retain_qa_icl",
    )
    question_sets = {
        name: [normalized_question(row["question"]) for row in rows[name]]
        for name in qa_task_names
    }
    for name, questions in question_sets.items():
        if len(questions) != len(set(questions)):
            raise ValueError(f"Duplicate normalized question within MUSE split: {name}")
    question_overlaps = {
        (left, right): len(set(question_sets[left]) & set(question_sets[right]))
        for index, left in enumerate(qa_task_names)
        for right in qa_task_names[index + 1 :]
    }
    if any(question_overlaps.values()):
        raise ValueError(f"Question leakage across MUSE QA/ICL splits: {question_overlaps}")
    forget_pairs = {(row["question"].strip(), row["answer"].strip()) for row in rows["knowmem_forget_qa"]}
    retain_pairs = {(row["question"].strip(), row["answer"].strip()) for row in rows["knowmem_retain_qa"]}
    if forget_pairs & retain_pairs:
        raise ValueError("Duplicate question-answer pair between MUSE forget and retain evaluation splits")
    return rows, hashes


def load_official_evaluator() -> str:
    repo_dir = ROOT / "muse_bench_official"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--quiet", EVALUATOR_REPO, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--quiet", "origin", EVALUATOR_COMMIT], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", "--detach", EVALUATOR_COMMIT], check=True)
    current = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    if current != EVALUATOR_COMMIT:
        raise ValueError(f"MUSE evaluator revision mismatch: {current}")
    sys.path.insert(0, str(repo_dir))
    return current


def load_official_tokenizer(hf_token: str):
    api = HfApi(token=hf_token)
    tokenizer_info = api.model_info(TOKENIZER_REPO, revision=TOKENIZER_REV, files_metadata=True)
    if tokenizer_info.sha != TOKENIZER_REV:
        raise ValueError(f"Official gated tokenizer revision mismatch: {tokenizer_info.sha}")
    local_dir = ROOT / "official_llama2_tokenizer"
    local_dir.mkdir(parents=True, exist_ok=True)
    expected_sha256 = {
        "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
        "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
        "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
    }
    for filename, expected_hash in expected_sha256.items():
        downloaded = Path(hf_hub_download(
            repo_id=TOKENIZER_REPO,
            filename=filename,
            revision=TOKENIZER_REV,
            token=hf_token,
            local_dir=str(local_dir),
        ))
        if sha256_file(downloaded) != expected_hash:
            raise ValueError(f"Official tokenizer content hash mismatch: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.vocab_size != 32000:
        raise ValueError(f"Unexpected official Llama 2 tokenizer vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def load_target_model(hf_token: str):
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly two CUDA GPUs (T4 x2); found {torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"This locked baseline expects two Tesla T4 GPUs; found {gpu_names}")
    model_info = HfApi().model_info(MODEL_REPO, revision=MODEL_REV, files_metadata=True)
    if model_info.sha != MODEL_REV:
        raise ValueError(f"MUSE target checkpoint revision mismatch: {model_info.sha}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REV,
        token=hf_token,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="auto",
        max_memory=MODEL_GPU_MAX_MEMORY,
    )
    model.eval()
    placement = getattr(model, "hf_device_map", {})
    devices = {str(value).lower() for value in placement.values()}
    invalid = {value for value in devices if value not in {"0", "1", "cuda:0", "cuda:1"}}
    if invalid:
        raise RuntimeError(f"CPU/disk/off-device placement is forbidden: {sorted(invalid)}")
    if not devices or not devices.issubset({"0", "1", "cuda:0", "cuda:1"}):
        raise RuntimeError(f"Unexpected model device placement: {placement}")
    print(f"GPU_PASS names={gpu_names}; placement={placement}; dtype=float32; no offload")
    return model, gpu_names, placement


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate(prompt: str, model, tokenizer, max_new_tokens: int) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = output_ids[:, len(input_ids[0]):]
    return tokenizer.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]


def score(reference: str, prediction: str, scorer) -> dict[str, float]:
    values = scorer.score(reference, prediction)
    return {
        "rouge1": float(values["rouge1"].fmeasure),
        "rouge1_recall": float(values["rouge1"].recall),
        "rougeL": float(values["rougeL"].fmeasure),
        "rougeL_recall": float(values["rougeL"].recall),
    }


def get_prefix_before_words_occur(text: str, words: list[str]) -> str:
    for word in words:
        text = text.split(word)[0]
    return text


def make_icl_prompt(rows: list[dict]) -> str:
    return "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in rows)


def verify_official_parity(rows, model, tokenizer, official_verbmem, official_knowmem, scorer) -> None:
    verb_rows = rows["verbmem_forget"][:2]
    official_agg, official_log = official_verbmem(
        model=model,
        tokenizer=tokenizer,
        prompts=[row["prompt"] for row in verb_rows],
        gts=[row["gt"] for row in verb_rows],
        max_new_tokens=128,
    )
    for index, (row, result) in enumerate(zip(verb_rows, official_log)):
        gt_ids = tokenizer(row["gt"], return_tensors="pt", add_special_tokens=True).input_ids[:, :128]
        gt_short = tokenizer.batch_decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        own_prediction = generate(row["prompt"], model, tokenizer, 128)
        own_scores = score(gt_short, own_prediction, scorer)
        if own_prediction != result["response"] or abs(own_scores["rougeL"] - result["rougeL"]) > 1e-12:
            raise RuntimeError(f"Official C1 parity smoke failed on row {index}")
    print(f"OFFICIAL_PARITY_PASS C1 two rows; official mean ROUGE-L={official_agg['mean_rougeL']:.8f}")

    icl = rows["knowmem_forget_qa_icl"]
    qa_rows = rows["knowmem_forget_qa"][:2]
    official_agg, official_log = official_knowmem(
        model=model,
        tokenizer=tokenizer,
        questions=[row["question"] for row in qa_rows],
        answers=[row["answer"] for row in qa_rows],
        icl_qs=[row["question"] for row in icl],
        icl_as=[row["answer"] for row in icl],
        max_new_tokens=32,
    )
    general_prompt = make_icl_prompt(icl)
    for index, (row, result) in enumerate(zip(qa_rows, official_log)):
        prompt = general_prompt + f"Question: {row['question']}\nAnswer: "
        own_prediction = get_prefix_before_words_occur(
            generate(prompt, model, tokenizer, 32), ["\n\n", "\nQuestion", "Question:"]
        )
        own_scores = score(row["answer"], own_prediction, scorer)
        if own_prediction != result["response"] or abs(own_scores["rougeL"] - result["rougeL"]) > 1e-12:
            raise RuntimeError(f"Official C2/C4 parity smoke failed on row {index}")
    print(f"OFFICIAL_PARITY_PASS C2 two rows; official mean ROUGE-L={official_agg['mean_rougeL']:.8f}")


def load_resume(path: Path, context_hash: str) -> dict[int, dict]:
    completed = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("context_sha256") != context_hash:
            raise ValueError(f"Resume context mismatch in {path.name}:{line_number}")
        index = int(row["row_index"])
        if index in completed:
            raise ValueError(f"Duplicate saved row in {path.name}: {index}")
        completed[index] = row
    return completed


def run_task(task: str, input_rows: list[dict], icl_rows: list[dict] | None, model, tokenizer, scorer, max_new_tokens: int) -> list[dict]:
    output_path = RESULT_ROOT / f"{task}.jsonl"
    context = {
        "protocol_sha256": PROTOCOL_SHA256,
        "task": task,
        "model_revision": MODEL_REV,
        "tokenizer_revision": TOKENIZER_REV,
        "dataset_revision": DATASET_REV,
        "max_new_tokens": max_new_tokens,
        "icl_sha256": canonical_hash(icl_rows) if icl_rows is not None else None,
    }
    context_hash = canonical_hash(context)
    completed = load_resume(output_path, context_hash)
    mode = "a" if output_path.exists() else "w"
    general_prompt = make_icl_prompt(icl_rows) if icl_rows is not None else ""
    with output_path.open(mode, encoding="utf-8") as stream:
        for index, row in enumerate(input_rows):
            expected_hashes = {
                "question_sha256": prompt_hash(row["question"]) if "question" in row else None,
                "reference_sha256": prompt_hash(row.get("answer", row.get("gt", ""))),
                "prompt_sha256": prompt_hash(row.get("prompt", "")) if "prompt" in row else None,
            }
            if index in completed:
                saved = completed[index]
                if any(saved.get(key) != value for key, value in expected_hashes.items()):
                    raise ValueError(f"Saved row input hash mismatch for {task} row {index}")
                continue
            if "prompt" in row:
                prompt = row["prompt"]
                gt_ids = tokenizer(row["gt"], return_tensors="pt", add_special_tokens=True).input_ids[:, :max_new_tokens]
                reference = tokenizer.batch_decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                prediction = generate(prompt, model, tokenizer, max_new_tokens)
            else:
                prompt = general_prompt + f"Question: {row['question']}\nAnswer: "
                reference = row["answer"]
                prediction = get_prefix_before_words_occur(
                    generate(prompt, model, tokenizer, max_new_tokens), ["\n\n", "\nQuestion", "Question:"]
                )
            result = {
                "context_sha256": context_hash,
                "task": task,
                "row_index": index,
                **expected_hashes,
                "prediction": prediction,
                "scores": score(reference, prediction, scorer),
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed[index] = result
            if (index + 1) % 10 == 0 or index + 1 == len(input_rows):
                print(f"PROGRESS {task}: {index + 1}/{len(input_rows)} rows")
    if set(completed) != set(range(len(input_rows))):
        raise ValueError(f"Incomplete task coverage for {task}: {len(completed)}/{len(input_rows)}")
    return [completed[index] for index in range(len(input_rows))]


def bootstrap_mean(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 100 or not np.isfinite(array).all():
        raise ValueError(f"Expected 100 finite values for row bootstrap; got n={len(array)}")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(array), size=(N_BOOTSTRAP, len(array)))
    sample_means = array[sample_indices].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975], method="linear")
    return {
        "n_rows": int(len(array)),
        "mean_percent": float(array.mean() * 100),
        "ci95_row_bootstrap_percent": [float(low * 100), float(high * 100)],
        "seed": int(seed),
        "resamples": N_BOOTSTRAP,
    }


def write_outputs(rows, data_hashes, model, gpu_names, placement, evaluator_sha, started, elapsed_seconds):
    summary = {}
    for seed_offset, task in enumerate(("verbmem_forget", "knowmem_forget", "knowmem_retain")):
        item_path = RESULT_ROOT / f"{task}.jsonl"
        item_results = [json.loads(line) for line in item_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(item_results) != 100 or len({item["row_index"] for item in item_results}) != 100:
            raise ValueError(f"Final row coverage failure for {task}")
        summary[task] = {
            "mean_rougeL": bootstrap_mean([item["scores"]["rougeL"] for item in item_results], BOOTSTRAP_SEED + seed_offset),
            "mean_rougeL_recall": bootstrap_mean([item["scores"]["rougeL_recall"] for item in item_results], BOOTSTRAP_SEED + seed_offset + 10),
        }

    model_map = getattr(model, "hf_device_map", {})
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "accelerate": __import__("accelerate").__version__,
        "scipy": __import__("scipy").__version__,
        "rouge_score": __import__("importlib.metadata", fromlist=["version"]).version("rouge-score"),
        "pyarrow": __import__("pyarrow").__version__,
        "gpus": gpu_names,
        "model_placement": model_map,
        "no_cpu_or_disk_offload": all(str(value).lower() in {"0", "1", "cuda:0", "cuda:1"} for value in model_map.values()),
        "precision": "float32",
        "max_memory_per_gpu": MODEL_GPU_MAX_MEMORY,
        "cuda_max_memory_allocated_bytes": [int(torch.cuda.max_memory_allocated(index)) for index in range(2)],
    }
    if not environment["no_cpu_or_disk_offload"]:
        raise ValueError("Model placement receipt shows CPU or disk offload")
    manifest = {
        "artifact_kind": "muse_news_target_baseline_inference_only",
        "status": "COMPLETE_BASELINE" if all(summary[task]["mean_rougeL"]["n_rows"] == 100 for task in summary) else "INCOMPLETE",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "protocol_sha256": PROTOCOL_SHA256,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REV, "license_tag": None, "weights_redistributed": False},
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REV, "access_route": "official gated repo using Kaggle HF_TOKEN", "file_sha256": {
            "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
            "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
            "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
            "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
        }},
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "file_sha256": data_hashes, "rows": EXPECTED_ROWS},
        "evaluator": {"repo": "jaechan-repo/muse_bench", "revision": evaluator_sha},
        "generation": {"do_sample": False, "precision": "float32", "verbmem_max_new_tokens": 128, "knowmem_max_new_tokens": 32},
        "bootstrap": {"method": "individual-row percentile", "resamples": N_BOOTSTRAP, "confidence_level": 0.95, "base_seed": BOOTSTRAP_SEED, "cluster_robust": False},
        "results": summary,
        "environment": environment,
        "interpretation": "Inference-only baseline. These scores are not unlearning effects. CIs resample QA rows and are not source-cluster robust.",
    }
    (RESULT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (RESULT_ROOT / "protocol.json").write_text(json.dumps(json.loads(Path("/kaggle/working/protocol.json").read_text()), indent=2) + "\n", encoding="utf-8")
    manifest["output_files_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(RESULT_ROOT.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    (RESULT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    import zipfile
    archive_path = ROOT / "muse_news_target_baseline.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(RESULT_ROOT.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Output ZIP CRC failure at {bad}")
    print(json.dumps(manifest, indent=2))
    print(f"OUTPUT_ARCHIVE {archive_path} sha256={sha256_file(archive_path)}")


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    clock = time.time()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_source = Path("/kaggle/working/protocol_source.json")
    if not protocol_source.exists():
        raise RuntimeError("Notebook setup did not stage the locked protocol at /kaggle/working/protocol_source.json")
    if hashlib.sha256(protocol_source.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Locked protocol hash mismatch")
    token = require_hf_token()
    os.environ["HF_TOKEN"] = token
    tokenizer = load_official_tokenizer(token)
    print("TOKENIZER_ACCESS_PASS; official repository and all four pinned file hashes match")
    rows, data_hashes = fetch_data()
    print("DATA_PREFLIGHT_PASS; all pinned MUSE splits and required schemas are valid")

    evaluator_sha = load_official_evaluator()
    from metrics.verbmem import eval as official_verbmem
    from metrics.knowmem import eval as official_knowmem
    scorer = rouge_scorer.RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=False,
    )
    model, gpu_names, placement = load_target_model(token)
    verify_official_parity(rows, model, tokenizer, official_verbmem, official_knowmem, scorer)

    run_task(
        "verbmem_forget", rows["verbmem_forget"], None, model, tokenizer, scorer, 128
    )
    run_task(
        "knowmem_forget", rows["knowmem_forget_qa"], rows["knowmem_forget_qa_icl"], model, tokenizer, scorer, 32
    )
    run_task(
        "knowmem_retain", rows["knowmem_retain_qa"], rows["knowmem_retain_qa_icl"], model, tokenizer, scorer, 32
    )
    write_outputs(rows, data_hashes, model, gpu_names, placement, evaluator_sha, started, time.time() - clock)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
