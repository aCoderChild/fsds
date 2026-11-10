#!/usr/bin/env python3
"""Exercise the Kaggle access preflight with mocked secret/HF clients."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts/kaggle_8b_access_preflight.py"
PRIVATE_TOKEN = "hf_fake_not_a_real_credential"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def load_script():
    spec = importlib.util.spec_from_file_location("r1_access_preflight_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def run_case(
    *,
    secret_error: bool = False,
    config: dict | None = None,
    download_error: Exception | None = None,
):
    module = load_script()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        output = root / "receipt.json"
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(config or {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096}),
            encoding="utf-8",
        )

        class Secrets:
            def get_secret(self, name):
                assert name == "HF_TOKEN"
                if secret_error:
                    raise RuntimeError(f"secret service failed with {PRIVATE_TOKEN}")
                return PRIVATE_TOKEN

        class Api:
            def __init__(self, token):
                assert token == PRIVATE_TOKEN

            def model_info(self, repo_id, revision, token):
                assert repo_id == module.MODEL_ID and token == PRIVATE_TOKEN
                assert revision == module.PINNED_REVISION
                return types.SimpleNamespace(sha=REVISION)

        def download(*, repo_id, filename, revision, token):
            assert repo_id == module.MODEL_ID and filename == "config.json"
            assert revision == REVISION and token == PRIVATE_TOKEN
            if download_error is not None:
                raise download_error
            return str(config_path)

        sys.modules["kaggle_secrets"] = types.SimpleNamespace(UserSecretsClient=Secrets)
        sys.modules["huggingface_hub"] = types.SimpleNamespace(HfApi=Api, hf_hub_download=download)
        previous = os.environ.get("R1_ACCESS_PREFLIGHT_OUTPUT")
        os.environ["R1_ACCESS_PREFLIGHT_OUTPUT"] = str(output)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                if secret_error or download_error or config and config.get("num_hidden_layers") != 32:
                    try:
                        module.main()
                    except Exception:
                        pass
                else:
                    assert module.main() == 0
        finally:
            if previous is None:
                os.environ.pop("R1_ACCESS_PREFLIGHT_OUTPUT", None)
            else:
                os.environ["R1_ACCESS_PREFLIGHT_OUTPUT"] = previous
            sys.modules.pop("kaggle_secrets", None)
            sys.modules.pop("huggingface_hub", None)
        receipt = json.loads(output.read_text(encoding="utf-8"))
        assert PRIVATE_TOKEN not in stdout.getvalue()
        assert PRIVATE_TOKEN not in output.read_text(encoding="utf-8")
        return receipt


def main() -> None:
    passed = run_case()
    assert passed["status"] == "PASS_HF_ACCESS_PREFLIGHT"
    assert passed["weights_downloaded"] is False and passed["training_started"] is False

    secret_failure = run_case(secret_error=True)
    assert secret_failure["status"] == "FAIL_HF_ACCESS_PREFLIGHT"
    assert secret_failure["failed_stage"] == "retrieve_kaggle_secret"
    assert secret_failure["weights_downloaded"] is False

    architecture_failure = run_case(config={"model_type": "llama", "num_hidden_layers": 16, "hidden_size": 4096})
    assert architecture_failure["status"] == "FAIL_HF_ACCESS_PREFLIGHT"
    assert architecture_failure["failed_stage"] == "download_and_validate_small_config"

    gated_failure = run_case(download_error=RuntimeError(f"gated repository denied: {PRIVATE_TOKEN}"))
    assert gated_failure["status"] == "FAIL_HF_ACCESS_PREFLIGHT"
    assert gated_failure["failed_stage"] == "download_and_validate_small_config"
    assert gated_failure["error_type"] == "RuntimeError"
    assert gated_failure["weights_downloaded"] is False
    assert gated_failure["research_data_loaded"] is False
    assert gated_failure["training_started"] is False

    print("PASS: mocked HF access preflight success, secret failure, architecture mismatch, and no credential leakage")


if __name__ == "__main__":
    main()
