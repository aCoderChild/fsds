"""Pinned model/precision profiles for the R1 pilot and main study."""
from __future__ import annotations

from dataclasses import dataclass


PILOT_PROFILE = "llama32_1b_pilot_fp16"
MAIN_PROFILE = "llama31_8b_main_qlora_nf4"

# The 8B commit was resolved from the authenticated repository metadata during
# the 2026-11-04 access preflight. The historical 1B run did not record its
# commit, so a new 1B stage must be explicitly pinned before it can start.
PINNED_MODEL_REVISIONS = {
    PILOT_PROFILE: None,
    MAIN_PROFILE: "0e9e39f249a16976918f6564b8830bc894c89659",
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    role: str
    model_id: str
    quantization: str
    required_packages: tuple[str, ...] = ()

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "model_id": self.model_id,
            "quantization": self.quantization,
            "compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4" if self.quantization == "4bit_nf4" else None,
            "bnb_4bit_use_double_quant": True if self.quantization == "4bit_nf4" else None,
            "bnb_4bit_compute_dtype": "float16" if self.quantization == "4bit_nf4" else None,
        }


PROFILES = {
    PILOT_PROFILE: ModelProfile(
        name=PILOT_PROFILE,
        role="feasibility_pilot",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        quantization="none",
    ),
    MAIN_PROFILE: ModelProfile(
        name=MAIN_PROFILE,
        role="main_study",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="4bit_nf4",
        required_packages=("bitsandbytes",),
    ),
}


# The notebook checks the downloaded config for the selected profile before
# enabling any model stage. Keep these checks separate from the precision
# profile so model-run manifests remain stable and architecture validation is
# explicit and testable.
EXPECTED_CONFIG_FIELDS = {
    PILOT_PROFILE: {
        "model_type": "llama",
        "num_hidden_layers": 16,
        "hidden_size": 2048,
    },
    MAIN_PROFILE: {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
    },
}


def get_model_profile(name: str = PILOT_PROFILE) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown R1 model profile {name!r}; choose one of {sorted(PROFILES)}") from exc


def get_pinned_model_revision(name: str = PILOT_PROFILE) -> str | None:
    try:
        return PINNED_MODEL_REVISIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown R1 model profile {name!r}; choose one of {sorted(PROFILES)}") from exc


def validate_model_config(name: str, config: dict) -> None:
    """Fail closed unless the pinned repository config matches the profile."""
    expected = EXPECTED_CONFIG_FIELDS.get(name)
    if expected is None:
        raise ValueError(f"No model-config validation is defined for {name!r}")
    if not isinstance(config, dict):
        raise ValueError("Model config must be a JSON object")
    mismatches = {
        field: {"expected": value, "actual": config.get(field)}
        for field, value in expected.items()
        if config.get(field) != value
    }
    if mismatches:
        raise ValueError(f"Model config does not match profile {name}: {mismatches}")


def model_load_options(name: str, torch, *, device_map=None) -> dict:
    """Build pinned load options without importing optional quantization packages for the pilot."""
    profile = get_model_profile(name)
    options = {"dtype": torch.float16}
    if profile.quantization == "4bit_nf4":
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if device_map is not None:
        options["device_map"] = device_map
    return options
