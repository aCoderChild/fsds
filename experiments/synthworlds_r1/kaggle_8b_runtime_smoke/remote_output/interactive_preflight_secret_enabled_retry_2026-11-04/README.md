# Kaggle 8B access preflight retry

On 2026-11-04, the interactive Kaggle notebook re-ran the no-GPU access preflight after the user enabled `HF_TOKEN`. The cell retrieved the secret and resolved repository metadata, then received HTTP 403 (`GatedRepoError`) while downloading only `config.json` from `meta-llama/Llama-3.1-8B-Instruct`.

The executed cell used the corrected local preflight branch, which does not downgrade Kaggle's installed `huggingface_hub`. No GPU was selected; no model weights or research data were downloaded; no training or hypothesis evaluation ran. The Kaggle session was stopped after the check.

This is an access diagnostic, not an 8B runtime smoke or research result. The follow-up identity-only cell called Hugging Face `whoami` without printing or saving the token and identified the token account as `bunnoob2005`. That account could resolve the repository revision but still received HTTP 403 for the config. The account identity and second result are recorded in `observation.json`.

The exact Hugging Face model page currently presents an access form. It says requesting access shares the primary Hugging Face email and username with the repository authors; it also asks for profile details and contains a checkbox accepting the Llama license and acknowledging the Meta privacy policy. The form was expanded for inspection only. No personal data was entered, no license checkbox was selected, and nothing was submitted. The account owner must complete that step and ensure access is granted for `meta-llama/Llama-3.1-8B-Instruct` before another no-GPU preflight. No GPU time should be spent until the config preflight passes.
