#!/usr/bin/env python3
"""Submit Kaggle notebooks; uploads only unless --execute explicitly supplied."""
import argparse
import os
import subprocess
from prepare_kaggle import prepare, SPEC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["01", "02", "03", "all"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME"))
    args = parser.parse_args()
    if not args.username:
        raise SystemExit("Set secret KAGGLE_USERNAME in GitHub Actions.")
    if not os.environ.get("KAGGLE_KEY") and not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Set KAGGLE_API_TOKEN (preferred) or the legacy KAGGLE_USERNAME/KAGGLE_KEY pair.")
    if (not os.environ.get("KAGGLE_API_TOKEN") and
            os.environ.get("KAGGLE_KEY", "").startswith("KGAT_")):
        raise SystemExit("New-style token detected in KAGGLE_KEY: use KAGGLE_API_TOKEN instead.")
    root, stages = prepare(args.username, args.stage, args.execute)
    for stage in stages:
        slug = SPEC[stage][0]
        cmd = ["kaggle", "kernels", "push", "-p", str(root / stage)]
        if not args.execute:
            cmd += ["--no-run"]
        else:
            cmd += ["--accelerator", "NvidiaTeslaT4"]
        print("Submitting private Kaggle notebook:", f"{args.username}/{slug}",
              "(RUN)" if args.execute else "(UPLOAD ONLY)", flush=True)
        subprocess.run(cmd, check=True)
    print("Submission complete. Check Kaggle notebook status and logs.")

if __name__ == "__main__":
    main()
