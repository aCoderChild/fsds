#!/usr/bin/env python3
"""Gate and launch existing Kaggle notebook 02/03 after prior stages finish.

No model artifact or password goes through GitHub. This is a one-iteration
scheduler, invoked hourly or manually in GitHub Actions. It does not ever
relaunch a FAILED notebook, and never creates new Kaggle IDs.
"""
import csv
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from prepare_kaggle import prepare, SPEC

USER = os.environ.get("KAGGLE_USERNAME","")
EXPECTED_SOURCE_SHA="679b7cc3cc953e4be5f964fc10dea468ad442595a0572cb5c53fe8823e629295"
MIN_GPU_HOURS=6

def call(args,check=True):
    p=subprocess.run(args,text=True,capture_output=True)
    if p.returncode and check:
        # Do not echo environment/auth data or URLs with embedded tokens.
        raise RuntimeError(f"Command failed with exit {p.returncode}: {args[0]} {args[1]} {args[2]}")
    return p

def status(stage):
    ref=f"{USER}/{SPEC[stage][0]}"
    result=call(["kaggle","kernels","status",ref],check=False)
    body=(result.stdout+" "+result.stderr).upper()
    if result.returncode:
        print(f"Stage {stage} status unavailable: {result.returncode}")
        return "UNKNOWN"
    for word in ("COMPLETE","ERROR","FAILED","CANCELLED","RUNNING","QUEUED","PENDING","STARTING"):
        if word in body:
            return word
    return "UNKNOWN"

def remaining_hours():
    p=call(["kaggle","quota"])
    for line in p.stdout.splitlines():
        if line.strip().startswith("GPU"):
            m=re.search(r"\b(\d+(?:\.\d+)?)h\b",line)
            # Table columns: used, remaining, total. Take second value.
            values=re.findall(r"(\d+(?:\.\d+)?)h",line)
            if len(values)>=3:
                return float(values[1])
    return None

def output(stage,names):
    """Download ONLY manifest and competence files, never large model data."""
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)
        ref=f"{USER}/{SPEC[stage][0]}"
        pat=r"(^|/)("+("|".join(re.escape(x) for x in names))+r")$"
        p=call(["kaggle","kernels","output",ref,"-p",tmp,"--file-pattern",pat],
               check=False)
        if p.returncode:
            print(f"Stage {stage}: inspection download unavailable ({p.returncode})")
            return {}
        out={}
        for n in names:
            hits=list(target.rglob(n))
            if len(hits)==1:
                if n.endswith(".json"):
                    out[n]=json.loads(hits[0].read_text(encoding="utf-8"))
                elif n.endswith(".csv"):
                    with hits[0].open(newline="",encoding="utf-8") as f:
                        out[n]=list(csv.DictReader(f))
        return out

def source_ready():
    files=output("01",["source_manifest.json","sft_sanity.csv"])
    manifest=files.get("source_manifest.json",{})
    rows=files.get("sft_sanity.csv",[])
    if manifest.get("source_sha256")!=EXPECTED_SOURCE_SHA:
        print("Stage 01 is not completed against the new 3K public source. No downstream launch.")
        return False
    if (manifest.get("forget"),manifest.get("retain"),manifest.get("holdout"))!=(512,2048,512):
        print("Stage 01 split counts do not match registered experiment.")
        return False
    slices=defaultdict(list)
    for row in rows:
        try:slices[(row["split"],row["language"],row["probe"])].append(int(row["correct"]))
        except (KeyError,ValueError):continue
    thresholds={("forget","en","direct"):.5,("forget","de","direct"):.5,
                ("forget","en","multi"):.25,("forget","de","multi"):.25}
    for key,threshold in thresholds.items():
        hits=slices[key]
        acc=sum(hits)/len(hits) if hits else None
        print("SFT_GATE",key,"n=",len(hits),"accuracy=",acc,"minimum=",threshold)
        if not hits or len(hits)<40 or acc<threshold:
            print("Competence gate failed. No unlearning will be launched.")
            return False
    print("SFT competence gate PASSED for supervised EN/DE probes.")
    return True

def unlearning_ready():
    files=output("02",["unlearning_manifest.json"])
    info=files.get("unlearning_manifest.json",{})
    valid=(info.get("source_sha256")==EXPECTED_SOURCE_SHA and
           set(info.get("methods",[]))=={"ga","gd","npo"})
    print("Stage 02 verified same-source GA/GD/NPO:",valid)
    return valid

def launch(stage):
    remaining=remaining_hours()
    print("Remaining Kaggle GPU quota:",remaining,"hours")
    if remaining is None or remaining<MIN_GPU_HOURS:
        print("Insufficient/unverifiable remaining quota; not starting GPU experiment.")
        return
    root,_=prepare(USER,stage,True)
    p=call(["kaggle","kernels","push","-p",str(root/stage),"--accelerator","NvidiaTeslaT4"])
    print(p.stdout)
    print("Submitted existing notebook stage:",stage,"to Kaggle (NOT a training-completion claim)")

def main():
    assert re.fullmatch(r"[A-Za-z0-9_-]+",USER or ""), "Missing valid KAGGLE_USERNAME secret"
    if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")):
        raise RuntimeError("Kaggle API auth unavailable")
    s1=status("01")
    print("Stage 01 status:",s1)
    if s1!="COMPLETE":
        print("No action; Notebook 01 has not finished successfully.")
        return
    if not source_ready():
        print("Stopping; source completeness or competence not established.")
        return
    s2=status("02")
    print("Stage 02 status:",s2)
    if s2 in ("RUNNING","QUEUED","PENDING","STARTING"):
        return
    if s2 in ("ERROR","FAILED","CANCELLED"):
        print("Stage 02 failed; manual debugging required. No automatic retry.")
        return
    if s2!="COMPLETE":
        launch("02")
        return
    if not unlearning_ready():
        # Kaggle no-run notebooks may report COMPLETE with no usable outputs.
        # A mismatching manifest is a hard stop to avoid overwriting past results.
        print("Stage 02 has COMPLETE status but missing/stale manifest; manual review required.")
        return
    s3=status("03")
    print("Stage 03 status:",s3)
    if s3 in ("RUNNING","QUEUED","PENDING","STARTING","COMPLETE","ERROR","FAILED","CANCELLED"):
        print("No new evaluation run; Stage 03 already has a status requiring review.")
        return
    launch("03")

if __name__=="__main__":
    main()
