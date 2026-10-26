#!/usr/bin/env python3
"""CPU-only validation: exactly same source loader as running Kaggle notebook.

Real MQuAKE records (official, published) are downloaded and split; artificial
strings are supplied ONLY as translation stubs for shape/testing. Stub values
are NEVER saved to Kaggle, W&B or benchmark outputs. Training translation runs
NLLB on actual original questions.
"""
import json, random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NBS=["01_mquake_t_source_translate_sft.ipynb",
     "02_mquake_t_unlearning_baselines.ipynb",
     "03_mquake_t_crosslingual_evaluation.ipynb"]
files=[ROOT/"notebooks"/n for n in NBS]
nb=[json.loads(p.read_text(encoding="utf-8")) for p in files]
for p,book in zip(files,nb):
    assert book.get("nbformat")==4
    assert all("source" in c for c in book["cells"])
    print("VALID_NOTEBOOK",p.name,"cells",len(book["cells"]))
loader=["".join(c.get("source",[])) for c in nb[0]["cells"]
        if "def build_payload(" in "".join(c.get("source",[]))]
assert len(loader)==1
ns={"__name__":"__main__"}
exec(compile(loader[0],"nb01:source", "exec"),ns)
raw,sha=ns["load_source"]()
cases,reasons=ns["build_cases"](raw,512,2048,512,seed=ns["SEED"])
assert len(cases)==3072
jobs=ns["get_translation_jobs"](cases)
langs=set(ns["LANGUAGES"])
assert langs=={"en","de","ru","zh","tr","vi"}
assert set(jobs)==langs-{"en"}
# Shape-validation stubs, not new benchmark QA or factual training data.
stubs={lang:{q:q+" [TRANSLATION_TEST_ONLY:"+lang+"]" for q in queries}
       for lang,queries in jobs.items()}
pack=ns["build_payload"](cases,stubs,sha,reasons)
assert pack["schema_version"]==4
assert set(pack["evaluation_languages"])==langs
assert pack["train_languages"]==("en","de")
assert len(pack["forget_train"])==512
assert all(x["lang"]=="en" and x["relation"]=="target"
           for x in pack["forget_train"])
assert all(x["lang"] in ("en","de") for x in pack["train"])
assert all(x["split"]!="holdout" for x in pack["train"])
assert all(r["multi_question_"+lang] for r in cases for lang in langs)
assert all(r["target_question_"+lang] for r in cases for lang in langs)
assert all(r["support_steps"] and r["depth"]>=2 for r in cases)
assert Counter(r["split"] for r in cases)=={"forget":512,"retain":2048,"holdout":512}
assert all("new_answer" not in r for r in cases)
print("DATA_VALIDATION",len(raw),"PUBLIC",len(cases),"SELECTED")
print("SOURCE_SHA",sha)
print("DEPTH_COUNTS",dict(Counter(r["depth"] for r in cases)))
print("TRANSLATION_JOBS", {k:len(v) for k,v in jobs.items()})
print("SFT_ROWS",len(pack["train"]),"FORGET_TARGETS",len(pack["forget_train"]),
      "RETAIN_REFERENCE_ROWS",len(pack["retain_reference"]))
print("PASS: identical source facts across six languages; EN+DE train, EN-only forget")
