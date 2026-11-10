#!/usr/bin/env python3
"""Merge completed and partial language-specific batch files; verify input hashes first."""
import json,sys
from pathlib import Path
from build_v2 import OUT, validate_sources, die, load_jsonl, write_jsonl
from ingest_translation_batch import LANGS, refresh_manifests

def main():
    validate_sources()
    output=OUT/'merged_partial';output.mkdir(exist_ok=True)
    for lang in LANGS:
        for stem,key in (('qa-sm','instance_id'),('qa-sm-docs','qid'),('qa-sm-facts','qid'),('sft_natural','sft_id')):
            records={}
            for b in sorted((OUT/'batches').glob('batch_*')):
                f=b/f'{stem}_{lang}_batch{b.name[-4:]}.jsonl'
                if not f.exists():continue
                for r in load_jsonl(f):
                    k=r[key]
                    if k in records and stem in ('qa-sm-docs','qa-sm-facts'):
                        t=('doc_' if stem=='qa-sm-docs' else 'facts_')+lang
                        if records[k][t]!=r[t]:die(f'Conflicting canonical {stem}/{lang}/{k}')
                    records[k]=r
            if records:write_jsonl(output/f'{stem}_{lang}_partial.jsonl',[records[k] for k in sorted(records)])
    refresh_manifests()
if __name__=='__main__':
    try:main()
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
