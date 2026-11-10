#!/usr/bin/env python3
"""Restore and validate full NLLB drafts; never approve or publish a corpus."""
import ast
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0,str(ROOT))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews, norm

def sha(b): return hashlib.sha256(b).hexdigest()
job = HERE / 'kaggle_full_nllb_candidates'
output = HERE / 'evidence/full_nllb_candidates/candidates.json'
tree = ast.parse((job/'translate.py').read_text(encoding='utf-8'))
raw = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
inputs = json.loads(raw)
bindings = json.loads((job/'source_bindings.json').read_text(encoding='utf-8'))['bindings']
data = json.loads(output.read_text(encoding='utf-8'))
assert data['status'] == 'UNREVIEWED_DIAGNOSTIC_ONLY'
assert data['model_id'] == 'facebook/nllb-200-distilled-1.3B' and data['model_revision'] == '7be3e24664b38ce1cac29b8aeed6911aa0cf0576'
assert data['input_sha256'] == sha(raw.encode())
_, by_lang = repair.source_manifest()
selected = repair.select_flagged_rows(by_lang,checked_translation_reviews())
assert len(inputs)==len(bindings)==len(selected)==len(data['records'])==799
answers = {}
training = ROOT.parent
for filename in ('atomic_reuse.jsonl','atomic_translation_requests.jsonl','composition_reuse.jsonl'):
    for line in (training/filename).read_text(encoding='utf-8').splitlines():
        row=json.loads(line); val=row.get('answer_en') or row.get('answer')
        if val:
            if row['sft_id'] in answers and answers[row['sft_id']] != val: raise ValueError('Conflicting answers')
            answers[row['sft_id']]=val
required=('nonempty','target_script_present','protected_names_preserved','protected_numbers_preserved','no_unrestored_marker')
seen=set(); counts=Counter(); records=[]
for inp,binding,item,rec in zip(inputs,bindings,selected,data['records']):
    row=item['row']; key=(item['language'],row['kind'],row['prompt_id'])
    assert key not in seen;seen.add(key)
    for obj in (inp,binding,rec): assert (obj['language'],obj['kind'],obj['prompt_id'])==key
    source=row['text_en']; assert binding['source_sha256']==sha(source.encode())
    names=sorted({n for n in row['protected_entity_names'] if n in source},key=lambda n:(-len(n),n))
    expected_mapping=dict(zip(names,['Zorvani','Kelvoria']))
    assert binding['names_to_surrogates']==expected_mapping
    masked=source
    for name, surrogate in expected_mapping.items(): masked=masked.replace(name,surrogate)
    assert inp['masked_source']==masked and binding['masked_sha256']==sha(masked.encode())
    expected_counts={v:source.count(k) for k,v in expected_mapping.items()}
    assert inp['surrogate_counts']==expected_counts
    target=rec['masked_translation']; assert isinstance(target,str)
    basic=bool(target.strip()) and all(target.count(k)==v for k,v in expected_counts.items())
    assert rec['surrogate_integrity'] is basic
    robust=basic and all(target.count(s)==expected_counts.get(s,0) for s in ('Zorvani','Kelvoria'))
    restored=target
    for name,surrogate in expected_mapping.items(): restored=restored.replace(surrogate,name)
    checks=repair.deterministic_checks(key[0],{**row,'text_translation':restored})
    answer=answers.get(key[2]); assert answer
    leak=bool(norm(answer) and norm(answer) in norm(restored))
    accepted=robust and all(checks.get(k) is True for k in required) and not leak
    counts['structural_pass' if accepted else 'structural_fail']+=1
    counts['exact_answer_mentions']+=int(leak)
    records.append({'language':key[0],'kind':key[1],'prompt_id':key[2], 'restored_translation':restored,
                    'structural_pass':accepted,'surrogate_integrity':robust,'checks':checks,'exact_answer_mention':leak})
out=output.parent/'restored_diagnostics.json'
out.write_text(json.dumps({'status':'UNREVIEWED_DIAGNOSTIC_ONLY','candidate_sha256':sha(output.read_bytes()),'counts':dict(counts),'records':records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'UNREVIEWED_DIAGNOSTIC_ONLY','counts':dict(counts),'output':str(out),
                  'limitation':'Structural checks and exact matching do not establish semantic accuracy or rule out indirect disclosure.'},indent=2))
