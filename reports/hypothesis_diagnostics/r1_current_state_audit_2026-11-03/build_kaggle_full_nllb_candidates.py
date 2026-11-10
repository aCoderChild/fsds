#!/usr/bin/env python3
"""Build an answer-blind, source-bound NLLB diagnostic for every final-review flag."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0, str(ROOT))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews

def digest(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

_, by_lang = repair.source_manifest()
selected = repair.select_flagged_rows(by_lang, checked_translation_reviews())
assert len(selected) == 799
rows, bindings = [], []
for item in selected:
    lang, row = item['language'], item['row']
    source = row['text_en']
    names = sorted({n for n in row['protected_entity_names'] if n in source}, key=lambda n: (-len(n), n))
    assert not any(c.isdigit() for c in source)
    assert len(names) <= 2 and all(not (a in b) for a in names for b in names if a != b)
    replacements = ['Zorvani', 'Kelvoria']
    assert not any(s in source for s in replacements)
    mapping = dict(zip(names, replacements))
    masked = source
    for name, surrogate in mapping.items(): masked = masked.replace(name, surrogate)
    assert all(name not in masked for name in names)
    key = {'language':lang, 'kind':row['kind'], 'prompt_id':row['prompt_id']}
    rows.append({**key, 'masked_source':masked, 'surrogate_counts':{v:source.count(k) for k,v in mapping.items()}})
    bindings.append({**key, 'source_sha256':digest(source), 'masked_sha256':digest(masked), 'names_to_surrogates':mapping})
payload = json.dumps(rows, ensure_ascii=False, separators=(',',':'))
out = HERE / 'kaggle_full_nllb_candidates'
out.mkdir(exist_ok=True)
(out / 'source_bindings.json').write_text(json.dumps({'status':'PRIVATE_LOCAL_BINDINGS_NO_ANSWERS','bindings':bindings},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
code = '''import hashlib,json,sys,subprocess
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','-q','transformers>=4.45,<4.52','sentencepiece','accelerate>=0.34,<2'],check=True)
import torch,transformers
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
assert torch.cuda.is_available()
raw=__RAW__
assert hashlib.sha256(raw.encode()).hexdigest()=='__HASH__'
rows=json.loads(raw)
model_id='facebook/nllb-200-distilled-1.3B';revision='7be3e24664b38ce1cac29b8aeed6911aa0cf0576'
codes={'fr':'fra_Latn','de':'deu_Latn','hi':'hin_Deva','th':'tha_Thai'}
model=AutoModelForSeq2SeqLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16).cuda().eval()
tok=AutoTokenizer.from_pretrained(model_id,revision=revision,src_lang='eng_Latn')
records=[];out=Path('/kaggle/working/candidates.json')
for lang in codes:
    chunk=[r for r in rows if r['language']==lang]
    for start in range(0,len(chunk),8):
        batchrows=chunk[start:start+8]
        batch=tok([r['masked_source'] for r in batchrows],return_tensors='pt',padding=True,truncation=False).to('cuda')
        with torch.inference_mode():
            ids=model.generate(**batch,forced_bos_token_id=tok.convert_tokens_to_ids(codes[lang]),max_new_tokens=160,num_beams=4,do_sample=False)
        for row,translated in zip(batchrows,tok.batch_decode(ids,skip_special_tokens=True)):
            counts=row['surrogate_counts']
            ok=bool(translated.strip()) and all(translated.count(k)==v for k,v in counts.items())
            records.append({'language':lang,'kind':row['kind'],'prompt_id':row['prompt_id'],'masked_translation':translated,'surrogate_integrity':ok})
        out.write_text(json.dumps({'status':'UNREVIEWED_DIAGNOSTIC_ONLY','input_sha256':'__HASH__','model_id':model_id,'model_revision':revision,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'records':records},ensure_ascii=False,indent=2))
        print('Generated',len(records),'/',len(rows),flush=True)
'''.replace('__RAW__',repr(payload)).replace('__HASH__',digest(payload))
(out / 'translate.py').write_text(code,encoding='utf-8')
meta={'id':'bunnoob2005/synthworlds-full-nllb-candidate-draft','title':'SynthWorlds full NLLB candidate draft','code_file':'translate.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out / 'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'selected':len(rows),'input_sha256':digest(payload),'script_sha256':digest(code),'bindings_sha256':hashlib.sha256((out/'source_bindings.json').read_bytes()).hexdigest(),'status':'UNREVIEWED_DIAGNOSTIC_ONLY'},indent=2))
