#!/usr/bin/env python3
"""Bind the 187 masked candidate rows to a private, independent Mistral review kernel."""
import ast,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
INPUT=PARENT/'kaggle_review/review_input.json'
TEMPLATE=HERE/'review_template.py'

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 rows=json.loads(INPUT.read_text(encoding='utf-8'))
 if len(rows)!=187 or len({r['review_key'] for r in rows})!=187:raise ValueError('expected 187 unique inputs')
 raw=json.dumps(rows,ensure_ascii=False,separators=(',',':'))
 if any(set(r)!={'review_key','language','kind','source_question_en','translated_question'} for r in rows):raise ValueError('unexpected input fields')
 if any(re.search(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]',r['source_question_en']) is None and re.search(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]',r['translated_question']) is not None for r in rows):raise ValueError('unexpected placeholders')
 text=TEMPLATE.read_text(encoding='utf-8'); tree=ast.parse(text); lines=text.splitlines(); vals={'INPUT_SHA256':f'INPUT_SHA256 = "{sha(raw.encode())}"','RAW_INPUT':'RAW_INPUT = '+repr(raw)};found=set()
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in vals:
     if n.lineno!=n.end_lineno or t.id in found:raise ValueError('unexpected template assignment')
     lines[n.lineno-1]=vals[t.id];found.add(t.id)
 if found!=set(vals):raise ValueError('template constants missing')
 script='\n'.join(lines)+'\n';ast.parse(script)
 out=HERE/'review.py';out.write_text(script,encoding='utf-8')
 (HERE/'review_input.json').write_text(raw+'\n',encoding='utf-8')
 meta={'id':'bunnoob2005/synthworlds-mistral-cross-review-2026-11-04','title':'SynthWorlds Mistral Cross Review 2026-11-04','code_file':'review.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[]}
 (HERE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 manifest={'status':'PREPARED_BLINDED_DIRECT_CROSS_REVIEW_DIAGNOSTIC_ONLY','rows':len(rows),'input_sha256':sha(raw.encode()),'review_script_sha256':sha(out.read_bytes()),'model_id':'mistralai/Mistral-7B-Instruct-v0.3','model_revision':'c170c708c41dac9275d15a8fff4eca08d52bab71','answers_or_splits_sent':False,'qwen_decisions_or_backtranslations_sent':False,'human_translation_reviewed':False,'release_approved':False,'training_eligible':False}
 (HERE/'build_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
