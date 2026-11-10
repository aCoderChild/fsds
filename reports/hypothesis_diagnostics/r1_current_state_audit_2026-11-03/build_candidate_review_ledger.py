#!/usr/bin/env python3
"""Create an auditable diagnostic ledger; no release decisions are made."""
import hashlib,json,subprocess,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
for name in ('verify_full_candidate_review.py','verify_nllb600_flagged_repairs.py','verify_nllb600_repair_review.py'):
 subprocess.run([sys.executable,'-B',str(HERE/name)],check=True,stdout=subprocess.DEVNULL)
first=HERE/'evidence/full_candidate_review/review.json';second=HERE/'evidence/nllb600_repair_review/review.json'
first_data=json.loads(first.read_text());second_data=json.loads(second.read_text())
first_bind=json.loads((HERE/'kaggle_full_candidate_review/candidate_bindings.json').read_text())['bindings']
second_bind=json.loads((HERE/'kaggle_nllb600_repair_review/bindings.json').read_text())['bindings']
def key(x):return (x['language'],x['kind'],x['prompt_id'])
initial={key(x):x for x in first_data['records']}; fb={key(x):x for x in first_bind}; replacements={key(x):x for x in second_data['records']};sb={key(x):x for x in second_bind}
assert len(initial)==len(fb)==799 and len(replacements)==len(sb)==100 and set(replacements)<=set(initial)
def passed(r):return r['decision'].get('semantic_pass') is True and r['decision'].get('fluency_pass') is True
rows=[];counts=Counter()
for k,record in initial.items():
 if passed(record):choice='first_screen';decision=record;binding=fb[k];review_file=first
 elif k in replacements and passed(replacements[k]):choice='adaptive_repair_screen';decision=replacements[k];binding=sb[k];review_file=second
 else:choice='unresolved';decision=record;binding=fb[k];review_file=first
 counts[choice]+=1
 rows.append({'language':k[0],'kind':k[1],'prompt_id':k[2], 'diagnostic_selection':choice,
              'candidate_sha256':binding['candidate_sha256'],'restoration_method':binding['restoration_method'],
              'review_artifact_sha256':sha(review_file.read_bytes()),
              'review_record_sha256':sha(json.dumps(decision,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()),
              'release_approved':False})
assert counts=={'first_screen':619,'adaptive_repair_screen':33,'unresolved':147}
out=HERE/'evidence/candidate_review_ledger.json'
out.write_text(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_RELEASE_APPROVAL','counts':dict(counts),'records':rows,
 'limitations':['Single-model direct screens include known errors.','Repair selection is adaptive.','Heuristic restoration is not independently validated.','No candidate has completed the frozen release review.']},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_RELEASE_APPROVAL','counts':dict(counts),'ledger_sha256':sha(out.read_bytes())},indent=2))
