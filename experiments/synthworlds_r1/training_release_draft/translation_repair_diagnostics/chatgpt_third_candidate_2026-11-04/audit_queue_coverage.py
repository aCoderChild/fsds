#!/usr/bin/env python3
"""Independently verify that the 187 third-pass rows have no exact-row pass in prior routes."""
import collections, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
WORK=ROOT/'chatgpt_fullflagged_queue_2026-11-04'
TG=ROOT/'translategemma4b_qwen7b_all_flagged_2026-11-03'

def read(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

queue=read(HERE/'requests.jsonl')
candidates=read(HERE/'proposed_translations.jsonl')
prior_chat=read(WORK/'remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl')
fullmap=read(WORK/'independent_review_inputs_2026-11-04/full799.private_map.jsonl')
tg=read(TG/'repair_candidates.jsonl')
tg_complement=read(WORK/'tg_complement_verified_reviews_v1.jsonl')
q_by_key={r['source_review_key']:r for r in queue}
if len(q_by_key)!=187 or len(candidates)!=187: raise ValueError('queue/candidate row count is not 187')
if len({r['source_review_key'] for r in candidates})!=187: raise ValueError('duplicate candidate keys')
chat_by_key={r['review_key']:r for r in prior_chat}
map_by_key={r['review_key']:r for r in fullmap}
for row in queue:
    key=row['source_review_key']
    old=chat_by_key.get(key)
    if old is None: raise ValueError(f'{key}: not present in source-bound full review')
    if old.get('review_status')=='AI_ONLY_REPAIR_REVIEW_PASS': raise ValueError(f'{key}: existing ChatGPT candidate pass omitted')
    m=map_by_key[key]
    tg_row=next((r for r in tg if (r['language'],r['prompt_id'],r['kind'])==(row['language'],m['prompt_id'],row['kind'])),None)
    if tg_row and tg_row.get('deterministic_pass') is True and tg_row.get('qwen_pass') is True:
        raise ValueError(f'{key}: existing TranslateGemma candidate pass omitted')
    if any((r['language'],r['prompt_id'],r['kind'])==(row['language'],m['prompt_id'],row['kind']) and r['review_status']=='AI_ONLY_REPAIR_REVIEW_PASS' for r in tg_complement):
        raise ValueError(f'{key}: prior complement-screen pass omitted')
# Report identical input/candidate payloads. The current batch intentionally retains every row identity.
by_candidate={r['source_review_key']:r for r in candidates}
payloads=[(r['language'],r['kind'],sha(r['source_question_en']),by_candidate[r['source_review_key']]['translated_question']) for r in queue]
payload_counts=collections.Counter(payloads)
report={
 'status':'PASS_EXACT_ROW_CANDIDATE_ROUTE_COVERAGE',
 'rows':len(queue),
 'exact_row_prior_route_passes':0,
 'prior_routes_checked':['full799 ChatGPT/NLLB/Qwen14 diagnostic','TranslateGemma deterministic+Qwen7 preliminary pass','75-row standardized complement final Qwen14 screen'],
 'previous_complement_pass_overlaps_exact_row':0,
 'unique_language_kind_source_candidate_payloads':len(payload_counts),
 'duplicate_payload_rows_retained':len(payloads)-len(payload_counts),
 'duplicates_note':'Every row remains source-ID bound; same prompt text across different IDs is not merged because entity-specific grammatical context is hidden by placeholders.',
 'request_sha256':hashlib.sha256((HERE/'requests.jsonl').read_bytes()).hexdigest(),
 'candidate_sha256':hashlib.sha256((HERE/'proposed_translations.jsonl').read_bytes()).hexdigest(),
 'release_approved':False,'training_eligible':False,
}
(HERE/'queue_coverage_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
