#!/usr/bin/env python3
"""Read-only integrity verification of an unapproved SynthWorlds-SM v2 pilot bundle."""
import collections,json,re,sys
from pathlib import Path
from build_v2 import ROOT,SOURCE,AUDIT,OUT,FILES,validate_sources,load_json,load_jsonl,sha,die
from ingest_translation_batch import LANGS,numeric_tokens

def verify():
    man,expected=validate_sources()
    p=load_json(OUT/'progress_manifest.json');stage1=load_json(OUT/'stage1_stage2_report.json')
    audit=load_json(OUT/'pilot_audit_report.json')
    assert p['source_checksums']==expected
    assert stage1['source_checksums']==expected
    assert stage1['proposed_split_counts']=={'forget':78,'retain':26,'holdout':26}
    assert stage1['cross_split_exact_chain_fact_collisions']==0
    assert stage1['fact_group_count']==112
    assert stage1['entity_disjoint'] is False
    assert set(p['proposed_assignments'])==set(r['source_qa_id'] for r in load_jsonl(OUT/'verified_pilot_assignments.jsonl'))
    assert p['split_status']=='PROPOSED_NOT_APPROVED'
    assert p['do_not_train'] is True
    sft_en={r['sft_id']:r for r in load_jsonl(OUT/'sft_natural_en.jsonl')}
    quarantined={r['source_qa_id'] for r in load_jsonl(OUT/'sft_quarantined_cases.jsonl')}
    assert len(sft_en)==206 and len(set(sft_en))==206 and '0080-qa-sm' in quarantined
    qa={r['instance_id']:r for r in load_jsonl(SOURCE/FILES['qa'])}
    doc={r['qid']:r for r in load_jsonl(SOURCE/FILES['documents'])}
    facts={r['qid']:r for r in load_jsonl(SOURCE/FILES['facts'])}
    holdout=set()
    for iid,a in p['proposed_assignments'].items():
        if a['proposed_split']=='holdout':holdout.update([a['target_fact_id'],a['r2_fact_id']])
    for key,rec in sft_en.items():
        assert rec['fact_id'] not in holdout
        assert rec['source_qa_id'] not in quarantined
        assert rec['proposed_split'] in ('forget','retain')
        assert rec['status']=='SFT_CANDIDATE_NOT_APPROVED'
        assert rec['messages']==[{'role':'user','content':rec['question_en']},{'role':'assistant','content':rec['answer']}]
        assert rec['fact_group_id']==p['proposed_assignments'][rec['source_qa_id']]['fact_group_id']
        assert rec['target_fact_id']==p['proposed_assignments'][rec['source_qa_id']]['target_fact_id']
        assert not any(z in rec for z in ('gold_answers','query','gold_docs','mhop_qa'))
        assert '<bridge>' not in rec['question_en']
    counts={}
    for lang in LANGS:
        seen={'qa':{},'documents':{},'facts':{},'sft':{}}
        for folder in sorted((OUT/'batches').glob('batch_*')):
            for category,stem,key in [('qa','qa-sm','instance_id'),('documents','qa-sm-docs','qid'),('facts','qa-sm-facts','qid'),('sft','sft_natural','sft_id')]:
                f=folder/f'{stem}_{lang}_batch{folder.name[-4:]}.jsonl'
                if not f.exists():continue
                for r in load_jsonl(f):seen[category][r[key]]=r
        for qid,r in seen['documents'].items():
            assert r['doc']==doc[qid]['doc'] and r['qid']==qid
            assert r['human_review_status']=='pending'
            assert numeric_tokens(r['doc'])==numeric_tokens(r['doc_'+lang])
        for qid,r in seen['facts'].items():
            assert r['facts']==facts[qid]['facts'] and r['qid']==qid
            assert r['human_review_status']=='pending'
            assert len(r['facts'])==len(r['facts_'+lang])
            for en,tr in zip(r['facts'],r['facts_'+lang]):
                z=en.split(' → ');v=tr.split(' → ')
                assert len(z)==len(v)
                assert all(z[i]==v[i] for i in range(0,len(z),2))
                assert numeric_tokens(en)==numeric_tokens(tr)
        complete=set();partial=set()
        for iid,r in seen['qa'].items():
            orig=qa[iid]
            assert all(r[k]==v for k,v in orig.items())
            assert r['human_review_status']=='pending'
            assert len(r['gold_docs_'+lang])==len(orig['gold_qids'])
            assert len(r['decomposed_questions_'+lang])==len(json.loads(orig['mhop_qa'])['decomposed_questions'])
            if all(r['gold_docs_'+lang]) and not r['translation_notes']:
                complete.add(iid)
                for qid,trdoc in zip(orig['gold_qids'],r['gold_docs_'+lang]):
                    assert qid in seen['documents']
                    assert trdoc==seen['documents'][qid]['doc_'+lang]
            else:
                partial.add(iid)
                assert r['translation_notes']
                for qid,trdoc in zip(orig['gold_qids'],r['gold_docs_'+lang]):
                    if not trdoc:assert qid not in seen['documents'] or orig['gold_docs'][orig['gold_qids'].index(qid)]!=doc[qid]['doc']
        for iid, r in seen['sft'].items():
            original=sft_en[iid]
            assert r['answer']==original['answer']
            assert r['question_en']==original['question_en']
            assert r['fact_id']==original['fact_id']
            assert r['human_review_status']=='pending'
            assert r['status']=='SFT_CANDIDATE_NOT_APPROVED'
            assert r['messages'][0]['content']==r['question_'+lang]
            assert r['messages'][1]['content']==r['answer']
            assert '<bridge>' not in r['question_'+lang]
        for cat in ('qa','documents','facts','sft'):
            m=p['translations'][lang][cat]
            actual_complete=(complete if cat=='qa' else set(seen[cat]))
            expected_ids=set(p['expected_source_ids'][cat]);expected_pending=expected_ids-actual_complete
            assert set(m['completed_ids'])==actual_complete
            assert set(m['pending_ids'])==expected_pending
            assert m['completed_count']==len(actual_complete)
            assert m['pending_count']==len(expected_pending)
            assert set(m['partial_ids'])==(partial if cat=='qa' else set())
        counts[lang]={cat:len(seen[cat]) for cat in seen}
    for rel,metadata in p['output_files'].items():
        path=OUT/rel
        assert path.is_file() and sha(path)==metadata['sha256'],f'Output SHA mismatch {rel}'
    for fn in ['build_v2.py','ingest_translation_batch.py','verify_v2.py']:
        source=(OUT/'scripts'/fn).read_text()
        assert ('/'+'mnt'+'/data') not in source,f'Nonportable script: {fn}'
    return {'verification':'PASS','sources_and_outputs_hashed':'PASS','split_unchanged':'PASS',
            'holdout_sft_exclusion':'PASS','qa_gold_document_reuse':'PASS_for_completed_records',
            'unfinished_gold_passages':'EMPTY_AND_FLAGGED','translation_semantic_certification':'NOT_PERFORMED',
            'record_counts_by_language':counts}
if __name__=='__main__':
    try:print(json.dumps(verify(),indent=2))
    except Exception as exc:print('VALIDATION FAILED: '+str(exc),file=sys.stderr);sys.exit(1)
