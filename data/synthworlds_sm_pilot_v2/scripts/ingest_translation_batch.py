#!/usr/bin/env python3
"""Ingest a small, explicit multilingual translation payload into v2 (never machine-fabricate text).

Run `python data/synthworlds_sm_pilot_v2/scripts/build_v2.py` first.
Then `python data/synthworlds_sm_pilot_v2/scripts/ingest_translation_batch.py 0001`.
Payloads are data/synthworlds_sm_pilot_v2/translation_payloads/batch_XXXX.json.
Requires unchanged original sources in data/synthworlds_sm and audits in data/synthworlds_sm_reaudit.
No network dependency. This validates structure, not translation correctness.
"""
import collections, copy, hashlib, json, re, sys
from pathlib import Path
from build_v2 import ROOT,SOURCE,AUDIT,OUT,FILES,validate_sources,load_json,load_jsonl,sha,die,entity_node,write_json,write_jsonl
LANGS=('fr','de','hi','th')
CATS=('qa','documents','facts','sft')


def numeric_tokens(s): return collections.Counter(re.findall(r'\d{4}-\d{2}-\d{2}|\d+',s))
def check_numbers(en,tr,where):
    if numeric_tokens(en)!=numeric_tokens(tr):die(f'Numbers/dates changed at {where}: en={numeric_tokens(en)} tr={numeric_tokens(tr)}')
def protected_names(english, graph):
    # Graph-renamed entity surfaces: the strings are protected, even when syntactically ordinary words.
    ren=graph.get('entity_rename_map',{})
    pool={entry['new_name'] for entry in ren.values() if entry.get('new_name')}
    pool.update(entity_node(graph['graph_qa'][k])[0] for k in ('n1','n2','n3'))
    return {name for name in pool if name and name in english}
def check_question(english,translated,graph,where):
    if not isinstance(translated,str) or not translated.strip():die(f'Missing translation {where}')
    check_numbers(english,translated,where)
    for name in protected_names(english,graph):
        if name not in translated:die(f'Entity name changed in {where}: {name!r}')
    if len(re.findall(r'<bridge>',english))!=len(re.findall(r'<bridge>',translated)) or len(re.findall(r'</bridge>',english))!=len(re.findall(r'</bridge>',translated)):
        die(f'Bridge markup count changed in {where}')
    if re.findall(r'<bridge>(.*?)</bridge>',english)!=re.findall(r'<bridge>(.*?)</bridge>',translated):
        die(f'Bridge entity changed in {where}')

def build_index_and_files(payload, batch):
    man,actual=validate_sources() # no writes before source validation
    for name in ('sft_natural_en.jsonl','verified_pilot_assignments.jsonl','stage1_stage2_report.json'):
        if not (OUT/name).is_file():die(f'Missing prerequisite output; run build_v2.py: {name}')
    qa={r['instance_id']:r for r in load_jsonl(SOURCE/FILES['qa'])}
    docs={r['qid']:r for r in load_jsonl(SOURCE/FILES['documents'])}
    facts={r['qid']:r for r in load_jsonl(SOURCE/FILES['facts'])}
    sft_en={r['sft_id']:r for r in load_jsonl(OUT/'sft_natural_en.jsonl')}
    props={r['source_qa_id']:r for r in load_jsonl(OUT/'verified_pilot_assignments.jsonl')}
    if len(sft_en)!=206:die('English SFT records have changed or count is wrong')
    existing_files=sorted((OUT/'batches').glob('batch_*/*_*.jsonl')) if (OUT/'batches').is_dir() else []
    if any(f.parent.name=='batch_'+batch for f in existing_files):die(f'Batch {batch} already exists; do not overwrite it')
    previous_docs={lang:{} for lang in LANGS}
    for lang in LANGS:
        for file in sorted((OUT/'batches').glob('batch_*/qa-sm-docs_'+lang+'_batch*.jsonl')):
            for r in load_jsonl(file):previous_docs[lang][r['qid']]=r['doc_'+lang]
    if set(payload)!= {'batch_id','method','languages'} and not {'batch_id','method','languages'}<=set(payload):die('Missing payload fields')
    if payload['batch_id']!=batch:die(f'Payload batch ID changed: {payload["batch_id"]} != {batch}')
    if set(payload['languages'])!=set(LANGS):die('All four language entries required')
    prepared={};preflight=[]
    for lang in LANGS:
        data=payload['languages'][lang]
        if set(data)!={'qa','documents','facts'}:die(f'{lang}: category fields required')
        if not set(data['qa'])<=set(qa) or not set(data['documents'])<=set(docs) or not set(data['facts'])<=set(facts):die(f'{lang}: unknown source ID')
        docmap={**previous_docs[lang]}
        drows=[];frows=[];qrows=[];srows=[]
        for qid,tr in sorted(data['documents'].items()):
            if not isinstance(tr,str) or not tr.strip():die(f'Empty doc translation {lang}/{qid}')
            if qid in docmap and docmap[qid]!=tr:die(f'{lang}/{qid}: conflicting canonical translation')
            check_numbers(docs[qid]['doc'],tr,f'{lang}/document/{qid}')
            # Check every graph-linked synthetic entity surface occurring in the document.
            protected=set()
            for q in qa.values():
                if qid in q['gold_qids']:
                    graph=json.loads(q['mhop_qa'])
                    protected.update(protected_names(docs[qid]['doc'],graph))
            for name in protected:
                if name not in tr:die(f'{lang}/document/{qid}: lost named entity {name!r}')
            row=copy.deepcopy(docs[qid]);row['doc_'+lang]=tr
            row['human_review_status']='pending';row['translation_notes']=[];row['translation_status']='translated_pending_human_review';row['translation_batch']=batch
            drows.append(row);docmap[qid]=tr
        for qid,tr in sorted(data['facts'].items()):
            src=facts[qid]['facts']
            if len(tr)!=len(src):die(f'{lang}/fact/{qid}: changed fact-list length')
            for i,(base,text) in enumerate(zip(src,tr)):
                orig_parts=base.split(' → '); parts=text.split(' → ')
                if len(orig_parts)!=len(parts):die(f'{lang}/fact/{qid}/{i}: relation/qualifier structure changed')
                for j,part in enumerate(orig_parts):
                    if j%2==0 and part!=parts[j]:die(f'{lang}/fact/{qid}/{i}: subject/object/value changed at segment {j}')
                    if j%2==1 and not parts[j].strip():die(f'{lang}/fact/{qid}/{i}: empty relation translation')
                check_numbers(base,text,f'{lang}/fact/{qid}/{i}')
            row=copy.deepcopy(facts[qid]);row['facts_'+lang]=tr
            row['human_review_status']='pending';row['translation_notes']=[];row['translation_status']='translated_pending_human_review';row['translation_batch']=batch
            frows.append(row)
        for iid,translated in sorted(data['qa'].items()):
            original=qa[iid];m=json.loads(original['mhop_qa']);d=m['decomposed_questions']
            if set(translated) !={'query','decomposed_questions'} or set(translated['decomposed_questions'])!=set(d):
                die(f'{lang}/qa/{iid}: incomplete query/Q1/Q2 keys')
            check_question(original['query'],translated['query'],m,f'{lang}/qa/{iid}/query')
            for key,txt in d.items():check_question(txt,translated['decomposed_questions'][key],m,f'{lang}/qa/{iid}/{key}')
            gold=[];notes=[]
            for i,qid in enumerate(original['gold_qids']):
                if qid not in docs or original['gold_docs'][i]!=docs[qid]['doc']:
                    gold.append('');notes.append({'code':'SOURCE_GOLD_DOC_QID_MAPPING_MISMATCH','qid':qid,'index':i})
                elif qid not in docmap:
                    gold.append('');notes.append({'code':'CANONICAL_DOC_TRANSLATION_PENDING','qid':qid,'index':i})
                else:gold.append(docmap[qid])
            row=copy.deepcopy(original)
            row['query_'+lang]=translated['query']
            row['decomposed_questions_'+lang]=translated['decomposed_questions']
            row['gold_docs_'+lang]=gold
            row['translation_notes']=notes
            row['translation_batch']=batch
            row['human_review_status']='pending'
            row['translation_status']='translated_pending_human_review' if not notes else 'partial_pending_canonical_gold_docs'
            if iid in props:
                row['proposed_split']=props[iid]['proposed_split'];row['fact_group_id']=props[iid]['fact_group_id']
            # All original source QA fields remain equal at the value and type level.
            if any(row[k]!=value for k,value in original.items()):die(f'{lang}/qa/{iid}: source content mutated')
            qrows.append(row)
            for hop,key in (('r1','Q1'),('r2','Q2')):
                sfid=iid+':'+hop
                if sfid not in sft_en:continue # holdout and quarantined facts do not produce SFT
                eng=sft_en[sfid]
                question=translated['decomposed_questions'][key]
                if hop=='r2':question=re.sub(r'</?bridge>', '',question)
                if '<bridge>' in question or '</bridge>' in question:die(f'Bridge markup not removed in SFT {lang}/{sfid}')
                rec=copy.deepcopy(eng)
                rec['question_'+lang]=question
                rec['messages_en']=rec['messages']
                rec['messages']=[{'role':'user','content':question},{'role':'assistant','content':rec['answer']}]
                rec['translation_batch']=batch
                rec['human_review_status']='pending'
                rec['translation_status']='translated_pending_human_review'
                rec['status']='SFT_CANDIDATE_NOT_APPROVED'
                srows.append(rec)
        if len({r['instance_id'] for r in qrows})!=len(qrows):die('Duplicate QA in batch')
        prepared[lang]={'qa':qrows,'documents':drows,'facts':frows,'sft':srows}
        preflight.append({'language':lang,'qa_records':len(qrows),'qa_complete_docs':sum(r['translation_status']=='translated_pending_human_review' for r in qrows),'docs':len(drows),'facts':len(frows),'sft':len(srows)})
    return prepared,preflight

def refresh_manifests():
    # Recompute from actual written files, never from optimistic payload counts.
    man,actual=validate_sources()
    prior=load_json(OUT/'stage1_stage2_report.json')
    en_sft=load_jsonl(OUT/'sft_natural_en.jsonl')
    qa=load_jsonl(SOURCE/FILES['qa']); docs=load_jsonl(SOURCE/FILES['documents']); facts=load_jsonl(SOURCE/FILES['facts'])
    ids={'qa':sorted(r['instance_id'] for r in qa),'documents':sorted(r['qid'] for r in docs),
         'facts':sorted(r['qid'] for r in facts),'sft':sorted(r['sft_id'] for r in en_sft)}
    batches=sorted(p for p in (OUT/'batches').glob('batch_*') if p.is_dir())
    status={}; QAindex={r['instance_id']:r for r in qa}; docindex={r['qid']:r['doc'] for r in docs}
    all_issues=[];completed_batches=[]
    for lang in LANGS:
        seen={cat:{} for cat in CATS}
        for b in batches:
            for cat,pattern,key in [('qa',f'qa-sm_{lang}_batch*.jsonl','instance_id'),('documents',f'qa-sm-docs_{lang}_batch*.jsonl','qid'),
                                    ('facts',f'qa-sm-facts_{lang}_batch*.jsonl','qid'),('sft',f'sft_natural_{lang}_batch*.jsonl','sft_id')]:
                for file in sorted(b.glob(pattern)):
                    for row in load_jsonl(file):
                        iid=row[key]
                        if iid not in ids[cat]:die(f'Invalid ID from output: {lang}/{cat}/{iid}')
                        if iid in seen[cat] and cat in ('documents','facts'):
                            tkey={'documents':'doc_'+lang,'facts':'facts_'+lang}[cat]
                            if seen[cat][iid][tkey]!=row[tkey]:die(f'Canonical translation conflict: {lang}/{cat}/{iid}')
                        seen[cat][iid]=row
        complete_qa=set();partial_qa=set();crossdocs=[]
        for iid,r in seen['qa'].items():
            originals=QAindex[iid]
            if any(r[k]!=v for k,v in originals.items()):die(f'English QA field mutated: {iid}/{lang}')
            for i,qid in enumerate(originals['gold_qids']):
                raw=originals['gold_docs'][i];can=docindex.get(qid)
                if qid in seen['documents'] and raw==can:
                    expected=seen['documents'][qid]['doc_'+lang]
                    if r['gold_docs_'+lang][i] and r['gold_docs_'+lang][i]!=expected:
                        die(f'Gold document translation mismatch: {lang}/{iid}/{qid}')
                    if not r['gold_docs_'+lang][i]:crossdocs.append({'qa_id':iid,'qid':qid,'language':lang,'reason':'CANONICAL_DOC_AVAILABLE_BUT_QA_BATCH_NEEDS_REFRESH'})
            if all(r['gold_docs_'+lang]) and not r['translation_notes']:
                for i,qid in enumerate(originals['gold_qids']):
                    if qid not in seen['documents'] or r['gold_docs_'+lang][i]!=seen['documents'][qid]['doc_'+lang]:
                        die(f'Completed QA does not match canonical document {lang}/{iid}')
                complete_qa.add(iid)
            else:partial_qa.add(iid)
        status[lang]={}
        for cat in CATS:
            expected=set(ids[cat]);completed=(complete_qa if cat=='qa' else set(seen[cat]))
            partial=(partial_qa if cat=='qa' else set())
            status[lang][cat]={
                'expected_count':len(expected),'completed_count':len(completed),
                'pending_count':len(expected-completed),'partial_count':len(partial),
                'completed_ids':sorted(completed),'partial_ids':sorted(partial),'pending_ids':sorted(expected-completed),
                'status':'COMPLETE_PENDING_HUMAN_REVIEW' if len(completed)==len(expected) else 'PARTIAL_PENDING_HUMAN_REVIEW'
            }
        all_issues.extend(crossdocs)
    output_meta={}
    for file in sorted(OUT.rglob('*')):
        if not file.is_file() or file.name in ('progress_manifest.json','sha256sums.txt','pilot_audit_report.json','README.md'):continue
        output_meta[str(file.relative_to(OUT))]={'sha256':sha(file),'bytes':file.stat().st_size,
                                                 'rows':sum(1 for r in file.open('rb') if r.strip()) if file.suffix=='.jsonl' else None}
    eligible=load_jsonl(OUT/'verified_pilot_assignments.jsonl')
    exclusions=load_jsonl(OUT/'excluded_type_a.jsonl')
    quarantine=load_jsonl(OUT/'sft_quarantined_cases.jsonl')
    hashes={name:sha(AUDIT/name) for name in ('candidate_split_manifest.json','type_a_target_eligibility.jsonl','proposed_split_assignments.jsonl','stage1_reaudit_report.json')}
    manifest={'schema_version':'v2-partial-1','status':'PARTIAL_TRANSLATION_SFT_CANDIDATE_NOT_APPROVED',
              'source_checksums':actual,'audit_file_sha256':hashes,
              'split_status':'PROPOSED_NOT_APPROVED',
              'proposed_split_counts':prior['proposed_split_counts'],
              'proposed_assignments':{r['source_qa_id']:{k:r[k] for k in ('proposed_split','fact_group_id','target_fact_id','r2_fact_id')} for r in eligible},
              'excluded_type_a_ids':sorted(r['instance_id'] for r in exclusions),
              'excluded_other_graph_type_ids':sorted(set(ids['qa'])-set(r['source_qa_id'] for r in eligible)-set(r['instance_id'] for r in exclusions)),
              'sft_quarantined_ids':sorted(r['source_qa_id'] for r in quarantine),
              'languages':{'en':'canonical_source_unchanged',**{l:'human_review_status=pending' for l in LANGS}},
              'expected_source_ids':ids,
              'translations':status,
              'batches':[], 'translation_model_method':'GPT-6 authored limited translation batches, carried forward seed batch 0001 from v1 plus new QA batch 0002. No external MT API. Python structural validation only.',
              'output_files':output_meta,
              'known_gold_doc_backfill_issues':all_issues,
              'do_not_train':True,'human_review_status':'pending',
              'note':'Pending IDs list includes partially translated QA IDs; no linguistic review is claimed.'}
    for folder in batches:
        batch_id=folder.name.split('_')[-1]
        meta_file=folder/'batch_report.json'
        bm=load_json(meta_file) if meta_file.is_file() else {}
        manifest['batches'].append({'batch_id':batch_id,'report':bm,'files':sorted(str(f.relative_to(OUT)) for f in folder.glob('*.jsonl'))})
    write_json(OUT/'progress_manifest.json',manifest)
    audit={'source_validation':'PASS','split_validation':'PASS_PROPOSAL_ONLY',
           'source_counts':{cat:item['rows'] for cat,item in actual.items()},
           'eligible_type_a':prior['eligible_type_a'],'excluded_type_a':prior['type_a_excluded'],
           'excluded_other_graph_types':prior['other_graph_types_excluded'],
           'split_counts':prior['proposed_split_counts'],
           'entity_overlap_pairs':prior['entity_overlap_pairs'],'entity_disjoint':False,
           'cross_split_exact_chain_fact_collisions':0,
           'en_sft_records':len(en_sft),'unique_en_sft_fact_ids':len(set(r['fact_id'] for r in en_sft)),
           'sft_quarantined_cases':quarantine,
           'by_language':{lang:{cat:{k:status[lang][cat][k] for k in ('expected_count','completed_count','pending_count','partial_count')} for cat in CATS} for lang in LANGS},
           'translation_model_method':manifest['translation_model_method'],
           'human_review_status':'pending',
           'translation_validation_failure_count':0,
           'known_pending_gold_doc_mappings':all_issues,
           'translation_review_issues':['All machine-generated translations need human linguistic review.',
               'Only full gold-doc translations are counted as completed QA; partial QA question-only translations are retained as partial.',
               'Source facts provide no independently checkable relation PIDs, only graph metadata.',
               'Graph r2 qualifier in a proposed eligible case references an entity different from the selected source fact; the SFT QA case was quarantined.',
               'No passage-level documents or whole fact rows are used as SFT.'],
           'status':'PARTIAL_FOR_REVIEW_NOT_APPROVED', 'model_training_started':False}
    write_json(OUT/'pilot_audit_report.json',audit)
    lines=[]
    for file in sorted(OUT.rglob('*')):
        if file.is_file() and file.name!='sha256sums.txt':lines.append(f'{sha(file)}  {file.relative_to(OUT)}')
    (OUT/'sha256sums.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return audit

def main(batch_id):
    payload_path=OUT/'translation_payloads'/f'batch_{batch_id}.json'
    data=load_json(payload_path)
    rows,preflight=build_index_and_files(data,batch_id)
    folder=OUT/'batches'/f'batch_{batch_id}'
    folder.mkdir(parents=True,exist_ok=False)
    for lang in LANGS:
        for cat,stem in (('qa','qa-sm'),('documents','qa-sm-docs'),('facts','qa-sm-facts'),('sft','sft_natural')):
            if rows[lang][cat]:write_jsonl(folder/f'{stem}_{lang}_batch{batch_id}.jsonl',rows[lang][cat])
    write_json(folder/'batch_report.json',{'batch_id':batch_id,'method':data['method'],
              'counts':preflight,'human_review_status':'pending','validation':'structural_pass_only'})
    audit=refresh_manifests()
    print(json.dumps({'batch':batch_id,'status':'saved_pending_review','counts':preflight},ensure_ascii=False))

if __name__=='__main__':
    if len(sys.argv)!=2 or not re.fullmatch(r'\d{4}',sys.argv[1]):
        print('Usage: python ingest_translation_batch.py 0001',file=sys.stderr);sys.exit(2)
    try:main(sys.argv[1])
    except Exception as exc:print(str(exc),file=sys.stderr);sys.exit(1)
