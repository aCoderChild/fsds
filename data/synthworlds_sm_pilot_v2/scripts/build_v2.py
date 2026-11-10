#!/usr/bin/env python3
"""Verify SynthWorlds-SM Type-A split and build unapproved English single-hop SFT.

Portable layout:
  <repo>/data/synthworlds_sm/{qa-sm,qa-sm-docs,qa-sm-facts}.jsonl
  <repo>/data/synthworlds_sm_reaudit/{candidate_split_manifest.json,...}
  <repo>/data/synthworlds_sm_pilot_v2/scripts/build_v2.py
Never modifies the original source/audit files. Stops before writing upon integrity mismatch.
"""
from __future__ import annotations
import collections, copy, hashlib, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data'/'synthworlds_sm'
AUDIT = ROOT/'data'/'synthworlds_sm_reaudit'
OUT = ROOT/'data'/'synthworlds_sm_pilot_v2'
FILES = {'qa':'qa-sm.jsonl','documents':'qa-sm-docs.jsonl','facts':'qa-sm-facts.jsonl'}
AUDIT_FILES = ('candidate_split_manifest.json','type_a_target_eligibility.jsonl','proposed_split_assignments.jsonl','stage1_reaudit_report.json')


def die(message): raise RuntimeError('STOP -- ' + message)
def sha(path):
    with path.open('rb') as h: return hashlib.file_digest(h,'sha256').hexdigest()
def load_json(path):
    if not path.is_file(): die(f'Missing required input: {path}')
    return json.loads(path.read_text(encoding='utf-8'))
def load_jsonl(path):
    if not path.is_file(): die(f'Missing required input: {path}')
    with path.open(encoding='utf-8') as h: return [json.loads(row) for row in h if row.strip()]
def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8') as h:
        for row in rows: h.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
def entity_node(raw):
    m = re.fullmatch(r'(.*) \((Q\d+)\)',raw,flags=re.DOTALL)
    return (m.group(1),m.group(2)) if m else (raw,None)
def fact_id(qid,text): return 'derived-sha256:'+hashlib.sha256((qid+'\0'+text).encode()).hexdigest()
def validate_sources():
    man=load_json(AUDIT/'candidate_split_manifest.json')
    # Stage 1 must stop before any output modifications if an input is missing or changed.
    for name in AUDIT_FILES: 
        if not (AUDIT/name).is_file(): die(f'Missing required audit input {AUDIT/name}')
    actual={}
    for cat,name in FILES.items():
        path=SOURCE/name
        if not path.is_file():die(f'Missing required source input: {path}')
        val={'filename':name,'sha256':sha(path),'bytes':path.stat().st_size,
             'rows':sum(1 for row in path.open('rb') if row.strip())}
        for k in ('filename','sha256','bytes','rows'):
            if man['source_checksums'][cat][k]!=val[k]:
                die(f'{name} {k} mismatch: audit={man["source_checksums"][cat][k]} actual={val[k]}')
        actual[cat]=val
    return man,actual

def build():
    man, hashes = validate_sources()  # Absolutely no files written before source validation.
    Q=load_jsonl(SOURCE/FILES['qa']); D=load_jsonl(SOURCE/FILES['documents']); F=load_jsonl(SOURCE/FILES['facts'])
    E=load_jsonl(AUDIT/'type_a_target_eligibility.jsonl')
    A=load_jsonl(AUDIT/'proposed_split_assignments.jsonl')
    prior=load_json(AUDIT/'stage1_reaudit_report.json')
    qm={x['instance_id']:x for x in Q};fm={x['qid']:x for x in F}; dm={x['qid']:x for x in D}
    if any(len(m)!=len(rows) for m,rows in ((qm,Q),(fm,F),(dm,D))): die('Duplicate source IDs')
    em={x['instance_id']:x for x in E};am={x['instance_id']:x for x in A}
    if len(em)!=200 or len(am)!=130 or len(set(am))!=130:die('Audit eligibility/assignment cardinality mismatch')
    if set(am)!={iid for iid,e in em.items() if e['eligible']}:die('Assignments do not cover exactly eligible IDs')
    if set(em)!={iid for iid,q in qm.items() if q['question_graph_type']=='A'}:die('Type-A audit ID set mismatch')
    groups=man['group_assignments']; by_group={}
    for gid,g in groups.items():
        for iid in g['instance_ids']:
            if iid in by_group:die(f'Duplicate group membership {iid}')
            by_group[iid]=(gid,g['proposed_split'])
        if len(g['instance_ids'])!=g['size']:die(f'Group size mismatch {gid}')
    if set(by_group)!=set(am):die('Missing/extra IDs in indivisible groups')
    if collections.Counter(len(g['instance_ids']) for g in groups.values())!=collections.Counter({1:98,2:12,4:2}):die('Group structure changed')
    excluded={};valid={};all_chain={}; group_guard=collections.defaultdict(set)
    for iid in sorted(em):
        e=em[iid];q=qm[iid];g=json.loads(q['mhop_qa']);graph=g['graph_qa']; decomp=g['decomposed_questions']
        n1,n2,n3=(entity_node(graph['n'+str(k)]) for k in (1,2,3))
        if graph['graph_type']!='A' or q['question_graph_type']!='A':die(f'Bad Type A mapping: {iid}')
        if set(decomp)!={'Q1','Q2'}:die(f'Unrecognized decomposed questions: {iid}')
        r1,r2=graph['r1'],graph['r2']; r1f=fm[n1[1]]['facts'];r2f=fm[e['r2_source_qid']]['facts']
        pref1=graph['n1']+' → '+r1['relation_str']+' → '
        possible={s.split(' → ')[2] for s in r1f if s.startswith(pref1) and len(s.split(' → '))>=3}
        r1matches=[s for s in r1f if s.startswith(pref1) and len(s.split(' → '))==3 and s.split(' → ')[2]==graph['n2']]
        pref2=graph['n2']+' → '+r2['relation_str']+' → '
        r2matches=[s for s in r2f if s in e['r2_matched_fact_texts'] and s.startswith(graph['n2']+' → ') and len(s.split(' → '))>=3 and s.split(' → ')[-1]==graph['n3']]
        q1=decomp['Q1'];q2=decomp['Q2']
        # Two documented punctuation variants are handled as a surface-only normalization; do NOT change source strings.
        q1anchored=n1[0] in q1 or n1[0].rstrip('.') in q1
        bridgeparts=re.findall(r'<bridge>(.*?)</bridge>',q2)
        q2anchored=len(bridgeparts)==1 and bridgeparts[0].rstrip('.')==n2[0].rstrip('.')
        eligible=(q1anchored and q2anchored and len(possible)==1 and
                  len(r1matches)==1 and len(r2matches)==1 and len(q['gold_answers'])>=1)
        if bool(e['eligible'])!=eligible:die(f'Eligibility disagreement: {iid}; expected audit {e["eligible"]}, computed {eligible}')
        if not eligible:
            excluded[iid]={'instance_id':iid,'reason_codes':e['eligibility_reason_codes']}
            continue
        if r1['qid']!=n1[1] or r2['qid']!=e['r2_source_qid']:die(f'Relation subject mismatch {iid}')
        if n3[0] not in q['gold_answers']:die(f'Gold answer / n3 mismatch {iid}')
        id1=fact_id(n1[1],r1matches[0]);id2=fact_id(e['r2_source_qid'],r2matches[0])
        if [id1,id2]!=e['chain_fact_ids'] or id1!=e['target_fact_id'] or r2matches!=e['r2_matched_fact_texts']:
            die(f'Exact fact/source ID disagreement {iid}')
        a=am[iid]; gid,split=by_group[iid]
        if (a['fact_group_id'],a['proposed_split'],a['target_fact_id'])!=(gid,split,id1):die(f'Assignment altered {iid}')
        if a['chain_fact_ids']!=[id1,id2] or a['source_gold_qids']!=q['gold_qids']:die(f'Assignment evidence changed {iid}')
        if man['target_fact_ids'][iid]!=id1:die(f'Manifest target ID altered {iid}')
        for fid in [id1,id2]:group_guard['fact:'+fid].add((gid,split))
        for ans in q['gold_answers']:group_guard['gold:'+ans].add((gid,split))
        all_chain[iid]=(r1matches[0],r2matches[0],id1,id2)
        valid[iid]={'audit':e,'qa':q,'graph':graph,'decomp':decomp,'nodes':(n1,n2,n3),'group':gid,'split':split,'fact_ids':(id1,id2), 'facts':(r1matches[0],r2matches[0])}
    if len(valid)!=130 or len(excluded)!=70:die('Not exactly 130 eligible and 70 excluded Type-A QA')
    for key,gs in group_guard.items():
        if len(gs)!=1:die(f'Shared exact fact or answer splits indivisible groups: {key}: {gs}')
    counts=collections.Counter(v['split'] for v in valid.values())
    if counts!=collections.Counter({'forget':78,'retain':26,'holdout':26}):die(f'Proposed assignments changed: {counts}')
    # Every group must represent a fact/answer-connected component (not merely check splits).
    parent={iid:iid for iid in valid}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for key,grs in group_guard.items():
        members=[iid for iid,v in valid.items() if (key.startswith('fact:') and key[5:] in v['fact_ids']) or
                 (key.startswith('gold:') and key[5:] in v['qa']['gold_answers'])]
        for iid in members[1:]:parent[find(iid)]=find(members[0])
    components=collections.defaultdict(set)
    for iid in valid:components[find(iid)].add(iid)
    if {frozenset(x) for x in components.values()}!={frozenset(g['instance_ids']) for g in groups.values()}:
        die('Fact/answer-connected groups differ from assigned groups')
    # Entity overlaps are permitted and quantified separately.
    shared_entities=collections.Counter()
    ids=list(valid)
    for i,id1 in enumerate(ids):
        m1=json.loads(valid[id1]['qa']['mhop_qa']); s1=({node[1] for node in valid[id1]['nodes'] if node[1]} | set(valid[id1]['qa']['gold_qids']) | set(m1.get('entity_rename_map',{})) | {r.get('relation_subject_qid') for r in (m1['graph_qa']['r1'],m1['graph_qa']['r2']) if r.get('relation_subject_qid')})
        for id2 in ids[i+1:]:
            m2=json.loads(valid[id2]['qa']['mhop_qa']); s2=({node[1] for node in valid[id2]['nodes'] if node[1]} | set(valid[id2]['qa']['gold_qids']) | set(m2.get('entity_rename_map',{})) | {r.get('relation_subject_qid') for r in (m2['graph_qa']['r1'],m2['graph_qa']['r2']) if r.get('relation_subject_qid')})
            if s1&s2:
                shared_entities['all_pairs']+=1
                if valid[id1]['split']!=valid[id2]['split']:shared_entities['cross_split_pairs']+=1
    holdout_fids={fid for v in valid.values() if v['split']=='holdout' for fid in v['fact_ids']}
    holdout_facts={f for v in valid.values() if v['split']=='holdout' for f in v['facts']}
    # A source-fact match is not enough when graph qualification mentions a different entity.
    # Do not overwrite a source or alter assignments; quarantine mismatched QAs from SFT.
    sft_quarantine=[]
    for iid,v in sorted(valid.items()):
        r2=v['graph']['r2']; rel=r2['relation_str']; source_parts=v['facts'][1].split(' → ')
        if rel.startswith('[') and rel.endswith(']'):
            expected=rel[1:-1].split(' → ')
            actual=[re.sub(r' \(Q\d+\)$','',p) for p in source_parts[1:-1]]
            sem_ok=expected==actual
        else:
            sem_ok=(len(source_parts)>=3 and source_parts[1]==rel and source_parts[2]==v['graph']['n3'])
        if not sem_ok:
            sft_quarantine.append({'source_qa_id':iid,'reason':'R2_GRAPH_RELATION_VS_SOURCE_FACT_INCONSISTENT',
                                   'graph_relation_str':rel,'verified_source_r2_fact':v['facts'][1],
                                   'q2':v['decomp']['Q2'],'proposed_split_unchanged':v['split']})
    quarantined={r['source_qa_id'] for r in sft_quarantine}
    sft=[];excluded_qa=[]
    for iid,v in sorted(valid.items()):
        if v['split']=='holdout' or iid in quarantined:continue
        n1,n2,n3=v['nodes']
        for hop,qi,ans,src_qid,relation_id,subj,obj,fid,ftext in [
            ('r1','Q1',n2[0],n1[1],v['graph']['r1']['pid'],n1[1],n2[1],v['fact_ids'][0],v['facts'][0]),
            ('r2','Q2',n3[0],v['graph']['r2']['qid'],v['graph']['r2']['pid'],n2[1],n3[1],v['fact_ids'][1],v['facts'][1])]:
            if fid in holdout_fids or ftext in holdout_facts:die('Holdout fact in SFT: '+iid)
            question=v['decomp'][qi]
            if hop=='r2':question=re.sub(r'</?bridge>', '',question)
            if '<bridge>' in question or '</bridge>' in question or not question.strip() or not ans.strip():
                excluded_qa.append({'source_qa_id':iid,'hop':hop,'reason':'UNSAFE_QUESTION_OR_ANSWER'});continue
            rec={'sft_id':iid+':'+hop,'source_qa_id':iid,'target_fact_id':v['fact_ids'][0],
                'fact_id':fid,'fact_group_id':v['group'],'proposed_split':v['split'],
                'relation_id':relation_id,'hop':hop,'source_fact_record_qid':src_qid,
                'subject_entity_qid':subj,'object_entity_qid':obj,
                'n1_qid':n1[1],'n2_qid':n2[1],'n3_qid':n3[1],
                'question_en':question,'answer':ans,'source_fact_arrow_en':ftext,
                'status':'SFT_CANDIDATE_NOT_APPROVED','human_review_status':'not_applicable_source',
                'messages':[{'role':'user','content':question},{'role':'assistant','content':ans}]}
            if any(key in rec for key in ('query','gold_answers','gold_docs','mhop_qa')):die('Evaluation metadata entered SFT record')
            sft.append(rec)
    if excluded_qa:die(f'Unexpected unusable English SFT mapping: {excluded_qa}')
    if len(sft)!=2*(104-len(quarantined & {iid for iid,v in valid.items() if v['split']!='holdout'})) or len(set(r['sft_id'] for r in sft))!=len(sft):die('SFT case/hop count mismatch')
    if {r['fact_id'] for r in sft}&holdout_fids:die('Holdout fact ID leaked into SFT')
    # Writes begin only after all Stage 1 checks succeeded.
    OUT.mkdir(parents=True,exist_ok=True)
    write_jsonl(OUT/'sft_natural_en.jsonl',sft)
    write_jsonl(OUT/'verified_pilot_assignments.jsonl',[
        {'source_qa_id':iid,'proposed_split':v['split'],'fact_group_id':v['group'],
         'target_fact_id':v['fact_ids'][0],'r2_fact_id':v['fact_ids'][1]} for iid,v in sorted(valid.items())])
    write_jsonl(OUT/'excluded_type_a.jsonl',[excluded[iid] for iid in sorted(excluded)])
    write_jsonl(OUT/'sft_quarantined_cases.jsonl',sft_quarantine)
    report={'status':'VERIFIED_PROPOSAL_NOT_APPROVED','source_integrity':'PASSED',
            'source_checksums':hashes, 'eligible_type_a':len(valid),'type_a_excluded':len(excluded),
            'other_graph_types_excluded':len(Q)-len(E),'proposed_split_counts':dict(counts),
            'fact_group_count':len(groups),'group_size_histogram':dict(sorted(collections.Counter(len(g['instance_ids']) for g in groups.values()).items())),
            'cross_split_exact_target_fact_collisions':0,'cross_split_exact_chain_fact_collisions':0,
            'cross_split_same_gold_answer_collisions':0,'entity_overlap_pairs':dict(shared_entities),
            'entity_disjoint':False,'shared_entity_qid_basis':'graph nodes, gold_qids, entity_rename_map keys, and qualified relation subject QIDs', 'sft_en_qa_hop_records':len(sft),'unique_sft_fact_ids':len({r['fact_id'] for r in sft}),
            'holdout_chain_fact_count_excluded':len(holdout_fids),'excluded_sft_unusable':excluded_qa,'sft_quarantined_cases':sft_quarantine,
            'notes':['Source facts lack independent relation PIDs; relation IDs come from graph metadata.',
                     'Single-hop Q1/Q2 natural-language semantics are structurally checked, not independently human-certified.',
                     'r2 answers may equal final multi-hop answer entities; this is intrinsic to the required support fact.',
                     'No whole facts rows, full documents, composed multi-hop queries, or gold_answers fields in SFT.',
                     'Split remains proposed; SFT records remain candidates and are not approved.']}
    write_json(OUT/'stage1_stage2_report.json',report)
    print(json.dumps({'verification':'passed','split':dict(counts),'en_single_hop_records':len(sft),'unique_facts':len({r['fact_id'] for r in sft})}))
    return report
if __name__=='__main__':
    try:build()
    except Exception as exc:print(str(exc),file=sys.stderr);sys.exit(1)
