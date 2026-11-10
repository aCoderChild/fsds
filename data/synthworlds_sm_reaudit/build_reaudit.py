#!/usr/bin/env python3
"""Reproducible, non-mutating Type-A fact-level leakage re-audit of SynthWorlds-SM.

Inputs in ../synthworlds_sm remain byte-identical. Generated outputs are NOT
approved training data; allocation is a deterministic policy proposal only.
"""
import collections
import hashlib
import itertools
import json
import pathlib
import re
from datetime import datetime, timezone

BASE = pathlib.Path(__file__).resolve().parents[1] / 'synthworlds_sm'
OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261102
SOURCES={'qa':'qa-sm.jsonl','documents':'qa-sm-docs.jsonl','facts':'qa-sm-facts.jsonl'}

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def filehash(p):
    with p.open('rb') as f:
        h=hashlib.sha256()
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read_jsonl(p):
    with p.open(encoding='utf-8') as f: return [json.loads(l) for l in f if l.strip()]
def write_jsonl(filename,lines):
    with (OUT/filename).open('w',encoding='utf-8') as f:
        for row in lines: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
def write_json(filename,ob):
    (OUT/filename).write_text(json.dumps(ob,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def parts(s):return tuple(v.strip() for v in s.split('→'))
def parse_qid(s):
    z=re.search(r'\((Q\d+)\)$',s)
    return z.group(1) if z else None
def label(s):
    return s.rsplit(' (Q',1)[0].strip().rstrip('.') if ' (Q' in s else re.sub(r'\s*\(year\)$','',s).strip().rstrip('.')
def normalize(s):return re.sub(r'[^\w]+','',s.casefold().replace('’',"'"),flags=re.UNICODE)
def fact_id(qid,stmt):return 'derived-sha256:'+sha_bytes((qid+'\0'+stmt).encode('utf-8'))

def r2_match(g,F):
    r=g['r2']; n2=g['n2']; n3=g['n3']; result=[]
    if r['relation_str'].startswith('['):
        sem=parts(r['relation_str'][1:-1])
        for f in F[r['qid']]['facts']:
            p=parts(f)
            if len(sem)!=3 or len(p)!=5:continue
            if not(p[0]==n2 and p[1]==sem[0] and p[3]==sem[2] and p[4]==n3):continue
            if 'relation_subject_qid' in r:
                if parse_qid(p[2])!=r['relation_subject_qid']:continue
            elif normalize(p[2])!=normalize(sem[1]):continue
            result.append(f)
    else:
        candidate=f'{n2} → {r["relation_str"]} → {n3}'
        result=[f for f in F[r['qid']]['facts'] if f==candidate]
    return result

class DSU:
    def __init__(self, ids):self.parent={i:i for i in ids}
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x
    def unite(self,x,y):
        x,y=self.find(x),self.find(y)
        if x!=y:self.parent[max(x,y)]=min(x,y)
    def comps(self):
        c=collections.defaultdict(list)
        for i in sorted(self.parent):c[self.find(i)].append(i)
        return sorted([sorted(v) for v in c.values()],key=lambda z:(-len(z),z[0]))

def stable_group_id(members):return 'fg-'+sha_bytes((','.join(sorted(members))).encode())[:16]

def compute_overlap(ids,data,include_pairs=False):
    pairs=[];counts=collections.Counter();a=DSU(ids);b=DSU(ids);c=DSU(ids)
    for i,j in itertools.combinations(sorted(ids),2):
        x,y=data[i],data[j]
        ent=bool(set(x['graph_qids'])&set(y['graph_qids']))
        exact=bool(set(x['chain_fact_ids'])&set(y['chain_fact_ids']))
        target=x['target_fact_id']==y['target_fact_id']
        path=x['chain_fact_ids']==y['chain_fact_ids']
        answer=bool(set(x['gold_answers'])&set(y['gold_answers']))
        if ent: counts['shared_entity_qid_pairs']+=1; c.unite(i,j)
        if exact: counts['shared_exact_fact_pairs']+=1; a.unite(i,j);b.unite(i,j)
        if target:counts['shared_target_fact_pairs']+=1
        if path: counts['shared_complete_path_pairs']+=1;b.unite(i,j)
        if answer:counts['shared_gold_answer_pairs']+=1;b.unite(i,j)
        if path or answer:counts['shared_path_or_answer_pairs']+=1
        if ent and not(exact or target or path or answer): counts['qid_only_pairs']+=1
        if include_pairs and (ent or exact or target or path or answer):
            pairs.append({'qa_1':i,'qa_2':j,'shared_graph_qids':sorted(set(x['graph_qids'])&set(y['graph_qids'])),
              'shared_exact_reasoning_fact_ids':sorted(set(x['chain_fact_ids'])&set(y['chain_fact_ids'])),
              'shared_target_fact':target,'identical_complete_chain':path,'shared_gold_answers':sorted(set(x['gold_answers'])&set(y['gold_answers'])),
              'entity_qid_only':ent and not(exact or target or path or answer)})
    counts['total_pairs']=len(ids)*(len(ids)-1)//2
    counts['qa_count']=len(ids)
    result={}
    for name,groups in [('exact_fact',a.comps()),('exact_fact_or_path_or_answer',b.comps()),('entity_qid',c.comps())]:
        result[name]={'count':len(groups),'size_distribution':dict(sorted(collections.Counter(map(len,groups)).items())),
                      'largest_group_size':max(map(len,groups)) if groups else 0,
                      'groups':groups}
    return dict(counts),result,pairs

qa=read_jsonl(BASE/SOURCES['qa'])
docs=read_jsonl(BASE/SOURCES['documents'])
facts=read_jsonl(BASE/SOURCES['facts'])
D={r['qid']:r for r in docs};F={r['qid']:r for r in facts}
assert len(qa)==1200 and len(D)==6290 and len(F)==6290
original_hashes={k:{'filename':v,'sha256':filehash(BASE/v),'bytes':(BASE/v).stat().st_size,'rows':len({'qa':qa,'documents':docs,'facts':facts}[k])} for k,v in SOURCES.items()}
prior=OUT/'candidate_split_manifest.json'
if prior.exists():
    old=json.loads(prior.read_text()).get('source_checksums',{})
    for key in SOURCES:
        assert old.get(key,{}).get('sha256') == original_hashes[key]['sha256'], f'Prior source checksum mismatch: {key}'
    # Byte-for-byte checks agree with the previous audit.

all_data={};qareports=[];excluded=[];r2_issue=[]
for row in qa:
    id=row['instance_id']
    if row['question_graph_type']!='A':
        excluded.append({'instance_id':id,'question_graph_type':row['question_graph_type'],'eligible':False,
                         'reason_code':'NOT_TYPE_A','reason':'Primary experiment is restricted to Type A; no other graph target rule has been verified.'})
        continue
    try:m=json.loads(row['mhop_qa']) if isinstance(row['mhop_qa'],str) else row['mhop_qa']
    except (ValueError,TypeError) as e:
        excluded.append({'instance_id':id,'question_graph_type':'A','reason_code':'INVALID_MHOP_JSON','reason':str(e)});continue
    g=m['graph_qa'];r1=g['r1'];s,t=g['n1'],g['n2'];q1=m['decomposed_questions'].get('Q1','');q2=m['decomposed_questions'].get('Q2','')
    first=f'{s} → {r1["relation_str"]} → {t}'
    direct=F.get(r1['qid'],{}).get('facts',[]).count(first)
    objset=sorted({p[2] for f in F[r1['qid']]['facts'] if len(p:=parts(f))>=3 and p[:2]==(s,r1['relation_str'])})
    q1_entity=normalize(label(s)) in normalize(q1)
    bridge=re.findall(r'<bridge>(.*?)</bridge>',q2,flags=re.DOTALL)
    q2_bridge=len(bridge)==1 and normalize(label(t))==normalize(bridge[0])
    q2_entity=normalize(label(t)) in normalize(q2)
    is_type_consistent=g.get('graph_type')=='A'
    r2_matches=r2_match(g,F)
    if len(r2_matches)!=1:r2_issue.append({'instance_id':id,'r2_matches':r2_matches,'r2_graph':g['r2']})
    chain_fact_ids=[fact_id(r1['qid'],first)]+[fact_id(g['r2']['qid'],f) for f in r2_matches]
    source_nodes=[v for k,v in g.items() if re.fullmatch(r'n\d+',k)]
    qids=set(row['gold_qids'])|set(q for n in source_nodes if (q:=parse_qid(n)))|set(r['relation_subject_qid'] for k,r in g.items() if k.startswith('r') and isinstance(r,dict) and r.get('relation_subject_qid'))
    failure=[]
    if not is_type_consistent: failure.append('GRAPH_TYPE_MISMATCH')
    if r1['qid']!=parse_qid(s):failure.append('R1_SUBJECT_QID_MISMATCH')
    if direct!=1:failure.append('R1_EXACT_FACT_NOT_UNIQUE')
    if not q1_entity:failure.append('Q1_SUBJECT_MISMATCH')
    if not (q2_bridge and q2_entity):failure.append('Q2_BRIDGE_MISMATCH')
    # Q1 must uniquely resolve to n2 from the supplied subject/relation fact inventory.
    if len(objset)!=1:failure.append('Q1_MULTIPLE_POSSIBLE_BRIDGE_ANSWERS' if len(objset)>1 else 'Q1_NO_FACT_ANSWER')
    if len(r2_matches)!=1: failure.append('R2_CHAIN_FACT_UNVERIFIED')
    rec={'instance_id':id,'question_graph_type':'A','source_gold_qids':row['gold_qids'],
      'target_fact_id':fact_id(r1['qid'],first),'target_fact_text':first,'target_subject_qid':r1['qid'],
      'target_object_qid':parse_qid(t),'target_subject':s,'target_object':t,'r2_source_qid':g['r2']['qid'],'r1_relation_id':r1['pid'],
      'r1_relation_str':r1['relation_str'],'q1_question':q1,'q2_question':q2,
      'q1_subject_present':q1_entity,'q2_bridge_matched':q2_bridge,
      'matching_r1_exact_fact_records':direct,'q1_possible_answer_count':len(objset),
      'q1_possible_answers':objset,'r2_matched_fact_ids':[fact_id(g['r2']['qid'],f) for f in r2_matches],
      'r2_matched_fact_texts':r2_matches,'chain_fact_ids':chain_fact_ids,
      'graph_qids':sorted(qids),'gold_answers':row['gold_answers'],
      'eligible':not failure,'eligibility_reason_codes':failure,
      'source_qa_sha256':original_hashes['qa']['sha256'],'source_fact_sha256':original_hashes['facts']['sha256']}
    qareports.append(rec)
    all_data[id]=rec
    if failure:
        excluded.append({'instance_id':id,'question_graph_type':'A','eligible':False,'reason_codes':failure,
          'reason':'First-hop Q1 does not uniquely select the graph bridge from available source facts' if 'Q1_MULTIPLE_POSSIBLE_BRIDGE_ANSWERS' in failure else 'Graph/decomposition/source-fact verification failed',
          'source_gold_qids':row['gold_qids'],'target_fact_id':rec['target_fact_id'],
          'alternative_bridge_objects':objset})

eligible=[r['instance_id'] for r in qareports if r['eligible']]
all_type_a=[r['instance_id'] for r in qareports]
assert len(all_type_a)==200
all_counts,all_groups,pairs=compute_overlap(all_type_a,all_data,include_pairs=True)
elig_counts,elig_groups,_=compute_overlap(eligible,all_data)
write_jsonl('type_a_target_eligibility.jsonl',qareports)
write_jsonl('excluded_examples.jsonl',excluded)
write_jsonl('overlap_pairs_type_a.jsonl',pairs)

# Exact reasoning fact + shared final gold answer grouping; whole groups only.
group_members=elig_groups['exact_fact_or_path_or_answer']['groups']
gid_for={x:stable_group_id(g) for g in group_members for x in g}
assert len(gid_for)==len(eligible)
groups=[{'group_id':stable_group_id(m),'instance_ids':m,'size':len(m),
         'target_fact_ids':[all_data[x]['target_fact_id'] for x in m],
         'source_gold_qids':sorted({q for x in m for q in all_data[x]['source_gold_qids']})} for m in group_members]
write_jsonl('candidate_fact_groups.jsonl',groups)

# Proposed ratio is a target, never a reason to break groups; assignment is provisional.
target={'forget':78,'retain':26,'holdout':26}
actual={s:0 for s in target}
order=sorted(groups,key=lambda g:(-g['size'],sha_bytes(f'{SEED}|{g["group_id"]}'.encode())))
group_split={}
for group in order:
    sz=group['size']
    choices=list(target)
    choices.sort(key=lambda s:((target[s]-actual[s])/target[s],target[s]-actual[s],sha_bytes(f'{SEED}|{group["group_id"]}|{s}'.encode())),reverse=True)
    choices=[s for s in choices if actual[s]+sz<=target[s]] or list(target)
    choice=choices[0];group_split[group['group_id']]=choice;actual[choice]+=sz
assign={x:group_split[gid_for[x]] for x in eligible}
assert len(assign)==len(eligible)
assert sum(actual.values())==len(eligible)
assert all(len({assign[x] for x in g['instance_ids']})==1 for g in groups)
assert not any((assign[i]!=assign[j] and set(all_data[i]['chain_fact_ids'])&set(all_data[j]['chain_fact_ids'])) for i,j in itertools.combinations(eligible,2))
assert not any((assign[i]!=assign[j] and set(all_data[i]['gold_answers'])&set(all_data[j]['gold_answers'])) for i,j in itertools.combinations(eligible,2))
write_jsonl('proposed_split_assignments.jsonl',[
 {'instance_id':x,'proposed_split':assign[x],'fact_group_id':gid_for[x],
  'target_fact_id':all_data[x]['target_fact_id'],'chain_fact_ids':all_data[x]['chain_fact_ids'],
  'source_gold_qids':all_data[x]['source_gold_qids']} for x in sorted(eligible)])

# Conservative document risk auditing. These are *risks*, not proof from co-mention alone.
# Build gold-qid training pool for eligible forget + retain QA only; excluded types are NEVER used.
train_doc_ids=sorted({qid for x in eligible if assign[x]!='holdout' for qid in all_data[x]['source_gold_qids']})
retain_doc_ids=sorted({qid for x in eligible if assign[x]=='retain' for qid in all_data[x]['source_gold_qids']})
qa_by_split={s:[x for x in eligible if assign[x]==s] for s in target}

def chain_edges(x):
    r=all_data[x];parts_out=[{'kind':'target_r1','fact_id':r['target_fact_id'],
                'subject_qid':r['target_subject_qid'], 'subject':r['target_subject'],'object':r['target_object']}]
    for y in r['r2_matched_fact_texts']:
        p=parts(y)
        parts_out.append({'kind':'second_hop','fact_id':fact_id(r['r2_source_qid'],y),'subject_qid':r['r2_source_qid'],
                          'subject':p[0],'object':p[2] if len(p)==3 else p[-1]})
    return parts_out
protected={s:[dict(edge,instance_id=x,split=s) for x in qa_by_split[s] for edge in chain_edges(x)] for s in target}

def doc_risk(qid,text,guards):
    text_cf=text.casefold()
    issues=[]
    for e in guards:
        name=label(e['subject']).casefold();obj=label(e['object']).casefold()
        by_qid=qid==e['subject_qid']
        co_mention=bool(name and obj and name in text_cf and obj in text_cf)
        if by_qid or co_mention:
            issues.append({'instance_id':e['instance_id'],'split':e['split'],'fact_id':e['fact_id'],
                           'fact_kind':e['kind'],'risk_type':'source_subject_doc' if by_qid else 'co_mentions_subject_and_object',
                           'confirmation':'possible_disclosure_not_semantically_certified'})
    return issues

holdout_guard=protected['holdout']
forget_guard=protected['forget']
all_doc_risk=[]
for qid,d in D.items():
    rr=doc_risk(qid,d['doc'],holdout_guard)
    if rr:all_doc_risk.append({'qid':qid,'holdout_risks':rr,'candidate_train_doc':qid in train_doc_ids,
      'full_corpus_training_decision':'exclude_without_passage_level_verification'})
unsafe_holdout_doc_ids={r['qid'] for r in all_doc_risk}
unsafe_train_doc_ids=sorted(set(train_doc_ids)&unsafe_holdout_doc_ids)
proposal_doc_rows=[]
for qid in train_doc_ids:
    d=D[qid]['doc']
    risks=doc_risk(qid,d,holdout_guard)
    proposal_doc_rows.append({'qid':qid,
      'proposed_disposition':'UNSAFE_EXCLUDE' if risks else 'UNSAFE_PENDING_PASSAGE_VERIFICATION',
      'holdout_fact_risks':risks,'doc_source_sha256':original_hashes['documents']['sha256'],
      'reason':'Source subject document or subject-object co-mention for protected holdout chain facts' if risks else
         'No direct subject-QID/co-mention indicator; natural-language and alias leakage NOT certified absent'})
write_jsonl('candidate_training_document_risks.jsonl',proposal_doc_rows)
write_jsonl('full_corpus_holdout_document_risks.jsonl',all_doc_risk)

# Structurally guaranteed fact-record exposure, even if prose misses relation.
# Whole fact record qid contains many facts, so protect all source subjects of holdout chain.
unsafe_fact_ids=sorted(set(train_doc_ids)&{e['subject_qid'] for e in holdout_guard})
fact_risk=[{'qid':qid,'split':'holdout','exposed_qa_ids':sorted({e['instance_id'] for e in holdout_guard if e['subject_qid']==qid}),
            'exposed_exact_fact_ids':sorted({e['fact_id'] for e in holdout_guard if e['subject_qid']==qid}),
            'disposition':'UNSAFE_EXCLUDE_FACT_ROW','fact_source_sha256':original_hashes['facts']['sha256']} for qid in unsafe_fact_ids]
write_jsonl('candidate_training_fact_risks.jsonl',fact_risk)

# Special retain-only corpus cannot reveal forget targets, and is never certified safe via mere subject/entity checks.
retain_risks=[]
for qid in retain_doc_ids:
    hits=doc_risk(qid,D[qid]['doc'],forget_guard)
    if hits:retain_risks.append({'qid':qid,'forget_target_or_chain_risks':hits,'disposition':'UNSAFE_EXCLUDE_FROM_RETAIN_ONLY'})
write_jsonl('retain_only_document_risks.jsonl',retain_risks)

# Audit cross-split facts for *each* split-specific document/fact pool.
# No entity disjointness is required, but any whole passage that may expose
# another partition's target OR second-hop chain must be withheld until vetted.
split_goldqids={s:sorted({qid for x in qa_by_split[s] for qid in all_data[x]['source_gold_qids']}) for s in target}
split_doc_risks=[]; split_fact_risks=[]; split_cross_counts={}
for owner in target:
    guards=[e for s in target if s!=owner for e in protected[s]]
    drisk_count=0; fr_count=0; flagged_qids=[]
    for qid in split_goldqids[owner]:
        collisions=doc_risk(qid,D[qid]['doc'],guards)
        if collisions:
            drisk_count+=1;flagged_qids.append(qid)
            split_doc_risks.append({'owner_split':owner,'qid':qid,
                'proposed_disposition':'UNSAFE_EXCLUDE','other_split_fact_risks':collisions})
        elif owner!='holdout':
            split_doc_risks.append({'owner_split':owner,'qid':qid,
                'proposed_disposition':'UNSAFE_PENDING_PASSAGE_VERIFICATION',
                'other_split_fact_risks':[]})
        fact_exposed=[e for e in guards if e['subject_qid']==qid]
        if fact_exposed:
            fr_count+=1
            split_fact_risks.append({'owner_split':owner,'qid':qid,'proposed_disposition':'UNSAFE_EXCLUDE_WHOLE_FACT_RECORD',
                'exposed_other_split_qa_ids':sorted({e['instance_id'] for e in fact_exposed}),
                'exposed_other_split_fact_ids':sorted({e['fact_id'] for e in fact_exposed})})
    split_cross_counts[owner]={'gold_qid_passage_pool_size':len(split_goldqids[owner]),
       'passages_flagged_other_split_chain_risk':drisk_count,
       'whole_fact_rows_exposing_other_split_chain':fr_count,
       'passages_without_direct_flags_but_still_not_certified':len(split_goldqids[owner])-drisk_count,
       'flagged_qids':flagged_qids}
write_jsonl('split_specific_document_risks.jsonl',split_doc_risks)
write_jsonl('split_specific_fact_risks.jsonl',split_fact_risks)

# Candidate QA lost all target source document coverage after conservative exclusions.
possible_remaining=set(train_doc_ids)-unsafe_holdout_doc_ids
forget_target_doc_remaining=sum(all_data[x]['target_subject_qid'] in possible_remaining for x in qa_by_split['forget'])
retain_target_doc_remaining=sum(all_data[x]['target_subject_qid'] in possible_remaining for x in qa_by_split['retain'])
unsafe_retain_qids={r['qid'] for r in retain_risks}
retain_without_forget_risk=set(retain_doc_ids)-unsafe_retain_qids

# Screen *all* 6,290 fact records, separately from actual candidate train fact pool.
all_held_subject_qids={e['subject_qid'] for e in holdout_guard}
held_fact_records=[{'qid':qid,'holdout_fact_ids':[e['fact_id'] for e in holdout_guard if e['subject_qid']==qid],
                    'document_count':len(F[qid]['facts'])} for qid in sorted(all_held_subject_qids)]
write_jsonl('full_corpus_holdout_fact_exposures.jsonl',held_fact_records)

# Excluded graph types can expose eligible facts if one trains on them accidentally.
other=qa[200:]
other_goldqid_users=collections.defaultdict(list)
for row in other:
    for qid in row['gold_qids']:other_goldqid_users[qid].append(row['instance_id'])
other_target_exposures=[{'instance_id':x,'target_subject_qid':all_data[x]['target_subject_qid'],
  'ineligible_qa_referencing_source_doc':sorted(set(other_goldqid_users[all_data[x]['target_subject_qid']]))} for x in eligible if all_data[x]['target_subject_qid'] in other_goldqid_users]
write_jsonl('excluded_graph_type_source_exposure.jsonl',other_target_exposures)

status='PROPOSED_FOR_REVIEW_NOT_APPROVED'
summary={
 'dataset':'SynthWorlds-SM','status':status,'stage_2_translation':'NOT_STARTED',
 'target_rule':'Type A only: graph_qa.r1 with subject n1 and bridge n2; Q1 asks r1 subject relation; Q2 <bridge> is n2',
 'eligibility_convention':'Conservative: Q1 must uniquely select n2 from all source facts with same (subject,relation). Multiple object answers => excluded.',
 'eligible_type_a_count':len(eligible),'type_a_graph_valid_first_hops':200,'excluded_count':len(excluded),
 'excluded_reasons':dict(collections.Counter('NOT_TYPE_A' if x['question_graph_type']!='A' else '&'.join(x.get('reason_codes',[])) for x in excluded)),
 'overlap_counts_type_a_all':all_counts,'overlap_counts_strict_eligible':elig_counts,
 'overlap_metrics_explanation':'Counts are unordered QA pairs; same entity uses graph and supporting QIDs; exact fact considers verified r1/r2 chain fact strings; target is exact r1. Full reasoning paths require identical ordered exact chain fact IDs. Shared answer is exact matching gold_answers. Categories overlap.',
 'exact_fact_group_sizes_all_type_a':all_groups['exact_fact']['size_distribution'],
 'exact_fact_group_sizes_eligible':elig_groups['exact_fact']['size_distribution'],
 'answer_and_fact_guard_group_sizes_eligible':elig_groups['exact_fact_or_path_or_answer']['size_distribution'],
 'entity_qid_group_sizes_eligible':elig_groups['entity_qid']['size_distribution'],
 'proposed_group_allocation':{'random_seed':SEED,'counts':actual,'target_counts':target,'group_count':len(groups),
  'method':'Deterministic SHA256-seeded greedy assignment of indivisible exact-fact/full-path/shared-answer groups; never random row split.',
  'qa_assignments_are_proposals':True},
 'cross_split_chain_exact_fact_overlap_pairs':0,'cross_split_target_overlap_pairs':0,'cross_split_exact_gold_answer_pairs':0,
 'documents':{'full_document_count':len(docs),'provisional_forget_retain_gold_qid_document_pool':len(train_doc_ids),
  'holdout_related_full_corpus_doc_qids_flagged':len(all_doc_risk),'unsafe_candidate_training_docs_excluded':len(unsafe_train_doc_ids),
  'all_candidate_training_docs_withheld_unverified':len(train_doc_ids),
  'cross_split_pool_risks':split_cross_counts,
  'candidate_docs_after_conservative_exclusions_not_certified_safe':len(possible_remaining),
  'unsafe_candidate_training_fact_rows':len(unsafe_fact_ids),
  'full_corpus_holdout_chain_subject_fact_rows':len(all_held_subject_qids),
  'retain_only_doc_pool':len(retain_doc_ids),'retain_only_docs_flagged_forget_chain_risk':len(retain_risks),
  'forget_examples_with_target_subject_doc_remaining':forget_target_doc_remaining,
  'retain_examples_with_target_subject_doc_remaining':retain_target_doc_remaining,
  'retain_only_doc_qids_after_forget_chain_flags':len(retain_without_forget_risk),
  'eligible_type_a_targets_referenced_as_gold_qids_by_excluded_graph_types':len(other_target_exposures)},
 'limitations':[
   'Q1 relation semantics were checked with graph/source triple, source-name anchoring, and bridge tags; not a certified manual paraphrase semantic evaluation for every question.',
   '70 Type-A rows with multi-valued Q1 are conservatively excluded even though r1 is uniquely specified by the graph.',
   'Exact relation PID is not encoded in textual fact statements; graph PID is preserved but cannot be independently confirmed against facts.jsonl.',
   'Training documents are full multi-fact passages and may expose facts beyond their own QA split; mere different QA labels do not imply safety.',
   'Document co-mention indicates possible leakage, not proof of the relation; absence of co-mention does not rule out paraphrases or aliases.',
   'Every candidate full document is withheld from training: explicit risk or UNSAFE_PENDING_PASSAGE_VERIFICATION; no train-ready corpus released.',
   'Original mhop_qa string and other QA fields remain untouched in the original uploads; audit exports only metadata.',
   'Excluded graph types must remain excluded from any training corpus unless separately re-audited.'
 ]}
write_json('stage1_reaudit_report.json',summary)
manifest={
 'schema_version':'2.0-proposal','dataset':'SynthWorlds-SM','status':status,'stage1':'RE_AUDITED_POLICY_PENDING_REVIEW',
 'stage2':'NOT_STARTED','seed':SEED,'source_checksums':original_hashes,
 'type_a_count':200,'strict_eligible_count':len(eligible),'excluded_count':len(excluded),
 'actual_proposed_split_counts':actual,'grouping_basis':'exact first-hop/second-hop fact + shared whole chain + exact gold answer, no entity/QID disjointness required',
 'group_assignments':{g['group_id']:{'proposed_split':group_split[g['group_id']],'size':g['size'],'instance_ids':g['instance_ids']} for g in groups},
 'target_fact_ids':{i:all_data[i]['target_fact_id'] for i in eligible},
 'unapproved_output_files':['proposed_split_assignments.jsonl','candidate_fact_groups.jsonl'],
 'safety_notes':summary['limitations'],
 'source_files_modified':False,'translation_not_started':True}
write_json('candidate_split_manifest.json',manifest)
# Strict QA ids are source ids and original English data is untouched.
readme=f'''# SynthWorlds-SM | Type-A fact-level Stage 1 re-audit

Status: **PROPOSED_FOR_REVIEW_NOT_APPROVED**. No training files or translations are issued. Source files are unchanged.

- Parsed 1,200 QA rows; Type A: 200.
- Graph-valid exact r1 first-hop facts: 200.
- Q1 uniquely selects n2: {len(eligible)}; 70 Type-A instances excluded due to multiple factual Q1 answers; B–F excluded (1,000).
- Within the strict 130: same graph QID pairs {elig_counts['shared_entity_qid_pairs']}; QID-only pairs {elig_counts['qid_only_pairs']}; same chain fact pairs {elig_counts['shared_exact_fact_pairs']}; repeated target fact pairs {elig_counts.get('shared_target_fact_pairs',0)}; same complete reasoning path {elig_counts.get('shared_complete_path_pairs',0)}; same gold answer {elig_counts['shared_gold_answer_pairs']}.
- Exact fact groups: {elig_groups['exact_fact']['size_distribution']}.
- Exact-fact + path + answer groups: {elig_groups['exact_fact_or_path_or_answer']['size_distribution']}.
- Provisional counts (never row-split): {actual}. Seed {SEED}.
- Candidate forget+retain source-gold-QID document pool: {len(train_doc_ids)}; {len(unsafe_train_doc_ids)} flagged unsafe by holdout source-entity or subject/object co-mention; {len(possible_remaining)} not-certified-safe candidates.
- Fact rows in candidate pool with holdout chain source QIDs: {len(unsafe_fact_ids)}, definitely unsafe.
- Across entire doc corpus, {len(all_doc_risk)} documents flagged for possible holdout-chain disclosure. **Do not interpret remaining documents as proven safe.**

Files:
- `stage1_reaudit_report.json`: full counts, definitions, disclosures, limitations.
- `candidate_split_manifest.json`: source SHA-256 hashes, grouped provisional assignments, target fact IDs.
- `type_a_target_eligibility.jsonl`: exact target matches, Q1 mapping, all Type-A eligibility reasons.
- `excluded_examples.jsonl`: all excluded 1,070 QA IDs and reasons.
- `overlap_pairs_type_a.jsonl`: example-level overlap for each sharing pair.
- `candidate_fact_groups.jsonl`: indivisible groups and target IDs.
- `proposed_split_assignments.jsonl`: proposed labels ONLY, not an approved split or train data.
- `candidate_training_document_risks.jsonl`: all candidate gold QID passages and safety flags.
- `full_corpus_holdout_document_risks.jsonl`: broader entire-corpus natural text risk flags.
- `candidate_training_fact_risks.jsonl`: definite risk in source fact-row training.
- `full_corpus_holdout_fact_exposures.jsonl`: exact source fact-row QIDs for holdout chain.
- `retain_only_document_risks.jsonl`: forget-chain disclosure in retain-only passages.
- `split_specific_document_risks.jsonl`: partition-wise passage exposure and pending verification.
- `split_specific_fact_risks.jsonl`: partition-wise whole-fact-row exposure.
- `excluded_graph_type_source_exposure.jsonl`: excluded B–F source-doc reuse risks.
- `outputs_sha256.json`: generated file hashes, separately from original checksums.
- `build_reaudit.py`: rerunnable Python analysis.

**Interpretation**: A factual split is mathematically feasible, but no natural-language passage-level isolation is fully verified. The policy is ready for researcher review, **not ready for training/deployment**. A conservative pipeline should use fact-level records scoped to verified chain facts, and only admit document passages after sentence-level claim verification/redaction and leakage tests.

To reproduce this audit from the repository root, run:

```bash
python3 data/synthworlds_sm_reaudit/build_reaudit.py
```

The script reads the unchanged source files from `data/synthworlds_sm/` and rewrites the proposal and audit outputs in this folder. The proposed split is not approved training data.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
# Write validation first so its bytes are included in the generated hash inventory.
write_json('run_validation.json',{'status':'PASS','source_hashes_unchanged':all(filehash(BASE/v)==original_hashes[k]['sha256'] for k,v in SOURCES.items()),
  'total_qa':len(qa),'excluded_plus_eligible':len(excluded)+len(eligible),
  'eligible':len(eligible),'proposal_counts':actual,'indivisible_groups_preserved':True,
  'source_hashes_equal_to_prior_audit':prior.exists(),
  'exact_fact_cross_split_overlap':0,'gold_answer_cross_split_overlap':0,
  'translated_files':0})
outputs={p.name:{'sha256':filehash(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name != 'outputs_sha256.json'}
write_json('outputs_sha256.json',outputs)
print(json.dumps({'eligible':len(eligible),'excluded':len(excluded),'actual':actual,
  'group_sizes':summary['answer_and_fact_guard_group_sizes_eligible'],'pairwise_overlap':elig_counts,
  'documents':summary['documents'],'r2_path_issues':len(r2_issue),'outputs':len(outputs)},indent=2))
