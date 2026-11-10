#!/usr/bin/env python3
"""Revalidate fixed, unapproved Type-A split and build exact-fact training candidates.
Original source and audit files are read-only. No translation is fabricated here.
"""
import json,hashlib,collections,re,os,sys,datetime
from pathlib import Path
ROOT=Path('/mnt/data')
AUD=ROOT/'synthworlds_sm_reaudit'
OUT=ROOT/'synthworlds_sm_pilot_v1'
OUT.mkdir(exist_ok=True)
SOURCES={'qa':'qa-sm.jsonl','documents':'qa-sm-docs.jsonl','facts':'qa-sm-facts.jsonl'}
MAN=json.loads((AUD/'candidate_split_manifest.json').read_text())
REPORT=json.loads((AUD/'stage1_reaudit_report.json').read_text())

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return [json.loads(l) for l in open(path,encoding='utf-8') if l.strip()]
def dump(obj,filename):
 p=OUT/filename;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');return p

def jsonl(rows,filename):
 p=OUT/filename
 with p.open('w',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
 return p

def fail(msg):
 dump({'stage':'stopped_source_or_stage1_validation','error':msg},'validation_failure.json')
 raise RuntimeError(msg)

sources={}
for kind,name in SOURCES.items():
 path=ROOT/name
 if not path.is_file():fail(f'Missing source: {name}')
 expected=MAN['source_checksums'][kind]
 actual={'filename':name,'bytes':path.stat().st_size,'sha256':sha(path),'rows':sum(1 for _ in path.open('rb'))}
 sources[kind]=actual
 for k in ['filename','bytes','sha256','rows']:
  if actual[k]!=expected[k]:fail(f'SOURCE INTEGRITY FAILED: {name} {k}: expected {expected[k]!r} actual {actual[k]!r}')
Q=load(ROOT/SOURCES['qa']);F=load(ROOT/SOURCES['facts']);D=load(ROOT/SOURCES['documents'])
E=load(AUD/'type_a_target_eligibility.jsonl');A=load(AUD/'proposed_split_assignments.jsonl')
if len(set(x['instance_id'] for x in Q))!=len(Q):fail('Duplicate QA instance_id')
if len(set(x['qid'] for x in F))!=len(F):fail('Duplicate fact-record QID')
if len(set(x['qid'] for x in D))!=len(D):fail('Duplicate document QID')
q_by_id={x['instance_id']:x for x in Q};f_by_qid={x['qid']:x['facts'] for x in F}
if len(E)!=200 or len(A)!=130:fail('Expected audit 200 Type A cases and 130 proposed eligible assignments')
if len(set(x['instance_id'] for x in E))!=len(E):fail('Duplicate eligibility ID')
if len(set(x['instance_id'] for x in A))!=len(A):fail('Duplicate assignment ID')
if {x['instance_id'] for x in E}!={q['instance_id'] for q in Q if q['question_graph_type']=='A'}:fail('Eligibility misses or adds Type A IDs')
if set(x['instance_id'] for x in A)!={x['instance_id'] for x in E if x['eligible']}:fail('Assignments do not cover precisely eligible IDs')
if sum(x['eligible'] for x in E)!=130 or MAN['strict_eligible_count']!=130:fail('Eligible count mismatch')

def node_qid(node):
 m=re.fullmatch(r'(.*?)\s*\((Q\d+)\)',node)
 return (m.group(1),m.group(2)) if m else (node,None)

def has_target_question(q1,subj):
 # Syntax/subject check: independent fact-count verification below supplies uniqueness.
 return subj.rstrip(".").casefold() in q1.casefold()

def matched_r2(text,n2,n3,relation):
 return text.startswith(n2+' → '+relation+' → '+n3) and (len(text)==len(n2+' → '+relation+' → '+n3) or text[len(n2+' → '+relation+' → '+n3):].startswith(' → '))

E_by={x['instance_id']:x for x in E}
verified={};issue=[]; q1_variants=collections.Counter()
for q in Q:
 iid=q['instance_id']
 if q['question_graph_type']!='A':continue
 ev=E_by[iid]
 try:
  m=json.loads(q['mhop_qa']);g=m['graph_qa'];decomp=m['decomposed_questions']
  if g['graph_type']!='A':fail(f'{iid}: graph type mismatch')
  n1,n2,n3=g['n1'],g['n2'],g['n3'];n1_name,n1_id=node_qid(n1);n2_name,n2_id=node_qid(n2);_,n3_id=node_qid(n3)
  r1=g['r1'];r2=g['r2'];q1=decomp['Q1'];q2=decomp['Q2']
  if r1['qid']!=n1_id or r2['qid']!=ev['r2_source_qid']:fail(f'{iid}: relation subject/source QIDs disagree')
  if set(q['gold_qids'])!={n1_id,ev['r2_source_qid']}:fail(f'{iid}: gold QIDs disagree with graph')
  if not has_target_question(q1,n1_name):fail(f'{iid}: Q1 missing exact n1 synthetic entity name')
  bridges=re.findall(r'<bridge>(.*?)</bridge>',q2)
  if bridges!=[n2_name] and [x.rstrip('.').strip() for x in bridges]!=[n2_name.rstrip('.').strip()]:fail(f'{iid}: Q2 bridge mismatch: {bridges!r}, expected {[n2_name]!r}')
  prefix=n1+' → '+r1['relation_str']+' → '
  direct=[x for x in f_by_qid[n1_id] if x.startswith(prefix)]
  candidates={s[len(prefix):].split(' → ')[0] for s in direct}
  expected_r1=prefix+n2
  r1_matches=[x for x in direct if x==expected_r1]
  r2_matches=[x for x in f_by_qid[ev['r2_source_qid']] if x in ev['r2_matched_fact_texts'] and x.startswith(n2+' → ') and n3 in x]
  is_eligible=(len(candidates)==1 and next(iter(candidates),None)==n2 and len(r1_matches)==1 and len(r2_matches)==1)
  if is_eligible!=ev['eligible']:fail(f'{iid}: independently derived eligibility={is_eligible}, audit says {ev["eligible"]}')
  if ev['q1_possible_answer_count']!=len(candidates):fail(f'{iid}: Q1 possible-answer count mismatch')
  if is_eligible:
   if ev['target_fact_text']!=r1_matches[0]:fail(f'{iid}: target fact text mismatch')
   if ev['target_fact_id']!='derived-sha256:'+hashlib.sha256((n1_id+'\0'+r1_matches[0]).encode('utf-8')).hexdigest():fail(f'{iid}: target fact hash ID mismatch')
   if ev['r2_matched_fact_ids']!=['derived-sha256:'+hashlib.sha256((ev['r2_source_qid']+'\0'+r2_matches[0]).encode('utf-8')).hexdigest()]:fail(f'{iid}: r2 fact hash ID mismatch')
   if ev['r2_matched_fact_texts']!=r2_matches:fail(f'{iid}: support fact text mismatch')
   if ev['matching_r1_exact_fact_records']!=1 or len(ev['r2_matched_fact_ids'])!=1:fail(f'{iid}: fact match cardinality mismatch')
   if ev['source_qa_sha256']!=sources['qa']['sha256'] or ev['source_fact_sha256']!=sources['facts']['sha256']:fail(f'{iid}: audit embedded source hash differs')
   if len(ev['chain_fact_ids'])!=2 or ev['chain_fact_ids']!=[ev['target_fact_id'],ev['r2_matched_fact_ids'][0]]:fail(f'{iid}: chain fact ID mapping differs')
   verified[iid]={'qa':q,'graph':g,'audit':ev,'r1':r1_matches[0],'r2':r2_matches[0], 'nodes':(n1_id,n2_id,n3_id),'relation_ids':(r1['pid'],r2['pid'])}
  else:
   if len(candidates)>1:reason='Q1_MULTIPLE_POSSIBLE_BRIDGE_ANSWERS'
   elif not candidates:reason='Q1_NO_MATCHING_RELATION_FACT'
   else:reason='OTHER_MAPPING_AMBIGUITY'
   if reason not in ev['eligibility_reason_codes']:fail(f'{iid}: exclusion reason mismatch: {reason}')
   issue.append({'instance_id':iid,'reason':reason,'candidate_bridge_objects':sorted(candidates)})
  q1_variants[r1['relation_str']]+=1
 except KeyError as ex:fail(f'{iid}: missing required field {ex}')
if len(verified)!=130 or len(issue)!=70:fail('Unexpected eligibility totals')

# Proposed group mappings must equal file and must keep all exact chain facts and final answer keys together.
assign={x['instance_id']:x for x in A}
G=MAN['group_assignments'];group_lookup={}
for group_id,group in G.items():
 for iid in group['instance_ids']:
  if iid in group_lookup:fail(f'QA {iid} listed in multiple groups')
  group_lookup[iid]=(group_id,group['proposed_split'])
if set(group_lookup)!=set(assign):fail('Manifest group membership not equal to assignment IDs')
for iid,a in assign.items():
 v=verified[iid];ev=v['audit'];group_id,split=group_lookup[iid]
 if a['fact_group_id']!=group_id or a['proposed_split']!=split:fail(f'{iid}: group/split disagreement')
 if a['target_fact_id']!=ev['target_fact_id'] or a['chain_fact_ids']!=ev['chain_fact_ids']:fail(f'{iid}: audit fact ids differ')
 if a['source_gold_qids']!=v['qa']['gold_qids']:fail(f'{iid}: assignment QIDs differ')
 if MAN['target_fact_ids'].get(iid)!=ev['target_fact_id']:fail(f'{iid}: manifest target fact ID mismatch')
 if split not in ('forget','retain','holdout'):fail(f'{iid}: invalid split')

parent={x:x for x in verified}
def find(x):
 while parent[x]!=x:
  parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):parent[find(a)]=find(b)
first_seen={}
for iid,v in verified.items():
 for fid in v['audit']['chain_fact_ids']:
  if fid in first_seen:union(iid,first_seen[fid])
  else:first_seen[fid]=iid
 for ans in v['qa']['gold_answers']:
  k='gold:'+ans
  if k in first_seen:union(iid,first_seen[k])
  else:first_seen[k]=iid
actual_components=collections.defaultdict(set)
for iid in verified:actual_components[find(iid)].add(iid)
manifest_components=[set(gr['instance_ids']) for gr in G.values()]
if {frozenset(c) for c in actual_components.values()}!={frozenset(c) for c in manifest_components}:fail('Indivisible fact/answer group membership does not match independently reconstructed components')
expected=collections.Counter(a['proposed_split'] for a in A)
if expected!=collections.Counter({'forget':78,'retain':26,'holdout':26}) or dict(expected)!=MAN['actual_proposed_split_counts']:fail('Split counts altered or mismatch')
# Independent exact fact cross-split isolation: both fact ID and literal source text.
seen_fid={};seen_fact={};seen_answer={}
for iid,v in verified.items():
 split=assign[iid]['proposed_split'];ev=v['audit']
 for fid,ftext in zip(ev['chain_fact_ids'],(v['r1'],v['r2'])):
  for key,table in [(fid,seen_fid),(ftext,seen_fact)]:
   if key in table and table[key]!=split:fail(f'Cross-split exact fact detected: {iid}, {key}')
   table[key]=split
 for ans in v['qa']['gold_answers']:
  if ans in seen_answer and seen_answer[ans]!=split:fail(f'Cross-split answer detected: {iid}')
  seen_answer[ans]=split

stage1={'status':'VERIFIED_PROPOSAL_NOT_APPROVED','source_checksums':sources,'eligible_type_a_count':len(verified),'excluded_type_a_count':len(issue),'other_graph_types_excluded':len(Q)-len(E), 'proposed_counts':dict(expected),'indivisible_group_size_histogram':dict(sorted(collections.Counter(len(c) for c in actual_components.values()).items())),'indivisible_group_count':len(actual_components),'cross_split_exact_facts':0,'cross_split_target_facts':0,'cross_split_exact_answers':0,'graph_relation_pid_independently_verifiable_from_source_facts':False,'limitations':['Source facts contain relation strings but not PIDs; relation ID provenance is the parsed graph metadata.','Q1 relation semantics were checked against the prior audit and exact source relation cardinality; automated checks cannot fully establish semantic entailment for every paraphrase.','The 78/26/26 assignment is candidate-only and not approved.']}
dump(stage1,'stage1_verification_report.json')

# Fact statements are unique training units, not whole facts rows or documents.
# Preserve every qualifier and keep all uses as metadata. Holdout chain exact facts are forbidden.
holdout_fact_text={v[t] for iid,v in verified.items() if assign[iid]['proposed_split']=='holdout' for t in ['r1','r2']}
holdout_fids={fid for iid,v in verified.items() if assign[iid]['proposed_split']=='holdout' for fid in v['audit']['chain_fact_ids']}
records={};excluded_holdout=[]
for iid,v in verified.items():
 split=assign[iid]['proposed_split'];ev=v['audit'];n1,n2,n3=v['nodes']
 if split=='holdout':
  excluded_holdout.append(iid);continue
 for hop,fid,full_text,source_qid,sid,oid,pid in [
  ('r1',ev['target_fact_id'],v['r1'],n1,n1,n2,v['relation_ids'][0]),
  ('r2',ev['r2_matched_fact_ids'][0],v['r2'],ev['r2_source_qid'],n2,n3,v['relation_ids'][1])]:
  if fid in holdout_fids or full_text in holdout_fact_text:fail(f'Forbidden holdout fact found in training: {iid} {fid}')
  parts=full_text.split(' → ')
  if len(parts)<3:fail(f'Unsupported fact syntax for {iid} {full_text}')
  base=parts[0]+' → '+parts[1]+' → '+parts[2]
  if fid not in records:
   records[fid]={'fact_id':fid,'fact_en':full_text,'training_text_en':full_text,'source_fact_record_qid':source_qid,'subject_entity_qid':sid,'object_entity_qid':oid,'relation_en':parts[1],'qualified_segments_en':parts[3:],'split':split,'uses':[],'source_qa_ids':[],'target_fact_ids':[],'fact_group_ids':[],'hops':[],'relation_ids':[],'status':'SFT_CANDIDATE_NOT_APPROVED'}
  rec=records[fid]
  if rec['fact_en']!=full_text or rec['split']!=split or rec['source_fact_record_qid']!=source_qid or rec['subject_entity_qid']!=sid or rec['object_entity_qid']!=oid:fail(f'Conflicting metadata for repeated fact ID {fid}')
  use={'source_qa_id':iid,'target_fact_id':ev['target_fact_id'],'group_id':assign[iid]['fact_group_id'],'proposed_split':split,'relation_id':pid,'hop':hop,'subject_entity_qid':sid,'object_entity_qid':oid}
  rec['uses'].append(use)
  rec['source_qa_ids'].append(iid);rec['target_fact_ids'].append(ev['target_fact_id']);rec['fact_group_ids'].append(assign[iid]['fact_group_id']);rec['hops'].append(hop);rec['relation_ids'].append(pid)
for r in records.values():
 for field in ['source_qa_ids','target_fact_ids','fact_group_ids','hops','relation_ids']:r[field]=sorted(set(r[field]))
 r['uses'].sort(key=lambda x:(x['source_qa_id'],x['hop']))
 # Ensures nobody confuses fact object with independent evaluation answer labels.
 if any(k in r for k in ['query','gold_answers','gold_answer','mhop_qa','gold_docs']):fail('Prohibited evaluation data in SFT record')
FACTS=sorted(records.values(),key=lambda r:(r['split'],r['fact_id']))
jsonl(FACTS,'pilot_fact_sft_en.jsonl')
jsonl([{'instance_id':iid,'proposed_split':a['proposed_split'],'fact_group_id':a['fact_group_id'],'target_fact_id':a['target_fact_id'],'r1_fact_id':verified[iid]['audit']['target_fact_id'],'r2_fact_id':verified[iid]['audit']['r2_matched_fact_ids'][0]} for iid,a in sorted(assign.items())],'verified_type_a_qa_index.jsonl')
jsonl(issue,'excluded_ambiguous_type_a.jsonl')
qa_counts=collections.Counter(x['proposed_split'] for x in A)
rec_counts=collections.Counter(x['split'] for x in FACTS)
if sum(len(x['uses']) for x in FACTS)!=2*(qa_counts['forget']+qa_counts['retain']):fail('Missing candidate fact use')
assert not set(x['fact_id'] for x in FACTS)&holdout_fids
stage2={'status':'SFT_CANDIDATE_NOT_APPROVED','source_qa_count':qa_counts['forget']+qa_counts['retain'],'fact_use_count':sum(len(x['uses']) for x in FACTS),'unique_fact_record_count':len(FACTS),'fact_record_counts':dict(rec_counts),'holdout_qa_count_excluded':len(excluded_holdout),'holdout_chain_fact_id_count_excluded':len(holdout_fids),'cross_split_exact_fact_collisions':0,'exact_fact_source_uniqueness':'1 matched entry per canonical source row / eligible hop','whole_docs_or_whole_fact_rows_used':False,'multihop_queries_or_gold_answer_fields_used':False,'limitation':'The r2 fact necessarily contains the answer entity as its object; no separate multi-hop gold-answer field or evaluation question is included.','verification':'Fact triple and any qualifiers copied verbatim from source facts, matched against graph and prior audit.'}
dump(stage2,'stage2_fact_level_report.json')
(OUT/'validation_failure.json').unlink(missing_ok=True)
print(json.dumps({'stage1':stage1,'stage2':stage2},ensure_ascii=False,indent=2)[:6000])
