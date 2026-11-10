#!/usr/bin/env python3
"""Manually authored, reviewed-in-context translation seed batch. No MT API.
Run after build_pilot.py; preserves original source and all existing metadata.
Outputs only complete translations, never auto-populates unsupported IDs.
"""
import json,hashlib,re,collections,copy,datetime,zipfile
from pathlib import Path
ROOT=Path('/mnt/data');OUT=ROOT/'synthworlds_sm_pilot_v1';B=OUT/'batch_0001';B.mkdir(exist_ok=True)
LANGS=('fr','de','hi','th')

def loadj(p):return [json.loads(s) for s in open(p,encoding='utf-8') if s.strip()]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(x,p):p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def write_jsonl(rows,p):
 with open(p,'w',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')

Q={x['instance_id']:x for x in loadj(ROOT/'qa-sm.jsonl')}
D={x['qid']:x for x in loadj(ROOT/'qa-sm-docs.jsonl')}
F={x['qid']:x for x in loadj(ROOT/'qa-sm-facts.jsonl')}
P={x['fact_id']:x for x in loadj(OUT/'pilot_fact_sft_en.jsonl')}
S=loadj(OUT/'verified_type_a_qa_index.jsonl');A={x['instance_id']:x for x in S}
R1=A['0074-qa-sm']['r1_fact_id'];R2=A['0074-qa-sm']['r2_fact_id']
qa_id='0074-qa-sm';doc_ids=['Q41329','Q25989'];fact_qid='Q25989';pilot_ids=[R1,R2]
T={
'fr':{
 'qa': 'Où se trouve le fuseau horaire dans lequel se situe Solmira ?',
 'q1': 'Dans quel fuseau horaire se trouve Solmira ?',
 'q2': 'Où se trouve <bridge>Meridianflux</bridge> ?',
 'docs':{
  'Q25989':'Meridianflux est un fuseau horaire situé en Lunarisia, qui est une région géographique.',
  'Q41329':'''Solmira est une grande ville qui fait également office de district, de municipalité et de ville statutaire. Le chef de son gouvernement occupe la fonction de maire.

La plus ancienne mention écrite de Solmira date de 838 CE. La ville se trouve dans le fuseau horaire Meridianflux.

Solmira est membre de Global Connectivity Forum, une organisation internationale non gouvernementale ; son adhésion a commencé en 2031. Elle entretient des jumelages avec plusieurs autres municipalités : Valyndor (depuis 2022-07-06), Thalmaris (depuis 2031-06-03), Solenreach (depuis 2032-10-28) et Rivenwood (depuis 2034).

Solmira est décrite dans plusieurs sources publiées, notamment CartoSphere, un répertoire géographique publié en 2042 ; Dreamt Compilation, une édition publiée ; HoloGrid Study Group, une collection et un projet de recherche ; Legendarium, une œuvre écrite ; Qylarans, une encyclopédie consacrée aux groupes ethnoreligieux et aux nations ; RefereX, une encyclopédie ; et The Midnight Library, une œuvre littéraire.'''
 },
 'rel_r1':'se trouve dans le fuseau horaire','rel_r2':'localisation'
},
'de':{
 'qa':'Wo liegt die Zeitzone, in der sich Solmira befindet?',
 'q1':'In welcher Zeitzone liegt Solmira?',
 'q2':'Wo befindet sich <bridge>Meridianflux</bridge>?',
 'docs':{
  'Q25989':'Meridianflux ist eine Zeitzone in Lunarisia, einer geografischen Region.',
  'Q41329':'''Solmira ist eine Großstadt, die zugleich als Bezirk, Gemeinde und Statutarstadt fungiert. Das Oberhaupt ihrer Stadtregierung bekleidet das Amt des Bürgermeisters.

Die früheste schriftliche Erwähnung von Solmira stammt aus dem Jahr 838 CE. Die Stadt liegt in der Zeitzone Meridianflux.

Solmira ist Mitglied des Global Connectivity Forum, einer internationalen Nichtregierungsorganisation; die Mitgliedschaft begann im Jahr 2031. Die Stadt pflegt Städtepartnerschaften mit mehreren anderen Gemeinden: Valyndor (seit 2022-07-06), Thalmaris (seit 2031-06-03), Solenreach (seit 2032-10-28) und Rivenwood (seit 2034).

Solmira wurde in verschiedenen veröffentlichten Quellen beschrieben, darunter CartoSphere, ein 2042 veröffentlichtes Ortsverzeichnis; Dreamt Compilation, eine veröffentlichte Ausgabe; HoloGrid Study Group, eine Sammlung und ein Forschungsprojekt; Legendarium, ein schriftliches Werk; Qylarans, eine Enzyklopädie über ethnoreligiöse Gruppen und Nationen; RefereX, eine Enzyklopädie; sowie The Midnight Library, ein literarisches Werk.'''
 },
 'rel_r1':'liegt in der Zeitzone','rel_r2':'Standort'
},
'hi':{
 'qa':'Solmira जिस समय क्षेत्र में स्थित है, वह कहाँ स्थित है?',
 'q1':'Solmira किस समय क्षेत्र में स्थित है?',
 'q2':'<bridge>Meridianflux</bridge> कहाँ स्थित है?',
 'docs':{
  'Q25989':'Meridianflux एक समय क्षेत्र है जो Lunarisia में स्थित है, जो एक भौगोलिक क्षेत्र है।',
  'Q41329':'''Solmira एक बड़ा शहर है, जो एक ज़िले, एक नगरपालिका और एक वैधानिक शहर के रूप में भी कार्य करता है। इसकी सरकार का प्रमुख महापौर का पद संभालता है।

Solmira का सबसे पुराना लिखित उल्लेख 838 CE का है। यह शहर Meridianflux समय क्षेत्र में स्थित है।

Solmira, Global Connectivity Forum का सदस्य है, जो एक अंतरराष्ट्रीय गैर-सरकारी संगठन है; इसकी सदस्यता 2031 में शुरू हुई। इसका कई अन्य नगरपालिकाओं के साथ जुड़वाँ नगर संबंध है: Valyndor (2022-07-06 से), Thalmaris (2031-06-03 से), Solenreach (2032-10-28 से) और Rivenwood (2034 से)।

Solmira का उल्लेख कई प्रकाशित स्रोतों में किया गया है, जिनमें CartoSphere, 2042 में प्रकाशित एक भौगोलिक स्थान-सूची; Dreamt Compilation, एक प्रकाशित संस्करण; HoloGrid Study Group, एक संग्रह और शोध परियोजना; Legendarium, एक लिखित कृति; Qylarans, जातीय-धार्मिक समूहों और राष्ट्रों पर आधारित एक विश्वकोश; RefereX, एक विश्वकोश; और The Midnight Library, एक साहित्यिक कृति शामिल हैं।'''
 },
 'rel_r1':'समय क्षेत्र में स्थित है','rel_r2':'स्थान'
},
 'th':{
 'qa':'เขตเวลาที่ Solmira อยู่ตั้งอยู่ที่ใด?',
 'q1':'Solmira อยู่ในเขตเวลาใด?',
 'q2':'<bridge>Meridianflux</bridge> ตั้งอยู่ที่ใด?',
 'docs':{
  'Q25989':'Meridianflux เป็นเขตเวลาที่ตั้งอยู่ใน Lunarisia ซึ่งเป็นภูมิภาคทางภูมิศาสตร์',
  'Q41329':'''Solmira เป็นเมืองขนาดใหญ่ที่ทำหน้าที่เป็นทั้งเขต เทศบาล และเมืองที่มีสถานะตามกฎหมาย หัวหน้ารัฐบาลของเมืองดำรงตำแหน่งนายกเทศมนตรี

บันทึกเป็นลายลักษณ์อักษรที่เก่าแก่ที่สุดเกี่ยวกับ Solmira มีอายุย้อนไปถึงปี 838 CE เมืองนี้ตั้งอยู่ในเขตเวลา Meridianflux

Solmira เป็นสมาชิกของ Global Connectivity Forum ซึ่งเป็นองค์กรพัฒนาเอกชนระหว่างประเทศ โดยเริ่มเป็นสมาชิกในปี 2031 เมืองนี้มีความสัมพันธ์แบบเมืองพี่เมืองน้องกับเทศบาลอื่นหลายแห่ง ได้แก่ Valyndor (ตั้งแต่ 2022-07-06), Thalmaris (ตั้งแต่ 2031-06-03), Solenreach (ตั้งแต่ 2032-10-28) และ Rivenwood (ตั้งแต่ปี 2034)

Solmira ได้รับการกล่าวถึงในแหล่งข้อมูลที่เผยแพร่หลายแห่ง รวมถึง CartoSphere ซึ่งเป็นสารบบภูมิศาสตร์ที่เผยแพร่ในปี 2042; Dreamt Compilation ซึ่งเป็นฉบับตีพิมพ์; HoloGrid Study Group ซึ่งเป็นทั้งชุดสะสมและโครงการวิจัย; Legendarium ซึ่งเป็นงานเขียน; Qylarans ซึ่งเป็นสารานุกรมเกี่ยวกับกลุ่มชาติพันธุ์และศาสนาและประเทศต่าง ๆ; RefereX ซึ่งเป็นสารานุกรม; และ The Midnight Library ซึ่งเป็นงานวรรณกรรม'''
 },
 'rel_r1':'ตั้งอยู่ในเขตเวลา','rel_r2':'ที่ตั้ง'
 }
}

# A strict entity/number/date/tag audit of the seed translations, not an oracle for semantic quality.
ENTITY_LITERALS=['Solmira','Meridianflux','Lunarisia','Global Connectivity Forum','Valyndor','Thalmaris','Solenreach','Rivenwood','CartoSphere','Dreamt Compilation','HoloGrid Study Group','Legendarium','Qylarans','RefereX','The Midnight Library']
required_iso=re.findall(r'\b\d{4}-\d{2}-\d{2}\b',D['Q41329']['doc'])
required_nums=re.findall(r'\b\d{3,4}\b',D['Q41329']['doc'])
output_files=[]
for lang in LANGS:
 t=T[lang]
 for qid in doc_ids:
  orig=D[qid]['doc'];translated=t['docs'][qid]
  if not translated:raise AssertionError((lang,qid,'empty translation'))
  for name in ENTITY_LITERALS:
   if name in orig and name not in translated:raise AssertionError((lang,qid,'lost synthetic entity',name))
  if collections.Counter(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',orig))!=collections.Counter(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',translated)):
   raise AssertionError((lang,qid,'date mismatch'))
  if collections.Counter(re.findall(r'(?<![\d-])\d{3,4}(?![\d-])',orig))!=collections.Counter(re.findall(r'(?<![\d-])\d{3,4}(?![\d-])',translated)):
   raise AssertionError((lang,qid,'number mismatch'))
 docs=[]
 for qid in doc_ids:
  o=copy.deepcopy(D[qid]);o['doc_'+lang]=t['docs'][qid];docs.append(o)
 docmap={x['qid']:x['doc_'+lang] for x in docs}
 qa=copy.deepcopy(Q[qa_id]);decomp=json.loads(qa['mhop_qa'])['decomposed_questions']
 assert list(decomp)==['Q1','Q2']
 assert qa['gold_qids']==doc_ids
 assert all(qa['gold_docs'][i]==D[qid]['doc'] for i,qid in enumerate(qa['gold_qids']))
 qa['query_'+lang]=t['qa'];qa['decomposed_questions_'+lang]={'Q1':t['q1'],'Q2':t['q2']};qa['gold_docs_'+lang]=[docmap[qid] for qid in qa['gold_qids']]
 qa['proposed_split']=A[qa_id]['proposed_split'];qa['fact_group_id']=A[qa_id]['fact_group_id'];qa['translation_notes']=[]
 if any(z not in qa['query_'+lang] for z in ['Solmira']):raise AssertionError((lang,'QA entity preservation'))
 if '<bridge>Meridianflux</bridge>' not in t['q2']:raise AssertionError((lang,'bridge preservation'))
 assert all(qa[k]==Q[qa_id][k] for k in Q[qa_id])
 factrow=copy.deepcopy(F[fact_qid]);assert len(factrow['facts'])==1 and factrow['facts'][0]==P[R2]['fact_en']
 parts=factrow['facts'][0].split(' → ');assert len(parts)==3 and parts[1]=='location'
 factrow['facts_'+lang]=[' → '.join([parts[0],t['rel_r2'],parts[2]])];factrow['translation_notes']=[]
 translated_pilot=[]
 for fid in pilot_ids:
  base=copy.deepcopy(P[fid]);parts=base['fact_en'].split(' → ')
  assert len(parts)==3
  trrel=t['rel_r1'] if fid==R1 else t['rel_r2']
  base['fact_'+lang]=' → '.join([parts[0],trrel,parts[2]])
  base['training_text_'+lang]=base['fact_'+lang]
  base['relation_'+lang]=trrel;base['translation_notes']=[]
  assert [parts[0],parts[2]]==[base['fact_'+lang].split(' → ')[0],base['fact_'+lang].split(' → ')[2]]
  translated_pilot.append(base)
 files=[(f'qa-sm_{lang}_batch0001.jsonl',[qa]),(f'qa-sm-docs_{lang}_batch0001.jsonl',docs),(f'qa-sm-facts_{lang}_batch0001.jsonl',[factrow]),(f'pilot_fact_sft_{lang}_batch0001.jsonl',translated_pilot)]
 for name,rows in files:
  p=B/name;write_jsonl(rows,p);assert len(loadj(p))==len(rows);output_files.append(p)

all_ids={'qa':[x['instance_id'] for x in Q.values()],'documents':[x['qid'] for x in D.values()],'facts':[x['qid'] for x in F.values()],'pilot_fact_sft':[x['fact_id'] for x in P.values()]}
completed={'qa':[qa_id],'documents':sorted(doc_ids),'facts':[fact_qid],'pilot_fact_sft':sorted(pilot_ids)}
input_manifest=json.loads((OUT/'stage1_verification_report.json').read_text())
statuses={}
for lang in LANGS:
 statuses[lang]={}
 for category,ids in all_ids.items():
  pending=sorted(set(ids)-set(completed[category]))
  statuses[lang][category]={'completed_ids':sorted(completed[category]),'pending_ids':pending,'completed_count':len(completed[category]),'pending_count':len(pending),'total_ids':len(ids),'status':'PARTIAL_NOT_APPROVED'}
outputs={p.name:{'relative_path':str(p.relative_to(OUT)),'sha256':sha(p),'rows':sum(1 for _ in open(p,encoding='utf-8')),'size_bytes':p.stat().st_size} for p in output_files}
for name in ['pilot_fact_sft_en.jsonl','stage1_verification_report.json','stage2_fact_level_report.json','verified_type_a_qa_index.jsonl','excluded_ambiguous_type_a.jsonl']:
 p=OUT/name;outputs[name]={'relative_path':name,'sha256':sha(p),'rows':sum(1 for _ in open(p,encoding='utf-8')) if name.endswith('jsonl') else None,'size_bytes':p.stat().st_size}
manifest={'schema_version':'1.0-batched-translation','status':'PARTIAL_TRANSLATION_SFT_CANDIDATE_NOT_APPROVED','all_source_english_fields_unchanged':True,'translation_model_method':'GPT-6, in-conversation human-readable authored translations for seed batch 0001; no external translation API or automated full-dataset translation; structural Python validation','source_checksums':input_manifest['source_checksums'],'prior_audit_sha256':{name:sha(ROOT/'synthworlds_sm_reaudit'/name) for name in ['candidate_split_manifest.json','type_a_target_eligibility.jsonl','proposed_split_assignments.jsonl','stage1_reaudit_report.json']},'split_assignments_proposed_not_approved':{x['instance_id']:{'split':x['proposed_split'],'group_id':x['fact_group_id'],'target_fact_id':x['target_fact_id']} for x in S},'eligible_type_a_ids':sorted(A),'excluded_type_a_ids':sorted(z['instance_id'] for z in loadj(OUT/'excluded_ambiguous_type_a.jsonl')),'excluded_other_graph_type_ids':sorted(set(Q)-{z['instance_id'] for z in loadj(ROOT/'synthworlds_sm_reaudit/type_a_target_eligibility.jsonl')}),'holdout_fact_ids':sorted({fid for iid,x in A.items() if x['proposed_split']=='holdout' for fid in [x['r1_fact_id'],x['r2_fact_id']]}),'source_pilot_fact_ids':sorted(P),'translations':statuses,'batches':[{'batch_id':'0001','completed_ids_per_language':completed,'notes':'Both canonical gold_docs for QA 0074 are derived from the translated qid documents; no duplicate passage translation.'}],'output_files':outputs,'do_not_train':True}
write_json(manifest,OUT/'progress_manifest.json')
report={'stage1':'PASSED with unapproved assignments','stage2':'206 unique verified exact-fact SFT candidates; 104 QA; excluded 26 holdout QA','stage3':'PARTIAL: seed batch 0001 only; all other records pending','by_language':{lang:{cat:{'completed':d['completed_count'],'pending':d['pending_count']} for cat,d in statuses[lang].items()} for lang in LANGS},'semantic_review_recommended':True,'human_review_items':['Translations in batch 0001: verify idiomatic correctness of the four target languages, especially multilingual meaning of municipality/statutory city, time zone, and location relation.','Automatic structural checks do not prove semantic equivalence.','Missing translations for all pending IDs; no remaining IDs treated as translated.','Source relation PIDs exist in QA graph metadata only, not the facts text.'],'validation_failures':[],'non_blocking_source_format_notes':['QA 0017-qa-sm has a bridge surface name missing a trailing period present in the graph node, but graph-QID/first-hop fact uniquely identifies the bridge.','QA 0098-qa-sm omits a trailing period from the Q1 surface name; source graph and first-hop fact uniquely identify the same entity.','Some graph nodes are literals with no QID; canonical entity QIDs remain null rather than invented.'],'disallowed_training_data_sources':['full document passages','full per-QID facts rows','composed multi-hop QA question','evaluation gold_answers'],'status':'CANDIDATE_READY_FOR_REVIEW_ONLY'}
write_json(report,OUT/'pilot_audit_report.json')
for name in ['pilot_audit_report.json','build_pilot.py','translate_batch_0001.py']:
 p=OUT/name;manifest['output_files'][name]={'relative_path':name,'sha256':sha(p),'rows':None,'size_bytes':p.stat().st_size}
write_json(manifest,OUT/'progress_manifest.json')
# Portable, reproducible read-only output package. Include all relevant unmodified audit inputs for traceability, not all original source files.
with zipfile.ZipFile(ROOT/'synthworlds_sm_pilot_v1_partial.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='validation_failure.json':z.write(p,p.relative_to(OUT.parent))
print(json.dumps({'batch':completed,'counts':report['by_language']['fr'],'file_count':len(outputs),'archive_sha256':sha(ROOT/'synthworlds_sm_pilot_v1_partial.zip')},ensure_ascii=False,indent=2))
