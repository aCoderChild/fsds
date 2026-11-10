import hashlib, json, re, sys, subprocess
from pathlib import Path
subprocess.run([sys.executable,"-m","pip","install","-q","transformers>=4.45,<4.52","sentencepiece","accelerate>=0.34,<2"],check=True)
import torch, transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
assert torch.cuda.is_available(), "GPU required"
model_id="facebook/nllb-200-distilled-1.3B"
revision="7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
raw='[{"language":"fr","kind":"atomic","prompt_id":"0050-qa-sm:r1","source":"Who is [ENTITY_A] the godparent of?","translation":"De qui [ENTITY_A] est-il le parrain ou la marraine ?"},{"language":"fr","kind":"composition","prompt_id":"disjoint-b:0348-qa-sm:q2","source":"Who is [ENTITY_A]\'s stepparent?","translation":"Qui est le beau-parent d’[ENTITY_A] ?"},{"language":"de","kind":"atomic","prompt_id":"0169-qa-sm:r1","source":"Who is the editor of [ENTITY_A]?","translation":"Wer ist der Herausgeber oder die Herausgeberin von [ENTITY_A]?"},{"language":"de","kind":"atomic","prompt_id":"0196-qa-sm:r1","source":"Who was a doctoral student of [ENTITY_A]?","translation":"Wer war Doktorand oder Doktorandin von [ENTITY_A]?"},{"language":"de","kind":"atomic","prompt_id":"0152-qa-sm:r2","source":"In what work is [ENTITY_A] described?","translation":"In welchem Werk wird [ENTITY_A] beschrieben?"},{"language":"de","kind":"composition","prompt_id":"disjoint-b:0238-qa-sm:multi","source":"Which political office of the state that replaced the country whose capital is [ENTITY_A] is held by its head of state?","translation":"Welches politische Amt des Staates, der das Land mit der Hauptstadt [ENTITY_A] ablöste, wird von dessen Staatsoberhaupt bekleidet?"},{"language":"de","kind":"composition","prompt_id":"disjoint-b:0203-qa-sm:q3","source":"Who is [ENTITY_A]\'s stepparent?","translation":"Wer ist der Stiefelternteil von [ENTITY_A]?"},{"language":"hi","kind":"atomic","prompt_id":"0072-qa-sm:r1","source":"What is on the boundary of [ENTITY_A]?","translation":"[ENTITY_A] की सीमा पर क्या है?"},{"language":"hi","kind":"atomic","prompt_id":"0078-qa-sm:r2","source":"In which work is [ENTITY_A] described?","translation":"किस कृति में [ENTITY_A] का वर्णन किया गया है?"},{"language":"hi","kind":"composition","prompt_id":"disjoint-b:0300-qa-sm:q2","source":"Who is [ENTITY_A]\'s godparent?","translation":"[ENTITY_A] के धर्म-माता या धर्म-पिता कौन हैं?"},{"language":"th","kind":"atomic","prompt_id":"0044-qa-sm:r2","source":"On which physical feature is [ENTITY_A] located?","translation":"[ENTITY_A] ตั้งอยู่บนลักษณะภูมิประเทศใด?"},{"language":"th","kind":"atomic","prompt_id":"0152-qa-sm:r1","source":"What honorific prefix is used for [ENTITY_A]?","translation":"คำยกย่องที่ใช้เป็นคำนำหน้าชื่อของ [ENTITY_A] คืออะไร?"},{"language":"th","kind":"atomic","prompt_id":"0055-qa-sm:r1","source":"Who performed [ENTITY_A]?","translation":"ใครเป็นผู้แสดง [ENTITY_A]?"},{"language":"th","kind":"composition","prompt_id":"disjoint-b:0382-qa-sm:multi","source":"Who is the father of the creator of the noble title held by [ENTITY_A]?","translation":"บิดาของผู้สร้างบรรดาศักดิ์ที่ [ENTITY_A] ถือครองคือใคร?"},{"language":"th","kind":"composition","prompt_id":"disjoint-b:0300-qa-sm:q2","source":"Who is [ENTITY_A]\'s godparent?","translation":"พ่อทูนหัวหรือแม่ทูนหัวของ [ENTITY_A] คือใคร?"}]'
assert hashlib.sha256(raw.encode()).hexdigest()=="38ff32f0036ce23e2a2daf0fa3c3d12fcf692eb3454d2ee45a6336e2f6d000ee"
rows=json.loads(raw)
langs={"fr":"fra_Latn","de":"deu_Latn","hi":"hin_Deva","th":"tha_Thai"}
model=AutoModelForSeq2SeqLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16).cuda().eval()
out=[]
path=Path('/kaggle/working/backtranslations.json')
for row in rows:
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,src_lang=langs[row['language']])
    marker=r"\[(?:ENTITY|NUMBER)_[A-Z]+\]"
    placeholders=list(dict.fromkeys(re.findall(marker,row['translation'])))
    surrogates=['Zorvani','Kelvoria','Miradon','Tavrelin','Vornesia','Peldorin','Raviona','Selvarin']
    if len(placeholders)>len(surrogates) or any(s in row['translation'] for s in surrogates):
        raise ValueError('Surrogate collision or too many placeholders')
    substitutions=dict(zip(placeholders,surrogates))
    prepared=row['translation']
    for key,value in substitutions.items(): prepared=prepared.replace(key,value)
    batch=tokenizer(prepared,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        ids=model.generate(**batch,forced_bos_token_id=tokenizer.convert_tokens_to_ids('eng_Latn'),max_new_tokens=160,num_beams=4,do_sample=False)
    raw_translation=tokenizer.batch_decode(ids,skip_special_tokens=True)[0]
    translated=raw_translation
    for key,value in substitutions.items(): translated=translated.replace(value,key)
    marker_ok=sorted(re.findall(marker,translated))==sorted(re.findall(marker,row['translation']))
    out.append({"language":row['language'],"kind":row['kind'],"prompt_id":row['prompt_id'],"substitutions":substitutions,"raw_backtranslation":raw_translation,"backtranslation":translated,"marker_integrity":marker_ok})
    path.write_text(json.dumps({"status":"DIAGNOSTIC_BACKTRANSLATION_ONLY","input_sha256":"38ff32f0036ce23e2a2daf0fa3c3d12fcf692eb3454d2ee45a6336e2f6d000ee","model_id":model_id,"model_revision":revision,"torch":torch.__version__,"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(0),"records":out},ensure_ascii=False,indent=2))
    print('Back-translated',len(out),'/',len(rows),flush=True)
