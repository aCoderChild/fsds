#!/usr/bin/env python3
"""Build private NLLB back-translation for the masked 15-question probe."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / "kaggle_assistant_review/review.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
raw = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "raw" for t in n.targets))
assert len(json.loads(raw)) == 15
digest = hashlib.sha256(raw.encode()).hexdigest()
code = '''import hashlib, json, re, sys, subprocess
from pathlib import Path
subprocess.run([sys.executable,"-m","pip","install","-q","transformers>=4.45,<4.52","sentencepiece","accelerate>=0.34,<2"],check=True)
import torch, transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
assert torch.cuda.is_available(), "GPU required"
model_id="facebook/nllb-200-distilled-1.3B"
revision="7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
raw=__RAW__
assert hashlib.sha256(raw.encode()).hexdigest()=="__HASH__"
rows=json.loads(raw)
langs={"fr":"fra_Latn","de":"deu_Latn","hi":"hin_Deva","th":"tha_Thai"}
model=AutoModelForSeq2SeqLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16).cuda().eval()
out=[]
path=Path('/kaggle/working/backtranslations.json')
for row in rows:
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,src_lang=langs[row['language']])
    marker=r"\\[(?:ENTITY|NUMBER)_[A-Z]+\\]"
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
    path.write_text(json.dumps({"status":"DIAGNOSTIC_BACKTRANSLATION_ONLY","input_sha256":"__HASH__","model_id":model_id,"model_revision":revision,"torch":torch.__version__,"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(0),"records":out},ensure_ascii=False,indent=2))
    print('Back-translated',len(out),'/',len(rows),flush=True)
'''.replace('__RAW__',repr(raw)).replace('__HASH__',digest)
out = HERE / "kaggle_assistant_backtranslation"
out.mkdir(exist_ok=True)
(out / "backtranslate.py").write_text(code, encoding="utf-8")
metadata = {"id":"bunnoob2005/synthworlds-assistant-backtranslation-probe", "title":"SynthWorlds assistant backtranslation probe", "code_file":"backtranslate.py", "language":"python", "kernel_type":"script", "is_private":"true", "enable_gpu":"true", "enable_internet":"true", "dataset_sources":[], "competition_sources":[], "kernel_sources":[], "model_sources":[]}
(out / "kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(json.dumps({"input_sha256":digest,"script_sha256":hashlib.sha256(code.encode()).hexdigest(),"output":str(out)},indent=2))
