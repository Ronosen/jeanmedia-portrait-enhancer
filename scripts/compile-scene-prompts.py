"""Compile fixed ASCII CLIP prompt vectors; text weights are not shipped."""
import sys,json,re,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'worker'))
from scene_intent import prompts,prompt_hash
from runtime import session
root=ROOT/'qa/scene-intent';tok=json.loads((root/'tokenizer.json').read_text(encoding='utf-8'))['model'];vocab=tok['vocab']
ranks={tuple(m.split()):i for i,m in enumerate(tok['merges'])}
def encode(text):
 # The fixed prompt list contains only lowercase ASCII words and punctuation.
 if not text.isascii() or "'" in text:raise ValueError('Compiler only accepts fixed ASCII prompts')
 ids=[49406]
 for word in re.findall(r'[a-z]+|[0-9]|[^\s\w]',text.lower()):
  pieces=list(word);pieces[-1]+='</w>'
  while len(pieces)>1:
   pairs=list(zip(pieces,pieces[1:]));best=min(pairs,key=lambda x:ranks.get(x,float('inf')))
   if best not in ranks:break
   merged=[];i=0
   while i<len(pieces):
    if i<len(pieces)-1 and (pieces[i],pieces[i+1])==best:merged.append(pieces[i]+pieces[i+1]);i+=2
    else:merged.append(pieces[i]);i+=1
   pieces=merged
  ids.extend(vocab[piece] for piece in pieces)
 return ids+[49407]
model=session(root/'text_model_int8.onnx');vectors=[]
print([(i.name,i.shape) for i in model.get_inputs()],[(i.name,i.shape) for i in model.get_outputs()])
for text in prompts():
 ids=encode(text);feed={}
 for input in model.get_inputs():
  feed[input.name]=np.array([ids if input.name=='input_ids' else [1]*len(ids)],np.int64)
 output=model.run(None,feed)[0].reshape(-1);vectors.append(output/np.linalg.norm(output))
target=ROOT/'worker/models/scene_intent_prompts.npz';np.savez_compressed(target,embeddings=np.array(vectors,np.float32))
metadata={'upstream':'https://huggingface.co/Xenova/clip-vit-base-patch32','revision':'6b7c523703ae26e39be4b03774966f8571c7f523','license':'MIT','licenseFile':'CLIP-LICENSE','promptHash':prompt_hash(),'promptVectorsSha256':hashlib.sha256(target.read_bytes()).hexdigest(),'textModelSha256':'18845f2ccc35223bb7fec403383a131154b11ac0918df25cf51986df5efd3a21','compiler':'scripts/compile-scene-prompts.py','scope':'Zero-shot object evidence; no calibrated probability or intent guarantee.'}
(ROOT/'worker/models/SceneIntent-PROVENANCE.json').write_text(json.dumps(metadata,indent=2))
print('Saved',target,metadata['promptVectorsSha256'])
