"""Install pinned, hash-verified local CLIP vision weights and prompt tooling."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV='6b7c523703ae26e39be4b03774966f8571c7f523'
FILES=[('onnx/vision_model_int8.onnx','worker/models/clip_vision_int8.onnx','0ab0c1b3ace708e539633af1744d5a95247fe4e14d3e08ff197ef82a6cb9bd93'),
 ('onnx/text_model_int8.onnx','qa/scene-intent/text_model_int8.onnx','18845f2ccc35223bb7fec403383a131154b11ac0918df25cf51986df5efd3a21'),
 ('tokenizer.json','qa/scene-intent/tokenizer.json','f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd'),('preprocessor_config.json','qa/scene-intent/preprocessor_config.json','6f638fb9401a6d6296feff533ee7efe657b787c49f954f82f5906b36ef2a1b1f')]
def download(url,path,digest=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  tmp=path.with_suffix(path.suffix+'.download')
  with urllib.request.urlopen(url,timeout=60) as source,tmp.open('wb') as dest:
   while chunk:=source.read(1024*1024):dest.write(chunk)
  if digest and hashlib.sha256(tmp.read_bytes()).hexdigest()!=digest:raise ValueError('Model checksum mismatch')
  tmp.replace(path)
 actual=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest and actual!=digest:raise ValueError('Existing model checksum mismatch')
 print(path.name,path.stat().st_size,actual,flush=True)
for name,target,digest in FILES:download(f'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/{REV}/{name}',ROOT/target,digest)
download('https://raw.githubusercontent.com/openai/CLIP/main/LICENSE',ROOT/'worker/models/CLIP-LICENSE','987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db')
