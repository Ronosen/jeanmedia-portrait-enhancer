"""Local BiRefNet foreground alpha; no network access during photo processing."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

MODEL=Path(__file__).with_name('models')/'birefnet-general-lite.onnx'

@lru_cache(maxsize=1)
def model_session():
 from runtime import session,sha256
 if sha256(MODEL)!='5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333':raise ValueError('Subject separation model integrity check failed.')
 return session(MODEL)

def _infer_alpha(rgb):
 from runtime import progress
 progress('Separating subjects from background',28)
 model=model_session()
 x=np.asarray(Image.fromarray(rgb).resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32)/255
 x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
 outputs=model.run(None,{model.get_inputs()[0].name:np.ascontiguousarray(x.transpose(2,0,1)[None])})
 logits=outputs[0]
 if logits.shape!=(1,1,1024,1024) or not np.isfinite(logits).all():raise ValueError('Invalid subject separation output.')
 a=1/(1+np.exp(-np.clip(logits[0,0],-60,60)))
 a=cv2.resize(a,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
 # The foreground model is no longer needed during full-size reconstruction.
 # Keeping its arena alive alongside LaMa exhausted memory in photo batches.
 model_session.cache_clear()
 return np.clip(a,0,1)


def foreground_alpha(rgb):
 """Reuse exact raw inference only; prop decisions are always recalculated."""
 import hashlib,os,tempfile
 from runtime import sha256,progress
 digest=sha256(MODEL)
 if digest!='5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333':
  raise ValueError('Subject separation model integrity check failed.')
 key=hashlib.sha256(b'matte-v1'+str(rgb.shape).encode()+rgb.tobytes()+digest.encode()).hexdigest()
 cache=Path(__file__).parent/'.cache'/'matting';target=cache/(key+'.npy')
 try:
  if target.is_file() and target.stat().st_size<8_000_000:
   value=np.load(target,allow_pickle=False)
   if value.shape==rgb.shape[:2] and value.dtype==np.float32 and np.isfinite(value).all() and value.min()>=0 and value.max()<=1:
    progress('Reusing verified subject separation',28)
    return value
 except (OSError,ValueError):pass
 value=_infer_alpha(rgb).astype(np.float32)
 try:
  cache.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=cache,suffix='.tmp',delete=False) as stream:
   temporary=Path(stream.name);np.save(stream,value,allow_pickle=False)
  os.replace(temporary,target)
  # Keep this local derived-image cache bounded. It is excluded from builds.
  files=sorted(cache.glob('*.npy'),key=lambda item:item.stat().st_mtime,reverse=True)
  total=0
  for i,item in enumerate(files):
   total+=item.stat().st_size
   if i>=32 or total>128_000_000:item.unlink(missing_ok=True)
 except OSError:pass
 return value
