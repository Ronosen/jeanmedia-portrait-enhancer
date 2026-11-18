"""Read-only source audit using the installed app's worker; no pipeline changes."""
import sys,json,time,traceback,base64,io,os,hashlib
from pathlib import Path
bundle=Path(sys.argv[1]).resolve();root=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(bundle/'worker'))
import numpy as np
from PIL import Image,ImageDraw
from photo import load_photo,apply_correction,correction,export_photo
from adaptive import auto_edit
from preflight import inspect
from service import change_mask
from runtime import sha256,manifest
import runtime
runtime_info=manifest()
# Each isolated audit process uses immutable model files. Reuse the provenance
# already measured at startup; inference integrity checks remain enabled.
runtime.manifest=lambda:runtime_info

# Optional QA-only reuse of the exact raw inference tensor. This cache is keyed
# by source preview pixels and model bytes, never by filename or manual masks.
if os.environ.get('JEANMEDIA_QA_MATTE_CACHE'):
 import matting
 cache=Path(os.environ['JEANMEDIA_QA_MATTE_CACHE']);cache.mkdir(parents=True,exist_ok=True)
 model_digest=runtime_info['models'][matting.MODEL.name];infer=matting.foreground_alpha
 def cached_alpha(rgb):
  key=hashlib.sha256(rgb.tobytes()+model_digest.encode()).hexdigest();target=cache/(key+'.npy')
  if target.exists():
   value=np.load(target,allow_pickle=False)
   if value.shape!=rgb.shape[:2] or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Invalid QA inference cache')
   return value
  value=infer(rgb).astype(np.float32);np.save(target,value,allow_pickle=False);return value
 matting.foreground_alpha=cached_alpha

root.mkdir(parents=True,exist_ok=False)
rows=[]
preview_only=os.environ.get('JEANMEDIA_QA_PREVIEW_ONLY')=='1'
(root/'runtime.json').write_text(json.dumps({'bundle':str(bundle),'runtime':runtime_info,
 'workerHashes':{p.name:sha256(p) for p in (bundle/'worker').glob('*.py')}},indent=2))
sources=sorted(Path(os.environ.get('JEANMEDIA_QA_SOURCE_DIR','D:/test22')).iterdir())
if len(sys.argv)>3:
 by_name={p.stem:p for p in sources};sources=[by_name[n] for n in sys.argv[3:] if n in by_name]
for src in sources:
 if src.suffix.lower() not in {'.jpg','.jpeg','.png','.tif','.tiff'}:continue
 start=time.time();row={'name':src.name,'source':str(src)}
 try:
  digest=sha256(src);im,frames=load_photo(src);p=im.copy();p.thumbnail((1280,1280),Image.Resampling.LANCZOS)
  p.save(root/(src.stem+'-before.png'))
  preflight=inspect(p)
  full,out,plan,status=auto_edit(im,p,export=not preview_only)
  check_source=p if preview_only else im;check_output=out if preview_only else full
  core_changes=0
  if plan.get('layers',{}).get('foregroundAlpha'):
   import cv2
   from backdrop_finish import unpack
   from floor import decode_mask
   shape=plan['layers']['shape'];alpha=unpack(plan['layers']['foregroundAlpha'],shape)
   core=cv2.erode((alpha>=250).astype(np.uint8),np.ones((5,5),np.uint8))>0
   if plan.get('propRepairs'):core &= decode_mask(plan['propRepairs']['mask'],tuple(shape))==0
   core=Image.fromarray(core.astype(np.uint8)*255).resize(check_source.size,Image.Resampling.NEAREST)
   for y in range(0,check_source.height,256):
    box=(0,y,check_source.width,min(check_source.height,y+256));guard=np.asarray(core.crop(box))>0
    core_changes+=int(np.count_nonzero(np.any(np.asarray(check_source.crop(box))[guard]!=np.asarray(check_output.crop(box))[guard],axis=1)))
   assert core_changes==0,('Protected source core changed',core_changes)
  selected=np.array(change_mask(plan))>0
  outside=int(np.count_nonzero(np.any(np.array(p)[~selected]!=np.array(out)[~selected],axis=1)))
  assert outside==0,('Unselected cleanup pixels changed',outside)
  plan['review']=list(dict.fromkeys(plan.get('review',[])+preflight['warnings']))
  (root/(src.stem+'-plan.json')).write_text(json.dumps(plan),encoding='utf-8')
  change_mask(plan).save(root/(src.stem+'-mask.png'))
  lut=correction(p);finished=apply_correction(out if preview_only else full,lut)
  target=root/(src.stem+'-edited.png');export_photo(finished,target,300)
  rendered=finished.copy();rendered.thumbnail((1280,1280),Image.Resampling.LANCZOS);rendered.save(root/(src.stem+'-after.png'))
  pair=Image.new('RGB',(p.width*2,p.height+28),'#222');pair.paste(p,(0,28));pair.paste(rendered,(p.width,28));d=ImageDraw.Draw(pair);d.text((10,7),src.name+' ORIGINAL',fill='white');d.text((p.width+10,7),'EDITED',fill='white');pair.save(root/(src.stem+'-compare.jpg'),quality=96)
  assert sha256(src)==digest,'Original changed'
  with Image.open(target) as check:assert check.size==finished.size and check.info.get('icc_profile')
  row.update(dimensions=im.size,frames=frames,preflight=preflight,review=plan['review'],
   engine=plan['repairEngine'],outsideCleanupMaskChanges=outside,subjectCoreChangesBeforeTone=core_changes,changedFraction=float(selected.mean()),
   output=str(target),sourceSha256=digest,originalUnchanged=True,fullResolutionExport=not preview_only,
   outputDimensions=finished.size,previewOnly=preview_only,processingRevision=plan.get('processingRevision'),status=status)
 except Exception as error:row.update(error=str(error),trace=traceback.format_exc())
 row['seconds']=round(time.time()-start,1);rows.append(row)
 (root/'run.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print(json.dumps(row),flush=True)
