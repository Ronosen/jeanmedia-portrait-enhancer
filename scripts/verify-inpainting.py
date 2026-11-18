"""Full-resolution acceptance checks on the two reported failures."""
import sys,json,time,base64,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
import numpy as np
from PIL import Image
from photo import load_photo,export_photo
from adaptive import auto_edit
from service import change_mask
from runtime import sha256
report=[]
for name in ('IMG_1083','IMG_0671'):
 start=time.time();source=Path('D:/test22')/(name+'.JPG');digest=sha256(source)
 image,_=load_photo(source);preview=image.copy();preview.thumbnail((1280,1280),Image.Resampling.LANCZOS)
 full,after,plan,status=auto_edit(image,preview,export=True)
 assert plan['repairEngine']=='lama-v1' and plan.get('inpaintPatch')
 selected=np.asarray(change_mask(plan))>0
 protected=np.asarray(Image.open(io.BytesIO(base64.b64decode(plan['layers']['protected']))))>0
 if plan.get('propRepairs'):
  from floor import decode_mask
  protected &= decode_mask(plan['propRepairs']['mask'],protected.shape)==0
 before=np.asarray(preview);out=np.asarray(after)
 outside=int(np.count_nonzero(np.any(before[~selected]!=out[~selected],axis=1)))
 foreground=int(np.count_nonzero(np.any(before[protected]!=out[protected],axis=1)))
 assert outside==0 and foreground==0,(outside,foreground)
 rois={}
 if name=='IMG_0671':
  for label,box in [('face',(305,420,360,485)),('hair',(240,370,265,550)),('coat',(260,580,400,780)),('seat',(80,850,455,1037)),('boot',(770,940,800,960)),('hand',(95,810,140,835))]:
   x0,y0,x1,y1=box;count=int(np.count_nonzero(np.any(before[y0:y1,x0:x1]!=out[y0:y1,x0:x1],axis=2)));rois[label]=count;assert count==0,(label,count)
  a,b=before[1000:1040,64:75],out[1000:1040,64:75]
  count=int(np.count_nonzero(np.any(a!=b,axis=2)&(a.mean(axis=2)>160)));rois['bright_seat_foot']=count;assert count==0,count
 after.save('qa/'+name+'-final-preview.png')
 path=Path('qa/'+name+'-full.png')
 if path.exists():path.unlink()
 export_photo(full,path,300)
 with Image.open(path) as exported:
  assert exported.size==image.size and exported.info.get('icc_profile')
  # Check source fidelity in full-size tiles, not only downscaled previews.
  full_protected=Image.fromarray(protected.astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST)
  full_changes=0
  for top in range(0,image.height,256):
   box=(0,top,image.width,min(image.height,top+256))
   keep=np.asarray(full_protected.crop(box))>0
   full_changes+=int(np.count_nonzero(np.any(np.asarray(image.crop(box))[keep]!=np.asarray(exported.crop(box))[keep],axis=1)))
  assert full_changes==0,full_changes
 assert digest==sha256(source)
 Path('qa/'+name+'-final-plan.json').write_text(json.dumps(plan),encoding='utf-8')
 row={'input':name,'engine':plan['repairEngine'],'size':image.size,'seconds':round(time.time()-start,1),'outsideMaskChanges':outside,'protectedPreviewChanges':foreground,'protectedFullResolutionChanges':full_changes,'manualRegions':rois,'originalUnchanged':True,'review':plan['review']}
 report.append(row);Path('QA-INPAINTING.json').write_text(json.dumps({'results':report,'scope':'Automated preservation plus visual review of these two cases; not general photographic certification.'},indent=2),encoding='utf-8')
 print(name,'PASS',row['seconds'],flush=True)
