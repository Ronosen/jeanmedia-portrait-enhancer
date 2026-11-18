"""Run all supplied originals through real Auto Edit and retain review artifacts."""
import sys,json,time,traceback,io,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
import numpy as np
from PIL import Image,ImageDraw
from photo import load_photo,apply_correction,correction,export_photo
from adaptive import auto_edit
from service import change_mask
from runtime import sha256

root=Path('qa/all-images');root.mkdir(parents=True,exist_ok=True)
rows=json.loads((root/'report.json').read_text()) if len(sys.argv)>1 and (root/'report.json').exists() else []
for src in sorted(Path('D:/test22').glob('*.JPG')):
 if len(sys.argv)>1 and src.stem not in sys.argv[1:]: continue
 start=time.time();row={'name':src.name}
 try:
  digest=sha256(src);im,_=load_photo(src);p=im.copy();p.thumbnail((1280,1280),Image.Resampling.LANCZOS)
  p.save(root/(src.stem+'-before.png'))
  full,out,plan,status=auto_edit(im,p,export=True)
  selected=np.asarray(change_mask(plan))>0
  outside=int(np.count_nonzero(np.any(np.asarray(p)[~selected]!=np.asarray(out)[~selected],axis=1)))
  assert outside==0,outside
  (root/(src.stem+'-plan.json')).write_text(json.dumps(plan),encoding='utf-8')
  lut=correction(p);apply_correction(out,lut).save(root/(src.stem+'-after.png'))
  target=root/(src.stem+'-finished.jpg')
  if target.exists(): target.unlink()
  export_photo(apply_correction(full,lut),target,300)
  assert sha256(src)==digest
  row.update(status=status,review=plan.get('review'),outsideMaskChanges=outside,changedFraction=float(selected.mean()),export=str(target),sourceSha256=digest)
 except Exception as e:
  row.update(error=str(e),trace=traceback.format_exc())
 row['seconds']=round(time.time()-start,1);rows=[r for r in rows if r['name']!=src.name];rows.append(row);rows.sort(key=lambda r:r['name'])
 (root/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print(json.dumps(row),flush=True)
 tiles=Image.new('RGB',(4*440,((len(rows)+1)//2)*680),'#222222');d=ImageDraw.Draw(tiles)
 for i,r in enumerate(rows):
  for j,kind in enumerate(('before','after')):
   path=root/(Path(r['name']).stem+'-'+kind+'.png')
   if path.exists():
    with Image.open(path) as pic:
     pic.thumbnail((425,640));x=(i%2*2+j)*440;y=i//2*680;tiles.paste(pic,(x,y+25));d.text((x+5,y+5),r['name']+' '+kind,fill='white')
 tiles.save(root/'contact.jpg')
