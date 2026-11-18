"""Verify full-size protected cores after the expected global tone/sharpen step."""
import sys,json,base64
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import cv2,numpy as np
from PIL import Image
from photo import load_photo,apply_correction,correction
from floor import decode_mask
root=Path(sys.argv[1]);rows=[]
for name in sys.argv[2:]:
 plan=json.loads((root/(name+'-plan.json')).read_text())
 original,_=load_photo(Path('D:/test22')/(name+'.JPG'))
 preview=original.copy();preview.thumbnail((1280,1280),Image.Resampling.LANCZOS)
 expected=apply_correction(original,correction(preview))
 protected=cv2.imdecode(np.frombuffer(base64.b64decode(plan['layers'].get('edgeProtected',plan['layers']['protected'])),np.uint8),0)>0
 for section,names in [('floorRepairs',['defects','extension']),('propRepairs',['mask'])]:
  if plan.get(section):
   for key in names:protected &= decode_mask(plan[section][key],protected.shape)==0
 core=cv2.erode(protected.astype(np.uint8),np.ones((5,5),np.uint8))>0
 full=Image.fromarray(core.astype(np.uint8)*255).resize(original.size,Image.Resampling.NEAREST)
 count=0
 with Image.open(root/(name+'-edited.png')) as actual:
  for y in range(0,original.height,256):
   box=(0,y,original.width,min(original.height,y+256));keep=np.asarray(full.crop(box))>0
   count+=int(np.count_nonzero(np.any(np.asarray(expected.crop(box))[keep]!=np.asarray(actual.crop(box))[keep],axis=1)))
 assert count==0,(name,count)
 rows.append({'image':name,'protectedCoreChangesBeyondExpectedTone':count})
 print(name,'PASS',flush=True)
(root/'preservation.json').write_text(json.dumps(rows,indent=2))
