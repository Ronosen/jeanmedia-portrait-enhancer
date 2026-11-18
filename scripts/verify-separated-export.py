"""Verify exact source cores and deterministic saved-matte export."""
import sys,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
sys.path.insert(0,str(Path('worker').resolve()))
from photo import load_photo,correction,apply_correction
from studio import clean_studio
from backdrop_finish import unpack
from floor import decode_mask

root=Path(sys.argv[1]);rows=[]
for name in sys.argv[2:]:
 plan=json.loads((root/(name+'-plan.json')).read_text())
 source,_=load_photo(Path('D:/test22')/(name+'.JPG'))
 preview=source.copy();preview.thumbnail((1280,1280),Image.Resampling.LANCZOS)
 raw,_,_=clean_studio(source,preview,plan,export=True)
 shape=plan['layers']['shape'];alpha=unpack(plan['layers']['foregroundAlpha'],shape)
 core=cv2.erode((alpha>=250).astype(np.uint8),np.ones((5,5),np.uint8))>0
 if plan.get('propRepairs'):core &= decode_mask(plan['propRepairs']['mask'],tuple(shape))==0
 core=Image.fromarray(core.astype(np.uint8)*255).resize(source.size,Image.Resampling.NEAREST)
 core_changes=0
 for y in range(0,source.height,256):
  box=(0,y,source.width,min(source.height,y+256));selected=np.asarray(core.crop(box))>0
  core_changes+=int(np.count_nonzero(np.any(np.asarray(source.crop(box))[selected]!=np.asarray(raw.crop(box))[selected],axis=1)))
 finished=apply_correction(raw,correction(preview));different=0
 with Image.open(root/(name+'-edited.png')) as actual:
  for y in range(0,source.height,256):
   box=(0,y,source.width,min(source.height,y+256))
   different+=int(np.count_nonzero(np.any(np.asarray(actual.crop(box))!=np.asarray(finished.crop(box)),axis=2)))
 row={'image':name,'subjectCoreChangesBeforeTone':core_changes,'savedExportReplayDifferences':different}
 print(row,flush=True);rows.append(row)
 assert core_changes==0 and different==0,row
(root/'separation-preservation.json').write_text(json.dumps(rows,indent=2))
