"""Check group-cap preservation and actual saved-plan export replay."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path('worker').resolve()))
from photo import load_photo,apply_correction,correction
from studio import clean_studio
from backdrop_finish import unpack

root=Path(sys.argv[1]);rows=[]
for name in sys.argv[2:]:
 plan=json.loads((root/(name+'-plan.json')).read_text())
 original,_=load_photo(Path('D:/test22')/(name+'.JPG'))
 preview=original.copy();preview.thumbnail((1280,1280),Image.Resampling.LANCZOS)
 cap=unpack(plan['groupProtection']['capMask'],plan['layers']['shape'])>0
 cap_full=Image.fromarray(cap.astype(np.uint8)*255).resize(original.size,Image.Resampling.NEAREST)
 replay,_,_=clean_studio(original,preview,plan,export=True)
 # Compare source protection before sharpening: convolution legitimately
 # depends on changed background neighbours at the outside of a mask.
 cap_changes=0
 for y in range(0,original.height,256):
  box=(0,y,original.width,min(original.height,y+256));mask=np.asarray(cap_full.crop(box))>0
  cap_changes+=int(np.count_nonzero(np.any(np.asarray(original.crop(box))[mask]!=np.asarray(replay.crop(box))[mask],axis=1)))
 replay=apply_correction(replay,correction(preview));replay_changes=0
 with Image.open(root/(name+'-edited.png')) as actual:
  for y in range(0,original.height,256):
   box=(0,y,original.width,min(original.height,y+256))
   a=np.asarray(actual.crop(box));r=np.asarray(replay.crop(box))
   replay_changes+=int(np.count_nonzero(np.any(a!=r,axis=2)))
 row={'image':name,'people':plan['groupProtection']['peopleDetected'],
      'capChangesBeforeTone':cap_changes,'savedExportReplayDifferences':replay_changes}
 print(row,flush=True);rows.append(row)
 assert cap_changes==0 and replay_changes==0,row
(root/'group-preservation.json').write_text(json.dumps(rows,indent=2))
