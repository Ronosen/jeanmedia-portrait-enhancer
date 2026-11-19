"""Identify affected sources from a frozen corpus using the new panel detector."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--bundle',required=True)
parser.add_argument('--corpus',action='append',required=True)
parser.add_argument('--prefer',action='append',default=[])
parser.add_argument('--out',required=True)
args=parser.parse_args();bundle=Path(args.bundle).resolve()
sys.path.insert(0,str(bundle/'worker'))
import numpy as np
from PIL import Image
from background import person_probability,MODEL as PERSON_MODEL
from matting import MODEL as MATTE_MODEL
from runtime import sha256
from studio import background_basis
from backdrop_finish import unpack
from scene_objects import frame_panel_leaks
root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
scope=[];records={}
for value in args.corpus+args.prefer:
 folder=Path(value)
 if value in args.corpus:scope.extend(json.loads((folder/'scope.json').read_text())['sources'])
 if (folder/'run.json').exists():
  for row in json.loads((folder/'run.json').read_text()):
   if not row.get('error'):records[row['source']]=row
matte_hash=sha256(MATTE_MODEL);rows=[];affected=[]
for source in dict.fromkeys(scope):
 row=records.get(source);item={'source':source}
 if row is None:item['error']='No completed planning evidence.'
 else:
  folder=Path(row['artifactDirectory']);name=Path(source).stem
  plan=json.loads((folder/(name+'-plan.json')).read_text())
  if plan.get('separation'):
   rgb=np.array(Image.open(folder/(name+'-before.png')));h,w=rgb.shape[:2];yy,xx=np.mgrid[:h,:w]
   reference=np.clip(background_basis(xx/max(1,w-1),yy/max(1,h-1))@np.array(plan['coefficients']),plan['low'],plan['high'])
   key=hashlib.sha256(rgb.tobytes()+matte_hash.encode()).hexdigest()
   raw=np.load(Path('qa/raw-matte-cache')/(key+'.npy'),allow_pickle=False)
   pp=person_probability(rgb);geometry=unpack(plan['layers']['geometrySupport'],[h,w])>0
   reject,extent=frame_panel_leaks(rgb,reference,pp,geometry,raw,with_extent=True)
   old=unpack(plan['layers']['foregroundAlpha'],[h,w]);protected=unpack(plan['layers']['protected'],[h,w])>0
   clutter=unpack(plan['layers']['clutter'],[h,w])>0
   matte_changes=int((reject&(old>0)).sum())
   selection_changes=int((extent&~(protected&~reject)&~clutter).sum())
   item.update(panelPixels=int(extent.sum()),foregroundRejectPixels=matte_changes,newClutterPixels=selection_changes,
               independentPersonCoreOverlap=int((reject&(pp>.9)).sum()),evidenceDirectory=str(folder))
   if matte_changes or selection_changes:affected.append(source)
  else:item.update(panelPixels=0,reason='Preserved non-studio scene does not enter separation.prepare.')
 rows.append(item);print(Path(source).name,item.get('newClutterPixels',0),flush=True)
 (root/'scan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(root/'scope.json').write_text(json.dumps({'sources':affected,'method':'source-material-and-independent-person-evidence','bundle':str(bundle),'totalExamined':len(rows)},indent=2),encoding='utf-8')
(root/'provenance.json').write_text(json.dumps({'personModelSha256':sha256(PERSON_MODEL),'matteModelSha256':matte_hash,'workerHashes':{p.name:sha256(p) for p in (bundle/'worker').glob('*.py')}},indent=2),encoding='utf-8')
assert not any(r.get('error') or r.get('independentPersonCoreOverlap',0) for r in rows),rows
print('Affected',len(affected),'of',len(rows),flush=True)
