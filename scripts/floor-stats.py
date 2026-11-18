import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import numpy as np
from floor import decode_mask
root=Path(sys.argv[1])
for r in json.loads((root/'run.json').read_text()):
 n=Path(r['name']).stem;p=json.loads((Path(r['artifactDirectory'])/(n+'-plan.json')).read_text());f=p.get('floorRepairs',{});s=f.get('shape')
 if s:
  a=decode_mask(f['extension'],tuple(s));y=np.flatnonzero(a.any(axis=1));print(n,p['floor'],round(float((a>0).mean()),3),int(y[0]) if len(y) else None)
