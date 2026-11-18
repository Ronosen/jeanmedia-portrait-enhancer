import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import numpy as np
from backdrop_finish import unpack
for name,points in [('IMG_0203',[(.02,.82),(.03,.84),(.02,.83)])]:
 root=Path(sys.argv[1]);p=json.loads((root/(name+'-plan.json')).read_text());shape=p['layers']['shape']
 print(name,'floor',p['floor'],'partial',p.get('partialCleanup'))
 for key in ('rawForegroundAlpha','foregroundAlpha','support','clutter','backdrop'):
  if key in p['layers']:
   a=unpack(p['layers'][key],shape)
   print(key,[int(a[min(shape[0]-1,int(y*shape[0])),min(shape[1]-1,int(x*shape[1]))]) for x,y in points])
