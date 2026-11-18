import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import cv2,numpy as np
from PIL import Image
from backdrop_finish import unpack,pack
from separation import render
root=Path(sys.argv[1]);name=sys.argv[2];p=json.loads((root/(name+'-plan.json')).read_text());shape=p['layers']['shape']
source=Image.open(root/(name+'-before.png')).convert('RGB');rgb=np.asarray(source);s=unpack(p['layers']['support'],shape)>0
raw=unpack(p['layers']['rawForegroundAlpha'],shape);a=unpack(p['layers']['foregroundAlpha'],shape)
r=max(3,round(min(shape)*.012));k=np.ones((r*2+1,r*2+1),np.uint8)
expanded=cv2.dilate(s.astype(np.uint8),k)>0;core=cv2.erode(s.astype(np.uint8),k)>0
m=np.zeros(shape,np.uint8);m[expanded]=cv2.GC_PR_BGD;m[s]=cv2.GC_PR_FGD;m[core]=cv2.GC_FGD;m[raw>=250]=cv2.GC_FGD
cv2.grabCut(rgb,m,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5,cv2.GC_INIT_WITH_MASK)
keep=((m==cv2.GC_FGD)|(m==cv2.GC_PR_FGD))&expanded
support=s&keep;a[s&(raw<242)&~keep]=0
p['layers']['foregroundAlpha']=pack(a)
render(source,unpack(p['layers']['backdrop'],shape)>0,p).save(root/(name+'-support-trial.png'))
Image.fromarray(support.astype(np.uint8)*255).save(root/(name+'-support-trial-mask.png'))
