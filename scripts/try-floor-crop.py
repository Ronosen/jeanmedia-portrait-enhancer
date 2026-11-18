import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import numpy as np
from PIL import Image
from floor import decode_mask
from backdrop_finish import unpack
from neural import repair
root=Path(sys.argv[1]);name=sys.argv[2];p=json.loads((root/(name+'-plan.json')).read_text());shape=p['layers']['shape']
source=Image.open(root/(name+'-before.png')).convert('RGB');extension=decode_mask(p['floorRepairs']['extension'],tuple(shape))>0
protected=unpack(p['layers']['foregroundAlpha'],shape)>3;clutter=unpack(p['layers']['clutter'],shape)>0
row=max(0,round((p['floor']-.03)*shape[0]));crop=source.crop((0,row,shape[1],shape[0]));mask=(extension|protected|clutter)[row:]
settings={'plateOnly':True,'excludeForegroundContext':False};out=repair(crop,mask,np.zeros_like(mask),settings)
patch=source.copy();patch.paste(out,(0,row));result=Image.composite(patch,source,Image.fromarray((extension&~protected).astype(np.uint8)*255))
result.save(root/(name+'-floor-crop.png'));(root/(name+'-floor-crop.json')).write_text(json.dumps(settings))
