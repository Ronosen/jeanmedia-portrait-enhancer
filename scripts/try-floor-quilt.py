import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import numpy as np
from PIL import Image
from floor import decode_mask
from backdrop_finish import unpack
from inpainting import repair
root=Path(sys.argv[1]);name=sys.argv[2];p=json.loads((root/(name+'-plan.json')).read_text());shape=p['layers']['shape']
source=Image.open(root/(name+'-before.png')).convert('RGB');mask=decode_mask(p['floorRepairs']['extension'],tuple(shape))
protected=unpack(p['layers']['foregroundAlpha'],shape)>3
protected |= unpack(p['layers']['clutter'],shape)>0
mask[protected]=0
settings={};result=repair(source,mask,protected,p['floor'],settings)
result.save(root/(name+'-floor-quilt.png'));(root/(name+'-floor-quilt.json')).write_text(json.dumps(settings))
