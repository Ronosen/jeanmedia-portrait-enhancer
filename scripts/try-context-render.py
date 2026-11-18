import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
from PIL import Image
from separation import render
from backdrop_finish import unpack
root=Path(sys.argv[1]);name=sys.argv[2];plan=json.loads((root/(name+'-plan.json')).read_text())
plan['separation']['floorMethod']='context-crop-v2';plan.pop('floorPlate',None)
source=Image.open(root/(name+'-before.png')).convert('RGB')
render(source,unpack(plan['layers']['backdrop'],plan['layers']['shape'])>0,plan).save(root/(name+'-context-check.png'))
