import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('worker').resolve()))
import numpy as np
from PIL import Image,ImageDraw
from backdrop_finish import unpack
root=Path(sys.argv[1]);name=sys.argv[2];p=json.loads((root/(name+'-plan.json')).read_text());shape=p['layers']['shape']
sheet=Image.new('RGB',(1000,520),'#222');d=ImageDraw.Draw(sheet)
for j,key in enumerate(('rawForegroundAlpha','foregroundAlpha','support','clutter','backdrop')):
 a=unpack(p['layers'][key],shape);im=Image.fromarray(a).convert('RGB');im.thumbnail((200,490));sheet.paste(im,(j*200,25));d.text((j*200,5),key,fill='white')
sheet.save(root/(name+'-layers.jpg'))
