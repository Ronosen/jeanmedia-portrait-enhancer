"""Produce paired native-pixel inspection crops; never modifies originals."""
import sys,json
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(sys.argv[1]);bundle=Path(sys.argv[2]);sys.path.insert(0,str(bundle/'worker'))
from photo import load_photo
boxes=json.loads((root/'detail-points.json').read_text())
for name,points in boxes.items():
 if len(sys.argv)>3 and name not in sys.argv[3:]:continue
 if not (root/(name+'-edited.png')).exists():continue
 original,_=load_photo(Path('D:/test22')/(name+'.JPG'))
 with Image.open(root/(name+'-edited.png')) as edited:
  for label,x,y in points:
   side=640;left=max(0,min(original.width-side,round(x*original.width)-side//2));top=max(0,min(original.height-side,round(y*original.height)-side//2));box=(left,top,left+side,top+side)
   canvas=Image.new('RGB',(side*2,side+28),'#222');canvas.paste(original.crop(box),(0,28));canvas.paste(edited.crop(box),(side,28));d=ImageDraw.Draw(canvas);d.text((5,7),name+' '+label+' ORIGINAL',fill='white');d.text((side+5,7),'EDITED — native pixel crop',fill='white')
   canvas.save(root/(name+'-detail-'+label+'.png'))
