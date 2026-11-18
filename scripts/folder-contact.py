"""Build indexed before/after review sheets from an isolated folder audit."""
import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(sys.argv[1]);rows=json.loads((root/'run.json').read_text())
for start in range(0,len(rows),6):
 sheet=Image.new('RGB',(1200,970),'#252525');draw=ImageDraw.Draw(sheet)
 for j,row in enumerate(rows[start:start+6]):
  if row.get('error'):continue
  name=Path(row['name']).stem;folder=Path(row['artifactDirectory'])
  x=(j%3)*400;y=(j//3)*485
  for k,suffix in enumerate(('before','after')):
   with Image.open(folder/(name+'-'+suffix+'.png')) as source:
    source.thumbnail((196,450));sheet.paste(source,(x+k*200,y+25))
  draw.text((x+5,y+5),name+'   original | edited',fill='white')
 sheet.save(root/('contact-'+str(start//6+1)+'.jpg'),quality=94)
