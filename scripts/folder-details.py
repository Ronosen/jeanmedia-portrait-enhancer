"""Native-pixel comparisons of contact edges, fine subjects and floor repairs."""
import sys,json
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path('worker').resolve()))
from photo import load_photo
root=Path(sys.argv[1]);rows=json.loads((root/'run.json').read_text())
points={'IMG_1030':[(.74,.62),(.8,.9),(.08,.85)],'IMG_0671':[(.24,.65),(.55,.94)],
 'IMG_0164':[(.72,.2),(.5,.96)],'IMG_0203':[(.71,.17),(.5,.59),(.04,.88)],
 'IMG_0656':[(.60,.29),(.86,.69)],'IMG_0683':[(.63,.38),(.84,.84)],
 'IMG_0731':[(.52,.42)],'IMG_0747':[(.43,.3),(.50,.72),(.50,.97)],
 'IMG_0747-finished':[(.43,.3),(.50,.72),(.50,.97)],'IMG_0779':[(.52,.58),(.05,.54)],
 'IMG_0797':[(.54,.24)],'IMG_0881':[(.50,.65),(.50,.96)],'IMG_0957':[(.50,.3),(.05,.45)],
 'IMG_0966':[(.71,.21),(.50,.95)],'IMG_1083':[(.53,.35),(.06,.86)],
 'IMG_1097':[(.60,.47)],'IMG_1167':[(.59,.43),(.48,.94)],'IMG_1232':[(.52,.4)],'IMG_1241':[(.55,.25)]}
for row in rows:
 if row.get('error'):continue
 name=Path(row['name']).stem;folder=Path(row['artifactDirectory']);source,_=load_photo(row['source'])
 if len(sys.argv)>2 and name not in sys.argv[2:]:continue
 with Image.open(row['output']) as edited:
  for index,(x,y) in enumerate(points.get(name,points.get(name.replace('-finished',''),[(.5,.5)]))):
   side=640;left=max(0,min(source.width-side,round(x*source.width)-side//2));top=max(0,min(source.height-side,round(y*source.height)-side//2));box=(left,top,left+side,top+side)
   pair=Image.new('RGB',(1280,668),'#222');d=ImageDraw.Draw(pair)
   pair.paste(source.crop(box),(0,28));pair.paste(edited.crop(box),(640,28));d.text((5,7),name+' original / native pixels',fill='white');d.text((645,7),'edited / native pixels',fill='white')
   pair.save(folder/(name+'-detail-'+str(index+1)+'.jpg'),quality=97)
