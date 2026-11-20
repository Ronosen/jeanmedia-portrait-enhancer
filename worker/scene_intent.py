"""Local semantic scene skill: classify measured objects before removing them.

CLIP similarities are evidence, not calibrated probabilities or knowledge of the
photographer's intent. People/attached supports win; ambiguous detached objects
are retained and reported. Classification never turns a box into a repair mask.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageOps

VISION_SHA='0ab0c1b3ace708e539633af1744d5a95247fe4e14d3e08ff197ef82a6cb9bd93'
LABELS={
 'graduation gown':('protect',['a graduation gown hanging on a clothes hanger','a ceremonial graduation robe displayed in a portrait']),
 'graduation cap':('protect',['a graduation mortarboard cap','a person holding a graduation cap']),
 'balloon':('protect',['decorative balloons at a celebration','a gold or black party balloon']),
 'diploma':('protect',['a rolled graduation diploma with a ribbon','graduation certificates and books used as portrait props']),
 'flowers':('protect',['a bouquet of flowers used as a portrait prop','decorative flowers at a celebration']),
 'seat':('protect',['a white pedestal seat used in a portrait','a chair or stool supporting a person']),
 'softbox':('remove',['a photographic lighting softbox in a studio','photography lighting equipment intruding into the picture']),
 'loose bag':('remove',['a crumpled plastic shopping bag on the floor','an unwanted bag left on a studio floor']),
 'wallpaper':('remove',['patterned wallpaper exposed beside a photography backdrop','a wall with yellow decorative lines']),
 'cables':('remove',['loose electrical cables on a studio floor','a photography light stand and cable']),
 'backdrop':('uncertain',['a fabric photography backdrop','a dark studio background with shadows']),
 'person':('uncertain',['a person wearing clothes','a human hand or foot']),
 'other':('uncertain',['an unidentified object in a portrait','an object partly hidden behind a person']),
}
def prompts():return ['a photo of '+text+'.' for _,texts in LABELS.values() for text in texts]
def prompt_hash():return hashlib.sha256(json.dumps(prompts()).encode()).hexdigest()

@lru_cache(maxsize=1)
def resources():
 from runtime import MODELS,session,sha256
 path=MODELS/'clip_vision_int8.onnx';vectors=MODELS/'scene_intent_prompts.npz'
 metadata=json.loads((MODELS/'SceneIntent-PROVENANCE.json').read_text())
 if sha256(path)!=VISION_SHA or sha256(vectors)!=metadata['promptVectorsSha256'] or metadata['promptHash']!=prompt_hash():
  raise ValueError('Scene intent model integrity check failed')
 with np.load(vectors,allow_pickle=False) as data:embeddings=data['embeddings'].copy()
 if embeddings.shape!=(len(prompts()),512) or not np.isfinite(embeddings).all():raise ValueError('Invalid scene intent prompts')
 return session(path),embeddings

def classify(crop):
 model,text=resources()
 # Match upstream CLIP resize-shorter-side, centre crop, RGB normalization.
 image=ImageOps.fit(Image.fromarray(crop),(224,224),Image.Resampling.BICUBIC)
 pixels=np.asarray(image,dtype=np.float32)/255
 pixels=(pixels-np.array([.48145466,.4578275,.40821073],np.float32))/np.array([.26862954,.26130258,.27577711],np.float32)
 output=model.run(None,{model.get_inputs()[0].name:pixels.transpose(2,0,1)[None].copy()})[0].reshape(-1)
 if output.shape!=(512,) or not np.isfinite(output).all():raise ValueError('Invalid scene intent model output')
 output=output/max(1e-9,float(np.linalg.norm(output)))
 scores=(text@output).reshape(len(LABELS),2).mean(axis=1)
 order=np.argsort(scores)[::-1];winner=int(order[0]);label=list(LABELS)[winner];action=LABELS[label][0]
 competitor=max(float(scores[i]) for i,key in enumerate(LABELS) if LABELS[key][0]!=action)
 score=float(scores[winner]);margin=score-competitor
 if score<(.26 if action=='remove' else .23) or margin<(.02 if action=='remove' else .025):action='uncertain'
 return {'label':label,'action':action,'similarity':round(score,4),'margin':round(margin,4)}

def proposals(rgb,raw,decoration,equipment,person):
 h,w=person.shape;seen=np.zeros((h,w),bool)
 chroma=np.ptp(rgb,axis=2)
 r=max(2,round(min(h,w)*.004));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
 colour=cv2.morphologyEx((chroma>55).astype(np.uint8),cv2.MORPH_OPEN,kernel)>0
 # Separate pools stop a yellow wallpaper line joining an entire balloon.
 dark=(rgb.mean(axis=2)<70)&(chroma<35)
 dark=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_OPEN,kernel)>0
 pools=[raw>.05 if raw is not None else seen,colour,dark,decoration>0,equipment>0]
 candidates=[]
 for pool in pools:
  n,labels,stats,_=cv2.connectedComponentsWithStats(pool.astype(np.uint8))
  for i in range(1,n):
   x,y,bw,bh,area=stats[i]
   if not h*w*.0005<area<h*w*.30 or min(bw,bh)<8:continue
   if min(bw,bh)<max(bw,bh)*.12:continue
   part=labels==i
   if (part&person).sum()>max(4,area*.015):continue
   if (part&seen).sum()>area*.6:continue
   seen|=part;candidates.append((area,(int(x),int(y),int(bw),int(bh)),part))
 return sorted(candidates,key=lambda c:-c[0])[:16]

def assess(rgb,raw,decoration,equipment,person):
 keep=np.zeros(person.shape,np.float32);remove=np.zeros(person.shape,bool);decisions=[]
 guard=cv2.dilate(person.astype(np.uint8),np.ones((5,5),np.uint8))>0
 for area,(x,y,bw,bh),part in proposals(rgb,raw,decoration,equipment,guard):
  pad=max(4,round(max(bw,bh)*.08));box=[max(0,x-pad),max(0,y-pad),min(rgb.shape[1],x+bw+pad),min(rgb.shape[0],y+bh+pad)]
  result=classify(rgb[box[1]:box[3],box[0]:box[2]])
  if result['action']!='protect':
   # Frame-clipped balloons can look like an unidentifiable colour swatch
   # when tightly cropped. A second view adds contour context. It may only
   # promote a distinct coloured balloon, never wallpaper or arbitrary gear.
   context=max(12,round(max(bw,bh)*.8))
   wide=[max(0,x-context),max(0,y-context),min(rgb.shape[1],x+bw+context),min(rgb.shape[0],y+bh+context)]
   second=classify(rgb[wide[1]:wide[3],wide[0]:wide[2]])
   if result['action']=='remove':
    # Removal needs agreement in two views, not a single marginal crop.
    if second['label']!=result['label'] or second['similarity']<.245 or second['margin']<.01:
     result=dict(result,action='uncertain',contextDisagreed=True)
    else:result=dict(result,contextConfirmed=True)
   coloured=float(np.mean(np.ptp(rgb[part],axis=1)>55))>.7
   # A neutral balloon needs a compact observed silhouette, not a loose
   # rectangle across a dark backdrop or a triangular softbox corner.
   compact_dark=(area/(bw*bh)>.67 and .55<bw/bh<4.5 and
                 float(np.mean((rgb[part].mean(axis=1)<70)&(np.ptp(rgb[part],axis=1)<35)))>.85)
   agrees=(result['label']==second['label']=='balloon' and min(result['similarity'],second['similarity'])>.26
           and min(result['margin'],second['margin'])>.012)
   if second['label']=='balloon' and (second['action']=='protect' or agrees) and result['label'] in ('other','balloon') and (coloured or compact_dark):
    result=dict(second,action='protect',contextConfirmed=True)
  mask=part&~guard
  if result['action']=='protect':
   # Preserve the measured material, not the padded classification rectangle.
   keep[mask]=1
  elif result['action']=='remove':remove|=mask
  elif LABELS[result['label']][0]=='protect' and result['similarity']>=.26 and result['margin']>=.008:
   # Uncertain but plausible props must not be deleted by an older generic
   # equipment/outlier rule. Preserve their measured source material pending
   # review; the decision remains uncertain, not a claimed identification.
   keep[mask]=1
   result=dict(result,preservedForReview=True)
  # Uncertain classifications never create new removals. Existing source
  # protection is left to the caller, rather than promoting wall colour.
  decisions.append(dict(result,box=box,pixels=int(mask.sum())))
 remove &= (keep==0)&~guard
 return keep,remove,{'version':1,'model':'CLIP-ViT-B32-int8','objects':decisions,
  'uncertain':sum(d['action']=='uncertain' for d in decisions)}
