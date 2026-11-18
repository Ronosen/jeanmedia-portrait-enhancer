"""Extract foreground, reconstruct a subject-free plate, then composite source."""
import cv2
import numpy as np
from PIL import Image
from backdrop_finish import pack,unpack

def trim_support_edges(rgb,support):
 # White seats cast dark shadows immediately outside their vertical sides.
 # A fitted rectangle must stop at the bright-to-dark physical object edge,
 # rather than the farther dark-to-bright edge of its cast shadow.
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=support.shape
 n,labels,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8))
 result=support.copy()
 for i in range(1,n):
  x,y,bw,bh,area=stats[i]
  if area<h*w*.005 or bh<h*.10 or bw<w*.10:continue
  left=x+round(bw*.65);right=min(w-4,x+bw+3)
  start=y+round(bh*.2);end=min(h,y+bh)
  strengths=[]
  for col in range(max(4,left),right):
   difference=gray[start:end,col-3].astype(float)-gray[start:end,col+3]
   strengths.append((float(np.mean(difference>6)),float(np.median(difference)),col))
  candidates=[v for v in strengths if v[0]>.55 and v[1]>6]
  if candidates:
   boundary=max(v[2] for v in candidates)
   result[(labels==i)&(np.arange(w)[None,:]>boundary+1)]=False
 return result

def anchored_alpha(alpha,person,objects):
 if alpha.shape!=person.shape or not np.isfinite(alpha).all():raise ValueError('Invalid foreground alpha.')
 anchors=person>.5
 contact=cv2.dilate((person>.2).astype(np.uint8),np.ones((25,25),np.uint8))>0
 for obj in objects:
  if obj['class'] in (56,57,59):
   x0,y0,x1,y1=obj['box']
   if contact[y0:y1,x0:x1].any():anchors[y0:y1,x0:x1]=True
 count,labels=cv2.connectedComponents((alpha>.02).astype(np.uint8))
 linked=np.unique(labels[anchors&(alpha>.5)])
 keep=np.isin(labels,linked[linked>0])
 if not keep.any():raise ValueError('Needs Review · No subjects could be separated safely.')
 near=cv2.dilate(keep.astype(np.uint8),np.ones((5,5),np.uint8))>0
 result=np.where(near,alpha,0).astype(np.float32)
 for obj in objects:
  if obj['class']==0 and obj['confidence']>=.45:
   x0,y0,x1,y1=obj['box'];region=result[y0:y1,x0:x1]
   # Equipment can be labelled as a person by the object detector. A missing
   # person requires independent human-segmentation evidence in that box.
   independent=person[y0:y1,x0:x1]
   if independent.size and (independent>.5).mean()<.025:continue
   if region.size and (region>.5).mean()<.10:raise ValueError('Needs Review · A person is missing from the internal subject mask.')
 return result

def prepare(plan,rgb,alpha,probability,objects):
 from inpainting import removal_mask,frame_panels
 from studio import background_basis
 h,w=alpha.shape;yy,xx=np.mgrid[:h,:w]
 model=np.clip(background_basis(xx/max(1,w-1),yy/max(1,h-1))@np.asarray(plan['coefficients']),plan['low'],plan['high'])
 alpha=alpha.copy()
 if plan['layers'].get('support'):
  # A person-only matte can omit a white pedestal or supporting seat. Retain
  # independently confirmed support geometry in the extracted source layer.
  support=unpack(plan['layers']['support'],[h,w])>0
  if support.any():
   light_backdrop=np.median(model.mean(axis=2))>140
   n,labels,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8))
   for i in range(1,n):
    part=labels==i;x,y,bw,bh,area=stats[i]
    if area<h*w*.005:continue
    threshold=float(np.percentile(rgb[part].mean(axis=1),90))-30
    if threshold<155:continue
    white=part&(rgb.mean(axis=2)>threshold)&(np.ptp(rgb,axis=2)<50)
    contours,_=cv2.findContours(white.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:continue
    hull=np.zeros((h,w),np.uint8);cv2.fillPoly(hull,[cv2.convexHull(np.concatenate(contours))],1)
    white_columns=np.flatnonzero(white.any(axis=0))
    rejected=part&(hull==0)&((yy<y+bh*.18)|(light_backdrop&(xx>=white_columns[0])))
    support[rejected]=False;alpha[rejected&(alpha<.95)]=0
  trimmed=trim_support_edges(rgb,support)
  alpha[support&~trimmed&(alpha<.95)]=0
  support=trimmed
  alpha[support]=1
 # Soft floor reflections are background, not part of the extracted person.
 # Keep the measured edge beside opaque people/supports; discard remote haze.
 distance=cv2.distanceTransform((alpha<.5).astype(np.uint8),cv2.DIST_L2,3)
 alpha[(yy>=round(plan['floor']*h))&(alpha<.5)&(distance>4)]=0
 # Reconstruction may run right up to the true matte, never a padded silhouette.
 protected=alpha>.02
 panels=frame_panels(rgb,probability,model,objects,plan['floor'])
 equipment=np.zeros(alpha.shape,np.uint8)
 if plan.get('equipmentPlan'):
  from equipment import polygon_mask
  for item in plan['equipmentPlan']['repairs']:equipment |= polygon_mask(alpha.shape,item['polygon'])
 selected=removal_mask(rgb,protected,model,plan['floor'],equipment|(panels.astype(np.uint8)*255))
 # The old exclusion margin left wall/softbox slivers beside a person. Those
 # slivers then contaminated the reconstruction donors. Compositing the source
 # foreground lets us remove these damaged background pixels right to the matte.
 selected |= (np.max(np.abs(rgb.astype(float)-model),axis=2)>38)&~protected&(yy<round(plan['floor']*h))
 from background_objects import edge_clutter
 clutter=edge_clutter(rgb,protected,plan['floor'],selected)
 # A wall-removal component may continue into a legitimate floor reflection.
 # Below the horizon only explicit clutter and the floor repair planner may
 # select pixels; healthy floor is never part of a generic backdrop repair.
 selected[round(plan['floor']*h):]=False
 # Equipment polygons may extend below the estimated wall/floor horizon.
 # Keep their complete measured footprint selected, including its bottom tip.
 clutter |= ((equipment>0)|panels)&~protected
 selected |= clutter
 layers=plan['layers'];layers['foregroundAlpha']=pack(np.rint(alpha*255).astype(np.uint8))
 layers['person']=layers['foregroundAlpha'];layers['protected']=pack(protected.astype(np.uint8)*255)
 layers['edgeProtected']=layers['protected'];layers['backdrop']=pack(selected.astype(np.uint8)*255)
 layers['clutter']=pack(clutter.astype(np.uint8)*255)
 plan['separation']={'version':1,'model':'birefnet-general-lite','workflow':'extract-clean-composite','floorMethod':'context-crop-v2','backdropMethod':'multiscale-v1'}
 if selected.mean()>.04:
  message='Large background reconstruction needs review.'
  if message not in plan.setdefault('review',[]):plan['review'].append(message)
 return selected.astype(np.float32)

def composite(source,background,alpha):
 # Solid foreground pixels are copied exactly. Only the measured edge is soft.
 matte=alpha.copy();matte[matte>=250]=255;matte[matte<=3]=0
 full=Image.fromarray(matte).resize(source.size,Image.Resampling.BICUBIC)
 # Edge pixels in the photograph already contain the old background colour.
 # Recover their foreground contribution before adding the cleaned backdrop;
 # multiplying those mixed source pixels by alpha again leaves a visible rim.
 valid=matte==0
 if not valid.any():return source.copy()
 small=np.asarray(source.resize((matte.shape[1],matte.shape[0]),Image.Resampling.LANCZOS))
 _,labels=cv2.distanceTransformWithLabels((~valid).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
 colours=small[valid];old_background=colours[np.clip(labels-1,0,len(colours)-1)]
 old_background=Image.fromarray(old_background).resize(source.size,Image.Resampling.BILINEAR)
 result=background.copy()
 for top in range(0,source.height,256):
  box=(0,top,source.width,min(source.height,top+256))
  a=np.asarray(full.crop(box),dtype=np.float32)/255
  original=np.asarray(source.crop(box));clean=np.asarray(background.crop(box),dtype=np.float32)
  donor=np.asarray(old_background.crop(box),dtype=np.float32)
  foreground=np.clip((original.astype(np.float32)-(1-a[...,None])*donor)/np.maximum(a[...,None],1/255),0,255)
  mixed=np.rint(np.clip(foreground*a[...,None]+clean*(1-a[...,None]),0,255)).astype(np.uint8)
  mixed[a==1]=original[a==1]
  result.paste(Image.fromarray(mixed),box[:2])
 return result

def render(source,selected,plan):
 from neural import repair
 from floor import decode_mask,apply_floor_patches
 from backdrop_finish import finish
 from runtime import progress
 shape=plan['layers']['shape'];matte=unpack(plan['layers']['foregroundAlpha'],shape)
 foreground=matte>3;holes=(selected>0)|foreground
 if plan.get('floorRepairs'):holes |= decode_mask(plan['floorRepairs']['extension'],tuple(shape))>0
 plate=plan.setdefault('backgroundPlate',{'plateOnly':True,'excludeForegroundContext':False})
 floor_method=plan.get('separation',{}).get('floorMethod')
 source_floor=floor_method in ('source-texture-v1','context-crop-v1','context-crop-v2')
 if plan.get('floorRepairs') and not source_floor:
  plate.update(floorRepairs=plan['floorRepairs'],floor=plan['floor'],adaptiveRevision=plan.get('adaptiveRevision',3))
 progress('Removing subjects from background plate',60)
 background=repair(source,holes,np.zeros(shape,bool),plate)
 progress('Cleaning separated background',78)
 protected=matte>3
 if plan.get('floorRepairs') and source_floor:
  # Rebuild only missing floor pieces from adjacent original floor texture.
  # The subject-free neural plate can invent shadows or retain dark folds;
  # its floor prediction is therefore not used as visible replacement texture.
  patches=dict(plan['floorRepairs']);patches['engine']='texture-v1';plan['floorRepairs']['engine']='texture-v1'
  clutter=unpack(plan['layers']['clutter'],shape)>0 if plan['layers'].get('clutter') else np.zeros(shape,bool)
  extension=decode_mask(patches['extension'],tuple(shape))>0
  ground_clutter=clutter&(np.arange(shape[0])[:,None]>=round((plan['floor']-.08)*shape[0]))
  context_floor=(floor_method=='context-crop-v1' and extension.any()) or (floor_method=='context-crop-v2' and (extension.any() or ground_clutter.any()))
  if context_floor and plan['floor']<.98:
   from floor_context import reconstruct
   repaired,full_mask=reconstruct(source,extension,protected,clutter,plan['floor'],plan.setdefault('floorPlate',{'version':2 if floor_method=='context-crop-v2' else 1,'blendRevision':1}),selected>0)
   defects=dict(patches);defects['extension']=pack(np.zeros(shape,np.uint8))
   repaired=apply_floor_patches(repaired,plan['floor'],defects,protected|clutter)
   defect_mask=Image.fromarray(decode_mask(patches['defects'],tuple(shape))).resize(source.size,Image.Resampling.NEAREST)
   full_mask=Image.fromarray(np.maximum(np.asarray(full_mask),np.asarray(defect_mask)))
   plan['floorRepairs']['engine']='lama-crop-v1'
  else:
   repaired=apply_floor_patches(source,plan['floor'],patches,protected|clutter)
   floor_mask=(decode_mask(patches['defects'],tuple(shape))>0)|extension
   full_mask=Image.fromarray(floor_mask.astype(np.uint8)*255).resize(source.size,Image.Resampling.NEAREST)
  background=Image.composite(repaired,background,full_mask)
 elif plan.get('floorRepairs'):
  patches=dict(plan['floorRepairs'])
  if not plate.get('floorFillCloned'):patches['extension']=pack(np.zeros(shape,np.uint8))
  else:
   # Extend actual adjacent floor texture when the neural fill was rejected.
   # A blurred surface continuation erased grain and reflections in this area.
   patches['engine']='clone-v1'
   plan['floorRepairs']['engine']='clone-v1'
  background=apply_floor_patches(background,plan['floor'],patches,protected)
 background=finish(background,plan)
 progress('Combining protected subjects and background',90)
 result=composite(source,background,matte)
 if plan.get('propRepairs'):
  from props import heal_props
  result=heal_props(result,plan['propRepairs'])
 return result

def replay(image,preview,plan,export):
 import base64
 from floor import validate_floor_patches
 shape=[preview.height,preview.width]
 if plan.get('separation',{}).get('version')!=1 or plan.get('layers',{}).get('shape')!=shape:
  raise ValueError('Invalid saved subject separation.')
 encoded=plan['layers']['source']
 if not isinstance(encoded,str) or len(encoded)>8_000_000:raise ValueError('Invalid saved separation source.')
 raw=base64.b64decode(encoded,validate=True)
 source=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
 if source is None or source.shape!=np.asarray(preview).shape or not np.array_equal(source,np.asarray(preview)):
  raise ValueError('Saved separation belongs to a different source photo. Run Auto Edit again.')
 matte=unpack(plan['layers']['foregroundAlpha'],shape)
 if plan.get('floorRepairs'):validate_floor_patches(matte>3,plan['floor'],plan['floorRepairs'])
 selected=unpack(plan['layers']['backdrop'],shape)>0
 out=render(preview,selected,plan)
 full=render(image,selected,plan) if export else image
 return full,out,plan
