"""Match a reconstructed equipment hole to observed dark-backdrop donors.

Only correct measured colour drift; retain reconstructed texture and leave the
floor, foreground and healthy source backdrop outside the repair untouched.
"""
import cv2
import numpy as np
from PIL import Image
from backdrop_finish import unpack,pack

def match(source,background,plan):
 if plan.get('sceneCleanupRevision',0)<22 or 'coefficients' not in plan:return background
 from studio import background_basis
 h,w=plan['layers']['shape'];yy,xx=np.mgrid[:h,:w]
 ref=np.clip(background_basis(xx/max(1,w-1),yy/max(1,h-1))@np.asarray(plan['coefficients']),plan['low'],plan['high'])
 if np.median(ref.mean(axis=2))>=80 or np.median(np.ptp(ref,axis=2))>=35:return background
 protected=unpack(plan['layers']['protected'],[h,w])>0
 anchors=unpack(plan['layers'].get('anchoredForegroundAlpha',plan['layers']['protected']),[h,w])>3
 rows=np.nonzero(anchors)[0]
 if not len(rows):return background
 # Source-derived upper-body extent is a safety limit, not a floor horizon.
 upper=yy<float(np.median(rows))
 clutter=unpack(plan['layers']['clutter'],[h,w])>0
 ownership_surface=unpack(plan['layers']['wallSurface'],[h,w])>0 if plan['layers'].get('wallSurface') else np.ones((h,w),bool)
 selected=clutter&upper&~protected
 src=np.asarray(source.resize((w,h),Image.Resampling.LANCZOS)).astype(float)
 predicted=np.asarray(background.resize((w,h),Image.Resampling.LANCZOS)).astype(float)
 donor=~protected&~clutter&upper&(np.max(np.abs(src-ref),axis=2)<30)
 correction=np.zeros((h,w,3),np.float32);count=0;owned=np.zeros((h,w),bool)
 n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8))
 for i in range(1,n):
  if stats[i,cv2.CC_STAT_AREA]<h*w*.0005:continue
  part=labels==i;radius=max(9,round(min(h,w)*.08))|1
  ring=(cv2.dilate(part.astype(np.uint8),np.ones((radius,radius),np.uint8))>0)&donor
  if ring.sum()<60:continue
  # This specialist also owns already-correct dark repairs. A later generic
  # harmonizer must not reinterpret a smooth rail transition as colour drift.
  if plan.get('sceneCleanupRevision',0)>=25:owned|=part&ownership_surface
  # Fit illumination from actual nearby donors, not an extrapolation of a
  # global model that may have been biased by the removed bright frame.
  plane=np.stack([np.ones((h,w)),xx/max(1,w-1),yy/max(1,h-1)],axis=2)
  coefficients=np.linalg.lstsq(plane[ring],src[ring],rcond=None)[0]
  low=np.percentile(src[ring],5,axis=0)-3;high=np.percentile(src[ring],95,axis=0)+3
  local=np.clip(plane@coefficients,low,high)
  source_residual=(src-local)[ring];repair_residual=(predicted-local)[part]
  desired=np.median(source_residual,axis=0);actual=np.median(repair_residual,axis=0)
  spread=np.percentile(source_residual,90,axis=0)-np.percentile(source_residual,10,axis=0)
  drift=np.percentile(np.max(np.abs(repair_residual-desired),axis=1),97)
  if drift<max(12,float(np.max(spread))*2):continue
  repaired_spread=np.percentile(repair_residual,97,axis=0)-np.percentile(repair_residual,3,axis=0)
  gain=np.clip(spread/np.maximum(1,repaired_spread),.05,1)
  target=local[part]+desired+(repair_residual-actual)*gain
  correction[part]=np.clip(target-predicted[part],-100,100)
  if plan.get('sceneCleanupRevision',0)>=23:
   # Neural reconstruction already meets the original at the mask edge.
   # Fade colour transfer inside that measured hole so its correction cannot
   # introduce a new contour along the old apparatus silhouette.
   # Protected holes are composited back from the source; they are not a
   # junction with healthy background and must not keep a bright repair rim.
   foreground_guard=cv2.dilate(protected.astype(np.uint8),np.ones((9,9),np.uint8))>0
   # The upper-body safety cutoff is not a visible source/repair boundary.
   # Treat its excluded region as continuous when measuring the blend so
   # that a component crossing the cutoff cannot acquire a horizontal band.
   surface=part|foreground_guard|~upper
   # A rail shadow joins the backdrop beneath the apparatus. Measure that
   # downward join only: fading at side-panel edges retains a pale vertical
   # stripe instead of correcting the reconstructed panel completely.
   next_edge=np.minimum.accumulate(np.where(~surface,yy,h)[::-1],axis=0)[::-1]
   # Limit this transition to hardware above the measured foreground. Side
   # repairs lower in the scene keep full colour correction, including where
   # a panel mask terminates; otherwise they can retain a rectangular remnant.
   distance=np.where(next_edge<=np.percentile(rows,5),next_edge-yy,h)
   t=np.clip((distance-1)/max(5,min(h,w)*.035),0,1)
   correction[part]*=(t*t*(3-2*t))[part,None]
  count+=1
 plan['reconstructionTone']={'version':1,'components':count,'method':'observed-donor-residual-v1'}
 if plan.get('sceneCleanupRevision',0)>=25:
  # The specialist's upper-body safety cutoff is not a real wall edge. Keep
  # ownership continuous across an entire connected wall repair so the next
  # stage cannot create a new band at that internal cutoff. Actual floor
  # regions remain available to their own repair path.
  _,whole=cv2.connectedComponents((clutter&ownership_surface&~protected).astype(np.uint8))
  linked=np.unique(whole[owned]);linked=linked[linked>0]
  if len(linked):owned|=np.isin(whole,linked)
  plan['reconstructionTone'].update(version=2,ownedMask=pack(owned.astype(np.uint8)*255),owner='dark-backdrop-specialist')
 if not count:return background
 # Add a smoothly enlarged correction to the native neural result; this is
 # colour transfer of the reconstructed texture, never a solid RGB overlay.
 result=Image.new('RGB',background.size)
 channels=[Image.fromarray(correction[:,:,c],mode='F').resize(background.size,Image.Resampling.BILINEAR) for c in range(3)]
 for top in range(0,background.height,256):
  box=(0,top,background.width,min(background.height,top+256))
  pixels=np.asarray(background.crop(box)).astype(float)
  delta=np.stack([np.asarray(c.crop(box)) for c in channels],axis=2)
  result.paste(Image.fromarray(np.rint(np.clip(pixels+delta,0,255)).astype(np.uint8)),box[:2])
 return result
