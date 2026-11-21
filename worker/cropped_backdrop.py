"""Independent source evidence for cropped portraits on light studio backdrops."""
import cv2
import numpy as np

def frame_intrusions(rgb,protected,reference,revision=22):
 """Measured exposed studio borders; independent of any floor estimate."""
 h,w=protected.shape;safe=~cv2.dilate(protected.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
 material=(np.max(np.abs(rgb.astype(float)-reference),axis=2)>50)&safe
 material=cv2.morphologyEx(material.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 n,labels,stats,_=cv2.connectedComponentsWithStats(material);result=np.zeros((h,w),bool)
 for i in range(1,n):
  x,y,bw,bh,area=stats[i]
  side=(x==0 or x+bw==w) and bh>h*.5 and bw<w*.18
  if not (y==0 or side) or not h*w*.001<area<h*w*.23:continue
  part=labels==i;row=part.sum(axis=1);col=part.sum(axis=0)
  thin=np.median(row[row>0])<w*.18 or np.median(col[col>0])<h*.18
  if not thin or not (bw>w*.2 or bh>h*.5):continue
  result|=(cv2.dilate(part.astype(np.uint8),np.ones((5,5),np.uint8))>0)&safe
 # Include nearby source-visible border remnants even when a global fitted
 # surface was biased by that exposed wall. Grow only next to confirmed gear,
 # using observed upper-backdrop donors, never a rectangular frame template.
 rows=np.nonzero(protected)[0]
 if result.any() and len(rows) and np.median(reference.mean(axis=2))<80:
  upper=np.arange(h)[:,None]<np.median(rows)
  donors=safe&~result&upper&(rgb.mean(axis=2)<80)&(np.ptp(rgb,axis=2)<35)
  if donors.sum()>100:
   colour=np.median(rgb[donors],axis=0)
   different=np.max(np.abs(rgb.astype(float)-colour),axis=2)>25
   size=max(9,round(min(h,w)*.04))|1
   nearby=cv2.dilate(result.astype(np.uint8),np.ones((size,size),np.uint8))>0
   result |= nearby&different&upper&safe
 result=(cv2.dilate(result.astype(np.uint8),np.ones((9,9),np.uint8))>0)&safe
 if revision>=23 and len(rows):
  # Include the apparatus' dark contact shadow below a bright rail. Colour
  # contrast alone misses that shadow because it resembles the black cloth.
  # Follow the measured gear edge; never introduce a fixed horizon or floor.
  upper=np.arange(h)[:,None]<np.median(rows)
  radius=max(5,round(min(h,w)*.02))
  kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
  result|=(cv2.dilate((result&upper).astype(np.uint8),kernel)>0)&upper&safe
 return result

def assess(rgb,protected,reference):
 h,w=protected.shape;empty=np.zeros((h,w),bool)
 rows=np.flatnonzero(protected.sum(axis=1)>max(3,w*.01))
 # A visible full-body scene remains subject to the measured floor contour.
 cropped=len(rows)>0 and rows[-1]>=h*.98
 match=np.max(np.abs(rgb.astype(float)-reference),axis=2)<38
 safe=~cv2.dilate(protected.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
 if not cropped or np.median(reference.mean(axis=2))<135 or (match&safe).mean()<.25:
  return empty,empty,{'method':'cropped-light-material-v1','detected':False}
 outlier=(np.max(np.abs(rgb.astype(float)-reference),axis=2)>55)&safe
 outlier=cv2.morphologyEx(outlier.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 n,labels,stats,_=cv2.connectedComponentsWithStats(outlier)
 gear=empty.copy();accepted=[]
 for i in range(1,n):
  x,y,bw,bh,area=stats[i]
  if not (x==0 or x+bw==w):continue
  if not h*w*.002<area<h*w*.23:continue
  part=labels==i
  widths=part.sum(axis=1);observed=widths[widths>0]
  # A top corner light can join the exposed side wall. Measure the typical
  # source strip width instead of rejecting the whole connected component.
  panel=(bh>h*.5 and np.median(observed)<w*.18 and
         np.median(np.ptp(rgb[part],axis=1))>35)
  contours,_=cv2.findContours(part.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contour=max(contours,key=cv2.contourArea);hull=cv2.convexHull(contour)
  polygon=cv2.approxPolyDP(contour,cv2.arcLength(contour,True)*.018,True)
  solid=cv2.contourArea(contour)/max(1,cv2.contourArea(hull))
  pixels=rgb[part];dark=np.median(pixels.mean(axis=1))<65 and np.median(np.ptp(pixels,axis=1))<35
  polygonal=4<=len(polygon)<=6 and solid>.9 and dark
  if not (panel or polygonal):continue
  ring=(cv2.dilate(part.astype(np.uint8),np.ones((15,15),np.uint8))>0)&~part&safe
  if (ring&match).sum()<max(30,ring.sum()*.45):continue
  # Only measured pixels and a small source edge band, never a bounding box.
  edge=cv2.dilate(part.astype(np.uint8),np.ones((5,5),np.uint8))>0
  gear|=edge&safe
  accepted.append({'box':[int(x),int(y),int(bw),int(bh)],'kind':'edge-panel' if panel else 'polygonal-dark-equipment'})
 wall=(match|gear)&safe
 return wall,gear,{'method':'cropped-light-material-v1','detected':True,'objects':accepted,'floorPreserved':True}
