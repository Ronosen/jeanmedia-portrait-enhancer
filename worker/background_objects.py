"""Locate disconnected frame-edge studio clutter, keeping protected supports."""
import cv2
import numpy as np

def edge_clutter(rgb,protected,floor,removed_wall=None):
 h,w=protected.shape;result=np.zeros((h,w),np.uint8)
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 edges=cv2.morphologyEx(cv2.Canny(gray,8,24),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 # A box cut off by the frame has no closed contour. Recover its neutral bright
 # surface separately; broad white backdrops fail the size/position checks.
 bright=(gray>110)&(np.ptp(rgb,axis=2)<50)&~protected
 floor_pixels=gray[round(floor*h):][~protected[round(floor*h):]]
 dark_floor=len(floor_pixels)>0 and np.median(floor_pixels)<100
 for lower in (False,True):
  candidate=bright.copy()
  if lower:candidate[:round(h*.70)]=False
  count,components,box_stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
  for i in range(1,count):
   x,y,bw,bh,area=box_stats[i]
   thin_panel=lower and dark_floor and 1<=bw<=w*.03 and h*.05<bh<h*.3 and area>h*w*.0001
   box=w*.03<bw<w*.28 and h*.03<bh<h*.3 and h*w*.002<area<h*w*.08
   nearby=gray[min(h-1,y+bh):min(h,y+bh+max(4,round(h*.02))),max(0,x-round(w*.05)):min(w,x+bw)]
   local_box=box and nearby.size and float(np.median(gray[components==i]))-float(np.median(nearby))>30
   if (x==0 or x+bw==w) and y>h*.5 and (y<floor*h or (dark_floor or local_box) and y<h*.9) and (box or thin_panel):
    part=(components==i).astype(np.uint8)
    outline,_=cv2.findContours(part,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(result,[cv2.convexHull(np.concatenate(outline))],255)
 for contour in contours:
  hull=cv2.convexHull(contour);area=cv2.contourArea(hull)
  if not h*w*.003<area<h*w*.08:continue
  x,y,bw,bh=cv2.boundingRect(hull)
  if y<h*.48 or bw<w*.05 or bh<h*.04 or not .3<bw/max(1,bh)<3:continue
  # Floor tiles and rectangular reflections are surface texture, not boxes.
  # A background support/object must rise above the inferred floor boundary.
  if y>=floor*h-h*.03:continue
  if not (x<w*.22 or x+bw>w*.78):continue
  polygon=cv2.approxPolyDP(hull,cv2.arcLength(hull,True)*.035,True)
  if not 4<=len(polygon)<=6:continue
  region=np.zeros((h,w),np.uint8);cv2.fillPoly(region,[polygon],1);part=region>0
  pixels=rgb[part]
  if protected[part].mean()>.3 or np.mean((pixels.mean(axis=1)>110)&(np.ptp(pixels,axis=1)<50))<.7:continue
  result[part&~protected]=255
 result=cv2.dilate(result,np.ones((max(5,round(h*.10))|1,)*2,np.uint8))
 # Thin diagonal stand legs can cross the floor without differing strongly
 # from a dark mat. Select rods entering from a side edge, away from people.
 floor_edges=edges.copy();floor_edges[:round(max(.55,floor-.15)*h)]=0;floor_edges[protected]=0
 rods=cv2.HoughLinesP(floor_edges,1,np.pi/180,max(20,w//14),minLineLength=h*.04,maxLineGap=10)
 if rods is not None:
  for x0,y0,x1,y1 in rods[:,0]:
   dx=abs(x1-x0);dy=abs(y1-y0)
   if dx<8 or not .5<dy/dx<3:continue
   if not (min(x0,x1)<w*.04 and max(x0,x1)<w*.27 or max(x0,x1)>w*.96 and min(x0,x1)>w*.73):continue
   xs=np.linspace(x0,x1,50).astype(int);ys=np.linspace(y0,y1,50).astype(int)
   if np.median(gray[ys,xs])>55:continue
   cv2.line(result,(int(x0),int(y0)),(int(x1),int(y1)),255,max(7,round(w*.022)))
 # Remove strongly coloured reflections belonging to removed edge clutter.
 # Reflections matching saturated foreground clothing are retained.
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 fg=protected&(hsv[:,:,1]>150)&(hsv[:,:,2]>70)
 bins=np.bincount((hsv[:,:,0][fg]//10),minlength=18)
 hues=np.flatnonzero(bins>max(10,int(fg.sum()*.01)))
 matches=np.zeros((h,w),bool)
 for hue in hues:
  diff=np.abs(hsv[:,:,0].astype(int)-(hue*10+5));same=np.minimum(diff,180-diff)<13
  columns=(fg&same).any(axis=0).astype(np.uint8)[None,:]
  aligned=cv2.dilate(columns,np.ones((1,max(3,round(w*.16))|1),np.uint8))>0
  matches |= same&aligned
 yy,xx=np.mgrid[:h,:w]
 # A coloured edge object that is being removed can leave a matching coloured
 # reflection on neutral paper. Its hue can overlap skin, so foreground-hue
 # retention must not automatically preserve that identified clutter reflection.
 if removed_wall is not None:
  coloured=removed_wall&(yy<round(floor*h))&(yy>h*.5)&((xx<w*.2)|(xx>w*.8))&(hsv[:,:,1]>100)
  removed_bins=np.bincount(hsv[:,:,0][coloured]//10,minlength=18)
  for hue in np.flatnonzero(removed_bins>max(20,h*w*.0001)):
   diff=np.abs(hsv[:,:,0].astype(int)-(hue*10+5));matches[np.minimum(diff,180-diff)<13]=False
 for zone in [(yy<round(floor*h))&~protected,(yy>=round(floor*h))&~protected]:
  if zone.any():
   colour=np.median(rgb[zone],axis=0).astype(np.uint8)
   ch=cv2.cvtColor(colour.reshape(1,1,3),cv2.COLOR_RGB2HSV)[0,0]
   if ch[1]>100 and ch[2]>80:
    diff=np.abs(hsv[:,:,0].astype(int)-int(ch[0]));matches |= np.minimum(diff,180-diff)<13
 reflections=(yy>=round(floor*h))&((xx<w*.25)|(xx>w*.75))&~protected&~matches&(hsv[:,:,1]>65)&(hsv[:,:,2]>50)
 n,labels=cv2.connectedComponents(reflections.astype(np.uint8))
 seed=reflections&((xx<w*.09)|(xx>w*.94));linked=np.unique(labels[seed])
 result[np.isin(labels,linked[linked>0])]=255
 # Include desaturated cloth borders and cast-colour halos, otherwise the
 # reconstruction network sees those old fragments as valid donor context.
 result=cv2.dilate(result,np.ones((max(3,round(min(h,w)*.05))|1,)*2,np.uint8));result[protected]=0
 return result>0
