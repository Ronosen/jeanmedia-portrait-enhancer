"""Measured studio perimeter and object relationships; no filename templates."""
import cv2
import numpy as np


def backdrop_boundary(rgb, reference, protected):
    h,w=protected.shape;empty={'version':1,'detected':False,'method':'observed-rail-v1'}
    if min(h,w)<40 or reference.shape!=rgb.shape:return empty
    rows=np.nonzero(protected)[0]
    if not len(rows):return empty
    # Hardware must be above most of the actual foreground, not at a fixed
    # percentage of the photograph. Raised hands/caps remain protected.
    limit=float(np.percentile(rows,30))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,35,100)
    edges[np.arange(h)[:,None]>=np.full((h,w),limit)]=0
    edges[cv2.dilate(protected.astype(np.uint8),np.ones((5,5),np.uint8))>0]=0
    lines=cv2.HoughLinesP(edges,1,np.pi/720,max(15,round(w*.18)),minLineLength=max(20,round(w*.32)),maxLineGap=max(5,round(w*.05)))
    if lines is None:return empty
    candidates=[];x=np.arange(w);gap=max(3,round(min(h,w)*.012))
    for x0,y0,x1,y1 in lines[:,0]:
        if abs(x1-x0)<w*.32:continue
        slope=(y1-y0)/(x1-x0)
        if abs(slope)>.25:continue
        contour=y0+slope*(x-x0)
        valid=(contour>gap)&(contour<h-gap*4)&(contour<limit)
        if valid.mean()<.65:continue
        columns=np.flatnonzero(valid);cy=np.rint(contour[valid]).astype(int)
        below=np.clip(cy+gap*3,0,h-1);above=np.clip(cy-gap*2,0,h-1)
        safe=~protected[below,columns]&~protected[above,columns]
        if safe.sum()<w*.4:continue
        columns=columns[safe];cy=cy[safe];below=below[safe];above=above[safe]
        lower_error=np.max(np.abs(rgb[below,columns].astype(float)-reference[below,columns]),axis=1)
        upper_error=np.max(np.abs(rgb[above,columns].astype(float)-reference[above,columns]),axis=1)
        rail=rgb[cy,columns].astype(float)
        # A long rail alone is insufficient: the material beneath it must
        # match the intended backdrop, while the room above differs.
        evidence=(lower_error<38)&(upper_error>45)
        brightness=(rail.mean(axis=1)>95)&(np.ptp(rail,axis=1)<65)
        fraction=float(evidence.mean())
        if fraction<.6 or brightness.mean()<.35:continue
        candidates.append((fraction*len(columns)/w,contour,slope,int(y0-slope*x0)))
    if not candidates:return empty
    score,contour,slope,intercept=max(candidates,key=lambda item:item[0])
    return {'version':1,'detected':True,'method':'observed-rail-v1',
            'rows':np.clip(np.rint(contour+gap),0,h-1).astype(int).tolist(),
            'support':round(score,4),'slope':round(float(slope),5),'intercept':intercept}


def exposed_room(shape, context, protected):
    if not context.get('detected'):return np.zeros(shape,bool)
    h,w=shape;rows=np.asarray(context.get('rows',[]))
    if rows.shape!=(w,) or not np.isfinite(rows).all():return np.zeros(shape,bool)
    return (np.arange(h)[:,None]<rows[None,:])&~protected


def object_relationship(part, protected, context):
    """Evidence of room decoration, not a universal balloon-removal rule."""
    area=int(part.sum());empty={'outsideBackdrop':0.0,'nearSubject':False,'roomDecoration':False}
    if not area:return empty
    h,w=part.shape;radius=max(3,round(min(h,w)*.04))
    near=cv2.dilate(protected.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))>0
    contact=bool((part&near).any())
    outside=exposed_room(part.shape,context,np.zeros(part.shape,bool))
    fraction=float((part&outside).sum()/area)
    return {'outsideBackdrop':round(fraction,4),'nearSubject':contact,
            'roomDecoration':bool(context.get('detected') and fraction>.95 and not contact)}
