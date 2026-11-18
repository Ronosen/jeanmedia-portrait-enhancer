"""Recover large neutral box seats using connected right-angle edges."""
import cv2
import numpy as np


def support_contacts(alpha,support,props=None):
    """Seal internal joins between confirmed props and adjacent foreground.

    A hard prop/person join is already photographed foreground. Re-matting its
    two sides independently invents background cracks and stair-stepped edges.
    External prop edges without an adjacent foreground core stay unexpanded.
    """
    props=np.zeros_like(support,dtype=bool) if props is None else props
    hard=support|props
    if not hard.any():return np.zeros_like(hard)
    radius=max(3,round(min(support.shape)*.01))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2)
    outside=cv2.dilate(hard.astype(np.uint8),np.ones((3,3),np.uint8))==0
    adjacent=(alpha>.9)&outside
    contact=(cv2.dilate(hard.astype(np.uint8),kernel)>0)&(cv2.dilate(adjacent.astype(np.uint8),kernel)>0)
    if support.any() and props.any():
        contact |= (cv2.dilate(support.astype(np.uint8),kernel)>0)&(cv2.dilate(props.astype(np.uint8),kernel)>0)
    return contact


def support_source_masks(rgb,model,alpha,support,props):
    contact=support_contacts(alpha,support,props)
    radius=max(2,round(min(support.shape)*.005))
    distance=cv2.distanceTransform(contact.astype(np.uint8),cv2.DIST_L2,3)
    blend=np.clip(distance/radius,0,1)
    reach=max(3,round(min(support.shape)*.025))
    near=cv2.dilate(support.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(reach*2+1,)*2))>0
    surface=near&(rgb.mean(axis=2)-model.mean(axis=2)>10)
    exact=surface|props|(blend>=1)
    blend[exact]=1
    return exact,blend.astype(np.float32)


def attached_props(rgb,alpha,support,model):
    """Retain compact source objects contacting both a support top and a hand."""
    result=np.zeros_like(support,dtype=bool);h,w=support.shape
    person=(alpha>.5)&~support
    source_edges=cv2.dilate(cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),12,30),np.ones((3,3),np.uint8))>0
    n,labels,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8))
    for index in range(1,n):
        x,y,bw,bh,area=stats[index]
        if area<h*w*.005 or bh<h*.08:continue
        top=max(0,y-round(bh*.3));bottom=min(h,y+4)
        if y-top<4:continue
        region=np.s_[top:bottom,x:x+bw]
        contrast=np.max(np.abs(rgb[region].astype(float)-model[region]),axis=2)>25
        contrast &= ~person[region]
        contrast=cv2.morphologyEx(contrast.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,parts,boxes,_=cv2.connectedComponentsWithStats(contrast)
        hand=cv2.dilate(person[region].astype(np.uint8),np.ones((7,7),np.uint8))>0
        for part in range(1,count):
            px,py,pw,ph,size=boxes[part];mask=parts==part
            if size<max(20,area*.005) or ph<4 or py+ph<y-top-3:continue
            # A component cut by the search window is not a measured object
            # silhouette. Its convex hull would preserve a rectangle of wall.
            if py==0:continue
            if not (mask&hand).any():continue
            positive=(rgb[region].mean(axis=2)-model[region].mean(axis=2)>15)
            if positive[mask].mean()<.15:
                # A diffuse cast shadow can touch both the arm and pedestal.
                # A dark prop must have a physical source edge, not merely a
                # threshold crossing or the artificial edge of this crop.
                boundary=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
                boundary &= ~hand
                if not boundary.any() or source_edges[region][boundary].mean()<.45:continue
            # Source geometry, including dark printed details between bright
            # book covers, belongs to the contacting prop, not the backdrop.
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            hull=cv2.convexHull(np.concatenate(contours));local=np.zeros(mask.shape,np.uint8)
            cv2.fillPoly(local,[hull],1);result[region] |= local>0
    return result


def box_seats(rgb,person):
    h,w=person.shape;result=np.zeros_like(person)
    rows=np.flatnonzero(person.any(axis=1))
    if not len(rows):return result
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),8,24)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,max(25,w//12),minLineLength=min(h,w)*.10,maxLineGap=max(12,round(h*.025)))
    if lines is None:return result
    vertical=[];horizontal=[]
    for x0,y0,x1,y1 in lines[:,0]:
        if abs(x1-x0)<12 and abs(y1-y0)>h*.10:vertical.append((int((x0+x1)/2),min(y0,y1),max(y0,y1)))
        if abs(y1-y0)<max(15,abs(x1-x0)*.18) and abs(x1-x0)>w*.15:horizontal.append((min(x0,x1),max(x0,x1),int((y0+y1)/2)))
    merged=[]
    for x,top,bottom in sorted(vertical):
        for i,(mx,mt,mb) in enumerate(merged):
            if abs(mx-x)<10 and top<=mb+h*.08 and bottom>=mt-h*.08:
                merged[i]=(round((mx+x)/2),min(mt,top),max(mb,bottom));break
        else:merged.append((x,top,bottom))
    vertical=merged
    contact=cv2.dilate(person.astype(np.uint8),np.ones((31,31),np.uint8))>0
    for x,top,bottom in vertical:
        if x<w*.08 or x>w*.98 or top<rows[0]+(rows[-1]-rows[0])*.4:continue
        for left,right,y in horizontal:
            if abs(y-bottom)>h*.06 or min(abs(x-left),abs(x-right))>w*.03:continue
            width=right-left;height=y-top;area=width*height
            if not (w*.15<width<w*.85 and h*.10<height<h*.45 and .025*h*w<area<.30*h*w):continue
            if left<3 or right>w-3 or top<0 or y>=h-3:continue
            if np.count_nonzero(contact[top:y,left:right])<area*.002:
                top=max(int(rows[0]+(rows[-1]-rows[0])*.4),top-round(h*.10))
            surface=rgb[top:y,left:right];neutral=np.ptp(surface,axis=2)<35
            light=surface.mean(axis=2)>135
            if np.mean(neutral&light)<.65 or np.count_nonzero(contact[top:y,left:right])<area*.002:continue
            result[max(0,top-1):y+2,max(0,left-1):right+2]=True
            for sx,st,sb in vertical:
                if left-width*.45<sx<left-width*.12 and st<y and sb>top+height*.25:
                    side=np.array([[left,top],[sx,max(top,st-4)],[sx,y-round(height*.12)],[left,y]],np.int32)
                    cv2.fillPoly(result.view(np.uint8),[side],1)
    return result
