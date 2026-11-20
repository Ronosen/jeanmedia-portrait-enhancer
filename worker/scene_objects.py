"""Source-evidence masks for detached clutter and thin supporting structures."""
import cv2
import numpy as np


def frame_panel_leaks(rgb,reference,person,support,matte=None,with_extent=False):
    """Find tall frame-connected wall material using independent person evidence."""
    h,w=support.shape;result=np.zeros((h,w),bool)
    extent=np.zeros((h,w),bool)
    if person is None:return (result,extent) if with_extent else result
    rows=np.flatnonzero((person>.5).any(axis=1))
    if not len(rows):return (result,extent) if with_extent else result
    human=person>.5
    if matte is not None:human=(person>.9)|(matte>.02)
    support=cv2.dilate(support.astype(np.uint8),np.ones((1,max(3,round(w*.02))|1),np.uint8))>0
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.dilate(cv2.Canny(gray,15,40),np.ones((3,3),np.uint8))>0
    neutral=np.ptp(rgb,axis=2)<45
    material=neutral&(rgb.mean(axis=2)-reference.mean(axis=2)>45)
    n,labels,stats,_=cv2.connectedComponentsWithStats((material&~edges&~human&~support).astype(np.uint8))
    candidates=[];anchors={}
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if not (x<w*.05 or x+bw>w*.95) or bw>w*.16 or bh<h*.05 or area<h*w*.001:continue
        part=labels==i
        if float(person[part].mean())>.1:continue
        brightness=float(np.median(rgb[part].mean(axis=1)))
        side=0 if x<w*.05 else 1
        candidates.append((i,side,brightness))
        if y<rows[0]:anchors[side]=max(anchors.get(side,0),brightness)
    for i,side,brightness in candidates:
        if side not in anchors or brightness<max(80,anchors[side]*.4):continue
        # Continuation fragments are accepted only on the anchored side and
        # on observed bright material. Dark gowns never enter a filled hull.
        part=cv2.dilate((labels==i).astype(np.uint8),np.ones((7,7),np.uint8))>0
        result |= part&neutral&(rgb.mean(axis=2)>max(80,brightness*.4))&(rgb.mean(axis=2)-reference.mean(axis=2)>35)
        x,y,bw,bh,_=stats[i]
        outer=np.zeros((h,w),bool)
        if side==0:outer[y:y+bh,:x+bw]=True
        else:outer[y:y+bh,x:]=True
        # Wallpaper outside a confirmed white backing strip belongs to the
        # same exposed room edge. Dark cloth and independent people stay out.
        result |= outer&(rgb.mean(axis=2)>max(80,brightness*.4))&(person<.2)&(np.max(np.abs(rgb.astype(float)-reference),axis=2)>35)
        if side==0:extent[:y+bh,:x+bw]=True
        else:extent[:y+bh,x:]=True
    # All donor fragments of this measured panel must be removed from the
    # background plate, including gaps between its colour components. Reject
    # foreground only where independent source/material evidence agrees.
    for side,brightness in anchors.items():
        side_extent=extent.copy()
        if side==0:side_extent[:,w//2:]=False
        else:side_extent[:,:w//2]=False
        result |= side_extent&(rgb.mean(axis=2)>max(80,brightness*.4))&(np.max(np.abs(rgb.astype(float)-reference),axis=2)>35)
    result &= ~human&~support
    return (result,extent) if with_extent else result


def corner_equipment(rgb,protected):
    """Select dark overhead gear bounded by a measured diagonal source edge."""
    h,w=protected.shape;gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,12,36);edges[protected]=0
    lines=cv2.HoughLinesP(edges,1,np.pi/180,max(20,w//12),minLineLength=min(h,w)*.12,maxLineGap=12)
    result=np.zeros((h,w),np.uint8)
    if lines is None:return result>0
    for xa,ya,xb,yb in lines[:,0]:
        dx=float(xb-xa);dy=float(yb-ya)
        if abs(dx)<10 or abs(dy)<10 or not .15<abs(dy/dx)<4:continue
        top=xa-ya*dx/dy
        for side in (0,w-1):
            side_y=ya+(side-xa)*dy/dx
            reach=top if side==0 else w-1-top
            if not w*.04<reach<w*.45 or not h*.025<side_y<h*.35:continue
            mask=np.zeros((h,w),np.uint8)
            cv2.fillPoly(mask,[np.array([[side,0],[round(top),0],[side,round(side_y)]],np.int32)],1)
            if (mask&protected).sum()>max(3,mask.sum()*.01):continue
            boundary=np.zeros((h,w),np.uint8)
            cv2.line(boundary,(round(top),0),(side,round(side_y)),1,max(5,round(min(h,w)*.015)))
            inner=(boundary>0)&(mask>0)&~protected
            outer=(boundary>0)&(mask==0)&~protected
            if inner.sum()<30 or outer.sum()<30:continue
            # A gradual vignette has no abrupt measured two-sided edge.
            if np.median(gray[inner])>35 or np.median(gray[outer])-np.median(gray[inner])<10:continue
            result |= cv2.dilate(mask,np.ones((7,7),np.uint8))
    return (result>0)&~protected


def detached_clutter(raw,protected,people,floor,rgb=None,reference=None):
    result=np.zeros(protected.shape,bool)
    if raw is None:return result,0
    h,w=protected.shape
    if raw.shape!=(h,w):raise ValueError('Invalid scene saliency dimensions.')
    human_rows=[]
    for item in people:
        if item['class']==0 and item['confidence']>=.45:
            human_rows.append((item['box'][1]+item['box'][3])/2)
    if not human_rows and protected.any():human_rows=[float(np.median(np.nonzero(protected)[0]))]
    ceiling=min(floor*h,float(np.median(human_rows))) if human_rows else floor*h
    contact=cv2.dilate(protected.astype(np.uint8),np.ones((7,7),np.uint8))>0
    body_height=max((o['box'][3]-o['box'][1] for o in people if o['class']==0),default=h)
    count,labels,stats,_=cv2.connectedComponentsWithStats((raw>.02).astype(np.uint8))
    removed=0
    for label in range(1,count):
        x,y,bw,bh,area=stats[label]
        if not h*w*.001<area<h*w*.45 or y>=ceiling:continue
        part=labels==label
        if (raw[part]>.5).sum()<area*.25 or (protected&part).sum()>max(16,area*.02):continue
        touch=part&contact
        if bh>=body_height*.4:touch[y+max(1,round(bh/3)):]=False
        if touch.any():continue
        possible_person=False
        for item in people:
            if item['class']!=0 or item['confidence']<.7:continue
            x0,y0,x1,y1=item['box']
            if part[y0:y1,x0:x1].sum()>area*.6:possible_person=True;break
        if possible_person:continue
        if rgb is not None and reference is not None:
            reach=max(5,min(round(bw*.1),round(h*.03)))
            vicinity=cv2.dilate(part.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(reach*2+1,)*2))>0
            luminance=rgb.mean(axis=2);expected=reference.mean(axis=2)
            chroma=rgb.astype(float)-luminance[...,None];expected_chroma=reference-expected[...,None]
            shadow=vicinity&(luminance<expected-8)&(np.max(np.abs(chroma-expected_chroma),axis=2)<10)
            body_guard=cv2.dilate(protected.astype(np.uint8),np.ones((7,7),np.uint8))>0
            part=part|(shadow&~body_guard)
        result |= part;removed+=1
    result=cv2.dilate(result.astype(np.uint8),np.ones((7,7),np.uint8))>0
    return result&~protected,removed


def thin_supports(rgb,alpha,objects):
    """Recover paired source edges below a person, retaining their highlights."""
    h,w=alpha.shape;gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,12,36);edges[alpha>.5]=0
    near=cv2.dilate((alpha>.8).astype(np.uint8),np.ones((max(7,round(min(h,w)*.04))|1,)*2,np.uint8))>0
    result=np.zeros((h,w),np.uint8)
    for obj in objects:
        if obj['class']!=0 or obj['confidence']<.5:continue
        x0,y0,x1,y1=obj['box'];bh=y1-y0;bw=x1-x0
        region=np.zeros((h,w),np.uint8)
        left=max(0,round((x0+x1)/2-bw*.3));right=min(w,round((x0+x1)/2+bw*.3))
        region[round(y0+bh*.4):y1,left:right]=1
        local=edges.copy();local[region==0]=0
        lines=cv2.HoughLinesP(local,1,np.pi/180,20,minLineLength=max(25,bh*.07),maxLineGap=max(8,round(bh*.025)))
        if lines is None:continue
        candidates=[]
        for xa,ya,xb,yb in lines[:,0]:
            if ya==yb or abs(xb-xa)>abs(yb-ya)*.15:continue
            if ya>yb:xa,ya,xb,yb=xb,yb,xa,ya
            cx=round((xa+xb)/2);patch=gray[ya:yb,max(0,cx-5):min(w,cx+6)]
            if np.percentile(patch,95)<120 or np.percentile(patch,95)-np.percentile(patch,5)<55:continue
            sign=float(np.median(gray[ya:yb,min(w-1,cx+3)].astype(float)-gray[ya:yb,max(0,cx-3)]))
            if abs(sign)<8:continue
            candidates.append((int(xa),int(ya),int(xb),int(yb),sign))
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                if a[4]*b[4]>=0:continue
                overlap=min(a[3],b[3])-max(a[1],b[1])
                if overlap<max(20,bh*.05):continue
                top=min(a[1],b[1]);bottom=max(a[3],b[3])
                def at(line,y):return line[0]+(y-line[1])*(line[2]-line[0])/(line[3]-line[1])
                gaps=[abs(at(a,y)-at(b,y)) for y in (top,bottom)]
                if max(gaps)>max(8,bw*.08):continue
                tx=round((at(a,top)+at(b,top))/2)
                if not near[min(h-1,top),min(w-1,max(0,tx))]:continue
                pad=max(3,round(bw*.02));extend=round(bh*.035)
                top=max(round(y0+bh*.35),top-extend);bottom=min(y1,bottom+extend)
                polygon=np.array([[min(at(a,top),at(b,top))-pad,top],[max(at(a,top),at(b,top))+pad,top],
                                  [max(at(a,bottom),at(b,bottom))+pad,bottom],[min(at(a,bottom),at(b,bottom))-pad,bottom]],np.int32)
                cv2.fillPoly(result,[polygon],255)
    return result>0


def intentional_props(raw,protected,people,floor,equipment):
    """Keep ambiguous detached source objects; equipment needs positive evidence."""
    result=np.zeros(protected.shape,np.float32)
    if raw is None:return result
    h,w=protected.shape
    n,labels,stats,_=cv2.connectedComponentsWithStats((raw>.02).astype(np.uint8))
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if not h*w*.0003<area<h*w*.45 or y>=floor*h:continue
        part=labels==i;core=part&(raw>.5)
        if core.sum()<area*.25 or (part&protected).sum()>max(16,area*.02):continue
        # Equipment evidence must cover a meaningful part of the core, rather
        # than one overlapping edge of a balloon or hanging sleeve.
        if not core.any() or (equipment&core).sum()>core.sum()*.15:continue
        result[part]=raw[part]
    return result


def decorative_props(rgb,reference,protected,equipment,floor):
    """Conservatively retain distinct coloured decorations missed by saliency."""
    h,w=protected.shape
    chroma=np.ptp(rgb,axis=2).astype(float)
    reference_chroma=np.ptp(reference,axis=2)
    # A coloured backdrop is its own material, not a prop. Require both
    # extra chroma and substantial colour separation from the fitted wall.
    contrast=np.max(np.abs(rgb.astype(float)-reference),axis=2)
    material=(chroma>80)&(chroma>reference_chroma+45)&(contrast>45)
    material &= ~protected&~equipment
    # Decorations may rest on the floor; a horizon is not a prop boundary.
    material=cv2.morphologyEx(material.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # Break thin wallpaper lines away from thick decorations before finding
    # components. Otherwise one attached stripe admits an entire wall motif.
    radius=max(2,round(min(h,w)*.006))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2)
    solid=cv2.morphologyEx(material,cv2.MORPH_OPEN,kernel)
    material=(cv2.dilate(solid,kernel)>0)&(material>0)
    n,labels,stats,_=cv2.connectedComponentsWithStats(material.astype(np.uint8))
    result=np.zeros((h,w),np.uint8)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if not h*w*.0005<area<h*w*.12 or bw>w*.35 or bh>h*.65:continue
        part=labels==i
        if cv2.distanceTransform(part.astype(np.uint8),cv2.DIST_L2,3).max()<max(3,min(h,w)*.008):continue
        # Fill only measured closed contours, never an enclosing rectangle or
        # convex hull across the gaps of a number balloon.
        contours,_=cv2.findContours(part.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result,contours,-1,255,cv2.FILLED)
    # A neutral balloon beside a coloured decoration may have almost no
    # chroma or saliency. Require an observed circular edge and a coherent
    # dark interior separated from a brighter surrounding material.
    original=result.copy()
    n,_,boxes,_=cv2.connectedComponentsWithStats((original>0).astype(np.uint8))
    for x,y,bw,bh,area in boxes[1:]:
        radius=max(bw,bh)/2
        if min(bw,bh)<12 or max(bw,bh)/max(1,min(bw,bh))>2:continue
        pad=round(radius*3);x0=max(0,x-pad);y0=max(0,y-pad);x1=min(w,x+bw+pad);y1=min(h,y+bh+pad)
        gray=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
        circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,max(10,radius),param1=60,param2=25,
            minRadius=max(8,round(radius*.55)),maxRadius=round(radius*1.5))
        if circles is None:continue
        yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]]
        for cx,cy,r in circles[0]:
            distance=np.hypot(xx-cx,yy-cy);inside=distance<r*.6;ring=(distance>r*1.08)&(distance<r*1.3)
            if inside.sum()<30 or ring.sum()<30:continue
            low=float(np.median(gray[inside]));high=float(np.median(gray[ring]))
            if low>80 or high-low<20:continue
            neutral=np.ptp(rgb[y0:y1,x0:x1],axis=2)<25
            dark=((distance<r*1.15)&(gray<(low+high)/2)&neutral).astype(np.uint8)
            count,parts,stats,_=cv2.connectedComponentsWithStats(dark)
            seed=parts[min(gray.shape[0]-1,max(0,round(float(cy)))),min(gray.shape[1]-1,max(0,round(float(cx))))]
            if not seed or stats[seed,cv2.CC_STAT_AREA]<np.pi*r*r*.35:continue
            contours,_=cv2.findContours((parts==seed).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result[y0:y1,x0:x1],contours,-1,255,cv2.FILLED)
    return (result>0)&~equipment&~protected
