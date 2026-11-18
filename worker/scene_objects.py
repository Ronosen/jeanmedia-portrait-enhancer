"""Source-evidence masks for detached clutter and thin supporting structures."""
import cv2
import numpy as np


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
        x,y,bw,bh,area=stats[label];part=labels==label
        if not h*w*.001<area<h*w*.45 or y>=ceiling:continue
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
