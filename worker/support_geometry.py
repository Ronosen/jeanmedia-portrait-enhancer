"""Recover large neutral box seats using connected right-angle edges."""
import cv2
import numpy as np


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
