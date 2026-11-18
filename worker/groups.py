"""Per-person mask fusion for multi-person portraits, using local models only."""
import numpy as np
import cv2
from foreground import foreground_probability
from background import person_probability


def protect_group(rgb,probability,foreground,objects):
    h,w=probability.shape
    people=[o for o in objects if o['class']==0 and o['confidence']>=.45
            and (o['box'][2]-o['box'][0])*(o['box'][3]-o['box'][1])>h*w*.025]
    if len(people)<2:return probability,foreground,None
    if len(people)>12:raise ValueError('Needs Review · Group exceeds the automatic protection limit.')
    p=probability.copy()
    vicinity=cv2.dilate((probability>.1).astype(np.uint8),np.ones((9,9),np.uint8))>0
    f=np.where(vicinity,foreground,0);caps=np.zeros((h,w),np.uint8)
    for person in people:
        x0,y0,x1,y1=person['box'];px=max(8,round((x1-x0)*.12));py=max(8,round((y1-y0)*.07))
        left,top,right,bottom=max(0,x0-px),max(0,y0-py),min(w,x1+px),min(h,y1+py)
        crop=np.ascontiguousarray(rgb[top:bottom,left:right])
        local_p=person_probability(crop);local_f=foreground_probability(crop)
        # Local saliency is anchored to the detected person, rather than being
        # required to find every graduate in a single low-resolution image.
        anchored=(local_p>.50)|(probability[top:bottom,left:right]>.50)
        n,labels=cv2.connectedComponents((local_f>.03).astype(np.uint8))
        linked=np.unique(labels[anchored]);local_f=np.where(np.isin(labels,linked[linked>0]),local_f,0)
        vicinity=cv2.dilate((local_p>.1).astype(np.uint8),np.ones((9,9),np.uint8))>0
        local_f=np.where(vicinity,local_f,0)
        # Recover mortarboard corners that person/saliency networks frequently
        # omit. Only dark components linked to the detected head qualify.
        cap_bottom=min(bottom,y0+round((y1-y0)*.12))
        upper=crop[:cap_bottom-top]
        # A 65-level cutoff connects black caps to a dim grey backdrop. Use
        # the deeper blacks for recovery, retaining the existing head mask.
        dark=(cv2.cvtColor(upper,cv2.COLOR_RGB2GRAY)<45).astype(np.uint8)
        head_anchor=anchored[:len(upper)]
        head_bottom=min(bottom,y0+round((y1-y0)*.23))
        head_f=foreground_probability(np.ascontiguousarray(crop[:head_bottom-top]))
        n,cl,stats,_=cv2.connectedComponentsWithStats((head_f>.005).astype(np.uint8))
        head=np.zeros(head_f.shape,np.uint8)
        for i in range(1,n):
            component=cl==i
            if stats[i,cv2.CC_STAT_AREA]<head.size*.75 and np.any(component[:len(upper)]&head_anchor):head[component]=1
        head=head[:len(upper)]|head_anchor.astype(np.uint8)
        near_head=cv2.dilate(head,np.ones((31,31),np.uint8))>0
        recovered=(dark.astype(bool)&near_head).astype(np.uint8)
        # Gold tassels and coloured trim in this head region are foreground,
        # not neutral studio backdrop. Keep their thin disconnected strands.
        coloured=(np.ptp(upper,axis=2)>45)&(upper.mean(axis=2)>60)
        recovered[coloured&near_head]=1
        recovered[head>0]=1
        recovered=cv2.dilate(recovered,np.ones((3,3),np.uint8))
        caps[top:cap_bottom,left:right]|=recovered
        region_p=p[top:bottom,left:right];region_f=f[top:bottom,left:right]
        region_p[:]=np.maximum(region_p,local_p)
        region_f[:]=np.maximum(region_f,local_f)
        # Two independent person detections agree on this core. Do not discard
        # a black gown solely because the saliency model favours another face.
        core=(local_p>.95)&(probability[top:bottom,left:right]>.9)
        region_f[core]=np.maximum(region_f[core],.1)
    p[caps>0]=np.maximum(p[caps>0],.99);f[caps>0]=np.maximum(f[caps>0],.9)
    agreement=float(np.mean(f[p>.9]>.03)) if np.any(p>.9) else 0
    from backdrop_finish import pack
    return p,f,{'peopleDetected':len(people),'mode':'per-person mask fusion','agreement':agreement,'capMask':pack(caps*255)}
