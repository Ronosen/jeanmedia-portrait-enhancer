"""Adaptive low-frequency backdrop cleaning, retaining source grain and shading."""
import base64
import cv2
import numpy as np
from PIL import Image


def pack(pixels):
    return base64.b64encode(cv2.imencode('.png',pixels)[1]).decode('ascii')


def unpack(encoded,shape):
    if not isinstance(encoded,str) or len(encoded)>4_000_000:raise ValueError('Invalid backdrop finish')
    raw=base64.b64decode(encoded,validate=True)
    if len(raw)<24 or raw[:8]!=b'\x89PNG\r\n\x1a\n' or int.from_bytes(raw[16:20],'big')!=shape[1] or int.from_bytes(raw[20:24],'big')!=shape[0]:
        raise ValueError('Invalid backdrop finish dimensions')
    pixels=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
    if pixels is None or list(pixels.shape)!=list(shape):raise ValueError('Invalid backdrop finish')
    return pixels


def finish(image,plan):
    if not plan.get('adaptiveRevision'):return image
    shape=plan['layers']['shape'];h,w=shape
    cached=plan.get('backdropFinish')
    if cached and cached.get('pop')=='black' and cached.get('edgeFinish')!=1:
        cached=None
    if cached is None:
        protected=unpack(plan['layers']['protected'],shape)>0
        if plan.get('adaptiveRevision',0)>=3 and not plan.get('partialCleanup') and not plan.get('separation') and plan['layers'].get('source'):
            raw=base64.b64decode(plan['layers']['source'],validate=True)
            source=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if source is not None and source.shape[:2]==(h,w) and (~protected).any() and np.median(source[~protected])<65:
                core=cv2.erode(protected.astype(np.uint8),np.ones((31,31),np.uint8))>0
                trimap=np.where(protected,cv2.GC_PR_FGD,cv2.GC_BGD).astype(np.uint8)
                trimap[core]=cv2.GC_FGD
                if core.any() and (~protected).any():
                    cv2.setRNGSeed(0)
                    cv2.grabCut(np.ascontiguousarray(source),trimap,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),4,cv2.GC_INIT_WITH_MASK)
                    refined=(trimap==cv2.GC_FGD)|(trimap==cv2.GC_PR_FGD)
                    # Only refine the existing protection band, never invent
                    # new foreground outside it or remove most of the subject.
                    if refined.sum()>protected.sum()*.65:
                        protected=refined
                        if plan.get('groupProtection',{}).get('capMask'):
                            protected |= unpack(plan['groupProtection']['capMask'],shape)>0
                        plan['layers']['edgeProtected']=pack(refined.astype(np.uint8)*255)
        safe=~protected
        floor_row=round(plan['floor']*h)
        across_floor=False
        if plan.get('separation') and floor_row<h-20:
            check=np.asarray(image.resize((w,h),Image.Resampling.LANCZOS))
            wall=check[max(0,floor_row-round(h*.10)):max(1,floor_row-round(h*.02))][safe[max(0,floor_row-round(h*.10)):max(1,floor_row-round(h*.02))]]
            ground=check[min(h,floor_row+round(h*.02)):min(h,floor_row+round(h*.10))][safe[min(h,floor_row+round(h*.02)):min(h,floor_row+round(h*.10))]]
            if len(wall)>w and len(ground)>w:
                wc=np.median(wall,axis=0);fc=np.median(ground,axis=0)
                across_floor=np.ptp(wc)<35 and wc.mean()>160 and np.ptp(fc)<40 and np.max(np.abs(wc-fc))<45
        if not across_floor:safe[max(0,floor_row-8):]=False
        distance=cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,3)
        weight=np.clip((distance-5)/20,0,1)
        edge_weight=np.clip(distance/2,0,1)
        if plan.get('separation') and not across_floor:
            edge_weight *= np.clip((floor_row-np.arange(h)[:,None])/max(8,h*.025),0,1)
        if plan.get('separation'):weight=edge_weight
        rgb=np.asarray(image.resize((w,h),Image.Resampling.LANCZOS))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
        valid=safe.astype(np.float32)
        sigma=max(12,min(h,w)*.035)
        norm=cv2.GaussianBlur(valid,(0,0),sigma)
        broad=cv2.GaussianBlur(gray*valid,(0,0),sigma)/np.maximum(norm,.00001)
        low=cv2.GaussianBlur(gray,(0,0),1.2)
        residual=broad-low
        samples=np.abs(residual)[weight>.9]
        # A nearly clean background is left alone. Higher variation receives a
        # stronger local correction; broad lighting gradients remain intact.
        variation=float(np.percentile(samples,90)) if len(samples) else 0
        strength=float(np.clip(variation/3,.25,.80)) if variation>=.6 else 0.0
        delta=np.clip(residual*strength,-24,24)*weight
        pop='none';scale=4
        if plan.get('adaptiveRevision',0)>=3 and not plan.get('partialCleanup') and np.any(weight>.9):
            colour=np.median(rgb[weight>.9],axis=0)
            if np.ptp(colour)<35 and colour.mean()>160:
                if plan.get('separation'):
                    delta+=np.clip((242-broad)*.8,0,56)*edge_weight;scale=2
                else:delta+=np.clip((242-gray)*.45,0,24)*weight
                pop='white'
            elif colour.mean()<65:
                # A broad protected-edge fade left a bright silhouette around
                # people. Carry background darkening right to the protected
                # edge, using only a two-pixel blend outside the foreground.
                delta-=np.clip(gray*.32,0,18)*edge_weight;pop='black'
            elif np.ptp(colour)>40:pop='colour'
        delta=np.clip(delta,-60 if scale==2 else -30,60 if scale==2 else 30)
        delta[np.abs(delta)<.75]=0
        encoded=np.rint(delta*scale+128).astype(np.uint8)
        mask=((delta!=0)|((weight>0)&(pop=='colour'))).astype(np.uint8)*255
        cached={'version':1,'edgeFinish':1,'scale':scale,'shape':shape,'delta':pack(encoded),'mask':pack(mask),
                'strength':strength,'variation':variation,'pop':pop,'action':'clean' if mask.any() else 'preserve'}
        plan['backdropFinish']=cached
    if cached.get('version')!=1 or cached.get('shape')!=shape:raise ValueError('Invalid saved backdrop finish')
    scale=cached.get('scale',4)
    if scale not in (2,4):raise ValueError('Invalid backdrop finish scale')
    delta=(unpack(cached['delta'],shape).astype(np.float32)-128)/scale
    active=unpack(cached['mask'],shape)>0
    # A hard gate protects all unselected native pixels. Soft variation lives
    # inside that gate; the colour correction is identical in all RGB channels.
    correction=Image.fromarray(delta).resize(image.size,Image.Resampling.BILINEAR)
    mask=Image.fromarray(active.astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST)
    result=image.copy()
    for top in range(0,image.height,256):
        box=(0,top,image.width,min(image.height,top+256))
        rgb=np.asarray(image.crop(box));selected=np.asarray(mask.crop(box))>0
        change=np.asarray(correction.crop(box))
        change=np.clip(change,-rgb.min(axis=2).astype(float),255-rgb.max(axis=2).astype(float))
        out=rgb.copy();out[selected]=np.rint(np.clip(rgb[selected].astype(float)+change[selected,None],0,255)).astype(np.uint8)
        if cached.get('pop')=='colour':
            gray=out.astype(float)@np.array([.2126,.7152,.0722]);enhanced=gray[...,None]+(out-gray[...,None])*1.16
            out[selected]=np.rint(np.clip(enhanced[selected],0,255)).astype(np.uint8)
        result.paste(Image.fromarray(out),box[:2])
    return result
