"""Editable local masks shared by preview, inspection and export."""
import cv2,base64
import numpy as np
from PIL import Image
from backdrop_finish import unpack

def selection(plan,layer):
    h,w=plan['layers']['shape'];layers=plan['layers']
    protected=unpack(layers['protected'],(h,w))>0
    person=unpack(layers.get('person',layers['protected']),(h,w))>0
    kind=layer.get('target','subject');mask=np.zeros((h,w),np.float32)
    if kind=='subject':mask=person.astype(np.float32)
    elif kind=='background':mask=(~protected).astype(np.float32)
    elif kind=='backdrop':
        mask=(~protected).astype(np.float32);mask[round(plan.get('floor',1)*h):]=0
    elif kind=='floor':
        mask=(~protected).astype(np.float32);mask[:round(plan.get('floor',1)*h)]=0
    elif kind=='props':mask=(protected&~person).astype(np.float32)
    elif kind=='linear':mask=np.broadcast_to(np.linspace(0,1,h)[:,None],(h,w)).copy().astype(np.float32)
    elif kind=='radial':
        yy,xx=np.mgrid[:h,:w];mask=np.clip(1-(((xx/w-.5)/.4)**2+((yy/h-.5)/.4)**2),0,1).astype(np.float32)
    elif kind in ('colour','luminance'):
        raw=base64.b64decode(layers.get('source',''),validate=True)
        rgb=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape[:2]!=(h,w):raise ValueError('Run Auto Edit again to create range masks')
        tolerance=float(layer.get('tolerance',.2))
        if not .01<=tolerance<=1:raise ValueError('Invalid range tolerance')
        if kind=='colour':distance=np.max(np.abs(rgb.astype(float)-np.array(layer.get('colour',[255,255,255]))),axis=2)/255
        else:distance=np.abs(rgb.mean(axis=2)/255-float(layer.get('luminance',.5)))
        mask=np.clip(1-distance/tolerance,0,1).astype(np.float32)
    elif kind!='brush':raise ValueError('Unknown mask target')
    strokes=layer.get('strokes',[])
    if len(strokes)>2000:raise ValueError('Too many brush strokes')
    for stroke in strokes:
        x,y,r,mode=stroke
        if not all(np.isfinite(v) for v in (x,y,r)) or not(0<=x<=1 and 0<=y<=1 and .001<=r<=.3):raise ValueError('Invalid brush stroke')
        cv2.circle(mask,(round(x*(w-1)),round(y*(h-1))),max(1,round(r*min(h,w))),0 if mode=='subtract' else 1,-1)
    feather=float(layer.get('feather',.02))
    if not 0<=feather<=.2:raise ValueError('Invalid mask feather')
    if feather:mask=cv2.GaussianBlur(mask,(0,0),max(.1,min(h,w)*feather))
    return 1-mask if layer.get('invert') else mask

def apply(image,plan):
    layers=plan.get('localMasks',[])
    if not isinstance(layers,list) or len(layers)>32:raise ValueError('At most 32 masks are supported')
    result=image.copy()
    for layer in layers:
        if not layer.get('enabled',True):continue
        ev=float(layer.get('exposure',0));contrast=float(layer.get('contrast',0));saturation=float(layer.get('saturation',0))
        if not(-2<=ev<=2 and -1<=contrast<=1 and -1<=saturation<=1):raise ValueError('Invalid local adjustment')
        if ev==contrast==saturation==0:continue
        mask=Image.fromarray(selection(plan,layer)).resize(image.size,Image.Resampling.BILINEAR)
        for top in range(0,image.height,256):
            box=(0,top,image.width,min(image.height,top+256));rgb=np.asarray(result.crop(box)).astype(np.float32)/255
            edited=(rgb*2**ev-.5)*(1+contrast)+.5
            gray=edited@np.array([.2126,.7152,.0722],np.float32)
            edited=gray[...,None]+(edited-gray[...,None])*(1+saturation)
            amount=np.asarray(mask.crop(box))[...,None]
            out=np.rint(np.clip(rgb*(1-amount)+np.clip(edited,0,1)*amount,0,1)*255).astype(np.uint8)
            result.paste(Image.fromarray(out),box[:2])
    return result
