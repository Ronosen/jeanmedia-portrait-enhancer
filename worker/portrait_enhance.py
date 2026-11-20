"""Saved, adaptive colour finishing and local teeth/sclera whitening.

Uses OpenCV's open-source LAB conversion to increase chroma without changing
its angle. No gray-world white balance and no generative face restoration.
"""
import hashlib
import cv2
import numpy as np
from PIL import Image
from backdrop_finish import pack,unpack


def prepare(image, preview):
    from face_areas import detect,landmarks,feature_masks
    from runtime import progress
    progress('Measuring portrait colour and facial details',92)
    rgb=np.asarray(preview.convert('RGB'));h,w=rgb.shape[:2]
    light=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)[:,:,0]
    low,median,high=np.percentile(light,[5,50,95])
    plan={'version':1,'source':hashlib.sha256(rgb.tobytes()).hexdigest(),
          'size':list(image.size),'previewSize':list(preview.size),
          'tone':{'contrast':float(np.clip(1+(170-(high-low))/650,1.04,1.17)),
                  'gamma':float(np.clip(1+(median-145)/1000,1,1.08)),
                  'vibrance':float(np.clip((72-np.median(np.ptp(rgb.astype(float),axis=2)))/500,.04,.14))},
          'faces':[]}
    from detail_finish import measure
    plan['detail']=measure(image)
    if image.mode!='RGB':return plan
    for face in detect(rgb):
        face=face.copy();face[:14:2]*=image.width/w;face[1:14:2]*=image.height/h
        x,y,fw,fh=face[:4]
        box=(max(0,int(x-fw*.35)),max(0,int(y-fh*.35)),min(image.width,int(x+fw*1.35)),min(image.height,int(y+fh*1.35)))
        if box[2]<=box[0] or box[3]<=box[1]:continue
        crop=image.crop(box);crop.thumbnail((768,768),Image.Resampling.LANCZOS)
        sx=crop.width/(box[2]-box[0]);sy=crop.height/(box[3]-box[1])
        face[:14:2]=(face[:14:2]-box[0])*sx;face[1:14:2]=(face[1:14:2]-box[1])*sy
        face[2]=fw*sx;face[3]=fh*sy
        pixels=np.asarray(crop);points=landmarks(pixels,face)
        if points is None:continue
        if not ((points[:,0]>=-crop.width*.1)&(points[:,0]<crop.width*1.1)&(points[:,1]>=-crop.height*.1)&(points[:,1]<crop.height*1.1)).all():continue
        teeth,eyes,guard=feature_masks(pixels,points)
        guard=cv2.GaussianBlur(guard,(0,0),max(1,min(crop.size)*.035))
        plan['faces'].append({'box':list(box),'shape':[crop.height,crop.width],
            'teeth':pack(np.rint(teeth*255).astype(np.uint8)),
            'eyes':pack(np.rint(eyes*255).astype(np.uint8)),'guard':pack(guard),
            'teethPixels':int((teeth>0).sum()),'eyePixels':int((eyes>0).sum())})
    plan['retouchAssessment']={'version':2,'faces':len(plan['faces']),
        'teethRegions':sum(f['teethPixels']>0 for f in plan['faces']),
        'eyeRegions':sum(f['eyePixels']>0 for f in plan['faces'])}
    return plan


def validate(plan, preview):
    if plan.get('version')!=1 or plan.get('source')!=hashlib.sha256(np.asarray(preview.convert('RGB')).tobytes()).hexdigest():
        raise ValueError('Portrait finishing settings belong to a different photo')
    if plan.get('previewSize')!=list(preview.size):raise ValueError('Invalid portrait preview size')
    size=plan.get('size',[])
    if len(size)!=2 or any(type(v)!=int or not 1<=v<=80000 for v in size) or size[0]*size[1]>80000000:raise ValueError('Invalid portrait dimensions')
    tone=plan.get('tone',{})
    for key,limits in [('contrast',(1,1.2)),('gamma',(1,1.1)),('vibrance',(0,.2))]:
        if not isinstance(tone.get(key),(int,float)) or not np.isfinite(tone[key]) or not limits[0]<=tone[key]<=limits[1]:raise ValueError('Invalid portrait colour settings')
    detail=plan.get('detail')
    if detail is not None:
        if detail.get('version')!=1:raise ValueError('Invalid detail version')
        for key,limits in [('noise',(0,30)),('denoise',(0,.45)),('sharpen',(0,.35))]:
            value=detail.get(key)
            if not isinstance(value,(int,float)) or not np.isfinite(value) or not limits[0]<=value<=limits[1]:raise ValueError('Invalid detail settings')
    faces=plan.get('faces')
    if not isinstance(faces,list) or len(faces)>32:raise ValueError('Invalid face masks')
    for face in faces:
        box=face.get('box',[]);shape=face.get('shape',[])
        if len(box)!=4 or any(type(v)!=int for v in box) or not(0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1]):raise ValueError('Invalid face region')
        if len(shape)!=2 or any(type(v)!=int or not 1<=v<=768 for v in shape):raise ValueError('Invalid face mask size')
        for name in ('teeth','eyes','guard'):unpack(face[name],shape)


def color_tile(rgb, tone, face_guard):
    pixels=rgb.astype(np.float32)/255
    # Unlike the legacy brightening curve, this curve never raises shadows.
    contrast=tone['contrast'];pixels=pixels**tone['gamma']
    pixels+=(contrast-1)*pixels*(1-pixels)*(4*pixels-2)
    pixels=np.clip(pixels,0,1)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma=np.hypot(lab[:,:,1]-128,lab[:,:,2]-128)
    ycrcb=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    cr=ycrcb[:,:,1].astype(float);cb=ycrcb[:,:,2].astype(float)
    skin=np.clip((cr-130)/8,0,1)*np.clip((185-cr)/8,0,1)*np.clip((cb-80)/10,0,1)*np.clip((140-cb)/10,0,1)*np.clip(chroma/18,0,1)
    # Protect skin by a continuous colour weight across face AND body. Blending
    # original pixels inside a geometric face oval produced a visible boundary.
    pixels=pixels*(1-.65*skin[...,None])+rgb.astype(float)/255*(.65*skin[...,None])
    safe=1-np.maximum(skin,.85*face_guard)
    # Neutral shadows in both cloth and backdrop get a soft, detail-preserving
    # toe. Multiplying all RGB channels retains their colour ratio.
    neutral=np.clip((25-chroma)/16,0,1)
    shadow=np.clip(1-lab[:,:,0]/150,0,1)**1.4
    pixels*=1-(.28*neutral*shadow*(1-skin))[...,None]
    toned=np.rint(pixels*255).astype(np.uint8)
    restored=cv2.cvtColor(toned,cv2.COLOR_RGB2LAB).astype(np.float32)
    gain=1+tone['vibrance']*np.clip(1-chroma/95,0,1)*safe
    restored[:,:,1:]=128+(restored[:,:,1:]-128)*gain[...,None]
    vivid=cv2.cvtColor(np.rint(np.clip(restored,0,255)).astype(np.uint8),cv2.COLOR_LAB2RGB)
    # Exact neutral shadows must not be lifted by LAB round-trip quantisation.
    vivid=np.where(((neutral>.9)&(lab[:,:,0]<100))[...,None],np.minimum(vivid,toned),vivid)
    return vivid


def whiten(rgb, teeth, eyes, version=2):
    if version==1:
        amount=np.maximum(teeth*.55,eyes*.30)
        neutral=rgb.astype(np.float32).mean(axis=2)
        target=rgb.astype(np.float32)*.3+neutral[...,None]*.7
        target+=np.minimum(18,(255-neutral)*.18)[...,None]
        return np.rint(np.clip(rgb*(1-amount[...,None])+target*amount[...,None],0,255)).astype(np.uint8)
    # RGB averaging darkened yellow enamel while the small lift cancelled it.
    # Work in perceptual lightness instead; retain texture and never clip white.
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    tooth=np.clip(teeth,0,1);eye=np.clip(eyes,0,1)*(1-tooth)
    lift=np.minimum(42,np.maximum(0,238-lab[:,:,0])*.48)*tooth
    lift+=np.minimum(22,np.maximum(0,225-lab[:,:,0])*.34)*eye
    lab[:,:,0]+=lift
    neutralize=np.clip(.78*tooth+.55*eye,0,1)
    lab[:,:,1:]=128+(lab[:,:,1:]-128)*(1-neutralize[...,None])
    target=cv2.cvtColor(np.rint(np.clip(lab,0,255)).astype(np.uint8),cv2.COLOR_LAB2RGB)
    # A LAB round-trip must not change skin, pupils, lips or gaps in the mask.
    return np.where(((tooth+eye)>0)[...,None],target,rgb)


def apply(image, plan):
    masks={name:Image.new('L',image.size) for name in ('guard','teeth','eyes')}
    for face in plan['faces']:
        box=face['box'];size=plan['size']
        destination=[round(box[i]*image.size[i%2]/size[i%2]) for i in range(4)]
        width=destination[2]-destination[0];height=destination[3]-destination[1]
        if min(width,height)<1:continue
        for name,canvas in masks.items():
            mask=Image.fromarray(unpack(face[name],face['shape'])).resize((width,height),Image.Resampling.BILINEAR)
            old=canvas.crop(tuple(destination));canvas.paste(Image.fromarray(np.maximum(np.asarray(old),np.asarray(mask))),tuple(destination[:2]))
    result=Image.new(image.mode,image.size)
    for y in range(0,image.height,256):
        top=max(0,y-8);end=min(y+256,image.height)
        box=(0,top,image.width,min(end+8,image.height));tile=image.crop(box)
        areas={name:np.asarray(mask.crop(box),dtype=np.float32)/255 for name,mask in masks.items()}
        rgb=np.asarray(tile.convert('RGB'))
        toned=color_tile(rgb,plan['tone'],areas['guard'])
        if plan.get('detail'):
            from detail_finish import apply as finish_detail
            toned=finish_detail(toned,plan['detail'],areas['guard'])
        finished=Image.fromarray(whiten(toned,areas['teeth'],areas['eyes'],plan.get('retouchAssessment',{}).get('version',1)))
        if image.mode=='RGBA':finished.putalpha(tile.getchannel('A'))
        result.paste(finished.crop((0,y-top,image.width,end-top)),(0,y))
    return result
