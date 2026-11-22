"""Bounded repair inspection. Passing these checks is not client acceptance."""
import cv2
import numpy as np
from PIL import Image
from backdrop_finish import pack,unpack


def gradient(rgb):
    gray=rgb.astype(np.float32).mean(axis=2)
    return np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))/8


def inspect_arrays(source, result, repair, protected, surface):
    if source.shape!=result.shape or repair.shape!=source.shape[:2]:raise ValueError('Invalid repair inspection dimensions')
    if not np.isfinite(result).all():raise ValueError('Nonfinite reconstructed pixels')
    h,w=repair.shape;selected=repair&surface&~protected
    guard=cv2.dilate(protected.astype(np.uint8),np.ones((5,5),np.uint8))>0
    donor=surface&~repair&~guard
    suspicious=np.zeros((h,w),bool);issues=[];score=0.0
    source_gradient=gradient(source);result_gradient=gradient(result)
    n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8))
    radius=max(5,round(min(h,w)*.025))|1
    components=sorted(range(1,n),key=lambda i:-stats[i,cv2.CC_STAT_AREA])
    eligible=[i for i in components if stats[i,cv2.CC_STAT_AREA]>=max(12,h*w*.00015)]
    if len(eligible)>32:issues.append({'kind':'inspection-budget','components':len(eligible)-32});score+=.5
    for i in eligible[:32]:
        area=int(stats[i,cv2.CC_STAT_AREA])
        if area<max(12,h*w*.00015):continue
        part=labels==i
        ring=(cv2.dilate(part.astype(np.uint8),np.ones((radius,radius),np.uint8))>0)&donor
        if ring.sum()<max(20,area*.002):
            issues.append({'kind':'insufficient-donors','component':i,'pixels':area});score+=.5
            continue
        samples=result[ring].astype(float)
        low=np.percentile(samples,5,axis=0);high=np.percentile(samples,95,axis=0)
        # Patterned material/strong illumination is not a uniform colour target.
        stable=float(np.max(high-low))<55
        edge=part&~(cv2.erode(part.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~guard
        baseline=float(np.percentile(result_gradient[ring],97))
        bad_edges=edge&(result_gradient>max(15,baseline*3+4))
        seam_fraction=float(bad_edges.sum()/max(1,edge.sum()))
        drift=np.maximum(low-18-result.astype(float),result.astype(float)-high-18).max(axis=2)
        drift_fraction=float((part&(drift>0)).sum()/area) if stable else 0.0
        source_texture=float(np.std(source_gradient[ring]))
        repair_texture=float(np.std(result_gradient[part]))
        texture_lost=source_texture>4 and repair_texture<source_texture*.12 and area>h*w*.003
        reasons=[]
        if seam_fraction>.18 and bad_edges.sum()>8:reasons.append('repair-seam')
        if drift_fraction>.35:reasons.append('material-colour-mismatch')
        if texture_lost:reasons.append('lost-surface-texture')
        if reasons:
            suspicious|=part&~guard
            severity=seam_fraction+drift_fraction+float(texture_lost)
            score+=severity*min(1,area/max(1,h*w*.05))
            issues.append({'kind':'+'.join(reasons),'component':i,'pixels':area,
                           'seamFraction':round(seam_fraction,4),'colourDriftFraction':round(drift_fraction,4)})
    return {'version':1,'issues':issues,'score':round(score,5),'visualReviewRequired':True},suspicious


def harmonize_arrays(source, result, repair, protected, wall, donor_exclude=None):
    """Adjust only low-frequency drift within repaired, locally smooth wall."""
    h,w=repair.shape;out=result.astype(np.float32).copy()
    selected=repair&wall&~protected
    guard=cv2.dilate(protected.astype(np.uint8),np.ones((5,5),np.uint8))>0
    donor=wall&~repair&~guard
    if donor_exclude is not None:donor &= ~donor_exclude
    yy,xx=np.mgrid[:h,:w]
    plane=np.stack([np.ones((h,w)),xx/max(1,w-1),yy/max(1,h-1)],axis=2)
    # High-frequency neural texture is retained; only drift in its illumination
    # field is corrected. Patterned walls are excluded by source donor spread.
    blurred=cv2.GaussianBlur(result.astype(np.float32),(0,0),max(2,min(h,w)*.008))
    n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8))
    components=sorted(range(1,n),key=lambda i:-stats[i,cv2.CC_STAT_AREA])
    for i in components[:16]:
        part=labels==i
        if stats[i,cv2.CC_STAT_AREA]<max(16,h*w*.0003):continue
        radius=max(9,round(min(h,w)*.06))|1
        ring=(cv2.dilate(part.astype(np.uint8),np.ones((radius,radius),np.uint8))>0)&donor
        if ring.sum()<60:continue
        samples=source[ring].astype(float)
        low=np.percentile(samples,5,axis=0);high=np.percentile(samples,95,axis=0)
        if np.max(high-low)>45:continue
        fit=np.linalg.lstsq(plane[ring],samples,rcond=None)[0]
        target=np.clip(plane@fit,low-3,high+3)
        difference=target-blurred
        if np.percentile(np.abs(difference[part]),75)<10:continue
        # No writes outside the true repair mask or into a person/prop. The
        # correction is bounded rather than painting a constant RGB surface.
        distance=cv2.distanceTransform((part|guard).astype(np.uint8),cv2.DIST_L2,3)
        weight=np.clip((distance-1)/max(3,min(h,w)*.012),0,1)
        delta=np.clip(difference,-28,28)*weight[...,None]
        out[part]+=delta[part]
    return np.rint(np.clip(out,0,255)).astype(np.uint8)


def masks(plan):
    shape=plan['layers']['shape'];layers=plan['layers']
    protected=unpack(layers['foregroundAlpha'],shape)>3
    repair=unpack(layers['clutter'],shape)>0
    wall=unpack(layers['wallSurface'],shape)>0
    surface=wall.copy()
    if plan.get('floorRepairs'):
        floor=plan['floorRepairs']
        for key in ('defects','extension'):repair|=unpack(floor[key],shape)>0
        # Floor checks operate on explicitly selected floor repairs only;
        # they never redefine the floor horizon or select healthy floor.
        surface|=~wall
    return repair&~protected,protected,wall,surface


def specialist_owned(plan):
    shape=plan['layers']['shape']
    tone=plan.get('reconstructionTone',{})
    if plan.get('sceneCleanupRevision',0)<25 or not tone.get('ownedMask'):return np.zeros(shape,bool)
    return unpack(tone['ownedMask'],shape)>0


def generic_repair_mask(plan,repair):
    # A region has one colour-transfer/reconstruction owner. Preserve the r23
    # specialist's edge fade instead of stacking a second correction on it.
    return repair&~specialist_owned(plan)


def harmonize(source,background,plan,selection=None):
    h,w=plan['layers']['shape'];repair,protected,wall,_=masks(plan)
    all_repairs=repair.copy()
    repair=generic_repair_mask(plan,repair)
    if selection is not None:repair &= selection
    if not (repair&wall).any():return background
    src=np.asarray(source.resize((w,h),Image.Resampling.LANCZOS))
    small=np.asarray(background.resize((w,h),Image.Resampling.LANCZOS))
    adjusted=harmonize_arrays(src,small,repair,protected,wall,donor_exclude=all_repairs)
    delta=adjusted.astype(np.float32)-small
    channels=[Image.fromarray(delta[:,:,c]).resize(background.size,Image.Resampling.BILINEAR) for c in range(3)]
    mask=Image.fromarray((repair&wall&~protected).astype(np.uint8)*255).resize(background.size,Image.Resampling.NEAREST)
    result=background.copy()
    for top in range(0,background.height,256):
        box=(0,top,background.width,min(top+256,background.height))
        pixels=np.asarray(background.crop(box)).astype(np.float32)
        correction=np.stack([np.asarray(c.crop(box)) for c in channels],axis=2)
        result.paste(Image.fromarray(np.rint(np.clip(pixels+correction,0,255)).astype(np.uint8)),box[:2])
    return Image.composite(result,background,mask)


def inspect_and_retry(source, background, plan, repairer=None):
    shape=plan['layers']['shape'];h,w=shape
    repair,protected,wall,surface=masks(plan)
    src=np.asarray(source.resize((w,h),Image.Resampling.LANCZOS))
    def inspect(value):return inspect_arrays(src,np.asarray(value.resize((w,h),Image.Resampling.LANCZOS)),repair,protected,surface)
    previous=plan.get('repairInspection')
    # Replay the recorded decision. Never invent another repair during export.
    if previous:
        if previous.get('retryAccepted'):
            from neural import repair as neural_repair
            retry=plan['repairRetry'];mask=generic_repair_mask(plan,unpack(retry['mask'],shape)>0)
            base=background
            background=(repairer or neural_repair)(background,mask,protected,retry['settings'])
            background=harmonize(base,background,plan,mask)
        return background
    report,suspect=inspect(background)
    suspect=generic_repair_mask(plan,suspect)
    report.update(attempts=1,retryAccepted=False,state='needs-review' if report['issues'] else 'checks-passed')
    if suspect.any() and suspect.mean()<.4:
        from neural import repair as neural_repair
        from runtime import progress
        progress('Refining background repair',83)
        retry={'mask':pack(suspect.astype(np.uint8)*255),'settings':{'excludeForegroundContext':True,'adaptiveRevision':3}}
        try:
            candidate=(repairer or neural_repair)(background,suspect,protected,retry['settings'])
            candidate=harmonize(background,candidate,plan,suspect)
            alternate,_=inspect(candidate)
            report['attempts']=2;report['retryScore']=alternate['score']
            if alternate['score']<report['score']-.02:
                background=candidate;report.update(retryAccepted=True,issues=alternate['issues'],score=alternate['score'])
            plan['repairRetry']=retry
        except (ValueError,RuntimeError,MemoryError) as error:
            report['retryError']=type(error).__name__
    report['state']='needs-review' if report['issues'] else 'checks-passed'
    plan['repairInspection']=report
    if report['issues']:
        message='Background or floor repair needs review.'
        if message not in plan.setdefault('review',[]):plan['review'].append(message)
    return background


def enforce_source_cores(source,result,matte,plan):
    """Verify and restore opaque source pixels before permitted tonal grading."""
    normalized=matte.copy();normalized[normalized>=250]=255;normalized[normalized<=3]=0
    full=Image.fromarray(normalized).resize(source.size,Image.Resampling.BICUBIC)
    # Existing explicitly planned furniture surface healing remains allowed.
    prop=None
    if plan.get('propRepairs'):
        prop=Image.fromarray(unpack(plan['propRepairs']['mask'],plan['layers']['shape'])).resize(source.size,Image.Resampling.NEAREST)
    restored=0;out=result.copy()
    for top in range(0,source.height,256):
        box=(0,top,source.width,min(top+256,source.height))
        protected=np.asarray(full.crop(box))==255
        if prop is not None:protected &= np.asarray(prop.crop(box))==0
        original=np.asarray(source.crop(box));pixels=np.asarray(result.crop(box)).copy()
        changed=protected&np.any(pixels!=original,axis=2)
        restored+=int(changed.sum());pixels[changed]=original[changed]
        out.paste(Image.fromarray(pixels),box[:2])
    plan.setdefault('repairInspection',{})['opaqueSourcePixelsRestored']=restored
    if restored:
        message='Subject protection needs review.'
        if message not in plan.setdefault('review',[]):plan['review'].append(message)
    return out
