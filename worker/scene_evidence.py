"""Measure scene boundaries and local backdrop defects without frame-position presets."""
import cv2
import numpy as np


def floor_evidence(rgb, protected, require_coherence=False):
    h, w = protected.shape
    appearance=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    radius = max(3, round(h * .0125))
    # Independent columns prevent a chair, softbox or one dark corner from
    # becoming a full-width floor. Missing foreground samples are interpolated.
    profiles = []; observed = []; centres=[]
    for columns in np.array_split(np.arange(w), 8):
        values = np.full((h, 3), np.nan)
        for y in range(h):
            valid = ~protected[y, columns]
            if valid.sum() >= max(3, len(columns) * .4):
                values[y] = np.median(appearance[y, columns[valid]], axis=0)
        known = np.flatnonzero(np.isfinite(values[:, 0]))
        if len(known) < h * .5:
            continue
        observed.append(np.isfinite(values[:, 0]))
        centres.append(float(columns.mean()))
        for c in range(3):
            values[:, c] = np.interp(np.arange(h), known, values[known, c])
        profiles.append(cv2.GaussianBlur(values.astype(np.float32), (1, 0), 0, sigmaY=max(1, radius / 4)))
    result = {'method': 'column-consensus-v1', 'detected': False, 'confidence': 0.0}
    if len(profiles) < 3 or h <= 8 * radius:
        return 1.0, result
    profiles = np.array(profiles)
    rows = np.arange(3 * radius, h - 3 * radius)
    above = profiles[:, rows-radius]
    below = profiles[:, rows+radius]
    jump = below - above
    # Subtract the ordinary local lighting slope. A sustained material change
    # has a stronger central jump than either neighbouring interval.
    previous = above - profiles[:, rows-3*radius]
    following = profiles[:, rows+3*radius] - below
    excess = np.max(np.abs(jump - (previous + following) / 2), axis=2)
    magnitude = np.max(np.abs(jump), axis=2)
    votes = (excess > 4) & (magnitude > 5)
    observed = np.array(observed)
    visible=observed[:, rows-radius] & observed[:, rows+radius]
    votes &= visible
    # Separate edges can occur at the same height (wallpaper, sleeves, props).
    # They are not a material boundary unless their colour changes agree in
    # direction and scale across the visible witnesses.
    if require_coherence:
        length=np.linalg.norm(jump,axis=2)
        unit=jump/np.maximum(length[...,None],.001)
        similarity=np.einsum('prc,qrc->pqr',unit,unit)
        ratio=np.maximum(length[:,None,:],length[None,:,:])/np.maximum(.001,np.minimum(length[:,None,:],length[None,:,:]))
        coherent=(similarity>.7)&(ratio<3)&votes[:,None,:]&votes[None,:,:]
        best=np.argmax(coherent.sum(axis=1),axis=0)
        votes=coherent[best,:,np.arange(len(rows))].T
    agreement=votes.sum(axis=0)/np.maximum(1,visible.sum(axis=0))
    score=np.sum(np.where(votes,excess,0),axis=0)/np.maximum(1,votes.sum(axis=0))
    # Hidden subjects/props are missing evidence, not votes against the join.
    # Require separated visible witnesses so one object cannot set the horizon.
    centres=np.array(centres)[:,None]
    span=np.max(np.where(votes,centres,-w),axis=0)-np.min(np.where(votes,centres,w*2),axis=0)
    people = np.flatnonzero(protected.any(axis=1))
    # Exclude overhead wallpaper/rail transitions using this subject's extent,
    # rather than an assumed percentage of the photograph.
    eligible = (agreement >= .6)&(votes.sum(axis=0)>=2)&(span>w*.35)
    if len(people):
        eligible &= rows > np.percentile(np.nonzero(protected)[0], 10)
    candidates = np.flatnonzero(eligible & (score >= 4))
    if not len(candidates):
        return 1.0, result
    # First sustained transition below the subjects' heads; later transitions
    # belong to floor edges/tape and must not move the horizon to a preset row.
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > radius) + 1)
    group = groups[0]
    best = group[np.argmax(score[group])]
    row = int(rows[best])
    if len(people) and row > people[-1] + radius:
        return 1.0, result
    result.update(detected=True, confidence=float(agreement[best]), row=row,
                  contrast=float(np.median(magnitude[:, best])))
    return row / h, result


def wall_surface(rgb, protected, floor, analysis):
    """Track the visible join; a scalar horizon is only a search seed.

    Leave uncertain and occluded columns out of wall correction rather than
    extending one measured height across the photograph.
    """
    h,w=protected.shape
    appearance=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    wall=np.ones((h,w),bool)
    if not analysis.get('detected'):
        return wall, {'method':'visible-contour-v1','detected':False}
    radius=max(2,round(h*.006));seed=round(floor*h)
    reach=max(radius*5,round(h*.10));yy=np.arange(h)
    positions=np.arange(max(3*radius,seed-reach),min(h-3*radius,seed+reach))
    wall[:]=False
    if not len(positions):return wall, {'method':'visible-contour-v1','detected':False}
    measured=[]
    reference=~protected
    above=appearance[max(0,seed-6*radius):max(1,seed-3*radius)][reference[max(0,seed-6*radius):max(1,seed-3*radius)]]
    below=appearance[min(h,seed+3*radius):min(h,seed+6*radius)][reference[min(h,seed+3*radius):min(h,seed+6*radius)]]
    direction=np.median(below,axis=0)-np.median(above,axis=0) if len(above) and len(below) else np.zeros(3)
    direction=direction/max(1,float(np.linalg.norm(direction)))
    # Narrow strips follow local slopes and folds without allowing subject
    # pixels to vote. No interpolation is used as boundary evidence.
    for columns in np.array_split(np.arange(w),max(8,min(48,w//6))):
        valid=~protected[:,columns]
        observed=valid.sum(axis=1)>=max(2,len(columns)*.5)
        profile=np.zeros((h,3),np.float32)
        for row in np.flatnonzero(observed):
            profile[row]=np.median(appearance[row,columns[valid[row]]],axis=0)
        known=np.flatnonzero(observed)
        if len(known)<h*.1:continue
        for c in range(3):profile[:,c]=np.interp(yy,known,profile[known,c])
        profile=cv2.GaussianBlur(profile,(1,0),0,sigmaY=max(1,radius/3))
        jump=profile[positions+radius]-profile[positions-radius]
        slope=(profile[positions-radius]-profile[positions-3*radius]+profile[positions+3*radius]-profile[positions+radius])/2
        score=np.max(np.abs(jump-slope),axis=1)
        usable=observed[positions-radius]&observed[positions+radius]&(np.max(np.abs(jump),axis=1)>3)
        if np.linalg.norm(direction)>.5:
            usable &= (jump@direction)>np.linalg.norm(jump,axis=1)*.5
        score[~usable]=0
        # Follow the same material change, not a stronger fold ridge or floor
        # scratch elsewhere in the search neighbourhood.
        rank=score/(1+np.abs(positions-seed)/max(1,reach*.25))
        best=int(np.argmax(rank))
        if score[best]<3:continue
        row=int(positions[best]);guard=max(3,radius*2)
        wall[:max(0,row-guard),columns]=True
        measured.append([int(columns[0]),int(columns[-1])+1,row])
    accepted=[];wall[:]=False
    for i,segment in enumerate(measured):
        neighbours=np.array([s[2] for s in measured[max(0,i-3):i+4]])
        middle=float(np.median(neighbours));spread=float(np.median(np.abs(neighbours-middle)))
        if abs(segment[2]-middle)>max(radius*6,spread*3):continue
        accepted.append(segment)
    if accepted:
        centres=np.array([(s[0]+s[1]-1)/2 for s in accepted])
        heights=np.array([s[2] for s in accepted],np.float32)
        # Continue measured geometry across hidden subjects/equipment, keeping
        # one continuous edge rather than a staircase of strip boundaries.
        curve=np.interp(np.arange(w),centres,heights).astype(np.float32)
        curve=cv2.GaussianBlur(curve[None,:],(0,1),max(2,w/48)).ravel()
        wall=np.arange(h)[:,None] < (curve[None,:]-max(3,radius*2))
        ground=np.rint(curve+max(3,radius*2)).astype(int).tolist()
    else:ground=[]
    return wall, {'method':'visible-contour-v1','detected':bool(accepted),'segments':accepted,'groundRows':ground}


def texture_finish(gray,safe):
    """Bounded local texture correction; do not flatten material/lighting levels."""
    h,w=gray.shape;valid=safe.astype(np.float32);sigma=max(6,min(h,w)*.025)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    horizontal=((np.abs(gy)>16)&(np.abs(gy)>np.abs(gx)*2)&safe).astype(np.uint8)*255
    lines=cv2.HoughLinesP(horizontal,1,np.pi/180,max(20,w//16),minLineLength=max(35,w*.15),maxLineGap=12)
    boundary=np.zeros((h,w),np.uint8)
    if lines is not None:
        for x0,y0,x1,y1 in lines[:,0]:
            if abs(y1-y0)>max(2,abs(x1-x0)*.2):continue
            cv2.line(boundary,(int(x0),int(y0)),(int(x1),int(y1)),1,max(5,round(sigma*3)))
    safe=safe&(boundary==0);valid=safe.astype(np.float32)
    small=cv2.GaussianBlur(valid,(0,0),1.2)
    low=cv2.GaussianBlur(gray*valid,(0,0),1.2)/np.maximum(small,.001)
    norm=cv2.GaussianBlur(valid,(0,0),sigma)
    broad=cv2.GaussianBlur(gray*valid,(0,0),sigma)/np.maximum(norm,.001)
    residual=broad-low
    noise=np.abs(gray-low)[safe]
    threshold=max(2,float(np.median(noise))*2) if len(noise) else 2
    strength=np.clip((np.abs(residual)-threshold)/max(3,threshold*2),0,1)*.8
    weight=np.clip(cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,3)/4,0,1)
    delta=np.clip(residual*strength,-12,12)*weight
    delta[np.abs(delta)<.75]=0
    return delta.astype(np.float32),float(strength.max()),float(np.max(np.abs(residual[safe]))) if safe.any() else 0.


def local_finish(gray, safe, bounded=False):
    """Remove measured ridges while retaining broad illumination and grain."""
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    x = xx / max(1, w-1); y = yy / max(1, h-1)
    design = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)
    samples = np.flatnonzero(safe.ravel())[::8]
    if len(samples) < 50:
        return np.zeros_like(gray), 0.0, 0.0
    a = design.reshape(-1, 6)[samples]; b = gray.ravel()[samples]
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
    for _ in range(4):
        residual = a @ coefficients - b
        weights = np.sqrt(np.minimum(1, 3 / np.maximum(np.abs(residual), .01)))
        coefficients = np.linalg.lstsq(a * weights[:, None], b * weights, rcond=None)[0]
    illumination = design @ coefficients
    if bounded:
        # Occluded edges cannot justify extrapolating negative illumination.
        lo,hi=np.percentile(gray[safe],[10,90])
        illumination=np.clip(illumination,lo,hi)
    valid = safe.astype(np.float32)
    sigma = max(4, min(h, w) * .025)
    small_norm = cv2.GaussianBlur(valid, (0, 0), 1.2)
    low = illumination + cv2.GaussianBlur(((gray-illumination)*valid).astype(np.float32), (0, 0), 1.2) / np.maximum(small_norm, .001)
    residual = (low - illumination) * valid
    norm = cv2.GaussianBlur(valid, (0, 0), sigma)
    ridges = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), sigma) / np.maximum(norm, .001)
    # Illumination is smooth in both directions; folds have repeated local
    # changes. Measure their contrast rather than selecting a lower wall band.
    contrast = np.zeros_like(gray)
    for scale in (sigma, sigma*2.4, sigma*4.8):
        normal = cv2.GaussianBlur(valid, (0, 0), scale)
        mean = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), scale) / np.maximum(normal, .001)
        variance = cv2.GaussianBlur(residual.astype(np.float32)**2, (0, 0), scale) / np.maximum(normal, .001) - mean**2
        contrast = np.maximum(contrast, np.sqrt(np.maximum(0, variance)))
    confidence = np.clip((contrast - 2) / 3, 0, 1)
    confidence[~safe] = 0
    # Suppress isolated detection noise; corrections stay inside the measured
    # defect support. Opaque foreground and all floor pixels remain excluded.
    strength = confidence * np.clip(cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_L2, 3) / 2, 0, 1)
    delta = np.clip(-residual * .95, -60, 60) * strength
    delta[np.abs(delta) < .75] = 0
    return delta.astype(np.float32), float(confidence.max()), float(contrast[safe].max()) if safe.any() else 0.0
