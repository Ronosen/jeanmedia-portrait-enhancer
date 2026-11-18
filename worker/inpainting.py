"""Context-based object repair; fitted backdrop colours are never painted here."""
import cv2
import numpy as np
from PIL import Image


def frame_panels(rgb, probability, model, objects, floor=1.0):
    """Separate narrow studio-edge panels from a broad supporting seat."""
    height, width = probability.shape
    edges = cv2.dilate(cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 15, 40), np.ones((3,3),np.uint8))
    bright = ((rgb.mean(axis=2)-model.mean(axis=2) > 45) & (np.ptp(rgb,axis=2) < 45) & (edges == 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    result = np.zeros(probability.shape,np.uint8)
    known_props = np.zeros_like(result)
    for item in objects:
        if item["class"] in (56,57,58,59,60,75):
            x0,y0,x1,y1=item["box"];known_props[y0:y1,x0:x1]=1
    subject = cv2.dilate((probability > .2).astype(np.uint8), np.ones((7,7),np.uint8)).astype(bool)
    for index in range(1,count):
        x,y,w,h,area=stats[index]
        if not ((x == 0 or x+w == width) and y < floor*height and w < width*.16 and h < height*.35 and area > width*height*.001):
            continue
        component=labels==index
        if np.any(known_props[component]) or np.any(probability[component] > .5):
            continue
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(result,[cv2.convexHull(np.concatenate(contours))],1)
    result=cv2.dilate(result,np.ones((9,9),np.uint8))
    result[subject]=0
    return result.astype(bool)


def repair(image, mask, protected=None, floor=1.0, settings=None):
    """Quilt real donor patches into holes and retain their full-res coordinates."""
    import base64
    if mask.ndim != 2 or max(mask.shape)>1280 or min(mask.shape)<1:
        raise ValueError("Invalid object repair mask.")
    selected=mask>0
    protected=np.zeros(mask.shape,bool) if protected is None else protected.astype(bool)
    if protected.shape!=mask.shape or np.any(selected & protected):
        raise ValueError("Object removal overlaps a protected subject or prop.")
    if not selected.any():
        return image.copy()
    if selected.mean()>.45:
        raise ValueError("Needs Review · Too much background is missing for local repair.")
    height,width=mask.shape
    settings={} if settings is None else settings
    encoded=settings.get("donorMap")
    if not encoded and max(mask.shape)>512:
        scale=512/max(mask.shape)
        sw,sh=max(1,round(width*scale)),max(1,round(height*scale))
        small_mask=cv2.resize(selected.astype(np.float32),(sw,sh),interpolation=cv2.INTER_AREA)>0
        small_protected=cv2.resize(protected.astype(np.float32),(sw,sh),interpolation=cv2.INTER_AREA)>0
        small_mask[small_protected]=False
        small_settings={}
        repair(image.resize((sw,sh),Image.Resampling.LANCZOS),small_mask,small_protected,floor,small_settings)
        small=cv2.imdecode(np.frombuffer(base64.b64decode(small_settings["donorMap"]),np.uint8),cv2.IMREAD_UNCHANGED)
        xp=(np.arange(width)+.5)*sw/width-.5;yp=(np.arange(height)+.5)*sh/height-.5
        ix=np.clip(np.rint(xp).astype(int),0,sw-1);iy=np.clip(np.rint(yp).astype(int),0,sh-1)
        coords=np.zeros((height,width,3),np.uint16)
        coords[:,:,0]=np.clip(np.rint((small[iy[:,None],ix[None,:],0]+(xp-ix)[None,:]+.5)*width/sw-.5),0,width-1).astype(np.uint16)
        coords[:,:,1]=np.clip(np.rint((small[iy[:,None],ix[None,:],1]+(yp-iy)[:,None]+.5)*height/sh-.5),0,height-1).astype(np.uint16)
        # Narrow boundary slivers removed by conservative downsampling use their
        # nearest valid original donor, never a protected pixel.
        allowed=~(selected|protected)
        invalid=selected & ~allowed[coords[:,:,1],coords[:,:,0]]
        if invalid.any():
            _,nearest=cv2.distanceTransformWithLabels((~allowed).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            points=np.argwhere(allowed)
            labels=nearest[coords[:,:,1][invalid],coords[:,:,0][invalid]]-1
            coords[:,:,1][invalid]=points[labels,0];coords[:,:,0][invalid]=points[labels,1]
        settings["donorMap"]=base64.b64encode(cv2.imencode(".png",coords)[1]).decode("ascii")
        settings["contextGuide"]=small_settings["contextGuide"]
        return repair(image,mask,protected,floor,settings)
    if encoded:
        if not isinstance(encoded,str) or len(encoded)>12_000_000:
            raise ValueError("Invalid donor map.")
        raw=base64.b64decode(encoded,validate=True)
        if (raw[:8]!=b"\x89PNG\r\n\x1a\n" or len(raw)<24
                or int.from_bytes(raw[16:20],"big")!=width or int.from_bytes(raw[20:24],"big")!=height):
            raise ValueError("Invalid donor map dimensions.")
        coords=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
        if (coords is None or coords.shape!=(height,width,3) or coords.dtype!=np.uint16
                or np.any(coords[:,:,0]>=width) or np.any(coords[:,:,1]>=height) or np.any(coords[:,:,2])):
            raise ValueError("Invalid donor coordinates.")
        # Every donor must be original, unselected background, never a person.
        if np.any((selected|protected)[coords[:,:,1][selected],coords[:,:,0][selected]]):
            raise ValueError("Saved repair tries to copy a protected or damaged area.")
    else:
        source=np.asarray(image.resize((width,height),Image.Resampling.LANCZOS))
        work=source.copy()
        remaining=selected.copy()
        known=~(selected|protected)
        radius=min(12,(min(height,width)-1)//2)
        size=radius*2+1
        valid=cv2.erode(known.astype(np.uint8),np.ones((size,size),np.uint8),
                        borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        yy,xx=np.mgrid[:height,:width]
        coords=np.stack((xx,yy,np.zeros_like(xx)),axis=2).astype(np.uint16)
        rng=np.random.default_rng(31)
        pools={}
        for ground in (False,True):
            region=valid & ((yy>=floor*height-radius) if ground else (yy<floor*height+radius))
            points=np.argwhere(region)
            if len(points)>192:
                points=points[rng.choice(len(points),192,replace=False)]
            if len(points):
                bank=np.stack([source[y-radius:y+radius+1,x-radius:x+radius+1] for y,x in points])
                pools[ground]=(points,bank,np.zeros(len(points),np.float32))
        iterations=0
        while remaining.any():
            frontier=remaining & (cv2.dilate(known.astype(np.uint8),np.ones((3,3),np.uint8))>0)
            if not frontier.any():
                raise ValueError("Needs Review · No safe original texture surrounds this repair.")
            confidence=cv2.boxFilter(known.astype(np.float32),-1,(size,size),normalize=False)
            confidence[~frontier]=-1
            cy,cx=np.unravel_index(np.argmax(confidence),remaining.shape)
            pool=pools.get(bool(cy>=floor*height))
            if pool is None:
                raise ValueError("Needs Review · Too little clear original surface is available for repair.")
            points,bank,usage=pool
            y0,y1=max(0,cy-radius),min(height,cy+radius+1)
            x0,x1=max(0,cx-radius),min(width,cx+radius+1)
            by,bx=radius-(cy-y0),radius-(cx-x0)
            candidates=bank[:,by:by+y1-y0,bx:bx+x1-x0]
            reference=work[y0:y1,x0:x1]
            context=known[y0:y1,x0:x1]
            difference=candidates[:,context].astype(np.float32)-reference[context].astype(np.float32)
            score=np.mean(difference*difference,axis=(1,2))
            score+=((points[:,0]-cy)/height*12)**2+usage*.15
            chosen=int(np.argmin(score));usage[chosen]+=1
            local=remaining[y0:y1,x0:x1]
            work[y0:y1,x0:x1][local]=candidates[chosen][local]
            sy,sx=points[chosen]
            dy,dx=np.mgrid[sy-radius+by:sy-radius+by+y1-y0,sx-radius+bx:sx-radius+bx+x1-x0]
            coords[y0:y1,x0:x1,0][local]=dx[local]
            coords[y0:y1,x0:x1,1][local]=dy[local]
            known[y0:y1,x0:x1][local]=True
            remaining[y0:y1,x0:x1]=False
            iterations+=1
            if iterations%50==0:
                from runtime import progress
                progress("Matching original surface texture",75)
            if iterations>10000:
                raise ValueError("Needs Review · The repair is too fragmented.")
        settings["donorMap"]=base64.b64encode(cv2.imencode(".png",coords)[1]).decode("ascii")
    guide_encoded=settings.get("contextGuide")
    if guide_encoded:
        if not isinstance(guide_encoded,str) or len(guide_encoded)>12_000_000:
            raise ValueError("Invalid repair context.")
        raw=base64.b64decode(guide_encoded,validate=True)
        if (len(raw)<24 or raw[:8]!=b"\x89PNG\r\n\x1a\n"
                or not 1<=int.from_bytes(raw[16:20],"big")<=1280
                or not 1<=int.from_bytes(raw[20:24],"big")<=1280):
            raise ValueError("Invalid repair context dimensions.")
        guide=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if guide is None:
            raise ValueError("Invalid repair context pixels.")
    else:
        source=np.asarray(image.resize((width,height),Image.Resampling.LANCZOS))
        guide=cv2.inpaint(source,(selected|protected).astype(np.uint8)*255,7,cv2.INPAINT_NS)
        guide=cv2.GaussianBlur(guide,(0,0),2.5)
        settings["contextGuide"]=base64.b64encode(cv2.imencode(".png",guide)[1]).decode("ascii")
    original=np.asarray(image)
    # Match broad illumination with the surrounding context; transplant only
    # fine real surface texture. Large copied scratches/folds and patch-shaped
    # lighting jumps must not become repeated motifs in a reconstructed floor.
    low=cv2.GaussianBlur(original,(0,0),max(1.2,image.width/width))
    illumination=Image.fromarray(guide).resize(image.size,Image.Resampling.BILINEAR)
    result=image.copy()
    full_width,full_height=image.size
    xp=(np.arange(full_width,dtype=np.float32)+.5)*width/full_width-.5
    ix=np.clip(np.rint(xp).astype(int),0,width-1)
    for top in range(0,full_height,128):
        end=min(full_height,top+128)
        yp=(np.arange(top,end,dtype=np.float32)+.5)*height/full_height-.5
        iy=np.clip(np.rint(yp).astype(int),0,height-1)
        target=selected[iy[:,None],ix[None,:]]
        if not target.any():continue
        mx=(coords[iy[:,None],ix[None,:],0].astype(np.float32)+(xp-ix)[None,:]+.5)*full_width/width-.5
        my=(coords[iy[:,None],ix[None,:],1].astype(np.float32)+(yp-iy)[:,None]+.5)*full_height/height-.5
        patch=cv2.remap(original,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        base=cv2.remap(low,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        grain=np.clip(patch.astype(np.float32)-base.astype(np.float32),-6,6)
        reconstructed=np.clip(np.asarray(illumination.crop((0,top,full_width,end))).astype(np.float32)+grain,0,255).astype(np.uint8)
        tile=original[top:end].copy();tile[target]=reconstructed[target]
        result.paste(Image.fromarray(tile),(0,top))
    return result


def removal_mask(rgb, protected, model, floor, equipment_mask):
    height, width = protected.shape
    clearance = max(2, round(min(height, width)*.003))
    guard = cv2.dilate(protected.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*clearance+1,)*2)).astype(bool)
    difference = np.max(np.abs(rgb.astype(float)-model), axis=2)
    candidates = ((difference > 38) & ~guard).astype(np.uint8)
    row = round(floor*height)
    candidates[row:] = 0
    if row < height:
        area = np.zeros_like(guard)
        area[min(height,row+clearance):min(height,round(max(floor+.07,.88)*height))] = True
        samples = rgb[area & ~guard & (equipment_mask == 0)]
        if len(samples) > width*5:
            floor_color = np.median(samples, axis=0)
            floor_difference = np.max(np.abs(rgb.astype(float)-floor_color), axis=2)
            candidates[row:] = ((floor_difference[row:] > 65) & ~guard[row:]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    mask = np.zeros_like(candidates)
    for index in range(1, count):
        x,y,w,h,area = stats[index]
        boundary = x <= width*.08 or y <= height*.06 or x+w >= width*.92
        # Bottom-only material gaps belong to the separate floor repair planner.
        # Frame clutter starting above the horizon retains its complete outline.
        if (boundary and y < row) or (y+h < row and 3 <= area <= 100 and w <= 18 and h <= 18):
            component=(labels == index).astype(np.uint8)
            if boundary:
                contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(mask,[cv2.convexHull(np.concatenate(contours))],1)
            else:
                mask[labels == index] = 1
    # Detected equipment keeps its entire outline even across the floor line.
    # The floor itself is not selected for blanket replacement.
    mask |= (equipment_mask > 0).astype(np.uint8)
    # Isolated scuffs and hanging ties can be larger than sensor-dust spots.
    # Compare locally so normal backdrop shading does not become a repair.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    residual=cv2.absdiff(gray,cv2.medianBlur(gray,21))
    marks=((residual>20)&~guard).astype(np.uint8)
    marks[row:]=0
    n,l,s,_=cv2.connectedComponentsWithStats(marks)
    for i in range(1,n):
        x,y,w,h,area=s[i]
        if 4<=area<=1200 and w<width*.10 and h<height*.08:
            mask[l==i]=1
    mask = cv2.dilate(mask, np.ones((9,9), np.uint8))
    mask[guard] = 0
    return mask.astype(bool)
