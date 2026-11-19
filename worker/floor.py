"""Localized floor repairs that keep all unmasked floor pixels intact."""

import base64
import cv2
import numpy as np
from PIL import Image


def encode_mask(mask):
    encoded = base64.b64encode(cv2.imencode(".png", mask)[1]).decode("ascii")
    if len(encoded) > 32000:
        raise ValueError("Needs Review · Floor damage is too complex for automatic patching.")
    return encoded


def decode_mask(encoded, shape):
    if not isinstance(encoded, str) or len(encoded) > 32000:
        raise ValueError("Invalid floor patch mask.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("Invalid floor patch mask.") from error
    if (len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n"
            or int.from_bytes(raw[16:20], "big") != shape[1]
            or int.from_bytes(raw[20:24], "big") != shape[0]):
        raise ValueError("Invalid floor patch dimensions.")
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape or not np.isin(mask, (0, 255)).all():
        raise ValueError("Invalid floor patch mask.")
    return mask


def close_guard_gaps(extension,protected):
    """Fill only guard holes inside an already measured floor-extension band."""
    result=extension.copy();h,w=result.shape
    radius=max(8,round(min(h,w)*.04))
    near=cv2.dilate(protected.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))>0
    for y in np.flatnonzero(extension.any(axis=1)):
        columns=np.flatnonzero(extension[y])
        if len(columns)<w*.25:continue
        span=np.zeros(w,bool);span[columns[0]:columns[-1]+1]=True
        result[y] |= span&near[y]&~protected[y]
    return result


def front_edge_extension(rgb,available,row):
    """Locate a low-contrast mat front only when bottom material differs."""
    h,w=available.shape;result=np.zeros((h,w),np.uint8)
    start=max(row+8,round(h*.84));end=round(h*.90)
    if start>=end:return result
    sample=rgb[start:end][available[start:end]]
    bottom=rgb[round(h*.985):][available[round(h*.985):]]
    if len(sample)<w*4 or len(bottom)<w*.35:return result
    reference=np.median(sample,axis=0);outside=np.median(bottom,axis=0)
    # A healthy reflection or gradual floor shading is not a missing mat.
    difference=np.max(np.abs(bottom.astype(float)-reference),axis=1)
    if np.max(np.abs(reference-outside))<18 and not (np.percentile(difference,75)>25 and np.mean(difference>18)>.25):return result
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,18,45);edges[:max(row,round(h*.90))]=0;edges[~available]=0
    lines=cv2.HoughLinesP(edges,1,np.pi/180,max(25,w//8),minLineLength=w*.28,maxLineGap=max(12,w//15))
    candidates=[]
    if lines is None:return result
    for x0,y0,x1,y1 in lines[:,0]:
        if abs(x1-x0)<w*.28 or abs(y1-y0)>abs(x1-x0)*.16:continue
        xs=np.arange(min(x0,x1),max(x0,x1)+1)
        ys=np.rint(y0+(xs-x0)*(y1-y0)/(x1-x0)).astype(int)
        above=rgb[np.maximum(0,ys-10),xs].astype(float)
        below=rgb[np.minimum(h-1,ys+10),xs].astype(float)
        if np.median(np.max(np.abs(above-below),axis=1))<8:continue
        contrast=float(np.median(np.max(np.abs(above-below),axis=1)))
        candidates.append((float(np.median(ys)),int(x0),int(y0),int(x1),int(y1),contrast))
    if not candidates:return result
    strong=[v for v in candidates if v[-1]>=max(20,max(v[-1] for v in candidates)*.4)]
    if strong:candidates=strong
    first=min(candidates)[0]
    values=np.array([np.median(gray[y,available[y]]) if available[y].any() else np.nan for y in range(h)])
    reference=float(np.nanmedian(values[max(row,int(first-h*.12)):max(row+1,int(first-h*.06))]))
    folded=None
    for y in range(max(row,int(first-h*.06)),max(row,int(first)-5)):
        if np.isfinite(values[y:y+6]).all() and np.median(values[y:y+6])<reference-max(6,reference*.20):folded=y;break
    for middle,x0,y0,x1,y1,contrast in candidates:
        if middle>first+h*.025:continue
        yy,xx=np.mgrid[:h,:w]
        boundary=(folded-h*.025) if folded is not None else y0+(xx-x0)*(y1-y0)/(x1-x0)-max(3,round(h*.01))
        selected=(yy>=np.maximum(row,boundary))&available
        if abs(x1-x0)<w*.5 and abs(y1-y0)>abs(x1-x0)*.06:selected &= (xx>=min(x0,x1)-4)&(xx<=max(x0,x1)+4)
        result[selected]=255
    return result


def plan_floor_patches(rgb, protected, floor, context=False):
    height, width = protected.shape
    row = round(floor * height)
    clearance = max(8, round(min(height, width) * .018))
    guard = cv2.dilate(protected.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2*clearance+1, 2*clearance+1))).astype(bool)
    available = ~guard
    available[:min(height, row + clearance)] = False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.medianBlur(gray, 11)
    residual = cv2.absdiff(gray, smooth)
    noise = float(np.median(residual[available])) * 1.4826 if available.any() else 0.
    threshold = max(8., min(24., noise * 4)) if context else 18
    defect = (residual > threshold) & available
    count, labels, stats, _ = cv2.connectedComponentsWithStats(defect.astype(np.uint8))
    repairs = np.zeros((height, width), np.uint8)
    patches = 0
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        compact=area<=90 and w<=20 and h<=20
        scratch=context and area<=240 and max(w,h)<=64 and min(w,h)<=6
        if not (3 <= area and (compact or scratch) and x >= 8
                and y >= 8 and x+w+8 < width and y+h+8 < height):
            continue
        region = np.s_[y-8:y+h+8, x-8:x+w+8]
        core = labels[region] == label
        expanded = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        samples = rgb[region][~expanded].astype(float)
        # Other tiny scuffs should not make an otherwise healthy donor ring
        # look like a textured surface. Use its central distribution, while
        # still rejecting patterned floors and broad reflection boundaries.
        if context and len(samples):
            lo,hi=np.percentile(samples,[10,90],axis=0)
            samples=samples[np.all((samples>=lo)&(samples<=hi),axis=1)]
        if guard[region].any() or not len(samples) or samples.std(axis=0).max() > (10 if context else 7):
            continue
        repairs[region][expanded] = 255
        patches += 1
        if patches >= 128:
            break
    extension = np.zeros_like(repairs)
    review = ""
    if context:
        extension |= front_edge_extension(rgb,available,row)
        if extension.any():review='Reconstructed floor edge needs a texture/shadow check.'
        # Printed mat tape may be a dark band between two similarly light
        # materials, so it is not connected to a bottom colour-outlier blob.
        row_values=np.full(height,np.nan,dtype=np.float32)
        for y in range(max(row+clearance,round(height*.90)),height):
            donors=gray[y,available[y]]
            if len(donors)>width*.40:row_values[y]=np.median(donors)
        for y in range(max(row+clearance+20,round(height*.92)),height-12):
            top=row_values[y-20:y-6];bottom=row_values[y+6:y+12]
            if not (np.isfinite(row_values[y]) and np.isfinite(top).sum()>8 and np.isfinite(bottom).sum()>3):continue
            contrast=min(float(np.nanmedian(top)),float(np.nanmedian(bottom)))-row_values[y]
            if contrast>22:
                extension[max(row,y-24):][available[max(row,y-24):]]=255
                review='Reconstructed floor edge needs a texture/shadow check.'
                break
        band=round(height*.98)
        reference=np.median(rgb[round(height*.94):round(height*.97)],axis=0)
        abrupt=(np.max(np.abs(rgb.astype(float)-reference[None]),axis=2)>60)&available
        abrupt[:band]=False
        n,l,s,_=cv2.connectedComponentsWithStats(abrupt.astype(np.uint8))
        for i in range(1,n):
            x,y,w,h,area=s[i]
            if y+h==height and w>width*.15 and area>width*height*.0005:
                local=cv2.dilate((l==i).astype(np.uint8),np.ones((5,5),np.uint8))>0
                extension[local&available]=255
                review="Reconstructed floor edge needs a texture/shadow check."
        # A mat lip can occupy only a sliver and share a component with a
        # reflection. Detect its long straight boundary separately.
        edges=cv2.Canny(gray,45,100)
        edges[:round(height*.90)]=0;edges[guard]=0
        lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(25,width//6),
                              minLineLength=width*.28,maxLineGap=12)
        if lines is not None:
            for x0,y0,x1,y1 in lines[:,0]:
                if abs(x1-x0)<width*.28 or abs(y1-y0)>abs(x1-x0)*.16: continue
                xs=np.arange(min(x0,x1),max(x0,x1)+1)
                ys=np.rint(y0+(xs-x0)*(y1-y0)/(x1-x0)).astype(int)
                above=rgb[np.maximum(0,ys-5),xs].astype(float)
                below=rgb[np.minimum(height-1,ys+4),xs].astype(float)
                if np.median(np.max(np.abs(above-below),axis=1))<35: continue
                yy,xx=np.mgrid[:height,:width]
                boundary=y0+(xx-x0)*(y1-y0)/(x1-x0)-2
                selected=(yy>=boundary)&(xx>=xs[0]-4)&(xx<=xs[-1]+4)&available
                extension[selected]=255
                review="Reconstructed floor edge needs a texture/shadow check."
    color = np.zeros(3, dtype=float)
    sample_area = available.copy()
    sample_area[round(height*(.93 if floor>.85 else .90)):] = False
    samples = rgb[sample_area]
    if len(samples) > width * 8:
        color = np.median(samples, axis=0)
        outlier = (np.max(np.abs(rgb.astype(float)-color), axis=2) > 55) & available
        outlier[:max(row, round(height*.82))] = False
        count, labels, stats, _ = cv2.connectedComponentsWithStats(outlier.astype(np.uint8))
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            thin_edge = context and y > height*.96 and w > width*.15 and h < height*.04
            if y+h != height or (w < width*.5 and not thin_edge) or not ((.0005 if thin_edge else .005) < area/(width*height) < .18):
                continue
            low, high = np.percentile(samples, [5, 95], axis=0)
            regular = samples[np.all((samples >= low) & (samples <= high), axis=1)]
            if ((not context and area > width*height*.04) or not len(regular)
                    or regular.std(axis=0).max() > (60 if thin_edge else 40 if context else 12)):
                review = "A large or textured floor edge needs review."
                continue
            selected = labels == label
            # Require a clear material/colour change rather than a gradual shadow.
            ring = cv2.dilate(selected.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~selected
            ring &= available
            if np.count_nonzero(ring) < (w // 2 if thin_edge else width // 2):
                continue
            if not context and np.median(np.max(np.abs(rgb[ring].astype(float)-color), axis=1)) > 40:
                continue
            contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            simple = np.zeros_like(repairs)
            for contour in contours:
                outline = cv2.approxPolyDP(contour, 2, True)
                cv2.fillPoly(simple, [outline], 1)
            selected = simple.astype(bool) & available
            if context and area > width*height*.01 and w>width*.5:
                # Include the tape/mat lip immediately above exposed tiles.
                # Expand vertically along its existing sloping boundary, not
                # across the entire healthy floor or around the subject.
                band = max(4, round(height*.03))
                columns=np.flatnonzero(selected.any(axis=0))
                edges=np.array([np.flatnonzero(selected[:,column])[0] for column in columns],float)
                good=np.ones(len(columns),bool)
                for _ in range(4):
                    line=np.polyfit(columns[good],edges[good],1)
                    residual=np.abs(np.polyval(line,columns)-edges)
                    good=residual < max(5,float(np.median(residual))*2.5)
                    if np.count_nonzero(good)<width*.4:
                        break
                if np.count_nonzero(good)<width*.4:
                    review="The floor edge could not be located consistently."
                    continue
                yy,xx=np.mgrid[:height,:width]
                selected=yy >= np.maximum(row,np.polyval(line,xx)-band)
                selected &= available
                review = "Reconstructed floor edge needs a texture/shadow check."
            span = max(6, round(height * .06))
            for column in np.flatnonzero(selected.any(axis=0)):
                if context:
                    continue
                edge = int(np.flatnonzero(selected[:, column])[0])
                if edge-span < row or guard[edge-span:edge, column].any():
                    selected[:, column] = False
            if not selected.any():
                continue
            extension[selected] = 255
            patches += 1
    repairs[extension > 0] = 0
    try:
        encoded_extension = encode_mask(extension)
    except ValueError:
        extension[:] = 0
        encoded_extension = encode_mask(extension)
        review = "Mat-edge extension needs review; the original floor was retained there."
    patches = cv2.connectedComponents(((repairs > 0) | (extension > 0)).astype(np.uint8))[0] - 1
    return {"version": 1, "shape": [height, width],
            "defects": encode_mask(repairs), "extension": encoded_extension,
            "count": patches, "review": review, "color": color.tolist(),
            "engine": "context-v1" if context else "clone-v1"}


def validate_floor_patches(protected, floor, plan):
    shape = protected.shape
    if plan.get("version") != 1 or plan.get("shape") != list(shape):
        raise ValueError("Invalid floor patch settings.")
    defects = decode_mask(plan.get("defects"), shape)
    extension = decode_mask(plan.get("extension"), shape)
    forbidden = protected.copy()
    forbidden[:round(floor*shape[0])] = True
    if np.any(((defects > 0) | (extension > 0)) & forbidden):
        raise ValueError("Needs Review · A saved floor patch overlaps a protected area.")
    return defects, extension


def apply_floor_patches(image, floor, plan, protected=None):
    shape = plan.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(not isinstance(value, int) or not 1 <= value <= 1280 for value in shape)):
        raise ValueError("Invalid floor patch dimensions.")
    defects, extension = validate_floor_patches(np.zeros(shape, bool), floor, plan)
    result = image.copy()
    # Work on the original-resolution pixels for floor texture and reflections.
    full_defects = cv2.resize(defects, image.size, interpolation=cv2.INTER_NEAREST)
    full_extension = cv2.resize(extension, image.size, interpolation=cv2.INTER_NEAREST)
    pixels = np.asarray(image).copy()
    if full_defects.any():
        count, _, stats, _ = cv2.connectedComponentsWithStats(full_defects)
        pad = max(5, round(image.width / shape[1] * 5))
        for x, y, w, h, _ in stats[1:count]:
            region = np.s_[max(0, y-pad):min(image.height, y+h+pad),
                           max(0, x-pad):min(image.width, x+w+pad)]
            pixels[region] = cv2.inpaint(pixels[region], full_defects[region], min(pad, 12), cv2.INPAINT_TELEA)
    if full_extension.any() and plan.get('engine')=='texture-v1':
        # Use one continuous reflected strip, rather than independently cycling
        # a different donor list in every column (which produced vertical steps).
        small=np.asarray(image.resize((shape[1],shape[0]),Image.Resampling.LANCZOS))
        healthy=(extension==0)&(defects==0)
        healthy[:round(floor*shape[0])]=False
        if protected is not None:healthy &= ~protected.astype(bool)
        if not healthy.any():raise ValueError('Needs Review · No clean floor texture is available for this gap.')
        edges_y=np.flatnonzero(extension.any(axis=1))
        reference_end=max(round(floor*shape[0])+1,min(round(shape[0]*.90),int(edges_y[0])-round(shape[0]*.03)))
        reference_start=round(floor*shape[0])
        for column in range(shape[1]):
            pool=small[reference_start:reference_end,column][healthy[reference_start:reference_end,column]]
            if len(pool)<3:
                pool=small[:,column][healthy[:,column]]
            if len(pool)<3:continue
            centre=np.median(pool,axis=0);spread=np.median(np.max(np.abs(pool.astype(float)-centre),axis=1))
            limit=np.clip(spread*3+8,18,35)
            healthy[:,column] &= np.max(np.abs(small[:,column].astype(float)-centre),axis=1)<limit
        if not healthy.any():raise ValueError('Needs Review · No matching floor texture is available for this gap.')
        valid=healthy.astype(np.float32)
        sigma=max(6,shape[1]*.035)
        norm=cv2.GaussianBlur(valid,(0,0),sigma)
        low=cv2.GaussianBlur(small.astype(np.float32)*valid[:,:,None],(0,0),sigma)/np.maximum(norm[:,:,None],.00001)
        _,labels=cv2.distanceTransformWithLabels((~healthy).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        points=np.argwhere(healthy);span=max(6,round(image.height*.06))
        sy=image.height/shape[0];sx=image.width/shape[1]
        source=np.asarray(image)
        grain_base=cv2.GaussianBlur(source,(0,0),max(1,image.width/shape[1]*.6))
        boundary_colours=np.full((shape[1],3),np.nan,np.float32)
        for col in range(shape[1]):
            selected_rows=np.flatnonzero(extension[:,col])
            if not len(selected_rows):continue
            edge=int(selected_rows[0]);lo=max(round(floor*shape[0]),edge-8)
            band=small[lo:edge,col][healthy[lo:edge,col]]
            if len(band):boundary_colours[col]=np.median(band,axis=0)
        known=np.flatnonzero(np.isfinite(boundary_colours).all(axis=1))
        if len(known):
            for channel in range(3):boundary_colours[:,channel]=np.interp(np.arange(shape[1]),known,boundary_colours[known,channel])
            boundary_colours=cv2.GaussianBlur(boundary_colours[None],(0,0),3)[0]
        for x in np.flatnonzero(full_extension.any(axis=0)):
            rows=np.flatnonzero(full_extension[:,x]);edge=int(rows[0]);phase=(rows-edge)%(2*span)
            donor_y=np.clip(edge-1-np.minimum(phase,2*span-1-phase),round(floor*image.height),image.height-1)
            donor_x=np.full(len(rows),x,dtype=int)
            iy=np.clip(((donor_y+.5)/sy).astype(int),0,shape[0]-1);ix=min(shape[1]-1,int((x+.5)/sx))
            bad=~healthy[iy,ix]
            if bad.any():
                nearest=points[np.clip(labels[iy[bad],ix]-1,0,len(points)-1)]
                donor_y[bad]=np.clip((nearest[:,0]*sy+((donor_y[bad]+.5)%sy)-.5).astype(int),0,image.height-1)
                donor_x[bad]=np.clip((nearest[:,1]*sx+((x+.5)%sx)-.5).astype(int),0,image.width-1)
            # Transfer actual high-frequency grain onto the illumination at the
            # repaired boundary. This avoids duplicating shadows, tape or large
            # reflected objects when the donor strip passes close to a seat.
            dy_small=np.clip(((donor_y+.5)/sy).astype(int),0,shape[0]-1)
            dx_small=np.clip(((donor_x+.5)/sx).astype(int),0,shape[1]-1)
            residual=np.clip(source[donor_y,donor_x].astype(float)-grain_base[donor_y,donor_x],-12,12)
            edge_small=max(round(floor*shape[0]),min(shape[0]-1,int((edge-2)/sy)))
            target=low[edge_small,ix]
            if norm[edge_small,ix]<.0001:target=low[points[labels[edge_small,ix]-1,0],points[labels[edge_small,ix]-1,1]]
            if len(known):target=boundary_colours[ix]
            synthesised=np.clip(target+residual,0,255)
            blend=np.clip((rows-edge)/max(3,sy*2),0,1)[:,None]
            anchor=source[max(0,edge-1),x].astype(float)
            pixels[rows,x]=np.rint(anchor*(1-blend)+synthesised*blend).astype(np.uint8)
    elif full_extension.any() and plan.get('engine')=='surface-v1':
        small=np.asarray(Image.fromarray(pixels).resize((shape[1],shape[0]),Image.Resampling.LANCZOS))
        continued=cv2.inpaint(small,extension,7,cv2.INPAINT_NS)
        expanded=np.asarray(Image.fromarray(continued).resize(image.size,Image.Resampling.BILINEAR))
        pixels[full_extension>0]=expanded[full_extension>0]
    elif full_extension.any() and plan.get("engine") == "context-v1":
        from inpainting import repair
        pixels = np.asarray(repair(Image.fromarray(pixels), extension, protected, floor, plan)).copy()
    elif full_extension.any():
        span = max(6, round(image.height * .06))
        source = np.asarray(image)
        full_guard=cv2.resize(protected.astype(np.uint8),image.size,interpolation=cv2.INTER_NEAREST) if protected is not None else None
        color = np.asarray(plan.get("color"), dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 255):
            raise ValueError("Invalid floor donor colour.")
        for x in np.flatnonzero(full_extension.any(axis=0)):
            rows = np.flatnonzero(full_extension[:, x])
            edge = int(rows[0])
            if edge < span:
                raise ValueError("Needs Review · Too little floor texture is available for extension.")
            candidates = np.arange(edge-span, edge)
            healthy = np.max(np.abs(source[candidates, x].astype(float)-color), axis=1) < 35
            if full_guard is not None:healthy &= full_guard[candidates,x]==0
            if not healthy.any():
                valid=candidates>=round(floor*image.height)
                if full_guard is not None:valid &= full_guard[candidates,x]==0
                local=source[candidates[valid],x].astype(float)
                if len(local)>=6:
                    low,high=np.percentile(local,[25,75],axis=0)
                    if np.max(high-low)<20:
                        centre=np.median(local,axis=0)
                        healthy=valid&(np.max(np.abs(source[candidates,x].astype(float)-centre),axis=1)<25)
            candidates = candidates[healthy]
            if not len(candidates):
                # A reflection can occupy an entire column. Use clean adjacent
                # floor pixels in the same donor band, not the tape below it.
                left=max(0,x-span);right=min(image.width,x+span+1)
                local=source[edge-span:edge,left:right]
                good=np.max(np.abs(local.astype(float)-color),axis=2)<45
                if full_guard is not None:
                    good &= full_guard[edge-span:edge,left:right]==0
                pool=local[good]
                if len(pool):
                    pixels[rows,x]=pool[np.arange(len(rows))%len(pool)]
                    continue
                raise ValueError("Needs Review · No clean floor texture is available for this gap.")
            donors = candidates[-1 - ((rows-edge) % len(candidates))]
            pixels[rows, x] = source[donors, x]
    mask = (full_defects > 0) | (full_extension > 0)
    # The assignment makes unchanged pixels byte-for-byte identical.
    output = np.asarray(result).copy()
    output[mask] = pixels[mask]
    return Image.fromarray(output)

