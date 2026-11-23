"""Conservative automatic edge cropping before repair; original canvas stays canonical."""
import cv2
import numpy as np
from backdrop_finish import pack,unpack


def propose(protected, defects):
    if protected.shape!=defects.shape:raise ValueError('Crop evidence dimensions differ')
    h,w=protected.shape;full=[0,0,w,h]
    result={'version':1,'enabled':False,'previewSize':[w,h],'box':full,'margin':max(2,round(min(h,w)*.025))}
    if protected.sum()<max(16,h*w*.001):return dict(result,reason='insufficient-protection')
    margin=result['margin'];ys,xs=np.nonzero(protected)
    limits=[min(round(w*.18),max(0,int(xs.min())-margin)),
            min(round(h*.18),max(0,int(ys.min())-margin)),
            max(w-round(w*.18),min(w,int(xs.max())+margin+1)),
            max(h-round(h*.18),min(h,int(ys.max())+margin+1))]
    # Only edge-connected/proximate repair evidence justifies losing canvas.
    n,labels,stats,_=cv2.connectedComponentsWithStats(defects.astype(np.uint8))
    edge=max(1,round(min(h,w)*.01));bad=np.zeros((h,w),bool)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area<max(8,h*w*.0005):continue
        if x<=edge or y<=edge or x+bw>=w-edge or y+bh>=h-edge:bad|=labels==i
    l,t,r,b=full;minimum=max(8,h*w*.0007)
    for _ in range(4):
        candidates=[];region=bad[t:b,l:r]
        # Cumulative evidence makes the boundary follow the actual intrusion.
        for side,counts in enumerate((region.sum(axis=0),region.sum(axis=1),region.sum(axis=0)[::-1],region.sum(axis=1)[::-1])):
            maximum=[limits[0]-l,limits[1]-t,r-limits[2],b-limits[3]][side]
            if maximum<=0:continue
            cumulative=np.cumsum(counts)
            for depth in range(1,maximum+1):
                box=[l,t,r,b];box[side]+=(depth if side<2 else -depth)
                nl,nt,nr,nb=box
                if (nr-nl)*(nb-nt)<h*w*.75:continue
                area=depth*((b-t) if side in (0,2) else (r-l))
                removed=int(cumulative[depth-1]);score=removed-.65*(area-removed)
                if removed<minimum or score<minimum:continue
                candidates.append((score,-area,box))
        if not candidates:break
        _,_,box=max(candidates,key=lambda v:(v[0],v[1]));l,t,r,b=box
    if [l,t,r,b]==full:return dict(result,reason='no-safe-beneficial-edge-crop')
    kept=np.zeros((h,w),bool);kept[t:b,l:r]=True
    if np.any(protected&~kept):raise ValueError('Crop intersects protected content')
    return dict(result,enabled=True,box=[l,t,r,b],reason='measured-edge-repair',
                removedRepairPixels=int((bad&~kept).sum()),retainedFraction=round(float(kept.mean()),5))


def protection(plan):
    layers=plan['layers'];shape=layers['shape']
    result=unpack(layers['foregroundAlpha'],shape)>3
    clutter=unpack(layers['clutter'],shape)>0
    for key in ('confirmedSupport','intentionalProps','sourceContact','sourceContactBlend','thinSupport'):
        if layers.get(key):result|=unpack(layers[key],shape)>1
    raw=unpack(layers['rawForegroundAlpha'],shape) if layers.get('rawForegroundAlpha') else None
    if raw is not None:result|=(raw>5)&~clutter
    # An unresolved object is a crop veto, even if its silhouette was missed.
    for item in plan.get('sceneIntent',{}).get('objects',[]):
        if item.get('action')!='uncertain':continue
        if item.get('objectViewDisagreed') and item.get('objectViewLabel') in ('wallpaper','softbox','cables'):continue
        box=item.get('box',[])
        if len(box)==4:
            x0,y0,x1,y1=box;result[max(0,y0):min(shape[0],y1),max(0,x0):min(shape[1],x1)]=True
    if plan.get('floorAnalysis',{}).get('detected'):
        # Reserve ground immediately below feet/supports for contact shadows.
        # Cropping a clear-looking floor strip must not leave floating shoes.
        ys,xs=np.nonzero(result)
        if len(ys) and ys.max()>=round(plan['floor']*shape[0]):
            bottom=min(shape[0]-1,int(ys.max())+max(3,round(shape[0]*.05)))
            result[int(ys.max()):bottom+1,int(xs.min()):int(xs.max())+1]=True
    return result


def prepare(plan, original_size):
    layers=plan['layers'];shape=layers['shape'];guard=protection(plan)
    bad=unpack(layers['clutter'],shape)>0
    if layers.get('roomExterior'):bad|=unpack(layers['roomExterior'],shape)>0
    if plan.get('floorRepairs'):bad|=unpack(plan['floorRepairs']['extension'],shape)>0
    decision=propose(guard,bad);decision['originalSize']=list(original_size)
    layers['cropProtection']=pack(guard.astype(np.uint8)*255);plan['smartCrop']=decision
    keep=retained_mask(plan)
    if not decision['enabled']:return
    # Do not reconstruct/grade areas that will be discarded. Protection and
    # source-reference layers retain full coordinates for deterministic replay.
    for key in ('backdrop','clutter','wallSurface','roomExterior','detachedClutter'):
        if layers.get(key):layers[key]=pack(np.where(keep,unpack(layers[key],shape),0).astype(np.uint8))
    if plan.get('floorRepairs'):
        floor=plan['floorRepairs']
        for key in ('defects','extension'):floor[key]=pack(np.where(keep,unpack(floor[key],shape),0).astype(np.uint8))
        floor['count']=int(cv2.connectedComponents((unpack(floor['defects'],shape)>0).astype(np.uint8))[0]-1)+int((unpack(floor['extension'],shape)>0).any())
    if plan.get('propRepairs'):
        p=plan['propRepairs'];p['mask']=pack(np.where(keep,unpack(p['mask'],shape),0).astype(np.uint8))
    if (unpack(layers['backdrop'],shape)>0).mean()<=.04:
        plan['review']=[m for m in plan.get('review',[]) if m!='Large background reconstruction needs review.']


def validated_box(plan):
    if not plan:return None
    crop=plan.get('smartCrop');shape=plan.get('layers',{}).get('shape')
    if not crop or not crop.get('enabled'):return None
    if not shape or crop.get('version')!=1 or crop.get('previewSize')!=[shape[1],shape[0]]:raise ValueError('Invalid smart crop dimensions')
    h,w=shape;box=crop.get('box',[])
    if len(box)!=4 or any(type(v)!=int for v in box):raise ValueError('Invalid smart crop coordinates')
    l,t,r,b=box
    if not (0<=l<r<=w and 0<=t<b<=h) or (r-l)*(b-t)<h*w*.75:raise ValueError('Unsafe smart crop bounds')
    if l>round(w*.18) or t>round(h*.18) or w-r>round(w*.18) or h-b>round(h*.18):raise ValueError('Smart crop removes too much canvas')
    if not plan['layers'].get('cropProtection'):raise ValueError('Missing smart crop protection')
    guard=unpack(plan['layers']['cropProtection'],shape)>0
    known=unpack(plan['layers']['foregroundAlpha'],shape)>3
    if np.any(known&~guard):raise ValueError('Smart crop protection is incomplete')
    ys,xs=np.nonzero(guard);margin=max(2,round(min(h,w)*.025))
    if not len(ys):raise ValueError('Smart crop has no subject protection')
    if ((l and l>xs.min()-margin) or (t and t>ys.min()-margin)
        or (r<w and r<=xs.max()+margin) or (b<h and b<=ys.max()+margin)):
        raise ValueError('Smart crop is too close to a subject or prop')
    return box


def retained_mask(plan):
    h,w=plan['layers']['shape'];box=validated_box(plan)
    keep=np.ones((h,w),bool)
    if box:
        keep[:]=False;l,t,r,b=box;keep[t:b,l:r]=True
    return keep


def continuous_surface(surface,plan):
    if not plan.get('smartCrop',{}).get('enabled'):return surface
    return surface|~retained_mask(plan)


def output_box(size,plan):
    box=validated_box(plan)
    if box is None:return (0,0,size[0],size[1])
    h,w=plan['layers']['shape'];l,t,r,b=box
    return (round(l*size[0]/w),round(t*size[1]/h),round(r*size[0]/w),round(b*size[1]/h))


def output_size(size,plan):
    l,t,r,b=output_box(size,plan);return r-l,b-t


def apply(image,plan):
    if not plan or not plan.get('smartCrop',{}).get('enabled'):return image
    return image.crop(output_box(image.size,plan))


def context_canvas(source,plan):
    if not plan.get('smartCrop',{}).get('enabled'):return source
    from PIL import Image
    l,t,r,b=output_box(source.size,plan);piece=source.crop((l,t,r,b))
    canvas=Image.new(source.mode,source.size);canvas.paste(piece,(l,t))
    if l:canvas.paste(piece.crop((0,0,1,piece.height)).resize((l,piece.height),Image.Resampling.NEAREST),(0,t))
    if r<source.width:canvas.paste(piece.crop((piece.width-1,0,piece.width,piece.height)).resize((source.width-r,piece.height),Image.Resampling.NEAREST),(r,t))
    if t:canvas.paste(canvas.crop((0,t,source.width,t+1)).resize((source.width,t),Image.Resampling.NEAREST),(0,0))
    if b<source.height:canvas.paste(canvas.crop((0,b-1,source.width,b)).resize((source.width,source.height-b),Image.Resampling.NEAREST),(0,b))
    return canvas
