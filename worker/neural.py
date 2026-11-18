"""Local LaMa adapter. No downloads, uploads, prompts or source-file writes."""
import base64
from functools import lru_cache
import io

import cv2
import numpy as np
from PIL import Image
from runtime import MODELS, session, sha256, progress

MODEL = MODELS / "lama_fp32.onnx"
EXPECTED_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"


@lru_cache(maxsize=1)
def available():
    if not MODEL.is_file():
        return False
    if sha256(MODEL) != EXPECTED_SHA256:
        raise ValueError("The local reconstruction model failed its integrity check.")
    return True


@lru_cache(maxsize=1)
def model_session():
    if not available():
        raise ValueError("The local reconstruction model has not been installed.")
    progress("Loading local reconstruction model",75)
    model = session(MODEL)
    inputs = {item.name: item.shape for item in model.get_inputs()}
    if (set(inputs)!={"image","mask"} or any(len(shape)!=4 for shape in inputs.values())
            or inputs["image"][1:]!=[3,512,512] or inputs["mask"][1:]!=[1,512,512]):
        raise ValueError("Unexpected reconstruction model interface.")
    return model


def repair(image, mask, protected, settings):
    selected=mask>0
    if settings.get('floorFillCloned'):
        from floor import decode_mask
        selected &= decode_mask(settings['floorRepairs']['extension'],mask.shape)==0
    if mask.shape != protected.shape or np.any(selected & protected):
        raise ValueError("Reconstruction overlaps a protected person or prop.")
    if not selected.any():
        return image.copy()
    if selected.mean()>.04 and not settings.get("floorRepairs",{}).get("review"):
        message="Large background reconstruction needs review."
        review=settings.setdefault("review",[])
        if message not in review:review.append(message)
    h,w=mask.shape
    limit=.95 if settings.get('plateOnly') is True else .45
    if not 1<=max(h,w)<=1280 or selected.mean()>limit:
        raise ValueError("Needs Review · The missing area is too large to reconstruct safely.")
    scale=512/max(h,w)
    rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    encoded=settings.get("inpaintPatch")
    if encoded:
        if not isinstance(encoded,str) or len(encoded)>4_000_000:
            raise ValueError("Invalid saved reconstruction.")
        raw=base64.b64decode(encoded,validate=True)
        if (len(raw)<24 or raw[:8]!=b"\x89PNG\r\n\x1a\n" or int.from_bytes(raw[16:20],"big")!=rw
                or int.from_bytes(raw[20:24],"big")!=rh):
            raise ValueError("Invalid saved reconstruction size.")
        with Image.open(io.BytesIO(raw)) as cached:
            patch=cached.convert("RGB")
    else:
        progress("Reconstructing missing background locally",75)
        pixels=np.asarray(image.resize((rw,rh),Image.Resampling.LANCZOS))
        hole=cv2.resize(selected.astype(np.float32),(rw,rh),interpolation=cv2.INTER_AREA)>0
        if selected.sum()<=max(64,selected.size*.005):
            ring=(cv2.dilate(selected.astype(np.uint8),np.ones((15,15),np.uint8))>0)&~selected
            original_preview=np.asarray(image.resize((w,h),Image.Resampling.LANCZOS))
            if ring.any() and not np.any(ring&protected) and original_preview[ring].std(axis=0).max()<12:
                patch=Image.fromarray(cv2.inpaint(pixels,hole.astype(np.uint8)*255,3,cv2.INPAINT_TELEA))
                buffer=io.BytesIO();patch.save(buffer,"PNG")
                settings["inpaintPatch"]=base64.b64encode(buffer.getvalue()).decode("ascii")
                settings["inpaintMethod"]="local-heal"
                return repair(image,mask,protected,settings)
        if settings.get("excludeForegroundContext",True):
            foreground=cv2.resize(protected.astype(np.float32),(rw,rh),interpolation=cv2.INTER_AREA)>0
            foreground=cv2.dilate(foreground.astype(np.uint8),np.ones((5,5),np.uint8))>0
            # Keep people/props out of the colour/shape donor context. This is
            # only a temporary network input; source foreground is never edited.
            context=cv2.inpaint(pixels,(hole|foreground).astype(np.uint8)*255,7,cv2.INPAINT_NS)
            pixels=pixels.copy();pixels[foreground]=context[foreground]
        # Preserve the original aspect ratio; pad only the network input.
        pixels=np.pad(pixels,((0,512-rh),(0,512-rw),(0,0)),mode="symmetric")
        hole=np.pad(hole,((0,512-rh),(0,512-rw)),mode="symmetric")
        tensor=np.ascontiguousarray(pixels.transpose(2,0,1)[None],dtype=np.float32)/255
        missing=np.ascontiguousarray(hole[None,None],dtype=np.float32)
        model=model_session()
        progress("Reconstructing missing background locally",75)
        output=model.run(None,{"image":tensor,"mask":missing})[0]
        if output.shape != (1,3,512,512) or not np.isfinite(output).all():
            raise ValueError("Unexpected reconstruction output.")
        pixels=np.rint(np.clip(output[0].transpose(1,2,0)[:rh,:rw],0,255)).astype(np.uint8)
        patch=Image.fromarray(pixels)
        floor=settings.get('floorRepairs')
        if settings.get('adaptiveRevision') and floor:
            from floor import decode_mask,encode_mask
            extension=decode_mask(floor['extension'],mask.shape)>0
            if extension.mean()>.005:
                original=np.asarray(image.resize((w,h),Image.Resampling.LANCZOS))
                predicted=np.asarray(patch.resize((w,h),Image.Resampling.LANCZOS))
                donor=~(protected|selected)
                yy=np.arange(h)[:,None]
                donor &= (yy>int(settings['floor']*h)+5)&(yy<min(h,int((settings['floor']+.10)*h)))
                if donor.sum()>w*4:
                    samples=original[donor].mean(axis=1)
                    generated=predicted[extension].mean(axis=1)
                    if np.percentile(generated,25)<np.percentile(samples,10)-35 or np.percentile(generated,75)>np.percentile(samples,90)+35:
                        # Bad floor continuations must not replace the source
                        # merely because the network returned a finite tensor.
                        selected &= ~extension
                        if settings.get('adaptiveRevision',0)>=3:
                            floor['engine']='surface-v1'
                            floor['review']='Floor filled from adjacent texture; inspect contact shadows.'
                            settings['floorFillCloned']=True
                        else:
                            floor['extension']=encode_mask(np.zeros(mask.shape,np.uint8))
                            floor['review']='Floor-edge fill rejected: source retained for review.'
                            floor['count']=int(cv2.connectedComponents((decode_mask(floor['defects'],mask.shape)>0).astype(np.uint8))[0]-1)
                        settings['floorFillRejected']=True
        buffer=io.BytesIO();patch.save(buffer,"PNG")
        settings["inpaintPatch"]=base64.b64encode(buffer.getvalue()).decode("ascii")
        settings["inpaintModelSha256"]=EXPECTED_SHA256
        settings["inpaintBatchSize"]=1
        settings["inpaintMethod"]="lama-fp32"
        settings["excludeForegroundContext"]=bool(settings.get("excludeForegroundContext",True))
    # Neural output can never touch unselected source pixels, even if its
    # unmasked prediction alters a face or invents details elsewhere.
    full_mask=Image.fromarray(selected.astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST)
    if settings.get('adaptiveRevision') and not settings.get('plateOnly'):
        distance=cv2.distanceTransform(selected.astype(np.uint8),cv2.DIST_L2,3)
        feather=Image.fromarray(np.rint(np.minimum(1,distance/4)*255).astype(np.uint8)).resize(image.size,Image.Resampling.BILINEAR)
        full_mask=Image.fromarray(np.minimum(np.asarray(feather),np.asarray(full_mask)))
    return Image.composite(patch.resize(image.size,Image.Resampling.LANCZOS),image,full_mask)
