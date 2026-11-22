"""CPU alpha refinement from automatic trimaps; opaque source cores are fixed."""
import os
from pathlib import Path
import cv2
import numpy as np

os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).parent/'.cache'/'numba'))
os.environ.setdefault('NUMBA_NUM_THREADS',str(min(4,os.cpu_count() or 1)))


def dependency_status():
    from importlib.metadata import version
    import pymatting
    expected={'PyMatting':'1.1.16','numba':'0.68.1','llvmlite':'0.50.0','scipy':'1.18.1'}
    installed={name:version(name) for name in expected}
    return {'available':installed==expected and callable(pymatting.estimate_alpha_cf)
            and callable(pymatting.estimate_foreground_ml),'versions':installed}


def refine(rgb, alpha, solver=None):
    if alpha.shape!=rgb.shape[:2] or not np.isfinite(alpha).all():raise ValueError('Invalid edge matte')
    source=np.clip(alpha,0,1).astype(np.float32);result=source.copy()
    unknown=(source>.02)&(source<.98)
    report={'version':1,'engine':'pymatting-cf','tiles':0,'changedPixels':0,'review':[]}
    if not unknown.any():return result,report
    if solver is None:
        from pymatting import estimate_alpha_cf
        solver=estimate_alpha_cf
    # Bound sparse systems and memory; shared source halos prevent tile edges
    # from becoming artificial foreground/background constraints.
    tile=128;halo=16;maximum=96
    h,w=source.shape
    for top in range(0,h,tile):
        for left in range(0,w,tile):
            bottom=min(h,top+tile);right=min(w,left+tile)
            if not unknown[top:bottom,left:right].any():continue
            if report['tiles']>=maximum:
                report['review']=['Some edge regions exceeded the refinement budget.'];break
            y0=max(0,top-halo);x0=max(0,left-halo);y1=min(h,bottom+halo);x1=min(w,right+halo)
            a=source[y0:y1,x0:x1]
            trimap=np.full(a.shape,.5,np.float64);trimap[a<=.02]=0;trimap[a>=.98]=1
            if not (trimap==0).any() or not (trimap==1).any():continue
            try:
                estimate=np.asarray(solver(rgb[y0:y1,x0:x1].astype(np.float64)/255,trimap,
                                           cg_kwargs={'maxiter':150}),dtype=np.float32)
                if estimate.shape!=a.shape or not np.isfinite(estimate).all():raise ValueError('Invalid refined matte')
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                report['review']=['Some subject edges could not be refined safely.'];continue
            # Limit movement in uncertain pixels; neither opaque people/props
            # nor known background can be changed by the edge solver.
            estimate=np.clip(estimate,a-.25,a+.25)
            core=estimate[top-y0:bottom-y0,left-x0:right-x0]
            selected=unknown[top:bottom,left:right]
            region=result[top:bottom,left:right];region[selected]=np.clip(core[selected],0,1)
            report['tiles']+=1
        if report['tiles']>=maximum:break
    if report['tiles']>=maximum:report['review']=['Some edge regions exceeded the refinement budget.']
    report['changedPixels']=int(np.count_nonzero(np.abs(result-source)>1/255))
    return result,report


def foreground_colours(rgb, alpha, estimator=None):
    """Estimate edge foreground without borrowing a distant background pixel."""
    if estimator is None:
        from pymatting import estimate_foreground_ml
        estimator=estimate_foreground_ml
    value=np.asarray(estimator(rgb.astype(np.float64)/255,alpha.astype(np.float64)),dtype=np.float32)
    if value.shape!=rgb.shape or not np.isfinite(value).all():raise ValueError('Invalid edge foreground colours')
    return np.rint(np.clip(value,0,1)*255).astype(np.uint8)


def saved_foreground_colours(source,matte,plan):
    import base64
    from PIL import Image
    h,w=matte.shape
    encoded=plan.get('edgeForeground')
    if encoded:
        if not isinstance(encoded,str) or len(encoded)>8_000_000:raise ValueError('Invalid saved edge colours')
        raw=base64.b64decode(encoded,validate=True)
        if len(raw)<24 or raw[:8]!=b'\x89PNG\r\n\x1a\n' or int.from_bytes(raw[16:20],'big')!=w or int.from_bytes(raw[20:24],'big')!=h:raise ValueError('Invalid edge colour dimensions')
        value=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if value is None or value.shape!=(h,w,3):raise ValueError('Invalid saved edge colours')
        return value
    rgb=np.asarray(source.resize((w,h),Image.Resampling.LANCZOS))
    value=foreground_colours(rgb,matte.astype(float)/255)
    ok,png=cv2.imencode('.png',value)
    if not ok:raise ValueError('Could not encode edge colours')
    plan['edgeForeground']=base64.b64encode(png).decode('ascii')
    return value
