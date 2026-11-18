"""Reconstruct only damaged floor using a closer view of original floor context."""
import numpy as np
from PIL import Image
from neural import repair

def reconstruct(source,extension,foreground,clutter,floor,settings,removed=None):
 h,w=extension.shape
 yy=np.arange(h)[:,None]
 local_context=settings.get('version',1)>=2
 ground_clutter=clutter&(yy>=round((floor-(.08 if local_context else 0))*h))
 clutter_rows=np.flatnonzero(ground_clutter.any(axis=1))
 row=max(0,min(round((floor-.03)*h),int(clutter_rows[0])-round(h*.03) if local_context and len(clutter_rows) else h))
 if local_context and len(clutter_rows) and removed is not None:
  import cv2
  columns=cv2.dilate(ground_clutter.any(axis=0).astype(np.uint8)[None,:],np.ones((1,max(3,round(w*.10))|1),np.uint8))>0
  upper=max(0,round((floor-.15)*h))
  connected=removed&columns&(yy>=upper)&(yy<row)
  extra=np.flatnonzero(connected.any(axis=1))
  if len(extra):row=max(upper,min(row,int(extra[0])-round(h*.03)))
 if 'row' in settings and settings['row']!=row:raise ValueError('Saved floor context belongs to a different floor.')
 settings.update(row=row,plateOnly=True,excludeForegroundContext=False)
 holes=extension|foreground|clutter
 if local_context and removed is not None:holes |= removed
 holes=holes[row:]
 top=round(row/h*source.height)
 crop=source.crop((0,top,source.width,source.height))
 patch=repair(crop,holes,np.zeros_like(holes),settings)
 result=source.copy();result.paste(patch,(0,top))
 selected=(extension|ground_clutter)&~foreground
 if local_context and removed is not None:selected |= removed&(yy>=row)&(yy<round(floor*h))&~foreground
 opacity=selected.astype(np.float32)
 if local_context and settings.get('blendRevision',0)>=1:
  # The closer reconstruction can have slightly different illumination from
  # the whole-background plate. Blend inside its selected upper edge so two
  # valid predictions cannot form a hard rectangle across the backdrop.
  opacity *= np.clip((yy-row)/max(4,h*.06),0,1)
 mask=Image.fromarray(np.rint(opacity*255).astype(np.uint8)).resize(source.size,Image.Resampling.NEAREST)
 return result,mask
