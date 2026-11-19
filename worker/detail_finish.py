"""Bounded, non-generative native-resolution detail finishing."""
import cv2
import numpy as np


def measure(image):
    # Sample original-resolution patches: resized previews hide sensor noise.
    estimates=[]
    for fy in (.2,.5,.8):
        for fx in (.2,.5,.8):
            x=max(0,min(image.width-192,round(image.width*fx)-96))
            y=max(0,min(image.height-192,round(image.height*fy)-96))
            rgb=np.asarray(image.crop((x,y,min(x+192,image.width),min(y+192,image.height))).convert('RGB'))
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
            base=cv2.GaussianBlur(gray,(5,5),1)
            edge=np.maximum(np.abs(cv2.Sobel(base,cv2.CV_32F,1,0)),np.abs(cv2.Sobel(base,cv2.CV_32F,0,1)))
            smooth=(edge<12)&(gray>12)&(gray<240)
            if smooth.sum()>100:
                residual=gray-cv2.medianBlur(gray,3)
                estimates.append(float(np.median(np.abs(residual[smooth]))*1.4826))
    noise=float(np.median(estimates)) if estimates else 0.
    return {'version':1,'noise':min(noise,30.),'denoise':float(np.clip((noise-1.5)/10,0,.45)),
            'sharpen':float(np.clip(.35-noise*.015,.08,.35))}


def apply(rgb, settings, face_guard):
    source=rgb.astype(np.float32)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Small range-aware filter; no face synthesis or broad skin smoothing.
    smooth=cv2.bilateralFilter(source,5,8,1.2)
    pixels=source+(smooth-source)*settings['denoise']
    base=cv2.GaussianBlur(gray,(5,5),.9)
    detail=gray-base
    threshold=max(1.5,settings['noise']*1.8)
    gain=settings['sharpen']*(1-.8*face_guard)
    delta=np.sign(detail)*np.minimum(np.maximum(np.abs(detail)-threshold,0)*gain,3.)
    # A shared luma correction preserves colour; never overshoot local extrema.
    low=cv2.erode(gray,np.ones((3,3),np.uint8));high=cv2.dilate(gray,np.ones((3,3),np.uint8))
    delta=np.clip(delta,low-gray,high-gray)
    return np.rint(np.clip(pixels+delta[...,None],0,255)).astype(np.uint8)
