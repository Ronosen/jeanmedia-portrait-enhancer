"""Local face/iris geometry; no identity recognition or network requests.

ROI convention adapted from yakhyo/mediapipe-face-mesh-onnx, copyright
2026 Yakhyokhuja Valikhujaev (Apache-2.0; FaceLandmarker-LICENSE).
Contour indices follow MediaPipe FaceMesh, copyright 2021 MediaPipe Authors,
Apache-2.0. Inference, confidence checks and retouch masks are app-specific.
"""
from functools import lru_cache
import cv2
import numpy as np
from runtime import MODELS, session, sha256

MODELS_USED = {
    'face_detection_yunet_2023mar.onnx': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    'face_landmarker_Nx3x256x256.onnx': '111795f8703cdeb6d0c68a9f3cc966a0f23f8786bb00f4577a11f461fc4276ac',
}
MOUTH = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
EYES = [([33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246], 468),
        ([263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466], 473)]
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]


@lru_cache(maxsize=1)
def model_paths():
    for name, digest in MODELS_USED.items():
        if sha256(MODELS/name) != digest:
            raise ValueError('Face model integrity check failed: ' + name)
    return [str(MODELS/name) for name in MODELS_USED]


def detect(preview):
    h,w=preview.shape[:2]
    detector=cv2.FaceDetectorYN.create(model_paths()[0], '', (w,h), .85, .3, 5000)
    _, faces=detector.detect(cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
    if faces is None:return []
    return sorted([f for f in faces if min(f[2:4])>=24],key=lambda f:-float(f[-1]))[:32]


def landmarks(rgb, face):
    x,y,w,h=face[:4];cx=x+w/2;cy=y+h/2;side=1.5*max(w,h)
    eyes=face[4:8].reshape(2,2);eyes=eyes[np.argsort(eyes[:,0])]
    angle=float(np.degrees(np.arctan2(*(eyes[1]-eyes[0])[::-1])))
    transform=cv2.getRotationMatrix2D((float(cx),float(cy)),angle,256/side)
    transform[:,2]+=np.array([128-cx,128-cy])
    crop=cv2.warpAffine(rgb,transform,(256,256))
    model=session(model_paths()[1])
    points,logits=model.run(None,{model.get_inputs()[0].name:np.ascontiguousarray(crop.transpose(2,0,1)[None].astype(np.float32)/255)})
    if points.shape!=(1,478,3) or not np.isfinite(points).all() or not np.isfinite(logits).all():
        raise ValueError('Invalid face landmark output')
    if float(np.ravel(logits)[0]) < 4:return None
    inverse=cv2.invertAffineTransform(transform)
    return points[0,:,:2]@inverse[:,:2].T+inverse[:,2]


def polygon(shape, points, indices):
    result=np.zeros(shape,np.uint8)
    cv2.fillPoly(result,[np.rint(points[indices]).astype(np.int32)],1)
    return result>0


def feature_masks(rgb, points):
    """Only visible bright tooth enamel and sclera; never generate teeth/eyes."""
    shape=rgb.shape[:2]
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    light,a,b=lab.transpose(2,0,1);chroma=np.hypot(a-128,b-128)
    teeth=np.zeros(shape,np.float32);white=np.zeros(shape,np.float32)
    mouth=polygon(shape,points,MOUTH)
    width=np.linalg.norm(points[78]-points[308])
    if width>=12 and np.linalg.norm(points[13]-points[14])>width*.07 and mouth.sum()>=12:
        threshold=max(115,float(np.percentile(light[mouth],55)))
        eligible=mouth&(light>=threshold)&(chroma<44)&(a<146)&(b<172)
        if eligible.sum()>=4:
            teeth=cv2.GaussianBlur(eligible.astype(np.float32),(0,0),.65)*eligible
    for indices,center in EYES:
        eye=polygon(shape,points,indices)
        iris=np.zeros(shape,np.uint8)
        radius=max(1.5,float(np.linalg.norm(points[center+1:center+5]-points[center],axis=1).max())*1.2)
        cv2.circle(iris,tuple(np.rint(points[center]).astype(int)),int(np.ceil(radius)),1,-1)
        if eye.sum()<12:continue
        # A shut eyelid has little opening; reflective dark glasses do not
        # become a white eye just because the mesh predicts an eye contour.
        eye_width=np.linalg.norm(points[indices[0]]-points[indices[8]])
        opening=eye.sum()/max(1,eye_width**2)
        if eye_width<8 or opening<.09:continue
        threshold=max(100,float(np.percentile(light[eye],55)))
        eligible=eye&(iris==0)&(light>=threshold)&(chroma<36)&(a<146)&(b<162)
        white=np.maximum(white,cv2.GaussianBlur(eligible.astype(np.float32),(0,0),.55)*eligible)
    return teeth,white,polygon(shape,points,OVAL).astype(np.uint8)*255
