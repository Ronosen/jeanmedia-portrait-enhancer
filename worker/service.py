"""Small local project, diagnostic and audit operations. No network access."""
import base64
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from PIL import Image
import numpy as np
from runtime import manifest, sha256, MODELS


def change_mask(plan):
    layers = plan.get("layers", {})
    shape = layers.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(not isinstance(x, int) or not 1 <= x <= 1280 for x in shape):
        raise ValueError("No valid change map is available.")
    raw = layers.get("backdrop", "")
    if not isinstance(raw, str) or len(raw) > 3_000_000:
        raise ValueError("Invalid change map.")
    with Image.open(io.BytesIO(base64.b64decode(raw, validate=True))) as mask:
        if mask.size != (shape[1], shape[0]) or mask.mode != "L":
            raise ValueError("Invalid change map dimensions.")
        pixels = np.asarray(mask).copy()
    from floor import decode_mask
    if plan.get("floorRepairs"):
        for name in ("defects", "extension"):
            pixels = np.maximum(pixels, decode_mask(plan["floorRepairs"][name], tuple(shape)))
    if plan.get("propRepairs"):
        pixels = np.maximum(pixels, decode_mask(plan["propRepairs"]["mask"], tuple(shape)))
    if plan.get('backdropFinish'):
        from backdrop_finish import unpack
        pixels=np.maximum(pixels,unpack(plan['backdropFinish']['mask'],shape))
    if plan.get('separation'):
        from backdrop_finish import unpack
        alpha=unpack(layers['foregroundAlpha'],shape)
        pixels=np.maximum(pixels,((alpha>3)&(alpha<250)).astype(np.uint8)*255)
    return Image.fromarray(pixels)


def project(action, data):
    path = Path(data["path"])
    if path.suffix.lower() != ".jeanmedia":
        raise ValueError("Choose a .jeanmedia project file.")
    if action == "save-project":
        source = Path(data["source"]).resolve(strict=True)
        if source == path.resolve():
            raise ValueError("The original photo cannot be used as a project file.")
        saved = {"version": 1, "source": str(source), "sourceSha256": sha256(source),
                 "studioPlan": data.get("studioPlan"), "enhance": bool(data.get("enhance")),
                 "history": data.get("history", []), "runtime": manifest(),
                 "approved": bool(data.get("approved"))}
        if saved["studioPlan"]:
            change_mask(saved["studioPlan"])
        with path.open("x", encoding="utf-8") as stream:
            json.dump(saved, stream, allow_nan=False)
        return {"saved": True}
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Project exceeds the 16 MB limit.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("version") != 1 or not isinstance(saved.get("source"), str):
        raise ValueError("Unsupported project file.")
    source = Path(saved["source"])
    if not source.is_file() or sha256(source) != saved.get("sourceSha256"):
        raise ValueError("The project's original photo is missing or has changed.")
    if saved.get("studioPlan"):
        change_mask(saved["studioPlan"])
    return saved


def diagnostics():
    import cv2
    import onnxruntime as ort
    from runtime import session
    disk = shutil.disk_usage(MODELS)
    hardware = {"os": platform.platform(), "cpu": platform.processor(),
                "threads": os.cpu_count(), "freeDiskBytes": disk.free,
                "providers": ort.get_available_providers(), "modelFolder": str(MODELS),
                "modelBytes": sum(p.stat().st_size for p in MODELS.glob("*.onnx"))}
    if os.name == "nt":
        command = "$ErrorActionPreference='Stop'; $o=Get-CimInstance Win32_OperatingSystem; $g=Get-CimInstance Win32_VideoController; @{windows=$o.Caption;ramKB=$o.TotalVisibleMemorySize;gpu=@($g|Select-Object Name,AdapterRAM)}|ConvertTo-Json -Depth 4 -Compress"
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                    capture_output=True, text=True, timeout=15, creationflags=0x08000000)
            if result.returncode == 0:
                hardware.update(json.loads(result.stdout))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    tests = []
    for name in ("human_segmentation_pphumanseg_2023mar.onnx", "u2netp.onnx", "yolox_nano.onnx", "face_detection_yunet_2023mar.onnx", "face_landmarker_Nx3x256x256.onnx"):
        path = MODELS/name
        try:
            model = session(path)
            tests.append({"model": name, "ok": True, "providers": model.get_providers()})
            del model
        except Exception:
            tests.append({"model": name, "ok": False})
    hardware["selfTest"] = tests
    from neural import MODEL as reconstruction_model, available as reconstruction_available
    hardware["reconstructionModel"] = {"installed": reconstruction_model.is_file(),
                                      "integrityVerified": reconstruction_available(),
                                      "check": "SHA256; inference is checked when processing"}
    hardware["runtime"] = manifest()
    hardware["opencvThreads"] = cv2.getNumThreads()
    return hardware


def dispatch(action, data):
    if action=='mask-preview':
        from local_masks import selection
        mask=selection(data['studioPlan'],data['layer'])
        rgba=np.zeros((*mask.shape,4),np.uint8);rgba[:,:,:3]=(255,80,60);rgba[:,:,3]=np.rint(mask*150).astype(np.uint8)
        buffer=io.BytesIO();Image.fromarray(rgba).save(buffer,'PNG')
        return {'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')}
    if action in ("save-project", "open-project"):
        return project(action, data)
    if action == "diagnostics":
        return diagnostics()
    if action == "mask":
        import cv2
        mask=np.asarray(change_mask(data["studioPlan"]))
        outline=cv2.morphologyEx((mask>0).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        rgba=np.zeros((*mask.shape,4),np.uint8)
        rgba[:,:,:3]=(255,110,75)
        rgba[:,:,3]=outline*220
        buffer = io.BytesIO()
        Image.fromarray(rgba).save(buffer, "PNG")
        return {"url": "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")}
    raise ValueError("Unknown local operation.")


if __name__ == "__main__":
    try:
        payload = sys.stdin.read(16*1024*1024+1)
        if len(payload) > 16*1024*1024:
            raise ValueError("Request too large.")
        print(json.dumps(dispatch(sys.argv[1], json.loads(payload))))
    except Exception as error:
        sys.exit(str(error))
