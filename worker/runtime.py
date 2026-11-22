"""Local model adapter, progress reporting and reproducibility metadata."""
import hashlib
import json
import os
from pathlib import Path
import sys

MODELS = Path(__file__).with_name("models")


def progress(stage, value):
    print("@progress:" + json.dumps({"stage": stage, "percent": value}), file=sys.stderr, flush=True)


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if Path(path).name in ('birefnet-general-lite.onnx','lama_fp32.onnx'):
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
    if Path(path).name == "lama_fp32.onnx":
        # The app processes one photo at a time. Resolving the export's symbolic
        # batch dimension avoids expensive dynamic-shape initialization.
        options.add_free_dimension_override_by_name("batch",1)
    available = ort.get_available_providers()
    # CPU is the reproducible low-memory default; acceleration is opt-in.
    providers = ["CPUExecutionProvider"]
    if os.environ.get("JEANMEDIA_DEVICE") == "gpu" and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    try:
        return ort.InferenceSession(str(path), options, providers=providers)
    except Exception:
        if len(providers) == 1:
            raise
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest():
    from importlib.metadata import version,PackageNotFoundError
    import cv2
    import numpy
    import onnxruntime
    import PIL
    edge={}
    for package in ('PyMatting','numba','llvmlite','scipy'):
        try:edge[package]=version(package)
        except PackageNotFoundError:edge[package]=None
    return {"worker": 3, "python": sys.version.split()[0], "pillow": PIL.__version__,
            "edgeRefinement":edge,
            "opencv": cv2.__version__, "numpy": numpy.__version__, "onnxruntime": onnxruntime.__version__,
            "models": {p.name: sha256(p) for p in sorted(MODELS.glob("*.onnx"))}}
