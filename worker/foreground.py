"""Small salient-foreground model for preserving seats, shoes and contact props."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def foreground_probability(rgb):
    from runtime import session as model_session

    session = model_session(Path(__file__).with_name("models") / "u2netp.onnx")
    resized = np.asarray(Image.fromarray(rgb).resize((320, 320), Image.Resampling.LANCZOS),
                         dtype=np.float32)
    normalized = resized / max(float(resized.max()), 1)
    normalized = (normalized - np.array([.485, .456, .406], np.float32)) / np.array(
        [.229, .224, .225], np.float32)
    result = session.run(None, {
        session.get_inputs()[0].name: normalized.transpose(2, 0, 1)[None].copy()
    })[0]
    if result.shape != (1, 1, 320, 320) or not np.isfinite(result).all():
        raise ValueError("Unexpected foreground mask output.")
    mask = result[0, 0]
    spread = float(np.ptp(mask))
    if spread < 1e-6:
        raise ValueError("Needs Review · Foreground boundaries could not be resolved.")
    mask = (mask - mask.min()) / spread
    return np.clip(cv2.resize(mask, (rgb.shape[1], rgb.shape[0])), 0, 1)

