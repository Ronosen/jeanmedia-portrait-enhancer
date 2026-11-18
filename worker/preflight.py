"""Preview-scale quality checks. Warnings are estimates, never repair guarantees."""
import cv2
import numpy as np


def inspect(image):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    dark, clipped = float(np.mean(gray < 8)), float(np.mean(gray > 248))
    residual = gray.astype(float) - cv2.medianBlur(gray, 3)
    noise = float(np.median(np.abs(residual)) * 1.4826)
    warnings = []
    if min(image.size) < 256:
        warnings.append("Low resolution")
    if blur < 15:
        warnings.append("Possible severe blur")
    if clipped > .2:
        warnings.append("Clipped highlights")
    if dark > .6:
        warnings.append("Deep shadows")
    if noise > 10:
        warnings.append("High noise")
    return {"blurVariance": blur, "highlightFraction": clipped, "shadowFraction": dark,
            "noiseEstimate": noise, "warnings": warnings,
            "needsReview": bool(warnings), "crop": [0, 0, 1, 1]}
