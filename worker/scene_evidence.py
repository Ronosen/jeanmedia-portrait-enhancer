"""Measure scene boundaries and local backdrop defects without frame-position presets."""
import cv2
import numpy as np


def floor_evidence(rgb, protected):
    h, w = protected.shape
    radius = max(3, round(h * .0125))
    # Independent columns prevent a chair, softbox or one dark corner from
    # becoming a full-width floor. Missing foreground samples are interpolated.
    profiles = []; observed = []
    for columns in np.array_split(np.arange(w), 8):
        values = np.full((h, 3), np.nan)
        for y in range(h):
            valid = ~protected[y, columns]
            if valid.sum() >= max(3, len(columns) * .4):
                values[y] = np.median(rgb[y, columns[valid]], axis=0)
        known = np.flatnonzero(np.isfinite(values[:, 0]))
        if len(known) < h * .5:
            continue
        observed.append(np.isfinite(values[:, 0]))
        for c in range(3):
            values[:, c] = np.interp(np.arange(h), known, values[known, c])
        profiles.append(cv2.GaussianBlur(values.astype(np.float32), (1, 0), 0, sigmaY=max(1, radius / 4)))
    result = {'method': 'column-consensus-v1', 'detected': False, 'confidence': 0.0}
    if len(profiles) < 3 or h <= 8 * radius:
        return 1.0, result
    profiles = np.array(profiles)
    rows = np.arange(3 * radius, h - 3 * radius)
    above = profiles[:, rows-radius]
    below = profiles[:, rows+radius]
    jump = below - above
    # Subtract the ordinary local lighting slope. A sustained material change
    # has a stronger central jump than either neighbouring interval.
    previous = above - profiles[:, rows-3*radius]
    following = profiles[:, rows+3*radius] - below
    excess = np.max(np.abs(jump - (previous + following) / 2), axis=2)
    magnitude = np.max(np.abs(jump), axis=2)
    votes = (excess > 4) & (magnitude > 5)
    observed = np.array(observed)
    votes &= observed[:, rows-radius] & observed[:, rows+radius]
    agreement = votes.mean(axis=0)
    score = np.median(np.where(votes, excess, 0), axis=0)
    people = np.flatnonzero(protected.any(axis=1))
    # Exclude overhead wallpaper/rail transitions using this subject's extent,
    # rather than an assumed percentage of the photograph.
    eligible = agreement >= .6
    if len(people):
        eligible &= rows > np.percentile(np.nonzero(protected)[0], 10)
    candidates = np.flatnonzero(eligible & (score >= 4))
    if not len(candidates):
        return 1.0, result
    # First sustained transition below the subjects' heads; later transitions
    # belong to floor edges/tape and must not move the horizon to a preset row.
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > radius) + 1)
    group = groups[0]
    best = group[np.argmax(score[group])]
    row = int(rows[best])
    if len(people) and row > people[-1] + radius:
        return 1.0, result
    result.update(detected=True, confidence=float(agreement[best]), row=row,
                  contrast=float(np.median(magnitude[:, best])))
    return row / h, result


def local_finish(gray, safe):
    """Remove measured ridges while retaining broad illumination and grain."""
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    x = xx / max(1, w-1); y = yy / max(1, h-1)
    design = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)
    samples = np.flatnonzero(safe.ravel())[::8]
    if len(samples) < 50:
        return np.zeros_like(gray), 0.0, 0.0
    a = design.reshape(-1, 6)[samples]; b = gray.ravel()[samples]
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
    for _ in range(4):
        residual = a @ coefficients - b
        weights = np.sqrt(np.minimum(1, 3 / np.maximum(np.abs(residual), .01)))
        coefficients = np.linalg.lstsq(a * weights[:, None], b * weights, rcond=None)[0]
    illumination = design @ coefficients
    valid = safe.astype(np.float32)
    sigma = max(4, min(h, w) * .025)
    small_norm = cv2.GaussianBlur(valid, (0, 0), 1.2)
    low = illumination + cv2.GaussianBlur(((gray-illumination)*valid).astype(np.float32), (0, 0), 1.2) / np.maximum(small_norm, .001)
    residual = (low - illumination) * valid
    norm = cv2.GaussianBlur(valid, (0, 0), sigma)
    ridges = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), sigma) / np.maximum(norm, .001)
    # Illumination is smooth in both directions; folds have repeated local
    # changes. Measure their contrast rather than selecting a lower wall band.
    contrast = np.zeros_like(gray)
    for scale in (sigma, sigma*2.4, sigma*4.8):
        normal = cv2.GaussianBlur(valid, (0, 0), scale)
        mean = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), scale) / np.maximum(normal, .001)
        variance = cv2.GaussianBlur(residual.astype(np.float32)**2, (0, 0), scale) / np.maximum(normal, .001) - mean**2
        contrast = np.maximum(contrast, np.sqrt(np.maximum(0, variance)))
    confidence = np.clip((contrast - 2) / 3, 0, 1)
    confidence[~safe] = 0
    # Suppress isolated detection noise; corrections stay inside the measured
    # defect support. Opaque foreground and all floor pixels remain excluded.
    strength = confidence * np.clip(cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_L2, 3) / 2, 0, 1)
    delta = np.clip(-residual * .95, -60, 60) * strength
    delta[np.abs(delta) < .75] = 0
    return delta.astype(np.float32), float(confidence.max()), float(contrast[safe].max()) if safe.any() else 0.0
