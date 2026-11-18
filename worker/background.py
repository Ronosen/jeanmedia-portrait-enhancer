"""Conservative backdrop repair using a CPU portrait mask."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL = Path(__file__).with_name("models") / "human_segmentation_pphumanseg_2023mar.onnx"


def person_probability(rgb):
    from runtime import session as model_session

    session = model_session(MODEL)
    pixels = cv2.resize(rgb, (192, 192)).astype(np.float32) / 127.5 - 1
    scores = session.run(None, {session.get_inputs()[0].name: pixels.transpose(2, 0, 1)[None]})[0]
    if scores.shape != (1, 2, 192, 192) or not np.isfinite(scores).all():
        raise ValueError("Unexpected portrait mask output.")
    return cv2.resize(scores[0, 1], (rgb.shape[1], rgb.shape[0]))


def repair_mask(rgb, probability):
    height, width = probability.shape
    if not np.any(probability > 0.9):
        return None, "Needs Review · A subject could not be confidently located."
    # Leave generous clearance around the person and uncertain mask boundaries.
    radius = max(10, round(min(height, width) * 0.04))
    protected = cv2.dilate(
        (probability > 0.05).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    ).astype(bool)
    # Restrict automatic inpainting to small isolated marks on a smooth, neutral
    # backdrop near the top of the frame. The floor and seat area remain protected.
    mask = np.zeros((height, width), np.uint8)
    if min(height, width) < 64:
        return mask, "Needs Review · Image is too small for backdrop repair."
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    base = cv2.medianBlur(gray, 15)
    deviation = cv2.absdiff(gray, base)
    candidates = (deviation >= 16).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not (3 <= area <= 72 and w <= 12 and h <= 12):
            continue
        if x < 12 or y < 12 or x + w + 12 >= width or y + h + 12 >= height * 0.38:
            continue
        region = np.s_[y - 12:y + h + 12, x - 12:x + w + 12]
        if protected[region].any():
            continue
        local = labels[region] == label
        outline = cv2.dilate(local.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        surroundings = rgb[region][~outline]
        if np.max(np.ptp(surroundings, axis=0)) > 9:
            continue
        if np.ptp(surroundings.mean(axis=0)) > 22:
            continue
        # Preserve small saturated objects even when surrounded by a neutral field.
        if np.max(np.ptp(rgb[region][local], axis=1)) > 25:
            continue
        mask[region][outline] = 255
    return mask, ""


def clean_backdrop(image, mask):
    if mask is None or not mask.any():
        return image
    rgb = np.array(image)
    scaled = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
    count, _, stats, _ = cv2.connectedComponentsWithStats(scaled)
    for x, y, w, h, _ in stats[1:count]:
        pad = max(4, round(4 * image.width / mask.shape[1]))
        region = np.s_[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        rgb[region] = cv2.inpaint(rgb[region], scaled[region], min(15, pad), cv2.INPAINT_TELEA)
    return Image.fromarray(rgb)


def edge_repair(image, region):
    """Extend a dark backdrop from the right of an approved left-edge selection."""
    x, top, bottom = region
    width, height = image.size
    edge, y0, y1 = round(x * width), round(top * height), round(bottom * height)
    feather = max(2, round(min(width, height) * 0.018))
    if edge < feather or y1 - y0 < 2 * feather:
        raise ValueError("Select a larger area around the equipment.")
    rgb = np.array(image)
    strip = rgb[y0:y1, edge:edge + feather]
    if np.percentile(strip.max(axis=2), 95) > 100 or np.percentile(np.ptp(strip, axis=2), 95) > 32:
        raise ValueError("The right edge must sit on clean, dark backdrop.")
    source = cv2.GaussianBlur(strip, (0, 0), max(1, feather / 5))
    columns = edge - 1 - np.arange(edge)
    reflected = columns % (2 * feather)
    reflected = np.minimum(reflected, 2 * feather - 1 - reflected)
    patch = source[:, reflected].astype(np.float32)
    vertical = np.ones(y1 - y0)
    if y0:
        vertical[:feather] = np.linspace(0, 1, feather)
    if y1 < height:
        vertical[-feather:] = np.linspace(1, 0, feather)
    horizontal = np.minimum(1, (edge - 1 - np.arange(edge)) / feather)
    blend = (vertical[:, None] * horizontal[None, :])[..., None]
    roi = rgb[y0:y1, :edge]
    roi[:] = np.rint(roi * (1 - blend) + patch * blend).astype(np.uint8)
    return Image.fromarray(rgb)


def automatic_edge_region(rgb, probability, compact=False):
    """Find a left-edge equipment candidate with long frame ribs on a dark set."""
    height, width = probability.shape
    if min(height, width) < 128 or not np.any(probability > 0.9):
        raise ValueError("Needs Review · A subject could not be confidently located.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Require several long, high contrast ribs extending into the image from the
    # left. Folds and a uniformly lit backdrop alone must not trigger removal.
    lines = cv2.HoughLinesP(cv2.Canny(gray, 35, 100), 1, np.pi / 180,
                           max(20, round(width * (.035 if compact else .06))),
                           minLineLength=max(25, round(width * (.045 if compact else .09))),
                           maxLineGap=max(3, round(width * .014)))
    if lines is None:
        raise ValueError("Needs Review · No clear edge equipment was found.")
    subject_columns = np.flatnonzero(np.any(probability[:round(height * .60)] > .05, axis=0))
    limit = width * .40
    if compact and subject_columns.size:
        limit = min(limit, int(subject_columns[0]) - width * .025)
    strong = [line[0] for line in lines
              if max(line[0][0], line[0][2]) < limit
              and max(line[0][1], line[0][3]) < height * .60]
    touching = [line for line in strong if min(line[0], line[2]) < width * .035]
    if len(strong) < 3 or len(touching) < 2:
        raise ValueError("Needs Review · No clear edge equipment was found.")
    feather = max(2, round(min(width, height) * .018))
    right = max(max(line[0], line[2]) for line in strong) + feather * 2
    if right > width * .45:
        raise ValueError("Needs Review · Equipment extends too far into the photograph.")
    # Fainter diagonal edges define the dark outer fabric below the bright ribs.
    faint = cv2.HoughLinesP(cv2.Canny(gray, 8, 24), 1, np.pi / 180,
                           max(24, round(width * .05)),
                           minLineLength=max(30, round(width * .08)),
                           maxLineGap=max(3, round(width * .024)))
    lower = max(max(line[1], line[3]) for line in strong)
    if faint is not None:
        for x0, y0, x1, y1 in faint[:, 0]:
            if max(x0, x1) >= right or max(y0, y1) >= height * .60 or abs(x1 - x0) < width * .06:
                continue
            slope = (y1 - y0) / (x1 - x0)
            if -2 < slope < -.4 and max(y0, y1) > lower * .8:
                intercept = y0 - slope * x0
                if intercept < height * .62:
                    lower = max(lower, intercept)
    bottom = int(np.ceil(lower + feather * 2))
    pad = round(min(width, height) * .025)
    occupied = np.flatnonzero(np.any(probability[:, :right + pad] > .05, axis=1))
    if occupied.size:
        bottom = min(bottom, int(occupied[0]) - pad)
    if bottom < lower + feather or bottom <= 2 * feather:
        raise ValueError("Needs Review · Equipment is too close to the subject.")
    # Validate the fill source before returning a removal area.
    region = (right / width, 0.0, min(bottom / height, 1.0))
    strip = rgb[:bottom, right:right + feather]
    if np.percentile(strip.max(axis=2), 95) > 100 or np.percentile(np.ptp(strip, axis=2), 95) > 32:
        raise ValueError("Needs Review · A clean dark backdrop could not be found beside the equipment.")
    return region

