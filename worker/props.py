"""Conservative surface healing on recognised supporting furniture only."""
import cv2
import numpy as np
from PIL import Image
from floor import encode_mask, decode_mask


def plan_props(rgb, person, objects):
    height, width = person.shape
    guard = cv2.dilate(person.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    safe = np.zeros(person.shape, bool)
    boxes = []
    for item in objects:
        if item["class"] not in (56, 57, 59, 60) or item["confidence"] < .5:
            continue
        x0, y0, x1, y1 = item["box"]
        x0, y0, x1, y1 = max(0, x0+14), max(0, y0+14), min(width, x1-14), min(height, y1-14)
        if x1 > x0 and y1 > y0:
            safe[y0:y1, x0:x1] = True
            boxes.append([x0/width, y0/height, x1/width, y1/height])
    safe &= ~guard
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 11))
    candidates = ((residual > 20) & safe).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    mask = np.zeros_like(gray)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (3 <= area <= 48 and w <= 12 and h <= 12 and x >= 8 and y >= 8
                and x+w+8 < width and y+h+8 < height):
            continue
        roi = np.s_[y-8:y+h+8, x-8:x+w+8]
        core = labels[roi] == index
        expanded = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if not safe[roi].all() or rgb[roi][~expanded].std(axis=0).max() > 6:
            continue
        mask[roi][expanded] = 255
    return {"version": 1, "shape": [height, width], "mask": encode_mask(mask), "boxes": boxes,
            "count": cv2.connectedComponents((mask > 0).astype(np.uint8))[0]-1}


def validate_props(plan, rgb, person, objects):
    if plan.get("version") != 1 or plan.get("shape") != list(person.shape):
        raise ValueError("Invalid prop repair settings.")
    mask = decode_mask(plan.get("mask"), person.shape)
    # Re-derive safe repairs: a changed source or altered plan cannot paint a person.
    safe = decode_mask(plan_props(rgb, person, objects)["mask"], person.shape)
    if np.any((mask > 0) & (safe == 0)):
        raise ValueError("A prop repair overlaps an unapproved surface.")


def heal_props(image, plan):
    shape = plan.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(not isinstance(x, int) or not 1 <= x <= 1280 for x in shape):
        raise ValueError("Invalid prop repair dimensions.")
    mask = cv2.resize(decode_mask(plan.get("mask"), tuple(shape)), image.size, interpolation=cv2.INTER_NEAREST)
    output = np.asarray(image).copy()
    if mask.any():
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        pad = max(8, round(image.width / shape[1] * 8))
        for x, y, w, h, _ in stats[1:]:
            roi = np.s_[max(0, y-pad):min(image.height, y+h+pad), max(0, x-pad):min(image.width, x+w+pad)]
            fixed = cv2.inpaint(output[roi], mask[roi], min(12, pad), cv2.INPAINT_TELEA)
            output[roi][mask[roi] > 0] = fixed[mask[roi] > 0]
    return Image.fromarray(output)
