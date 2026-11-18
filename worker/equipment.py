"""Detect isolated studio lights at either frame edge and repair their backdrop."""

import cv2
import numpy as np
from PIL import Image

from background import automatic_edge_region


def polygon_mask(shape, polygon):
    height, width = shape
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 16:
        raise ValueError("Invalid equipment outline.")
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        raise ValueError("Invalid equipment coordinates.")
    mask = np.zeros((height, width), np.uint8)
    coordinates = np.rint(points * (width - 1, height - 1)).astype(np.int32)
    cv2.fillPoly(mask, [coordinates], 255)
    return mask


def donor_colors(rgb, mask, protected, side):
    height, width = mask.shape
    span = max(6, round(min(height, width) * .018))
    colors = np.zeros((height, 3), np.float32)
    good = np.zeros(height, bool)
    rows = np.flatnonzero(mask.any(axis=1))
    for y in rows:
        columns = np.flatnonzero(mask[y])
        if side == "left":
            start = int(columns[-1]) + 2
            end = min(width, start + span)
        else:
            end = int(columns[0]) - 1
            start = max(0, end - span)
        if end - start < span // 2 or protected[y, start:end].any():
            continue
        samples = rgb[y, start:end].astype(np.float32)
        if samples.std(axis=0).max() > 16:
            continue
        colors[y] = np.median(samples, axis=0)
        good[y] = True
    if not len(rows) or good[rows].mean() < .30 or np.count_nonzero(good) < 12:
        raise ValueError("Needs Review · Equipment is too close to the subject or a textured background.")
    known = np.flatnonzero(good)
    for channel in range(3):
        colors[:, channel] = np.interp(np.arange(height), known, colors[known, channel])
    colors = cv2.GaussianBlur(colors, (1, 0), 0, sigmaY=max(1, min(height, width) * .004))
    if np.max(np.percentile(colors[rows], 95, axis=0) - np.percentile(colors[rows], 5, axis=0)) > 100:
        raise ValueError("Needs Review · The surrounding backdrop is too uneven for automatic repair.")
    return colors


def validate_plan(rgb, probability, plan):
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("Unsupported equipment edit settings.")
    repairs = plan.get("repairs")
    if not isinstance(repairs, list) or not 1 <= len(repairs) <= 4:
        raise ValueError("Invalid equipment edit settings.")
    if not np.any(probability > .9):
        raise ValueError("Needs Review · A subject could not be confidently located.")
    height, width = probability.shape
    clearance = max(5, round(min(height, width) * .008))
    protected = cv2.dilate((probability > .05).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (2 * clearance + 1, 2 * clearance + 1)))
    result = []
    occupied = np.zeros((height, width), np.uint8)
    for repair in repairs:
        side = repair.get("side")
        if side not in ("left", "right"):
            raise ValueError("Invalid equipment edge.")
        mask = polygon_mask((height, width), repair.get("polygon"))
        if np.any(mask & protected) or np.any(mask & occupied):
            raise ValueError("Needs Review · The detected equipment overlaps a protected area.")
        columns = np.flatnonzero(mask.any(axis=0))
        if not len(columns) or len(columns) > width * .48:
            raise ValueError("Needs Review · Equipment extends too far into the photo.")
        if (side == "left" and columns[0] != 0) or (side == "right" and columns[-1] != width - 1):
            raise ValueError("Equipment must meet the frame edge.")
        colors = donor_colors(rgb, mask, protected, side)
        occupied |= mask
        result.append((mask, colors))
    return result


def detect_equipment(rgb, probability):
    height, width = probability.shape
    if not np.any(probability > .9):
        raise ValueError("Needs Review · A subject could not be confidently located.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    neutral = np.ptp(rgb, axis=2) < 65
    _, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats((gray < 90).astype(np.uint8))
    repairs = []
    for bright in (True, False):
        binary = ((gray > 190) & neutral) if bright else (gray < 100)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if not (.002 < area / (width * height) < .22 and .06 < w / width < .40
                    and .08 < h / height < .70):
                continue
            if x <= width * .02:
                side = "left"
            elif x + w >= width * .98:
                side = "right"
            else:
                continue
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(max(contours, key=cv2.contourArea))
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0 or area / hull_area < .76:
                continue
            approx = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * .018, True)
            if not 4 <= len(approx) <= 8:
                continue
            core_values = gray[component.astype(bool)]
            if bright and np.median(core_values) < 220:
                continue
            if not bright and y < height * .08:
                continue
            # Reject white shirts and bright backdrop regions. Studio lights
            # need a dark frame immediately outside their bright polygon.
            hull_mask = np.zeros_like(component)
            cv2.fillPoly(hull_mask, [hull], 1)
            rim_size = max(5, round(min(height, width) * .014))
            ring = cv2.dilate(hull_mask, np.ones((2 * rim_size + 1,) * 2, np.uint8)) - hull_mask
            ring_values = gray[ring.astype(bool)]
            if bright and (not len(ring_values) or np.mean(ring_values < 85) < .30):
                continue
            padding = max(5, round(min(height, width) * .05))
            if bright:
                nearby = cv2.dilate(hull_mask, np.ones((11, 11), np.uint8))
                merged = hull_mask.copy()
                reach = round(min(height, width) * .065)
                bounds = np.zeros_like(component)
                bounds[max(0, y-reach):y+h+reach, max(0, x-reach):x+w+reach] = 1
                for dark_label in np.unique(dark_labels[nearby.astype(bool)]):
                    if not dark_label or dark_stats[dark_label, cv2.CC_STAT_AREA] > width * height * .20:
                        continue
                    merged |= ((dark_labels == dark_label) & bounds.astype(bool)).astype(np.uint8)
                contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1 and np.any(merged != hull_mask):
                    hull_mask[:] = 0
                    cv2.fillPoly(hull_mask, [cv2.convexHull(contours[0])], 1)
                    padding = max(4, round(min(height, width) * .012))
            padded = cv2.dilate(hull_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * padding + 1, 2 * padding + 1)))
            contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = cv2.approxPolyDP(cv2.convexHull(contours[0]),
                                      cv2.arcLength(contours[0], True) * .005, True)[:, 0]
            if len(outline) > 16:
                continue
            if side == "left":
                outline[outline[:, 0] < width * .02, 0] = 0
            else:
                outline[outline[:, 0] > width * .98, 0] = width - 1
            polygon = (outline / (width - 1, height - 1)).tolist()
            repair = {"side": side, "polygon": polygon}
            try:
                validated = validate_plan(rgb, probability, {"version": 1, "repairs": [repair]})
            except ValueError:
                continue
            mask, colors = validated[0]
            difference = np.abs(rgb[component.astype(bool)].astype(float)
                                - np.median(colors[np.any(mask, axis=1)], axis=0)).mean()
            if difference < 45:
                continue
            if any(np.any(mask & polygon_mask((height, width), r["polygon"])) for r in repairs):
                continue
            repairs.append(repair)
            if len(repairs) == 4:
                break
    if not repairs:
        # Dark fabric can conceal a softbox face. Fall back to its visible frame.
        for compact in (False, True):
            try:
                edge, top, bottom = automatic_edge_region(rgb, probability, compact=compact)
                repair = {"side": "left",
                          "polygon": [[0.0, top], [edge, top], [edge, bottom], [0.0, bottom]]}
                validate_plan(rgb, probability, {"version": 1, "repairs": [repair]})
                repairs.append(repair)
                break
            except ValueError:
                continue
    if not repairs:
        raise ValueError("Needs Review · No isolated studio equipment could be safely removed.")
    plan = {"version": 1, "repairs": repairs}
    validate_plan(rgb, probability, plan)
    return plan


def apply_equipment_plan(image, preview_rgb, probability, plan):
    edits = validate_plan(preview_rgb, probability, plan)
    result = image.copy()
    for mask, colors in edits:
        source = preview_rgb.copy()
        replace = (probability > .05) | (mask > 0)
        source[replace] = np.broadcast_to(
            colors[:, None, :], source.shape)[replace].astype(np.uint8)
        fill = cv2.inpaint(source, mask, 5, cv2.INPAINT_TELEA)
        fill = cv2.GaussianBlur(fill, (0, 0), 1.2)
        enlarged = Image.fromarray(fill).resize(image.size, Image.Resampling.LANCZOS)
        rows = np.flatnonzero(mask.any(axis=1))
        for top in range(round(rows[0] * image.height / mask.shape[0]),
                         min(image.height, round((rows[-1] + 1) * image.height / mask.shape[0])), 256):
            end = min(image.height, top + 256)
            coordinates = np.minimum(mask.shape[0] - 1,
                                     (np.arange(top, end) * mask.shape[0] / image.height).astype(int))
            tile_mask = cv2.resize(mask[coordinates], (image.width, end - top),
                                  interpolation=cv2.INTER_NEAREST)
            # Feather within the detected area so edits cannot spill outside it.
            distance = cv2.distanceTransform(tile_mask, cv2.DIST_L2, 3)
            alpha = np.minimum(1, distance / max(2, image.width * .003))[..., None]
            patch = np.asarray(enlarged.crop((0, top, image.width, end))).astype(np.float32)
            tile = np.asarray(image.crop((0, top, image.width, end))).astype(np.float32)
            rendered = np.rint(np.clip(tile * (1 - alpha) + patch * alpha, 0, 255)).astype(np.uint8)
            result.paste(Image.fromarray(rendered), (0, top))
    return result

