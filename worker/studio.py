"""Reconstruct a plain studio backdrop with protected portrait pixels."""

import cv2
import numpy as np
from PIL import Image


def background_basis(x, y):
    return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)


def floor_boundary(rgb, protected):
    from scene_evidence import floor_evidence
    return floor_evidence(rgb, protected)[0]


def fit_backdrop(rgb, protected, floor):
    height, width = protected.shape
    yy, xx = np.mgrid[:height, :width]
    allowed = ~protected.copy()
    allowed[round(floor * height):] = False
    allowed[:round(height * .06)] = False
    allowed[:, :round(width * .08)] = False
    allowed[:, round(width * .92):] = False
    coordinates = np.argwhere(allowed)[::8]
    if len(coordinates) < 500:
        raise ValueError("Needs Review · Too little unobstructed backdrop is visible.")
    colors = rgb[coordinates[:, 0], coordinates[:, 1]].astype(np.float32)
    bins = np.floor(colors / 24).astype(int)
    ids = bins[:, 0] * 121 + bins[:, 1] * 11 + bins[:, 2]
    winner = np.bincount(ids).argmax()
    center = np.median(colors[ids == winner], axis=0)
    # Use a dominant colour family to exclude equipment and exposed room edges.
    chosen = np.max(np.abs(colors - center), axis=1) < 38
    if chosen.mean() < .32:
        raise ValueError("Needs Review · A plain, consistent studio backdrop could not be identified.")
    points = coordinates[chosen]
    values = colors[chosen]
    if np.ptp(points[:, 1]) < width * .35 or np.ptp(points[:, 0]) < height * .25:
        raise ValueError("Needs Review · The visible backdrop is too small to extend reliably.")
    design = background_basis(points[:, 1] / max(1, width-1),
                              points[:, 0] / max(1, height-1))
    weights = np.ones(len(points))
    for _ in range(4):
        weighted = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(design * weighted, values * weighted, rcond=None)[0]
        residual = np.max(np.abs(design @ coefficients - values), axis=1)
        weights = np.minimum(1, 12 / np.maximum(residual, 1))
    low = np.maximum(0, np.percentile(values, 3, axis=0) - 10)
    high = np.minimum(255, np.percentile(values, 97, axis=0) + 10)
    model = background_basis(xx / max(1, width-1), yy / max(1, height-1)) @ coefficients
    model = np.clip(model, low, high)
    return coefficients, low, high, model


def initial_protection(probability, objects, foreground_probability=None):
    if not np.any(probability > .9):
        raise ValueError("Needs Review · A person could not be confidently located.")
    if foreground_probability is not None:
        if foreground_probability.shape != probability.shape or not np.isfinite(foreground_probability).all():
            raise ValueError("Invalid detailed foreground mask.")
        if np.mean(foreground_probability[probability > .9] > .03) < .95:
            raise ValueError("Needs Review · The models disagree on the person boundary. Cleanup was skipped.")
    height, width = probability.shape
    radius = max(4, round(min(height, width) * (.010 if foreground_probability is not None else .015)))
    threshold = .5 if foreground_probability is not None else .025
    candidate = probability > threshold
    if foreground_probability is not None:
        candidate &= foreground_probability > .03
    _, components = cv2.connectedComponents(candidate.astype(np.uint8))
    seeded = np.unique(components[probability > .9])
    foreground = np.isin(components, seeded[seeded > 0]).astype(np.uint8)
    if foreground_probability is not None:
        _, salient_components = cv2.connectedComponents((foreground_probability > .35).astype(np.uint8))
        anchored = np.unique(salient_components[probability > .9])
        foreground |= np.isin(salient_components, anchored[anchored > 0]).astype(np.uint8)
    person = cv2.dilate(foreground,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
    protected = person.copy()
    contact = cv2.dilate(person, np.ones((2*radius+1, 2*radius+1), np.uint8))
    kept = []
    # COCO chairs, couches, plants, beds, tables and vases are intentional props.
    always_keep = {56, 57, 58, 59, 60, 75}
    for item in objects:
        if item["class"] == 0:
            continue
        x0, y0, x1, y1 = item["box"]
        if item["class"] in always_keep or contact[y0:y1, x0:x1].any():
            x0, y0 = max(0, x0-radius), max(0, y0-radius)
            x1, y1 = min(width, x1+radius), min(height, y1+radius)
            protected[y0:y1, x0:x1] = 1
            kept.append({"class": item["class"], "confidence": item["confidence"]})
    return person.astype(bool), protected.astype(bool), kept


def support_shapes(rgb, person, protected, model, floor):
    """Preserve broad support surfaces while separating thin wall-edge strips."""
    height, width = person.shape
    threshold = np.where(model.mean(axis=2) > 140, 25, 65)
    contrast = np.max(np.abs(rgb.astype(float) - model), axis=2) > threshold
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 10, 30), np.ones((3, 3), np.uint8))
    neutral = np.ptp(rgb, axis=2) < 45
    contrast &= neutral & (edges == 0)
    contrast[person] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8))
    rows = np.flatnonzero(person.any(axis=1))
    below = rows[0] + .4 * (rows[-1] - rows[0])
    padding = max(5, round(min(height, width) * .025))
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if x == 0 or x+w == width:
            continue
        if (below < y < rows[-1] + height * .08 and y+h < height * .98
                and area > width * height * .005 and w > width * .08
                and h > height * .06 and area / (w*h) > .55):
            candidates.append(label)
    contact = cv2.dilate(protected.astype(np.uint8), np.ones((45, 45), np.uint8)).astype(bool)
    linked = set(np.unique(labels[contact]))
    found_support = False
    confirmed = np.zeros_like(person)
    for label in candidates:
        if label not in linked:
            continue
        x, y, w, h, _ = stats[label]
        reference = np.median(rgb[labels == label], axis=0)
        if y >= floor * height and reference.mean() < 150:
            continue
        found_support = True
        surface = (labels == label).astype(np.uint8)
        accepted = {label}
        for _ in range(3):
            neighbours = cv2.dilate(surface, np.ones((25, 25), np.uint8)).astype(bool)
            for neighbour in np.unique(labels[neighbours]):
                if not neighbour or neighbour in accepted:
                    continue
                nx, ny, nw, nh, area = stats[neighbour]
                if (nx == 0 or nx+nw >= width or ny < y-height*.12
                        or ny+nh > y+h+height*.025 or area < width*height*.0001):
                    continue
                part = labels == neighbour
                if np.max(np.abs(np.median(rgb[part], axis=0) - reference)) < 85:
                    surface[part] = 1
                    accepted.add(int(neighbour))
        contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = cv2.convexHull(np.concatenate(contours))
        solid = np.zeros_like(surface)
        cv2.fillPoly(solid, [outline], 1)
        # Preserve the complete planar support, including its straight edges.
        # Edge detection deliberately removed those pixels from the colour
        # components; preserving only the components cut notches into seats.
        confirmed |= cv2.dilate(solid,np.ones((3,3),np.uint8)).astype(bool)
        solid = cv2.dilate(solid, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*padding+1, 2*padding+1)))
        protected |= solid.astype(bool)
    return protected, found_support, confirmed


def contact_protection(rgb, person, protected, model, floor):
    # Recover unrecognised seats and boxes by retaining contrasting components
    # joined to the subject. This also conservatively retains ambiguous contact.
    height, width = person.shape
    foreground = (np.max(np.abs(rgb.astype(float) - model), axis=2) > 30).astype(np.uint8)
    foreground[round(floor * height):] = 0
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    foreground[person] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    radius = max(11, round(min(height, width) * .03))
    close = cv2.dilate(person.astype(np.uint8), np.ones((radius, radius), np.uint8)).astype(bool)
    linked = np.unique(labels[close])
    person_rows = np.flatnonzero(person.any(axis=1))
    support_top = person_rows[0] + (person_rows[-1] - person_rows[0]) * .45
    for label in linked:
        x, y, w, h, area = stats[label]
        if not label:
            continue
        component = labels == label
        lower_area = np.count_nonzero(component[round(support_top):])
        # A seat can share a connected component with a cuff, dangling hair or
        # bright trim above the hip. Its top edge is not a safe rejection test.
        if (lower_area > width * height * .003 and lower_area > area * .55
                and area / max(1, w * h) > .20):
            contours, _ = cv2.findContours(component.astype(np.uint8),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solid = np.zeros_like(foreground)
            cv2.drawContours(solid, contours, -1, 1, cv2.FILLED)
            protected |= solid.astype(bool)
    return cv2.dilate(protected.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def refine_edges(rgb, protected, probability, foreground):
    radius = max(12, round(min(protected.shape) * .04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    core = cv2.erode(protected.astype(np.uint8), kernel).astype(bool)
    core |= protected & ((foreground > .75) | ((probability > .97) & (foreground > .3)))
    loose = cv2.dilate(protected.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    trimap = np.full(protected.shape, cv2.GC_BGD, np.uint8)
    trimap[loose] = cv2.GC_PR_BGD
    trimap[protected] = cv2.GC_PR_FGD
    trimap[core] = cv2.GC_FGD
    if not core.any() or not np.any(trimap == cv2.GC_BGD):
        raise ValueError("Needs Review · The foreground boundary could not be refined.")
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(rgb), trimap, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                2, cv2.GC_INIT_WITH_MASK)
    refined = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
    if np.count_nonzero(refined) < np.count_nonzero(protected) * .65:
        raise ValueError("Needs Review · The foreground boundary is too uncertain.")
    return cv2.dilate(refined.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def prepare_backdrop(rgb, probability, objects, clean_floor=False, plan=None,
                     foreground_probability=None, floor_mode=None, equipment_plan=None, floor_probability=None):
    from runtime import progress
    progress("Protecting subject", 45)
    detailed = foreground_probability is not None if plan is None else plan.get("version") in (2, 3)
    if detailed and foreground_probability is None:
        raise ValueError("Needs Review · Detailed foreground protection is unavailable.")
    person, protected, kept = initial_protection(probability, objects,
                                                foreground_probability if detailed else None)
    from scene_evidence import floor_evidence
    floor_witness=(floor_probability if floor_probability is not None else probability)>.5
    floor, floor_analysis = floor_evidence(rgb, floor_witness)
    if plan is None:
        coefficients, low, high, model = fit_backdrop(rgb, protected, floor)
        plan = {"version": 3 if detailed else 1, "coefficients": coefficients.tolist(),
                "low": low.tolist(), "high": high.tolist(),
                "floor": float(floor), "floorAnalysis": floor_analysis, "cleanFloor": bool(clean_floor)}
        if floor_mode is not None:
            if floor_mode not in ("preserve", "patch", "replace"):
                raise ValueError("Invalid floor treatment.")
            plan["floorMode"] = floor_mode
            plan["cleanFloor"] = floor_mode == "replace"
            plan["selectiveBackdrop"] = True
            plan["smartProps"] = True
            from neural import available
            plan["repairEngine"] = "lama-v1" if available() else "context-v1"
            plan["equipmentPlan"] = equipment_plan
            plan["adaptiveRevision"] = 3
    else:
        coefficients = np.asarray(plan.get("coefficients"), dtype=float)
        low, high = np.asarray(plan.get("low"), dtype=float), np.asarray(plan.get("high"), dtype=float)
        floor = plan.get("floor")
        if (plan.get("version") not in (1, 2, 3) or coefficients.shape != (6, 3)
                or low.shape != (3,) or high.shape != (3,)
                or not np.isfinite(coefficients).all() or not np.isfinite(low).all()
                or not np.isfinite(high).all() or np.any(low < 0) or np.any(high > 255)
                or np.any(low > high) or not isinstance(floor, (int, float))
                or not 0 < floor <= 1.0 or not isinstance(plan.get("cleanFloor"), bool)):
            raise ValueError("Invalid backdrop edit settings.")
        height, width = probability.shape
        yy, xx = np.mgrid[:height, :width]
        model = np.clip(background_basis(xx / max(1, width-1),
                                        yy / max(1, height-1)) @ coefficients, low, high)
    confirmed_support=np.zeros_like(protected)
    if detailed:
        protected, found_support, confirmed_support = support_shapes(rgb, person, protected, model, floor)
        if plan.get('adaptiveRevision'):
            from support_geometry import box_seats
            recovered=box_seats(rgb,person)
            protected |= recovered
            confirmed_support |= recovered
            found_support = found_support or recovered.any()
        known_seat = any(item["class"] in {56, 57, 59, 60} for item in kept)
        for item in objects:
            if item["class"] != 0 or item["confidence"] < .4:
                continue
            x0, y0, x1, y1 = item["box"]
            seated = (x1-x0) > (y1-y0) * .5 and y1 < protected.shape[0] * .95
            # Cropped group portraits can have a wide person box without a
            # visible seat. Preserve connected edge furniture in that case.
            if seated and not (known_seat or found_support):
                if not plan.get('adaptiveRevision'):
                    raise ValueError("Needs Review · The supporting seat could not be protected. Cleanup was skipped.")
                # The detailed source matte protects visible people/supports.
                # A wide person box alone is not evidence that an entire band
                # of backdrop, balloons and floor belongs to a hidden seat.
                plan['supportUncertain']=True
                plan.setdefault('review',[]).append('Supporting seat needs review.')
    else:
        protected = contact_protection(rgb, person, protected, model, floor)
    if plan["version"] == 3:
        protected = refine_edges(rgb, protected, probability, foreground_probability)
        protected |= confirmed_support
        if plan.get('adaptiveRevision'):
            # Fine body details missed by GrabCut must not enter a repair.
            guard=(probability>.10)&(foreground_probability>.10)
            protected |= cv2.dilate(guard.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    panels=np.zeros(protected.shape,bool)
    if plan.get("repairEngine") in ("context-v1", "lama-v1"):
        from inpainting import frame_panels
        panels = frame_panels(rgb, probability, model, objects, floor)
        panels &= ~confirmed_support
        protected[panels] = False
    mode = plan.get("floorMode", "replace" if plan["cleanFloor"] else "preserve")
    equipment_mask = np.zeros(protected.shape, np.uint8)
    if plan.get("repairEngine") in ("context-v1", "lama-v1") and plan.get("equipmentPlan"):
        from equipment import polygon_mask
        for item in plan["equipmentPlan"]["repairs"]:
            equipment_mask |= polygon_mask(protected.shape, item["polygon"])
        if np.any((equipment_mask > 0) & protected):
            raise ValueError("Needs Review · The equipment overlaps a protected subject or prop.")
    repair_protection = protected.copy()
    if mode not in ("preserve", "patch", "replace") or (mode == "replace") != plan["cleanFloor"]:
        raise ValueError("Invalid floor treatment.")
    if mode == "patch":
        from floor import plan_floor_patches, validate_floor_patches
        if "floorRepairs" not in plan:
            plan["floorRepairs"] = plan_floor_patches(rgb, protected | (equipment_mask > 0), floor,
                                                     context=plan.get("repairEngine") in ("context-v1", "lama-v1"))
        validate_floor_patches(protected, floor, plan["floorRepairs"])
    if plan.get("smartProps"):
        from props import plan_props, validate_props
        if "propRepairs" not in plan:
            plan["propRepairs"] = plan_props(rgb, person, objects)
        validate_props(plan["propRepairs"], rgb, person, objects)
    floor_row = round(floor * protected.shape[0])
    support_columns = np.flatnonzero(protected[:floor_row].any(axis=0))
    person_rows = np.flatnonzero(person.any(axis=1))
    support_top = round(person_rows[0] + (person_rows[-1] - person_rows[0]) * .45)
    # Dark legs, shoes and unknown seats can be missed by both small models.
    # Treat the lower-body band as uncertain foreground, not safe backdrop.
    if not detailed:
        protected[support_top:floor_row] = True
    if not plan["cleanFloor"]:
        protected[floor_row:] = True
    elif not detailed:
        # Unknown supporting boxes may continue below the visible backdrop.
        # Keep a contact corridor and its shadows beneath the people.
        margin = round(protected.shape[1] * .10)
        protected[floor_row:,
                  max(0, support_columns[0]-margin):min(protected.shape[1], support_columns[-1]+margin+1)] = True
    # Every protected pixel has exactly zero edit strength.
    distance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, 3)
    feather = 2 if plan["version"] == 3 else max(4, round(min(protected.shape) * (.008 if detailed else .03)))
    alpha = np.minimum(1, distance / feather).astype(np.float32)
    if np.count_nonzero(alpha > .5) < alpha.size * .05:
        raise ValueError("Needs Review · There is too little safe background to clean.")
    if plan.get("repairEngine") in ("context-v1", "lama-v1"):
        from inpainting import removal_mask
        selected = removal_mask(rgb, repair_protection, model, floor, equipment_mask | (panels.astype(np.uint8)*255))
        if plan.get('partialCleanup'):
            # In an uncertain seated scene, illumination differences are not
            # evidence of a missing wall. Limit reconstruction to equipment
            # actually detected above the protected band and isolated marks.
            n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8))
            allowed=np.zeros_like(selected)
            for index in range(1,n):
                part=labels==index
                if stats[index,cv2.CC_STAT_AREA]<selected.size*.002 or np.any(part&(equipment_mask>0)):
                    allowed |= part
            yy,xx=np.mgrid[:selected.shape[0],:selected.shape[1]]
            first=np.flatnonzero(person.any(axis=1))[0]
            allowed |= selected&((yy<first-selected.shape[0]*.03)|(xx<selected.shape[1]*.04)|(xx>selected.shape[1]*.97))
            selected=allowed
        alpha = selected.astype(np.float32)
        # This is the actual protection layer, not a blanket floor exclusion.
        protected = repair_protection
        plan["backdropAction"] = "inpaint"
    elif plan.get("selectiveBackdrop"):
        # Keep already-consistent backdrop pixels; feather only detected defects.
        deviation = np.max(np.abs(rgb.astype(float) - model), axis=2)
        damaged = ((deviation > 12) & (alpha > 0)).astype(np.uint8)
        damaged_fraction = np.count_nonzero(damaged) / max(1, np.count_nonzero(alpha > 0))
        if damaged_fraction < .08:
            damaged = cv2.dilate(damaged, np.ones((7, 7), np.uint8))
            local = cv2.GaussianBlur(damaged.astype(np.float32), (0, 0), 2)
            alpha *= np.clip(local, 0, 1)
            plan["backdropAction"] = "patch"
        else:
            # Widespread wrinkles need a coherent field, not disconnected islands
            # of fitted colour with visible boundaries between them.
            plan["backdropAction"] = "smooth"
    # Inspectable preview-scale layers; full-size rendering uses this same mask.
    import base64
    def encoded(mask):
        return base64.b64encode(cv2.imencode(".png", mask)[1]).decode("ascii")
    plan["layers"] = {"shape": list(protected.shape),
                      "person": encoded(person.astype(np.uint8)*255),
                      "floorWitness": encoded(floor_witness.astype(np.uint8)*255),
                      "source": encoded(rgb),
                      "support": encoded(confirmed_support.astype(np.uint8)*255),
                      "protected": encoded(protected.astype(np.uint8)*255),
                      "backdrop": encoded(np.rint(alpha*255).astype(np.uint8))}
    return plan, alpha, kept


def render_backdrop(image, alpha, plan):
    if plan.get('separation'):
        from separation import render
        return render(image,alpha,plan)
    if plan.get("repairEngine") in ("context-v1", "lama-v1"):
        from inpainting import repair
        import base64
        protected = cv2.imdecode(np.frombuffer(base64.b64decode(plan["layers"]["protected"]),np.uint8),cv2.IMREAD_GRAYSCALE)>0
        if plan["repairEngine"] == "lama-v1":
            from neural import repair as neural_repair
            from floor import decode_mask, encode_mask
            combined=alpha>0
            floor_repairs=plan.get("floorRepairs")
            if floor_repairs:
                combined |= decode_mask(floor_repairs["extension"],alpha.shape)>0
            result=neural_repair(image,combined,protected,plan)
            if floor_repairs:
                # Extension was reconstructed in the same scene-aware pass.
                remaining=dict(floor_repairs)
                if not plan.get('floorFillCloned'):
                    remaining["extension"]=encode_mask(np.zeros(alpha.shape,np.uint8))
                from floor import apply_floor_patches
                result=apply_floor_patches(result,plan["floor"],remaining,protected)
            if plan.get("propRepairs"):
                from props import heal_props
                result=heal_props(result,plan["propRepairs"])
            from backdrop_finish import finish
            result=finish(result,plan)
            return result
        if np.mean(alpha>0)>.04:
            raise ValueError("Needs Review · Large object removal needs the local reconstruction model.")
        if plan.get("floorRepairs"):
            from floor import decode_mask
            if np.mean(decode_mask(plan["floorRepairs"]["extension"],alpha.shape)>0)>.04:
                raise ValueError("Needs Review · Floor reconstruction needs the local reconstruction model.")
        result = repair(image, alpha, protected, plan["floor"], plan)
        if plan.get("floorMode") == "patch":
            from floor import apply_floor_patches
            result = apply_floor_patches(result, plan["floor"], plan["floorRepairs"], protected)
        if plan.get("propRepairs"):
            from props import heal_props
            result = heal_props(result, plan["propRepairs"])
        return result
    height, width = image.height, image.width
    coefficients = np.asarray(plan["coefficients"], dtype=float)
    low, high = np.asarray(plan["low"]), np.asarray(plan["high"])
    result = image.copy()
    strength = Image.fromarray(alpha).resize(image.size, Image.Resampling.BILINEAR)
    # Preserve the hard protected boundary after upsampling the soft blend.
    hard = Image.fromarray((alpha > 0).astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    x = np.arange(width, dtype=float) / max(1, width-1)
    for top in range(0, height, 256):
        end = min(height, top + 256)
        xx, yy = np.meshgrid(x, np.arange(top, end) / max(1, height-1))
        fill = np.clip(background_basis(xx, yy) @ coefficients, low, high)
        box = (0, top, width, end)
        amount = np.asarray(strength.crop(box)).copy()
        amount[np.asarray(hard.crop(box)) == 0] = 0
        original = np.asarray(image.crop(box)).astype(np.float32)
        output = original * (1 - amount[..., None]) + fill * amount[..., None]
        result.paste(Image.fromarray(np.rint(np.clip(output, 0, 255)).astype(np.uint8)), (0, top))
    if plan.get("floorMode") == "patch":
        from floor import apply_floor_patches
        result = apply_floor_patches(result, plan["floor"], plan["floorRepairs"])
    if plan.get("propRepairs"):
        from props import heal_props
        result = heal_props(result, plan["propRepairs"])
    return result


def clean_studio(image, preview, plan=None, clean_floor=False, export=False, floor_mode=None):
    from background import person_probability
    from equipment import apply_equipment_plan, detect_equipment
    from objects import detect_objects
    from foreground import foreground_probability
    from runtime import progress

    if plan and plan.get("runtime", {}).get("models"):
        from runtime import manifest
        current_models=manifest()["models"]
        if any(current_models.get(name)!=digest for name,digest in plan["runtime"]["models"].items()):
            raise ValueError("Local models changed. Run Auto Edit again before replaying this project.")

    if preview.mode != "RGB":
        raise ValueError("Studio backdrop cleanup needs an opaque photograph.")
    original_image = image
    original_preview = preview
    if plan and plan.get('separation'):
        from separation import replay
        return replay(image,preview,plan,export)
    if plan and plan.get("repairEngine")=="preserve-v1":
        if plan.get("layers",{}).get("shape")!=[preview.height,preview.width]:
            raise ValueError("Saved scene dimensions changed.")
        return image,preview,plan
    rgb = np.asarray(preview)
    progress("Finding people", 15)
    probability = person_probability(rgb)
    floor_probability=probability.copy()
    progress("Refining foreground", 28)
    detailed_foreground = foreground_probability(rgb) if plan is None or plan.get("version") in (2, 3) else None
    if plan is None:
        try:
            equipment = detect_equipment(rgb, probability)
        except ValueError:
            equipment = None
    else:
        equipment = plan.get("equipmentPlan")
    use_context = plan is None or plan.get("repairEngine") in ("context-v1", "lama-v1")
    if equipment and not use_context:
        preview = apply_equipment_plan(preview, rgb, probability, equipment)
    pixels = np.asarray(preview)
    progress("Finding props", 38)
    objects = detect_objects(pixels)
    separated=None
    raw_alpha=None
    from matting import MODEL,foreground_alpha
    if (plan is None and MODEL.is_file()) or (plan and plan.get('separation')):
        from separation import anchored_alpha
        raw_alpha=foreground_alpha(pixels)
        separated=anchored_alpha(raw_alpha,probability,objects)
        probability=separated
        detailed_foreground=separated
    group_info=None
    if detailed_foreground is not None and separated is None:
        from groups import protect_group
        probability,detailed_foreground,group_info=protect_group(pixels,probability,detailed_foreground,objects)
    try:
        plan, alpha, kept = prepare_backdrop(pixels, probability, objects, clean_floor, plan,
                                            detailed_foreground, floor_mode, equipment, floor_probability)
    except ValueError as error:
        ambiguous_support="supporting seat could not be protected" in str(error)
        if plan is not None or floor_mode is None or not ("plain, consistent studio backdrop" in str(error) or ambiguous_support):
            raise
        # An ordinary room is not a broken studio backdrop. Keep its surfaces
        # intact and let the normal tonal pipeline handle the photograph.
        from floor import encode_mask
        empty=encode_mask(np.zeros(pixels.shape[:2],np.uint8))
        person_mask=((probability>.1)&(detailed_foreground>.1)).astype(np.uint8)*255
        from backdrop_finish import pack
        plan={"version":3,"repairEngine":"preserve-v1","adaptiveRevision":3,"floorMode":"preserve",
              "layers":{"shape":list(pixels.shape[:2]),"backdrop":empty,"protected":pack(person_mask),'person':pack(person_mask),'source':pack(pixels)},
              "floorRepairs":{"version":1,"shape":list(pixels.shape[:2]),"count":0,"review":"",
                              "defects":empty,"extension":empty,"engine":"clone-v1","color":[0,0,0]},
              "review":["Cleanup withheld: the supporting seat could not be reliably protected." if ambiguous_support else "Non-studio scene preserved; automatic backdrop reconstruction was not applied."]}
        return image,preview,plan
    plan["equipmentPlan"] = equipment
    if separated is not None:
        from separation import prepare
        alpha=prepare(plan,pixels,separated,probability,objects)
        from backdrop_finish import pack
        plan['layers']['rawForegroundAlpha']=pack(np.rint(raw_alpha*255).astype(np.uint8))
    if group_info:plan['groupProtection']=group_info
    plan["protectedObjects"] = kept
    progress("Repairing backdrop and floor", 65)
    preview = render_backdrop(original_preview, alpha, plan)
    if export:
        image = render_backdrop(original_image, alpha, plan)
    return image, preview, plan

