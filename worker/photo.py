"""Local portrait worker: protected repairs, previews and separate full-size exports."""

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from background import clean_backdrop, edge_repair, person_probability, repair_mask
from equipment import apply_equipment_plan, detect_equipment
from studio import clean_studio

PREVIEW_SIZE = 1280
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG",
           ".tif": "TIFF", ".tiff": "TIFF"}
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def load_photo(path):
    with Image.open(path) as original:
        frames = getattr(original, "n_frames", 1)
        oriented = ImageOps.exif_transpose(original)
        profile = oriented.info.get("icc_profile")
        rgb = oriented.convert("RGB")
        if profile:
            color_input = oriented if oriented.mode in {"RGB", "CMYK", "L", "LAB"} else rgb
            rgb = ImageCms.profileToProfile(
                color_input, io.BytesIO(profile), SRGB, outputMode="RGB"
            )
        if "A" in oriented.getbands() or "transparency" in oriented.info:
            rgb.putalpha(oriented.convert("RGBA").getchannel("A"))
        return rgb, frames


def correction(preview):
    pixels = np.asarray(preview.convert("RGB"))
    if preview.mode == "RGBA":
        pixels = pixels[np.asarray(preview.getchannel("A")) > 127]
    if not pixels.size:
        return np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 3))
    pixels = pixels.reshape(-1, 3)
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722])
    # Scene colours, mixed lighting and coloured backdrops are not a gray card.
    # Keep a shared RGB curve instead of guessing independent channel gains.
    gains = np.ones(3)
    low, median, high = np.percentile(luminance, [5, 50, 95])
    gamma = np.clip(1 + (median - 118) / 720, 0.9, 1.1)
    if np.mean(luminance>248)>.15:
        gamma=max(1.0,gamma)  # Do not lift an already overexposed white set.
    contrast = np.clip(1 + (150 - (high - low)) / 450, 1, 1.25)
    values = np.arange(256, dtype=float)[:, None] / 255
    toned = np.clip(values * gains, 0, 1) ** gamma
    # A smooth curve raises midtone contrast while anchoring black and white.
    toned += (contrast - 1) * toned * (1 - toned) * (4 * toned - 2)
    return np.rint(np.clip(toned, 0, 1) * 255).astype(np.uint8)


def apply_correction(image, lut, portrait=None):
    if portrait is not None:
        from portrait_enhance import apply
        return apply(image, portrait)
    result = Image.new(image.mode, image.size)
    for y in range(0, image.height, 512):
        end = min(y + 512, image.height)
        top = max(0, y - 2)
        box = (0, top, image.width, min(end + 2, image.height))
        tile = image.crop(box)
        pixels = np.asarray(tile.convert("RGB"))
        toned = cv2.LUT(pixels, lut.reshape(256, 1, 3))
        if image.mode == "RGB":
            luma = cv2.cvtColor(toned, cv2.COLOR_RGB2GRAY).astype(np.float32)
            smooth = cv2.GaussianBlur(luma, (5, 5), 0.9)
            detail = luma - smooth
            # Ignore weak variations and limit the adjustment at hard edges.
            adjustment = np.sign(detail) * np.clip((np.abs(detail) - 2) * 0.45, 0, 7)
            toned = np.rint(np.clip(toned + adjustment[..., None], 0, 255)).astype(np.uint8)
        corrected = Image.fromarray(toned)
        if image.mode == "RGBA":
            corrected.putalpha(tile.getchannel("A"))
        result.paste(corrected.crop((0, y - top, image.width, end - top)), (0, y))
    return result


def preview_url(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def export_photo(image, output, dpi, quality=96):
    format_name = FORMATS.get(output.suffix.lower())
    if not format_name:
        raise ValueError("Choose JPEG, PNG or TIFF.")
    options = {"icc_profile": SRGB.tobytes(), "dpi": (dpi, dpi)}
    if format_name == "JPEG":
        if image.mode == "RGBA":
            flattened = Image.new("RGB", image.size, "white")
            flattened.paste(image, mask=image.getchannel("A"))
            image = flattened
        options.update(quality=quality, subsampling=0)
    if output.exists():
        raise FileExistsError("Choose a new filename; existing files are preserved.")
    # Publish only a complete file. Cancellation cannot leave a corrupt final copy.
    descriptor, temporary = tempfile.mkstemp(prefix=".jeanmedia-", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            image.save(stream, format_name, **options)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # atomic, refuses to overwrite on both platforms
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--auto-edit", action="store_true")
    parser.add_argument("--clean-backdrop", action="store_true")
    parser.add_argument("--remove-edge-equipment", nargs=3, type=float)
    parser.add_argument("--auto-remove-equipment", action="store_true")
    parser.add_argument("--equipment-plan", type=json.loads)
    parser.add_argument("--clean-studio", action="store_true")
    parser.add_argument("--studio-plan", type=json.loads)
    parser.add_argument("--clean-floor", action="store_true")
    parser.add_argument("--floor-mode", choices=("preserve", "patch", "replace"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--quality", type=int, default=96)
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--inspect-region", nargs=4, type=int)
    parser.add_argument("--request-stdin", action="store_true")
    args = parser.parse_args()
    if args.request_stdin:
        raw = sys.stdin.read(16*1024*1024+1)
        if len(raw) > 16*1024*1024:
            raise ValueError("Repair settings exceed the 16 MB limit.")
        request = json.loads(raw)
        if not isinstance(request, dict) or set(request) - {"studio_plan", "equipment_plan"}:
            raise ValueError("Invalid repair settings request.")
        for key, value in request.items():
            if value is not None and not isinstance(value, dict):
                raise ValueError("Invalid repair plan.")
            setattr(args, key, value)
    if args.auto_edit:
        if args.studio_plan or args.clean_studio or args.floor_mode or args.clean_floor or args.equipment_plan or args.auto_remove_equipment or args.remove_edge_equipment:
            raise ValueError("Automatic editing cannot be combined with another repair plan.")
        args.enhance = True
    if not 1 <= args.dpi <= 2400:
        raise ValueError("DPI must be between 1 and 2400.")
    if not 50 <= args.quality <= 100:
        raise ValueError("JPEG quality must be between 50 and 100.")
    if args.source.suffix.lower() not in FORMATS:
        raise ValueError("Choose a JPEG, PNG or TIFF photograph.")
    from runtime import progress
    from preflight import inspect
    progress("Opening photo", 3)
    if args.source.stat().st_size > 800 * 1024 * 1024:
        raise ValueError("Photo exceeds the 800 MB file limit.")
    with Image.open(args.source) as header:
        if header.width * header.height > 80_000_000:
            raise ValueError("Photo exceeds the 80 megapixel processing limit.")
        metadata = {"format": header.format, "orientation": header.getexif().get(274, 1),
                    "icc": bool(header.info.get("icc_profile")), "mode": header.mode}
    image, frames = load_photo(args.source)
    full_original = image
    preview = image.copy()
    preview.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    original_preview = preview
    preflight = inspect(preview)
    render_full = bool(args.output or args.inspect_region)
    lut = correction(preview) if args.enhance else None
    equipment_plan = args.equipment_plan
    if args.auto_remove_equipment or equipment_plan:
        if args.remove_edge_equipment or preview.mode != "RGB" or (args.auto_remove_equipment and equipment_plan):
            raise ValueError("Automatic equipment removal needs an opaque photo and no manual selection.")
        try:
            probability = person_probability(np.asarray(preview))
            if args.auto_remove_equipment:
                equipment_plan = detect_equipment(np.asarray(preview), probability)
            source_rgb = np.asarray(preview)
            preview = apply_equipment_plan(preview, source_rgb, probability, equipment_plan)
            if args.output:
                image = apply_equipment_plan(image, source_rgb, probability, equipment_plan)
        except Exception as error:
            raise ValueError(f"Automatic removal skipped. {error}") from error
    if args.remove_edge_equipment:
        right, top, bottom = args.remove_edge_equipment
        if not (0 < right <= 0.45 and 0 <= top < bottom <= 1) or preview.mode != "RGB":
            raise ValueError("Select equipment along the left edge of an opaque photo.")
        try:
            probability = person_probability(np.asarray(preview))
        except Exception as error:
            raise ValueError("Needs Review · Portrait protection could not run.") from error
        w, h = preview.size
        pad = round(min(w, h) * 0.025)
        guard = probability[max(0, round(top*h)-pad):round(bottom*h)+pad,
                            :round(right*w)+pad]
        if not (probability > 0.9).any() or (guard > 0.05).any():
            raise ValueError("Needs Review · Move the selection farther from the subject.")
        region = (right, top, bottom)
        preview = edge_repair(preview, region)
        if args.output:
            image = edge_repair(image, region)
    mask = None
    backdrop_status = ""
    studio_plan = args.studio_plan
    if args.auto_edit:
        from adaptive import auto_edit
        image, preview, studio_plan, backdrop_status = auto_edit(image, preview, render_full)
    elif args.clean_studio or studio_plan:
        if args.clean_studio and studio_plan:
            raise ValueError("Choose new backdrop cleanup or saved backdrop settings.")
        try:
            image, preview, studio_plan = clean_studio(
                image, preview, studio_plan, args.clean_floor, render_full, args.floor_mode)
            backdrop_status = "Needs Review · Backdrop cleaned and extended. Inspect hair, props and floor boundaries before export."
            if studio_plan.get("floorMode") == "patch":
                count = studio_plan["floorRepairs"]["count"]
                backdrop_status += (f" {count} localized floor repairs applied." if count
                                    else " Floor checked. No confident defects found.")
                if studio_plan["floorRepairs"].get("review"):
                    backdrop_status += " " + studio_plan["floorRepairs"]["review"]
        except Exception as error:
            raise ValueError(f"Backdrop cleanup skipped. {error}") from error
    if args.enhance and args.clean_backdrop and not studio_plan:
        if preview.mode == "RGBA":
            backdrop_status = "Needs Review · Backdrop cleanup was skipped for a transparent image."
        else:
            try:
                rgb = np.asarray(preview)
                mask, backdrop_status = repair_mask(rgb, person_probability(rgb))
                if mask is not None and not backdrop_status:
                    backdrop_status = ("Backdrop cleaned." if mask.any()
                                       else "No isolated backdrop marks qualified for repair.")
            except Exception:
                backdrop_status = "Needs Review · Portrait protection could not run. Backdrop preserved."
    if studio_plan and studio_plan.get('localMasks'):
        from local_masks import apply as apply_local_masks
        preview=apply_local_masks(preview,studio_plan)
        if render_full:image=apply_local_masks(image,studio_plan)
    portrait = studio_plan.get('portraitEnhancement') if studio_plan else None
    if portrait is not None:
        from portrait_enhance import validate
        validate(portrait, original_preview)
    review = []
    if studio_plan:
        review = list(studio_plan.get("review", []))
        if args.auto_edit:
            review.extend(preflight["warnings"])
            review = list(dict.fromkeys(review))
            studio_plan["review"] = review
        backdrop_status = "Needs review" if review else "Done"
    if args.mask_output:
        if not studio_plan:
            raise ValueError("Run Auto Edit before exporting a change mask.")
        from service import change_mask
        change_mask(studio_plan).save(args.mask_output.open("xb"), "PNG")
    if args.output:
        if review and not args.approved:
            raise ValueError("Review and approve this edit before saving.")
        progress("Saving full resolution", 85)
        finished = apply_correction(clean_backdrop(image, mask), lut, portrait) if lut is not None else image
        export_photo(finished, args.output, args.dpi, args.quality)
        progress("Saved", 100)
        print(json.dumps({"saved": True, "backdropStatus": backdrop_status,
                          "equipmentRegion": args.remove_edge_equipment,
                          "equipmentPlan": equipment_plan,
                          "studioPlan": studio_plan}))
    elif args.inspect_region:
        x, y, w, h = args.inspect_region
        if min(x, y) < 0 or not 1 <= w <= 1400 or not 1 <= h <= 1400 or x+w > image.width or y+h > image.height:
            raise ValueError("Inspection region is outside the photo.")
        box = (x, y, x+w, y+h)
        after = apply_correction(image, lut, portrait).crop(box) if lut is not None else image.crop(box)
        print(json.dumps({"before": preview_url(full_original.crop(box)), "after": preview_url(after)}))
    else:
        after = apply_correction(clean_backdrop(preview, mask), lut, portrait) if lut is not None else preview
        before = preview_url(original_preview)
        print(json.dumps({"width": image.width, "height": image.height, "frames": frames,
                          "before": before, "after": preview_url(after) if args.enhance or args.remove_edge_equipment or equipment_plan or studio_plan else before,
                          "backdropStatus": backdrop_status,
                          "equipmentRegion": args.remove_edge_equipment,
                          "equipmentPlan": equipment_plan,
                          "studioPlan": studio_plan, "preflight": preflight, "metadata": metadata,
                          "review": review}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            import datetime
            directory = Path(tempfile.gettempdir()) / "JeanMedia"
            directory.mkdir(exist_ok=True)
            log = directory / "errors.jsonl"
            if log.exists() and log.stat().st_size > 64_000:
                log.unlink()
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         "errorType": type(error).__name__, "worker": 3})+"\n")
        except OSError:
            pass
        sys.exit(str(error))

