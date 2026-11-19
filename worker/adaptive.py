"""Automatic studio-editing skill: assess each image and apply localized repairs.

The studio planner derives subject/prop protection, backdrop colour and the floor
boundary anew for every photo. Floor masks select defects and missing edges;
healthy floor pixels never enter a repair mask. Saved plans replay on export.
"""

from studio import clean_studio


def auto_edit(image, preview, export=False):
    original_image, original_preview = image, preview
    from neural import model_session
    model_session.cache_clear()
    image, preview, plan = clean_studio(
        image, preview, export=export, floor_mode="patch")
    floor = plan["floorRepairs"]
    plan['processingRevision']=16
    from portrait_enhance import prepare
    plan['portraitEnhancement']=prepare(original_image, original_preview)
    plan["assessment"] = {
        "version": 1,
        "floorAction": "repair" if floor["count"] else "preserve",
        "floorRepairCount": floor["count"],
        "floorUncertain": bool(floor.get("review")),
    }
    from runtime import manifest
    plan["runtime"] = manifest()
    plan["review"] = list(dict.fromkeys(plan.get("review",[])+([floor["review"]] if floor.get("review") else [])))
    status = "Needs review" if plan["review"] else "Done"
    return image, preview, plan, status
