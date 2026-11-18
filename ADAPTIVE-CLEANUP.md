# Adaptive cleanup revision 2

Auto Edit assesses each photo independently. After distraction reconstruction it reduces localized backdrop wrinkles using a masked, low-frequency luminance correction. It retains original fine grain and broad lighting gradients; subject/prop protection and the floor boundary exclude those areas from backdrop finishing. Nearly clean backdrops receive no finishing correction. Finishing settings replay on full-resolution export and are included in the Changes outline.

Automatic neutral-pixel white-balance guessing has been removed. RGB channels share the same tonal curve, avoiding independent channel gains that caused warmer skin or cyan/green whites. Highly clipped white sets do not receive an additional midtone lift. This does not recover clipped source detail or neutralize intentional coloured lighting.

Large neutral rectangular seats receive an additional geometric protection check. Fine subject boundaries receive a conservative guard, and reconstruction edges blend inside the selected mask. If a supporting seat remains uncertain, the safe upper backdrop can still be cleaned; the uncertain lower scene is preserved and flagged for review. Ordinary non-studio rooms retain their surroundings.

An additional floor detector handles dark printed mat tape between similarly light materials near the frame bottom. It requires a broad narrow dark band with lighter samples on both sides, and remains outside protected foreground.

Existing results can be reprocessed from their originals using Auto Edit. Saved revision-1 repair plans remain readable; rerun Auto Edit to use the new planner. The UI still uses one Auto Edit button.

## Limits

This update does not certify all items in the earlier 96-finding audit as fixed. Difficult furniture boundaries, backdrop remnants near a subject, generated texture, floor wear, ambiguous props and clipped highlights still need review. Strong drapery folds may be reduced rather than completely erased. The app does not replace the entire floor or paint a fitted solid background colour.

Validation results are recorded in `QA-ADAPTIVE.json` once the current checks finish. The original audit remains in `PIPELINE-DEFECTS-2026-11-13.md`.
