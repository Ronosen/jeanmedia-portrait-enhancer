# Intelligent edge crop (r26)

Auto Edit first plans the subject, props, studio boundary and repairs on the
original canvas. Before reconstruction it decides whether an unwanted outer
strip can be removed without crossing protected content. No marking is needed.

## Decision policy

- Crop only when measured unwanted material or missing floor reaches an edge.
  Healthy negative space and isolated interior defects do not justify cropping.
- Preserve all foreground/prop/support/contact masks. Retain unclassified raw
  foreground unless identified as clutter, and veto crops through unresolved
  object proposals even when their detailed silhouette is uncertain.
- Keep a 2.5% short-side margin (at least two preview pixels) around protection.
  Where feet/supports meet a detected floor, reserve additional ground below
  them for contact shadows.
- Limit each edge to 18% of its original dimension, retain at least 75% of the
  canvas, and require enough unwanted material to justify lost healthy space.
  There is no fixed crop position, filename exception or forced aspect ratio.
- If no safe beneficial crop exists, keep the framing and use normal repairs.
  A safe partial crop can coexist with normal repair of remaining areas.

Protection is limited by the source detection evidence; it is not proof that
every possible unseen/unrecognized prop was identified. Difficult cases still
need human review. Cropping never changes the original source file.

## Repair and output

Discarded areas are removed from backdrop/floor repair selection and inspection
donors. Reconstruction sees an edge-extended retained context rather than the
unwanted exterior, and crop edges do not become artificial blend boundaries.
Subject/prop source compositing and existing specialist repair ownership remain.

Internal layers, face masks and saved source references retain original-canvas
coordinates. Crop is applied to the delivered preview/export after finishing;
native bounds scale from the saved preview box, with no output upscaling.
Before/after sliders share the cropped framing for pixel alignment. Undo shows
the full original; Redo restores the edit. A small Auto-cropped label identifies
the decision. Native 100% inspection translates cropped display coordinates
back into the original canvas. Output dimensions/PPI reflect the actual crop.

Solo, batch, project replay and change-map exports use the same saved crop.
Old plans without crop metadata keep their dimensions. Crop bounds and protected
content are validated during replay; stale/tampered unsafe bounds are rejected.
Development scripts calling internal `adaptive.auto_edit` receive canonical
full-canvas images plus crop metadata; use `photo.py` or `smart_crop.apply` after
finishing for the delivered cropped output.

## Validation

Verification uses synthetic masks, geometry/control-flow unit tests, compilation,
native builds and portable dependency/model diagnostics. No source or sample
photos were processed, continuing the user's manual-testing instruction.
Photographic acceptance is pending. Rerun Auto Edit in r26 to apply cropping.
