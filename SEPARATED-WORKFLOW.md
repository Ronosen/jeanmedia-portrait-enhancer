# Internal subject protection and background reconstruction

Auto Edit now uses an internal foreground layer. The Masks editor, change-mask
export and mask-overlay controls are hidden from the app.

1. Local BiRefNet general lite predicts a detailed alpha matte. Independent
   person detections anchor the people; confirmed seats and pedestals join the
   protected foreground.
2. LaMa removes that foreground and detected clutter from a separate background
   plate. Damaged background pixels can be repaired right to the true matte.
3. Background finishing and localized floor repairs operate on that plate.
4. Original opaque subject pixels are composited back exactly. Soft edge pixels
   have their old background contribution removed before compositing.
5. Saved exports reuse the stored matte and reconstruction after verifying the
   source preview. Older edits can be rerun using Auto Edit.

BiRefNet general lite is a 224,005,088-byte local ONNX model from the rembg
release, with MIT attribution in `worker/models/BiRefNet-LICENSE`. Its SHA256 is
`5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333`.
No photo upload or model training is involved. CPU processing takes longer than
the previous small-mask pipeline; the app reports progress and supports cancel.

## Current corpus verification — revisions 13 and 14

The frozen corpus contains 49 originals from `D:/test22` and `D:/test 222`.
Existing `-finished` exports are excluded. The per-photo review is generated at
`qa/corpus-review/index.html`, with machine-readable findings in `ledger.json`.
Processing success and numerical preservation checks are separate from visual
acceptance; a completed render is not a claim that the image is perfect.

Detached salient objects excluded from the person layer are selected completely
for reconstruction. Clothing contacting a person's upper silhouette is retained,
as are detected people and intentional reflections. Paired vertical source edges
recover thin chrome supports omitted by the matte. Source-supported bright seat
surfaces are retained even when a measured rectangle describes only one panel;
confirmed cropped supports can reach the bottom frame. Native prop contacts
remain confined to measured foreground and its immediate antialiasing rim.

The backdrop fit checks alternate dominant colour families when a cropped seat
wins the largest colour bin but fails the wall-coverage test. The minimum sample
and spatial coverage guards still apply. Overhead corner gear requires a measured
diagonal edge with a dark-to-light transition; gradual vignettes are not selected.
Spare edge boxes use nearby floor contrast so distant bright tiles do not hide
them.

Backdrop finishing now uses bounded local texture correction instead of a global
surface residual. It excludes measured material boundaries and differing material
colours, preserving broad illumination and the photographed floor. Floor-edge
extension keeps its measured material boundary, filling only guard gaps beside
the extracted silhouette. Repairs stay on background pixels. Local floor plates blend inside their selection boundaries;
the foreground is excluded from that feather to avoid a halo beside shoes.

Saved older plans retain their rendering methods. Auto Edit offers a rerun when
the processing revision changes. QA caches contain exact model tensors keyed by
source pixels and model SHA256; no filename or hand-drawn mask influences the
production planner.

The initial full-size comparison set is in `qa/separation-final-20261117`. It covers
four graduates, two graduates with held props, a seated red-outfit portrait with
a softbox, and a light-backdrop portrait. These are regression examples, not a
claim of perfect segmentation on every photograph.

Known visual limitations still require inspection: thin cap/hair fringes,
pedestal borders, small blue-cloth remnants on the group images, scratched floor
texture and the large frayed floor edge. Healthy floor is retained; a missing or
complex floor edge can still require review.

## Revision 7 — 18 November 2026

The current folder audit is `qa/folder-v6-final/REVIEW.md`. It includes every image in
the audit snapshot of `D:/test22`, including already edited files. Original
hashes, full-resolution export dimensions, ICC profiles, unchanged cleanup
pixels and opaque foreground interiors are checked separately from visual QA.
Repeated matte inference uses exact float32 tensors keyed by preview pixels and
the pinned model digest; planning, reconstruction and export still execute.

Confirmed seat surfaces are filled continuously rather than preserving disjoint
colour components. White pedestal masks stop at physical bright-to-dark edges,
excluding adjacent cast shadows. Equipment masks keep their complete footprint
below the estimated floor horizon. Disconnected edge boxes and coloured clutter
reflections are detected separately from the supporting seat.

Missing floor edges now get a closer LaMa view of the original floor and its
lighting. Only damaged floor and detected clutter are composited from that
repair; healthy floor remains original. Saved projects cache this floor plate
and verify its crop geometry on export. Older plans remain replayable. Auto Edit
offers a rerun when an older processing revision is loaded.

An uncertain seat no longer turns a whole background band into opaque foreground.
Its review warning is retained without protecting balloons or wallpaper. Local
floor reconstruction blends into the background plate inside the selected mask.
A late paper/tile edge
below the shoes is not used as the wall/floor horizon. Inference sessions release
large memory arenas between subject matting and background reconstruction;
isolated audit workers prevent one failed photo from corrupting later results.

Visual acceptance remains open for fine fringes, source scuffs, ambiguous
supports and irrecoverable clipped highlights. Passing preservation checks is
not a promise of perfect reconstruction or photographic fidelity at every edge.

## Dark backdrop folds — revision 8

The floor fallback previously stopped cleanup at 80% of the frame in a dark,
low-contrast scene, leaving the lowest cloth folds untouched. A wider, sustained
brightness transition now locates a subtle dark-wall/dark-floor junction while
rejecting ordinary gradual lighting gradients.

New neutral-dark backdrop plans use a larger normalized smoothing scale and
stronger fold suppression up to the floor boundary. Source grain and broad
illumination remain; darkening uses the cleaned luminance so it does not turn
bright fold ridges into inverted dark stripes. Foreground and floor protection
remain enforced. Saturated backdrops retain the previous finishing treatment.

Finishing is cached in the saved plan. Existing plans replay unchanged; Auto
Edit is available for results made with older processing revisions. Local
regression comparisons are in `qa/backdrop-fold-v8`.

## Per-photo evidence — revision 9

The revision 8 floor search and 80% fallback have been replaced. The planner
checks material transitions across independent visible columns throughout the
photo. It excludes overhead transitions using subject geometry, rejects
interpolated samples hidden behind subjects, and records the measured row and
agreement in `floorAnalysis`. Without sufficient evidence it records no floor
boundary (1.0); it does not invent a donor band at a preset frame height.

New separated plans use `local-evidence-v2`. A robust illumination fit separates
broad lighting from local ridges; multiple contrast scales select defect areas
wherever they occur. Masked filtering excludes people and the detected floor.
Clean backgrounds can receive no finishing correction. The previous automatic
whole-backdrop darkening/smoothing is not applied to new plans. Cached older
plans still replay their saved appearance; Auto Edit creates revision 9 plans.

Regression tests move floor boundaries and isolated folds through different
positions, verify clean illumination is unchanged, and check subject/floor
preservation and saved replay. Real-photo verification is recorded separately
in `qa/adaptive-v9-verified`. These measurements do not guarantee that every
subtle junction is detectable: ambiguous floor repair can be withheld, and
other equipment/floor-edge detectors still have geometric priors. Visual
acceptance of fine edges and reconstruction remains necessary.

## Preserve the source join — revision 10

Revision 9 still used its measured scalar floor height as a horizontal cutoff.
That can introduce a false straight edge where the photographed cloth meets
the floor at different heights. New plans track locally visible transitions
in narrow strips and save the measured segments as `surfaceContour` and the
wall region as `layers.wallSurface`. A continuous contour follows neighbouring
measurements across hidden subjects/equipment, rejects isolated fold/scratch
peaks, and blends the repair inside the wall. A guard retains the original
visible join.

Generic wall reconstruction and finishing use that wall mask; floor patch
masks are restricted to confirmed ground below the local contour. Explicit
equipment removal may still cross the join. Older saved plans retain their
recorded rendering. Regression tests cover opposite slopes with uneven edges
and verify source junction, floor and subject preservation. Local full-photo
checks are in `qa/contour-v10-witness-regressions` and
`qa/contour-v10-witness-9044`. The reported graduation photo is a QA case only;
no filename, source coordinates or per-image exceptions enter the worker.

The floor witness uses the independent human segmentation mask, rather than
salient floor reflections or shadow haze in the extraction matte. Perceptual
colour transitions distinguish dark floor material from dark cloth. Consensus
counts only visible witnesses with spatial separation; occluded columns do not
vote against the boundary. The extraction matte still controls compositing and
subject protection.

## Validate an apparently tiny backdrop — revision 11

A false early floor transition can leave too little wall for a reliable fit.
When the fit specifically fails its sample-extent checks, the planner retries
with corroborated high-confidence foreground pixels from both models. The
retry requires separated visible witnesses whose colour transitions agree in
direction and magnitude. It accepts a later boundary only after the original
fit coverage checks pass. The retry is recorded in `floorAnalysis`; genuinely
insufficient samples still require review. Other valid plans keep their
original detection path. No filename-specific branches are used.

Compact source props touching a hand and a confirmed support top are added to
foreground protection, including the dark printed gaps between book covers.
The hull is constrained to the support's local geometry. Unattached objects
remain eligible for background cleanup.

Regression checks cover the retry, opposing unrelated edges, contact props and
insufficient samples. Full-resolution checks are recorded in
`qa/backdrop-extent-final`; they retain source hashes and compare subject cores,
unselected pixels and saved export replay.

## Preserve photographed prop joins — revision 12

Separate person and prop mattes left thin cracks along pedestal/gown contacts
and beneath books. New plans seal only contact regions with adjacent foreground
cores, recover source-bright shaded rims near confirmed supports, and record
`layers.sourceContact` plus a soft outer `sourceContactBlend`. Exterior edges
without contact are not expanded by the contact guard; local surface recovery
still requires source contrast against the fitted backdrop.

Rendering restores original native pixels in the exact protection mask after
compositing. A feathered outer transition avoids creating a new cutout edge at
the guard boundary. Saved older plans retain their rendering. Intentional prop
healing remains a separate operation.

Tests now cover shaded rims, unchanged external background, and native prop
boundary pixels without eroding away the boundary. The export verifier reports
`propBoundaryPixelsChecked` and `propBoundaryChangesBeforeTone` separately from
subject-core preservation. Real-image comparisons are in
`qa/prop-edge-verified-reported` and `qa/prop-edge-verified-seated`.

Measured box geometry now takes precedence over overlapping coarse colour/shadow
support regions. Contact props are measured outside the existing person mask;
components truncated at the search boundary cannot become protected hulls.
Dark props additionally need physical source-edge evidence, preventing diffuse
wall shadows from being treated as objects. White-backdrop regressions cover
oversized support/shadow regions and preserve the complete measured box.

## Reconcile floor evidence with protected foreground — revision 15

The planner validates people and contacting supports before refining the floor
measurement. Only person detections corroborated by segmentation, detailed
foreground inside those detections, and validated supports contribute to the
second measurement. Floor reflections and unrelated overhead equipment cannot
vote as people. The existing spatial consensus and backdrop coverage checks
still apply; there is no fixed floor row or filename-specific behavior.

Confirmed supports and photographed contact joins survive a successful floor
refit. Contacting shaded seat tops can use local material and physical edge
evidence when their colour resembles the fitted background. Frame-panel cleanup
selects exposed background without clearing the protected foreground matte.
Small detached floor clutter uses local contrast and a compact repair mask.

The corpus audit separately compares the original anchored subject core before
tone adjustments, so shrinking a later foreground mask cannot conceal damage.
It also checks final foreground cores, unselected pixels, native export dimensions
and original file hashes. Optional QA reconstruction reuse requires identical
render inputs, model/runtime provenance and rendering dependencies; the pipeline
still executes planning, compositing and native export for every source. Changed
inputs run reconstruction again. QA artifacts and photo caches remain untracked.

## Portrait colour and feature finishing — revision 16

After cleanup, new plans measure the source colour distribution and locate faces
using pinned local YuNet and MediaPipe Face Landmarker ONNX models. The original
native-resolution face crops supply tooth-enamel and sclera masks; iris geometry,
mouth aperture, brightness and colour checks exclude pupils, lips and dark gaps.
The saved, source-bound masks replay without rerunning detection during exports.

Colour finishing uses a shared RGB contrast curve that does not lift shadows,
a gentle toe for neutral black cloth/backdrops, and bounded LAB chroma gain.
Face guards and skin-colour checks limit colour shifts. It retains texture and
does not infer true material colours from a photograph lacking a reference.
The finishing stage is shared by preview, native export and 100% inspection;
older plans retain their legacy tone processing. New features remain behind
the single Auto Edit button, without exposing mask controls.

## Noise-aware detail finishing � revision 17

Original-resolution patches measure noise before resizing. The saved finishing
plan controls a small bilateral filter and thresholded luminance sharpening.
Face regions receive less sharpening, hard edges cannot exceed local extrema,
and overlapping tiles prevent processing seams. No face restoration or new
image detail is generated. Plans without detail settings keep revision-16 output.

## Backdrop folds and prop bases � revision 18

New plans fit smooth illumination from the measured wall surface, retain grain,
and bound the target to observed wall brightness. Deep neutral folds remain
eligible even when hue is unreliable. Corrections fade at the measured floor
boundary and exclude the protected subject/prop masks.

Short bright footrests require direct person contact rather than proximity alone.
The source layer includes a narrowly feathered contact shadow beneath each
measured supporting base. This prevents reconstructed floor from breaking its
outline or making it float. Planner and export both protect encoded alpha values
above 3; legacy 4/5-alpha floor overlaps are removed from repair masks on replay,
while opaque overlaps still fail validation. Existing projects keep cached
finishing until Auto Edit is explicitly rerun.
