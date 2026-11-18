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
