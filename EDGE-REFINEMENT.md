# Internal scene context and edge refinement (r24)

Auto Edit remains one button. All segmentation, trimaps and inspection masks
are internal. Processing remains local and original files remain unchanged.

## Scene context

A measured rail separating exposed room material from a matching studio
backdrop establishes the upper backdrop contour. The detector requires a long
observed edge, matching material below, different material above, and exclusion
of independently protected people/supports. It follows a sloped contour rather
than assuming the same image row for every photograph. No filename, sample
coordinates or client-specific templates are used.

Object identity and object role are separate. A balloon/flower entirely above
that boundary and away from the subject can be treated as room decoration.
Held, nearby, or boundary-crossing props retain protection. Confirmed portrait
props and source anchors are excluded when the backdrop is extended into room
space. Without adequate rail evidence no upper boundary is invented, and
uncertain frame-clipped decorations require review. This cannot establish the
photographer's intent with certainty.

## Edges

PyMatting 1.1.16 supplies closed-form alpha refinement and multilevel foreground
colour estimation. The app generates trimaps from its existing BiRefNet matte;
there is no manual marking step. Sparse solves are tiled and bounded. Only
uncertain alpha values can move, by at most 0.25; known background and opaque
source cores remain fixed. Colour estimation replaces nearest-background
sampling at soft edges, with bounded correction. Its result is stored for
consistent saved-plan/native export replay. Opaque source pixels are checked
and restored after repair, before permitted portrait colour/tonal finishing.
Explicitly planned furniture surface healing remains allowed.

Pinned Windows x64 / Python 3.12 dependencies and wheel hashes are recorded in
`worker/edge-requirements.txt`. Install with the bundled Python's pip using
`--target .worker-deps --no-deps --require-hashes --only-binary=:all:`. The
existing NumPy/Pillow versions are retained. Libraries carry their license
notices in the portable package; PyMatting is MIT licensed. The broader native
dependency/model commercial provenance audit remains open.

## Reconstruction and inspection

Locally smooth wall donors can correct bounded low-frequency colour drift in
repaired regions. This retains reconstructed detail and cannot select healthy
floor or edit a protected person. Patterned/strongly varying donors are excluded.
Existing measured floor masks still decide which floor pixels may be repaired.

The repair inspector checks selected regions for unusually sharp joins, colour
drift relative to nearby edited material, lost texture and insufficient donors.
It examines at most 32 repair components. A suspicious region can receive at
most one additional local LaMa repair. It is adopted only if its inspection
score improves. The mask, cached prediction and choice are saved; export does
not start an unbounded retry loop. Unresolved cases remain flagged for review.
The inspection score is a heuristic, not a calibrated probability or proof of
client readiness. It does not certify prop intent, anatomy, perspective or
every possible halo/reflection defect.

## Scope and validation

This build uses existing CPU ONNX models plus PyMatting. Grounded SAM 2 and
SDXL are not bundled or claimed installed: their hardware/resource evaluation
remains pending for the recorded 2 GB GPU / CPU-first Windows setup. No remote
photo service or automatic model downloads occur during editing.

At the user's request, no original, edited or sample photographs were processed
for r24 acceptance. Verification is limited to synthetic array/unit tests,
TypeScript/native build checks and portable dependency/model diagnostics.
Photographic acceptance is reserved for the user's manual testing. Older saved
plans keep their rendering behavior; rerun Auto Edit for the r24 changes.
