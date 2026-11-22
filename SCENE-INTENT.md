# Local prop and distraction assessment

Narrow support panels can be confirmed by bright source material and direct
subject contact even when a gown occludes their rectangular shape. Blue lighting
casts no longer disqualify an otherwise supported white seat. Tiny unanchored
side-frame colour fragments need foreground or semantic prop evidence rather
than receiving opaque protection from colour alone.
Dark reconstructed gear regions are checked against nearby observed backdrop
donors. Local illumination and residual-contrast transfer correct colour drift
while retaining reconstructed texture; healthy backdrop, floor and foreground
outside those repairs are not part of that transfer.

Pipeline revision 22 adds independent backdrop evidence for cropped portraits
on light studio sets. A subject mask reaching the frame bottom and a broad
observed backdrop material enable local wall smoothing without guessing a floor
horizon. Frame-connected dark polygonal equipment and thin coloured wall strips
require measured source silhouettes, a matching backdrop ring and no overlap
with the subject/support guard. Semantic prop protection still wins over these
equipment proposals. Full-body scenes and dark sets retain the conservative
floor-contour behavior. This is evidence-based detection, not filename rules or
a fixed repair position, and does not guarantee removal of every kind of gear.
Colour-only decoration protection also stops at a measured floor boundary, so
saturated floor reflections cannot become hard source-colour islands inside a
repair. Semantic evidence can still preserve a real prop placed on the floor.
Exposed thin studio borders connected to the top frame can be removed without
a floor estimate. When a proposed prop overlaps independent equipment evidence,
an isolated-object view prevents a nearby cap/book in the context crop from
protecting the entire rig. Weaker floor-material witnesses limit wall darkening
without redefining the reconstruction horizon. Background feathering follows
surface boundaries rather than leaving a wide old-background ring around people.

The prop and whitening features began in revision 21; new automatic edits now
use revision 24. The scene-intent skill runs inside
the application, before background reconstruction. It needs no API key or photo
upload and adds no mask controls to the interface.

The skill proposes detached objects from source matting, colour components,
compact dark components and measured equipment. A pinned CLIP ViT-B/32 ONNX model
compares object crops with fixed descriptions of graduation gowns, caps,
balloons, books/diplomas, flowers, seats, softboxes, bags, wallpaper and cables.
It records the label, similarity, competing-category margin, decision, crop and
selected pixel count in `studioPlan.sceneIntent`. Similarities are not calibrated
probabilities. A wider context view can confirm a partly visible balloon;
removal requires matching categories in both the tight and context crops.

Independent subject and supporting-geometry masks exclude people, their clothes
and attached supports from removal candidates. Confident props are protected on
their measured masks; confident clutter can enter reconstruction even when the
wall/floor boundary is unknown. A classification rectangle is never painted or
used as an object silhouette. Uncertain classifications authorize no additional
removal, retain existing protection and produce a review warning.
Likely but uncertain props also retain their measured source material pending
review, so an older outlier rule cannot silently delete that proposed prop.
Existing independent equipment rules still apply; the skill does not veto every previous
rule or infer the photographer's intent with certainty.

Limitations: the skill assesses at most 16 detached proposals per pass; occluded,
tiny, connected or unusual props can be missed. Dark-on-dark segmentation and
reconstruction blending remain separate problems. Incorrectly identified
equipment or missing masks still need review. It is not a guarantee that every
object will be classified correctly or every output will be client-ready.

## Repair boundaries

Revision 24 separates object identity from its role in the studio scene. A
measured sloped rail and material evidence define exposed room space; detached
balloons/flowers entirely above that boundary and away from the subject can
be removed as room decorations. Held, near-subject and boundary-crossing props
remain protected. With insufficient boundary evidence the app does not invent
a fixed upper crop or certify the object as unwanted. See
[EDGE-REFINEMENT.md](EDGE-REFINEMENT.md) for internal PyMatting, colour matching,
bounded inspection/retry behavior and the manual photographic acceptance limit.

New cleanup revision 23 includes the contact shadow of measured top-edge studio
hardware in its reconstruction mask. The margin follows detected equipment,
respects foreground protection and stays above the measured subject extent;
it does not introduce a fixed horizontal boundary or change floor repair.
Reconstruction colour transfer fades into the backdrop beneath the measured
top repair, avoiding a sharp join. Side-panel edges and the internal upper-body
safety cutoff do not trigger this fade. Protected prop holes do not cause it,
so they do not retain a bright reconstructed rim. Older saved plans keep their
previous rendering behavior. The native window title identifies revision r23.

## Teeth and eye whites

New portrait plans record `retouchAssessment.version = 2`. Visible enamel and
sclera receive bounded LAB lightness and tint correction, preserving local
texture. Irises, pupils, lips and unselected pixels receive no whitening. Closed
mouths, dark glasses, small or undetected features may correctly receive none;
the app does not invent teeth or eyes. Old saved plans retain the previous
whitening formula when replayed, and source photographs remain untouched.

## Reproduce model installation

Run `scripts/install-scene-intent.py` with the project Python runtime, then
`scripts/compile-scene-prompts.py` with the worker dependencies on `PYTHONPATH`.
The installer verifies pinned model and tokenizer hashes. The text encoder stays
in ignored QA tooling; only the 88.6 MB vision encoder and compiled prompt vectors
are distributed. Model weights, vectors and derived photo caches stay out of Git.
The manifest, MIT license and provenance are tracked. Portable self-tests check
the vision model and prompt-vector integrity.
