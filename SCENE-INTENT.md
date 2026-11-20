# Local prop and distraction assessment

New automatic edits use pipeline revision 21. The scene-intent skill runs inside
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
