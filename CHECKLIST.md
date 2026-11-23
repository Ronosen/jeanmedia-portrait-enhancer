# JeanMedia Portrait Enhancer — Full Build Checklist

**Goal:** A lightweight Windows desktop app that transforms imperfect client photographs into professional, client-ready edits while preserving client identity, face, hair, clothes, pose, body, and important original props. Automate safe repairs, require approval for significant changes, and keep original photos intact.

**Status:** Active project checklist, adopted 2026-11-11. Checked items have implementation evidence; unchecked items are pending or need acceptance verification. A working preview alone does not establish photographic fidelity or client-ready quality.

**Current product decision:** Keep the one-button **Auto Edit** interface requested by the user. Checklist proposals for multiple modes, dashboards, toggles and manual tools remain pending design decisions; they do not override that choice. Significant replacements still require explicit approval. Keep technical explanations out of the editing interface.

**Current gaps:** Difficult reconstructed edges, ambiguous supports and textured floor repair still require visual review. Manual editing modes are intentionally deferred under the one-button interface. Custom sizing/upscaling, complete commercial license clearance, a signed installer and broad client acceptance are not complete. See the dated implementation updates below; early checklist items are not a claim that every scene is solved.

**Verification:** Worker regression tests, native build checks and packaged-runtime photo audits are separate from clean-machine and client acceptance. Do not mark an output visually perfect just because processing and preservation checks pass.

**Evidence:** `src/main.tsx` (single-button UI, import, comparison, undo/redo, save), `src-tauri/src/main.rs` (Python IPC), `worker/photo.py` (import, tonal correction, color-managed separate export), `worker/adaptive.py` (per-photo orchestration), `worker/studio.py` and `worker/floor.py` (protected backdrop and localized floor repair), and worker regression tests. Model licenses are included, but this is not a completed commercial license audit.

**Studio repair correction (2026-11-12):** The colour-fit renderer was inadequate for object reconstruction. The approved local LaMa model is installed and checksum verified. Auto Edit now reconstructs selected background/floor regions; small marks on smooth surfaces use local healing first. Two real 4000 × 6000 exports passed protected-pixel and outside-mask containment checks, including the main seat's small support. See [QA-INPAINTING.json](QA-INPAINTING.json). Large fills require review before saving. This does not complete general photographic or release acceptance.

## 1. Product requirements & boundaries
- [x] Confirm app name: JeanMedia Portrait Enhancer.
- [x] Choose offline-first Windows desktop delivery.
- [x] Support single-photo editing and batch queues.
- [x] Provide a Solo/Batch start page and sequential open/edit/save batch processing into the source folder's edited subfolder.
- [ ] Define three modes: Quick Clean, Studio Repair, and Advanced Manual Review.
- [ ] Require preservation of face, expression, hair, body, pose, clothing, accessories and footwear.
- [ ] Preserve important props; allow surface cleaning with no additions by default.
- [ ] Require user approval before major prop replacement or identity-sensitive edits.
- [x] Keep original files unmodified; store edits in a new project or output folder.
- [ ] Mark generatively rebuilt areas in an inspectable edit mask/history.
- [x] Specify no-guarantee behavior: uncertain reconstructions must be flagged for review.

## 2. Hardware and developer environment
- [x] Record Windows version, CPU, GPU, VRAM, RAM, and storage space.
- [x] Detect CUDA/NVIDIA acceleration if available; support CPU fallback.
- [ ] Set separate low-VRAM and GPU-accelerated model presets.
- [x] Install Git, Python, Node.js, Rust/Tauri toolchain and suitable build tools.
- [ ] Create repository and reproducible locked dependency environments.
- [x] Use Python 3.11+ where compatible with selected packages; verify supported ranges.
- [x] Define model weights download/cache folder and disk-budget screen.
- [x] Add a hardware diagnostic and dependency self-test.
- [x] Never download huge models without showing size and requesting user consent.

## 3. Architecture
- [x] Frontend: Tauri + TypeScript + React (or lightweight equivalent).
- [x] Backend: isolated Python worker service or subprocess API.
- [x] Image processing: Pillow/OpenCV + NumPy.
- [x] Inference: PyTorch/ONNX Runtime where compatible.
- [ ] Organize model adapters so segmentation, object removal and inpainting can be swapped.
- [x] Use a task queue with cancellation, progress, retries and error messages.
- [x] Cache masks, previews, and generated intermediate results for reversible edits.
- [x] Never apply destructive edits directly to the original source photo.
- [x] Save an editable project file with original reference, masks, parameters, layers and history.
- [x] Log versions, settings and model identifiers for reproducibility.

## 4. File import and preflight
- [x] Import JPEG, PNG and TIFF (as supported).
- [x] Read image width/height, EXIF orientation, ICC profile, and metadata.
- [x] Correct orientation without unexpectedly changing framing.
- [x] Generate fast low-resolution preview and preserve full-resolution source.
- [ ] Inspect severe blur, noise, clipping, shadows and compression artifacts.
- [ ] Identify camera/studio edge intrusions, prop defects, unwanted objects and backdrop problems.
- [x] Flag images too damaged for reliable reconstruction.
- [x] Identify crop boundary and preserve useful negative space.
- [ ] Expose side-by-side original and planned edit masks before execution.

## 5. Subject detection, preservation and masking — CRITICAL
- [ ] Identify every person in image, not just primary face.
- [ ] Segment face, hair (including individual strands), skin, hands, body, clothing and boots.
- [ ] Detect contact and occlusion points between person, props and floor.
- [ ] Use conservative protected masks with adjustable padding.
- [ ] Offer manual brush: Protect / Erase / Repair / Refine Hair.
- [ ] Never run generative inpainting inside a protected subject mask by default.
- [ ] Keep original subject geometry, perspective, pose and accessories.
- [x] Store an original-pixel reference layer for pixel-difference checks.
- [x] For global exposure/color correction, preserve subject structure without claiming pixel-for-pixel identity.
- [ ] Flag uncertain hair boundaries and subject/prop overlap for review.

## 6. Studio cleanup and background detection
- [ ] Detect visible softboxes, light stands, cables, curtain edges, wall corners and backdrop rolls.
- [ ] Detect visible tiles, exposed floor edges, tape, seams and scratches.
- [ ] Detect accidental extra boxes/stools/equipment, without selecting client-owned props by mistake.
- [ ] Build proposed removal masks with confidence labels.
- [ ] Allow manual approve/reject each unwanted object.
- [ ] Identify intended backdrop style: dark, light, colored or patterned.
- [ ] Preserve original backdrop wherever it already looks good.
- [ ] Smooth wrinkles, marks and inconsistent patches without flattening intentional texture.
- [ ] Respect shadows and photographic depth.

## 7. Smart patching and object removal
- [x] Attempt simple clone/heal/texture repair for tiny marks and dust.
- [x] Use inpainting for isolated objects and small missing areas.
- [x] Use more advanced masked generative reconstruction only where needed.
- [ ] Build coherent wall-to-floor transitions and perspective.
- [ ] Continue backdrop color, gradient, texture and illumination naturally.
- [ ] Recreate floor texture without repeating patterns or distortion.
- [ ] Maintain original ground/contact shadows and realistic reflections.
- [ ] Prevent copied limbs, extra fingers, unwanted jewelry, duplicated shoes or invented objects.
- [ ] Validate boundaries with edge and color consistency checks.
- [ ] Offer manual masking and a second attempt for failed patches.
- [x] Show changed-region overlay after every repair.

## 8. Smart Props Enhancement
- [ ] Detect key props: stools, chairs, benches, boxes, flowers, tables and decorative pieces.
- [ ] Distinguish intentionally placed props from accidental studio equipment.
- [x] Default Props mode = Enhance.
- [x] Default Optional additions = No.
- [ ] Default Major replacement = Requires approval.
- [ ] Remove scratches, dust, chips and minor stains from existing props.
- [ ] Maintain exact shape, dimensions, position, construction and color unless approved.
- [ ] Preserve contact points, occlusions, shadows and reflections.
- [ ] Flag structurally broken or visually unsuitable props for approval, not silent replacement.
- [ ] Show a before/after close-up of every edited prop.

## 9. Color, light and detail finishing
- [x] Apply non-destructive white balance and exposure corrections.
- [ ] Recover shadow/highlight detail without unrealistic contrast.
- [ ] Preserve natural skin tone and natural skin texture.
- [ ] Add selective sharpening to hair, fabric and boots without halos.
- [ ] Use denoising and JPEG artifact reduction conservatively.
- [x] Avoid heavy face restoration by default; do not invent facial details.
- [ ] Improve black clothing detail without turning black into gray.
- [ ] Preserve the original outfit and prop colors unless user explicitly chooses grading.
- [ ] Offer adjustable light/color/texture intensity controls.
- [ ] Optionally upscale after reconstruction with a user-visible detail-risk warning.

## 10. UX and controls
- [ ] Dashboard: New Edit, Recent Projects, Batch Queue, Settings, Hardware Status.
- [x] Drag-and-drop upload.
- [x] Before/after split-slider and full-size 100% inspection.
- [ ] Presets: Quick Clean, Studio Repair, Background Patch, Props Clean, Manual.
- [ ] Toggle: Strict Subject Protection (on by default).
- [ ] Toggle: Auto Background Patching (on in Studio Repair).
- [ ] Toggle: Smart Props Enhancement (on, enhancement only).
- [ ] Toggle: Optional Prop Additions (off by default).
- [ ] Toggle: Approval for Major Replacements (on by default).
- [ ] Adjustable image crop, exposure, backdrop and shadow controls.
- [ ] Manual brush and rectangle masks with undo/redo.
- [ ] Highlight proposed changes, confidence, and reason for each repair.
- [x] Progress/Cancel/Retry with full error recovery.
- [x] Remember project preferences without silently reusing old client images.

## 11. Export, print, and color management
- [x] Save edited output separately from original.
- [x] Export JPG and PNG; optional TIFF for high-quality print workflows.
- [ ] Support original resolution, controlled upscaling, and custom pixel dimensions.
- [x] Preserve or convert ICC profiles intentionally (e.g., sRGB export).
- [ ] Include print-size calculator (A4, A3, custom cm/in, intended PPI).
- [x] Never imply changing DPI metadata creates extra detail.
- [x] Support adjustable JPEG quality and PNG alpha when applicable.
- [ ] Provide batch filename templates and destination folder selection.
- [x] Export mask/change map for audit and re-editing.
- [ ] Check no mismatched colors in soft proof / export preview.

## 12. Open-source model evaluation & licenses
- [ ] Evaluate SAM 2 for precise subject and object segmentation.
- [ ] Evaluate rembg with a commercially permitted model for quick rough matting.
- [ ] Evaluate LaMa/IOPaint as a prototype or reference for local inpainting; original IOPaint repo is archived.
- [ ] Evaluate ComfyUI for advanced masked reconstruction; integrate as a replaceable backend.
- [ ] Evaluate Real-ESRGAN only as an optional finishing/upscaling step.
- [ ] Check license of each model **weight**, code package, and embedded asset independently.
- [x] Do not assume MIT-licensed rembg makes every bundled model commercially usable.
- [ ] Check whether GPL dependencies require architecture changes or source obligations.
- [ ] Record model names, revisions, sources, SHA256 hashes and attribution.
- [ ] Provide offline download/cache, version pinning and uninstall paths.

## 13. Security, privacy and reliability
- [x] Process client portraits locally by default; ask consent before any cloud upload.
- [x] Avoid collecting or retaining facial embeddings unnecessarily.
- [ ] Secure project files and cached images from unintended access.
- [ ] Provide Delete Project / Clear Cache / Delete Generated Previews controls.
- [x] Keep local processing/network state obvious to the user.
- [x] Validate file types, image dimensions and resource use.
- [x] Ensure failure never destroys the source image.
- [ ] Add graceful CPU fallback, OOM recovery and optional tiled processing.
- [x] Add crash logs without private image content.

## 14. Testing with the supplied studio portrait
- [ ] Input = supplied full-body portrait with black backdrop, white seating box and visible studio equipment.
- [ ] Softbox on left is removed, without altering hair.
- [ ] Curtain/wall on right is replaced by matching backdrop.
- [x] Floor tiles and edge tape are removed.
- [ ] Black floor is repaired with perspective-consistent surface and shadows.
- [ ] Unwanted right-hand white box is removed.
- [ ] Main white seating box remains exactly in position, with surface blemishes cleaned.
- [ ] Face, hairstyle, pose, black coat and white trim are unchanged in shape/design.
- [ ] Glossy black boots retain shape, highlights and heel proportions.
- [ ] Finger anatomy is preserved with no additions.
- [ ] No seams, duplicated details, unnatural blur or halo edges are visible at 100% zoom.
- [ ] No unexpected object added automatically.
- [ ] Major uncertain changes appear in an approval queue.
- [ ] Export has correct dimensions, color profile and no unintended cropping.

## 15. General QA and acceptance gates
- [ ] Create test set of at least 30 varied real photographs with permission: dark/light backdrops, crowded scenes, hair detail, chairs, flowers, shadows and reflective boots.
- [ ] Measure success of object detection and preservation masks separately.
- [x] Test original-vs-output protected-region geometry and detail differences.
- [x] Confirm edits do not leak outside approved masks except permitted global color grading.
- [ ] Compare output at full resolution, not just thumbnails.
- [ ] Evaluate on CPU-only and on low-/high-VRAM machines.
- [ ] Measure peak RAM/VRAM and processing time by stage.
- [ ] Collect human approval scores for client-ready quality.
- [x] Define pass/fail checks for identity preservation, patch coherence, contact shadows and prop retention.
- [x] Do not auto-export a high-risk result as client-ready without approval.

## 16. Release and development phases
- [ ] Phase 1 / MVP: import, preserve subject mask, remove small unwanted objects, preview and export.
- [ ] Phase 2: advanced seamless backdrop and floor patching, automatic cleanup suggestions.
- [ ] Phase 3: Smart Props Enhancement and approval UI.
- [ ] Phase 4: color/detail presets, batch processing and print-ready export.
- [ ] Phase 5: reliability, model-license audit, optimized packaging and signed Windows installer.
- [ ] Add first-run hardware check and optional model downloads.
- [ ] Build rollback/backup for model updates.
- [x] Write user manual, troubleshooting guide and developer API docs.
- [ ] Verify repeatable install on another Windows computer.

## 17. Nonnegotiable delivery criteria
- [x] Original photo remains available unchanged.
- [ ] Client identity/clothing/body/pose are preserved to photographic fidelity.
- [ ] Background patch is coherent at 100% zoom.
- [ ] Props are retained unless explicitly approved for change.
- [x] UI clearly differentiates safe auto-edits and approval-required transformations.
- [x] App does not promise perfect results for images lacking recoverable information.
- [ ] License review completed before commercial client work.
- [ ] Final export is correctly color-managed and print-ready for requested dimensions.

## Internal-mask workflow update — 2026-11-17
- [x] Hide mask editing, mask export and mask overlay controls.
- [x] Install pinned local BiRefNet general lite with MIT attribution.
- [x] Extract subjects and confirmed supports into an internal foreground layer.
- [x] Reconstruct and clean a separate background plate, then composite original subjects.
- [x] Remove the previous background contribution from soft edge pixels.
- [x] Verify four full-size examples and deterministic saved-matte exports.
- [ ] Resolve all remaining fine fringes, pedestal borders and difficult floor defects across the full photo collection.

## Folder regression update — 2026-11-18
- [x] Recover continuous seat surfaces and protect their contact edges.
- [x] Keep complete equipment outlines below the floor boundary.
- [x] Reconstruct missing floor with a closer original-context view.
- [x] Detect disconnected frame-edge boxes separately from supporting seats.
- [x] Release large inference allocations between matting and reconstruction.
- [x] Invalidate older edits for an explicit Auto Edit rerun (revision 7).
- [x] Run original-hash, export, mask-containment and opaque-interior checks per input.
- [ ] Certify all fine edges, ambiguous supports and reconstructed texture as visually flawless.

Evidence and per-image remaining inspection points: `qa/folder-v6-final/REVIEW.md`.

## Dark backdrop fold regression — 2026-11-18
- [x] Locate weak dark-wall/floor junctions instead of stopping at a fixed 80% row.
- [x] Remove broad lower-backdrop folds with source foreground and floor protection.
- [x] Preserve backdrop grain and gradual lighting instead of painting a solid colour.
- [x] Keep saturated backdrops and cached older edits compatible.
- [x] Allow Auto Edit reruns for older results (processing revision 8).

## Open-source references
- SAM 2: https://github.com/facebookresearch/sam2
- rembg: https://github.com/danielgatis/rembg
- IOPaint (archived): https://github.com/Sanster/IOPaint
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN

**Next acceptance step:** Resolve the three skipped originals and advanced floor reconstruction, then expand to 30 varied consented originals and collect full-resolution human quality scores. Complete model/native-dependency license provenance and signed packaging, and verify installation on a separate Windows machine. UI proposals for multiple modes/manual controls remain pending decisions under the one-button product requirement.

## Client finishing update � 2026-11-20
- [x] Internal face landmarks protect irises and lips during restrained teeth/sclera whitening.
- [x] Adaptive chroma and neutral-black finishing, with continuous skin protection.
- [x] Measure noise on native-resolution source patches; reduce sharpening on noisy photos.
- [x] Bounded detail enhancement with local-extrema protection and mild edge-aware noise reduction.
- [x] Save finishing parameters for deterministic export and preserve older project rendering.
- [ ] Certify every reconstruction as ready for delivery without visual review.

These are implemented operations, not guarantees of recovering clipped highlights, motion blur or missing original detail. Seven-photo packaged-build acceptance results are recorded separately after execution.


## r24 implementation update - 2026-11-22

- [x] Separate object identity from scene role using a measured backdrop/room boundary.
- [x] Preserve held/nearby/straddling props rather than always protecting balloon identity.
- [x] Add bounded internal PyMatting edge and foreground-colour refinement.
- [x] Add local repair illumination matching, seam/colour/texture/donor checks and at most one retry.
- [x] Save retry decisions for replay and verify opaque source pixels before tonal finishing.
- [ ] User manual photographic acceptance of r24; no photo pipeline tests requested.
- [ ] Grounded SAM 2 and SDXL resource evaluation/integration; not bundled in r24.

Implementation checks are synthetic tests and code/build checks, not proof of client-ready output. See EDGE-REFINEMENT.md.


## r25 - Backdrop seam regression guard

The specialized dark-backdrop correction now owns its repair regions, including already-good transitions. Later generic harmonization and retry passes cannot overwrite or redraw them, and their removed source material cannot contaminate generic donors. Other repairs remain eligible. Rerun Auto Edit for revision 25. Verification uses synthetic stage-interaction tests and build/runtime checks only; no photographs were processed. User manual photographic acceptance is pending.


## r26 - Intelligent edge crop

Auto Edit can crop measured unwanted edge strips when subjects, props and safety margins remain intact; otherwise it uses normal repair. Feet/supports reserve floor contact space. Discarded regions are excluded from repairs and donors. The saved smartCrop box uses original preview coordinates; internal layers and face masks stay on the original canvas. Delivered preview/export, native inspection and output dimensions share the same crop. Before/after comparison uses matched framing; Undo shows the full original. No manual crop marking is required. Originals remain unchanged. See SMART-CROP.md.

No photo pipeline tests were run; verification uses synthetic masks/geometry, code/build and runtime checks. User manual acceptance remains pending.
