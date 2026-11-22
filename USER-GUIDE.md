# JeanMedia user guide

In the development folder, double-click **JeanMedia.exe**. It opens the selected
portable build in the adjacent `release` folder. In a downloaded portable ZIP,
extract the whole folder and double-click its **jeanmedia.exe**; keep the bundled
Python, worker and dependencies beside it.

Choose **Solo Edit** on the start page. Open or drop one or more JPEG, PNG or TIFF photographs, then press **Auto Edit**. Photos are processed sequentially on this computer. Original files are never overwritten.

New edits measure native-resolution noise and apply restrained noise reduction and detail sharpening without inventing facial detail. Sharpening is reduced in noisy areas and face regions.

New edits also deepen faded neutral blacks, add gentle vibrance, and brighten
visible teeth and eye whites using internal face and iris masks. Skin colours,
lips, dark tooth gaps, pupils and irises are protected from whitening. Small,
closed, obscured or uncertain features are left alone. Colour finishing improves
the supplied photo; exact original material colours require a calibrated reference.
These settings are saved with the edit and used for preview, inspection and export.
Older projects retain their previous finishing until Auto Edit is run again.

New Auto Edit results also smooth measured backdrop folds and protect short supporting footrests and their contact shadows. Run Auto Edit again on older results to use these changes.

Select a photo in the queue to view it. Scroll the mouse wheel inside the picture area to zoom toward the pointer, then drag to pan. **Fit** resets the view; switching photos also resets it. The before/after comparison stays aligned while zooming. Press **100%** or double-click a part of the image for original-resolution inspection. Undo/Redo are per photo. Auto Edit retries unfinished or undone items.

In Solo Edit, a **Needs review** result requires your approval before saving. An error means cleanup stopped; the photo has not been successfully edited. Protecting an uncertain seat or subject takes priority over filling that area. Large or complex floor reconstructions require review; insufficient context can still prevent repair. Inspect hair, hands, shoes, seats, shadows and filled areas before approving.

**Save** writes a new original-resolution JPEG, PNG or TIFF. Existing files are refused. A complete temporary file is published atomically so cancellation cannot leave a damaged final export. A cancelled save may leave a `.jeanmedia-*.tmp` beside the destination; that temporary file can be deleted when no save is running. The source is not used for temporary output.

The **â‹¯** menu holds projects, manual queue export, print metadata, JPEG quality and hardware diagnostics. **Save queue** skips unapproved review results and reports skipped or failed entries. Batch names have a sequence number. TIFF uses its filename extension. PNG preserves alpha when the source supports it; automatic studio cleanup currently requires opaque RGB input.

A `.jeanmedia` project stores the source reference and hash, repair layers, model hashes, settings, approval and edit history. Keep the original alongside the project. Reopening refuses a missing or changed original. Save projects using a new name. Projects contain private source references and masks: store them in your own protected folders. Clear session discards in-memory previews; it does not delete originals, saved projects or exports. Delete those explicitly through Windows Explorer when required.

Print width/height in cm are calculated from original pixels and selected PPI. Changing PPI metadata creates no extra image detail. Outputs are intentionally converted to sRGB. Custom crop, resizing, AI upscaling and soft proofing are not implemented.

If a photo fails, select it to read the short error and press Auto Edit to retry. Cancel stops the current worker and remaining queue entries. No cloud upload or model download happens during editing. Hardware checks load the five small protection/detection/face models and verify the reconstruction model checksum. The approved local LaMa model reconstructs substantial holes; tiny smooth-surface marks use local healing first. Model loading and processing can take tens of seconds or longer on this CPU. The default CPU preset is appropriate for the audited 16 GB RAM / AMD 2 GB GPU system. CUDA is used only if explicitly configured and an ONNX CUDA provider is installed; otherwise CPU remains available.

Technical crash logs contain only timestamp, exception type and worker version, in the current user's temporary `JeanMedia/errors.jsonl`. No image data or source filename is logged. A bounded local subject-mask cache is stored in the portable app’s worker/.cache/matting folder to speed repeated Auto Edit operations. It contains derived masks, not original photos, and is excluded from release packages. You can delete that cache while the app is closed; it will be regenerated.

# Troubleshooting and release limits

- **Supporting seat could not be protected:** cleanup was skipped. Current lightweight models cannot reliably identify every white-on-white or occluded seat. Do not treat the original preview as an edited result.
- **Plain backdrop could not be identified:** patterned or crowded scenes are outside the current automatic reconstruction capability.
- **Review before saving:** inspect the preview and explicit review reasons; approve only if acceptable.
- **Could not start worker:** use the project launcher or the packaged portable app; a source-checkout executable requires its local Python environment.
- **File already exists:** choose a different filename. No overwrite mode is provided.

This is a development build. Local masked LaMa reconstruction is implemented. Semantic hair/face/body-part parsing, manual brushes, signed installer distribution and broad human quality acceptance are still pending. It is not certified client-ready for every portrait.

Detached salient objects are conservatively protected as possible photo props, including hanging gowns and balloons. Equipment requires independent removal evidence. This protects ambiguous objects but may also retain unwanted clutter; semantic recognition of every graduation prop is not guaranteed. Repeated Auto Edit on exactly the same pixels can reuse subject inference; new photos still require inference and background reconstruction.

When the wall/floor boundary cannot be measured reliably, broad backdrop smoothing is withheld and the original surface texture and contact shadows are preserved. Independently detected equipment may still be repaired. Some backdrop folds can remain in these uncertain scenes.


# Automatic Batch Edit

Choose **Batch Edit**, **Choose folder**, then **Start batch**. Supported photographs in that folder are processed one at a time: open, Auto Edit, save the full-resolution JPEG, then open the next. Subfolders are not scanned, so the output folder is not loaded again.

Results are saved beside the originals in **edited**, using the source basename. Existing files are preserved: a numbered name is used when needed. The current Solo PPI and JPEG-quality preferences are used and can be changed under Batch Export settings before starting.

Batch automatically saves results carrying review flags, as requested for unattended processing. **Saved · Review** identifies those files. Inspect them before client delivery. A local batch report in edited records saved filenames, review notes and errors after every attempt. Unreadable photos are recorded as failed and the next file is attempted; disk-space or report-write failures stop the run.

**Stop after current** finishes and saves the active photo, then stops. **Resume batch** skips already saved items and retries unfinished/failed items. Switching modes preserves both sessions while the app remains open. Closing the app ends the run; completed files and the report remain. Selecting the folder again after restarting begins a new batch and uses numbered output names to preserve earlier exports.


Pipeline revision 21 adds a local semantic prop check before reconstruction. It compares measured object crops with graduation gowns, caps, balloons, diplomas/books, flowers, seats and common studio distractions. Removal requires agreement between tight and wider crops; plausible uncertain props stay protected for review. It cannot reliably infer every photographer's intent, and equipment, reflections or repair seams can remain. New portrait plans also use a stronger bounded teeth/eye-white correction; old saved plans retain their earlier whitening behavior. See SCENE-INTENT.md for details. The hardware self-test now includes the local CLIP vision model and prompt-vector integrity.


Pipeline revision 22 repairs cropped light-backdrop equipment independently of floor detection, checks isolated objects before treating equipment as a prop, and prevents saturated floor reflections from becoming protected colour islands. It preserves weaker floor-edge evidence and avoids retaining a wide old-background ring at the subject edge. Dark repaired equipment areas use nearby source backdrop donors to correct colour drift while retaining reconstructed texture. Existing saved plans retain their earlier compositing behavior until Auto Edit is run again.


## r24 - Scene context and protected edges

Rerun Auto Edit to apply the new pipeline. Internal masks distinguish detached room decorations above a measured backdrop rail from held/nearby portrait props. PyMatting refines uncertain edges and foreground colour. Repair checks can trigger one local retry; unresolved problems remain marked for review. First use may take longer while numerical routines initialize. No manual mask control is required. Existing projects retain their old rendering.

No source or sample photographs were processed for r24 acceptance, at the user's request. Visual testing is reserved for the user; this build is not a client-readiness certificate. Grounded SAM 2 and SDXL are not bundled. See EDGE-REFINEMENT.md for exact scope.


## r25 - Backdrop seam regression guard

The specialized dark-backdrop correction now owns its repair regions, including already-good transitions. Later generic harmonization and retry passes cannot overwrite or redraw them, and their removed source material cannot contaminate generic donors. Other repairs remain eligible. Rerun Auto Edit for revision 25. Verification uses synthetic stage-interaction tests and build/runtime checks only; no photographs were processed. User manual photographic acceptance is pending.
