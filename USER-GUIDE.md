# JeanMedia user guide

Open or drop one or more JPEG, PNG or TIFF photographs, then press **Auto Edit**. Photos are processed sequentially on this computer. Original files are never overwritten.

Select a photo in the queue to view it. Drag the comparison slider, press **100%**, or double-click a part of the image for original-resolution inspection. **Changes** shows thin repair outlines; **Hide outlines** returns to the clean preview. Undo/Redo are per photo. Auto Edit retries unfinished or undone items.

A **Needs review** result requires your approval before saving. An error means cleanup stopped; the photo has not been successfully edited. Protecting an uncertain seat or subject takes priority over filling that area. Large or complex floor reconstructions require review; insufficient context can still prevent repair. Inspect hair, hands, shoes, seats, shadows and filled areas before approving.

**Save** writes a new original-resolution JPEG, PNG or TIFF. Existing files are refused. A complete temporary file is published atomically so cancellation cannot leave a damaged final export. A cancelled save may leave a `.jeanmedia-*.tmp` beside the destination; that temporary file can be deleted when no save is running. The source is not used for temporary output.

The **⋯** menu holds projects, batch export, change-mask export, print metadata, JPEG quality and hardware diagnostics. **Save queue** skips unapproved review results and reports skipped or failed entries. Batch names have a sequence number. TIFF uses its filename extension. PNG preserves alpha when the source supports it; automatic studio cleanup currently requires opaque RGB input.

A `.jeanmedia` project stores the source reference and hash, repair layers, model hashes, settings, approval and edit history. Keep the original alongside the project. Reopening refuses a missing or changed original. Save projects using a new name. Projects contain private source references and masks: store them in your own protected folders. Clear session discards in-memory previews; it does not delete originals, saved projects or exports. Delete those explicitly through Windows Explorer when required.

Print width/height in cm are calculated from original pixels and selected PPI. Changing PPI metadata creates no extra image detail. Outputs are intentionally converted to sRGB. Custom crop, resizing, AI upscaling and soft proofing are not implemented.

If a photo fails, select it to read the short error and press Auto Edit to retry. Cancel stops the current worker and remaining queue entries. No cloud upload or model download happens during editing. Hardware checks load the three small protection/detection models and verify the reconstruction model checksum. The approved local LaMa model reconstructs substantial holes; tiny smooth-surface marks use local healing first. Model loading and processing can take tens of seconds or longer on this CPU. The default CPU preset is appropriate for the audited 16 GB RAM / AMD 2 GB GPU system. CUDA is used only if explicitly configured and an ONNX CUDA provider is installed; otherwise CPU remains available.

Technical crash logs contain only timestamp, exception type and worker version, in the current user's temporary `JeanMedia/errors.jsonl`. No image data or source filename is logged. Photographs and masks are not automatically persisted outside a project/export you choose to save.

# Troubleshooting and release limits

- **Supporting seat could not be protected:** cleanup was skipped. Current lightweight models cannot reliably identify every white-on-white or occluded seat. Do not treat the original preview as an edited result.
- **Plain backdrop could not be identified:** patterned or crowded scenes are outside the current automatic reconstruction capability.
- **Review before saving:** inspect the preview and explicit review reasons; approve only if acceptable.
- **Could not start worker:** use the project launcher or the packaged portable app; a source-checkout executable requires its local Python environment.
- **File already exists:** choose a different filename. No overwrite mode is provided.

This is a development build. Local masked LaMa reconstruction is implemented. Semantic hair/face/body-part parsing, manual brushes, signed installer distribution and broad human quality acceptance are still pending. It is not certified client-ready for every portrait.
