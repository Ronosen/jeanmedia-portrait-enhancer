# Local worker API

`worker/photo.py SOURCE` returns JSON with bounded before/after PNG data URLs, dimensions, preflight metrics, metadata and review state. Use `--auto-edit` for the combined workflow; it enables tonal finishing and computes a fresh protected repair plan. Opaque RGB is required for automatic studio cleanup. Maximum source size: 800 MB / 80 MP. Previews: maximum 1280 pixels per side.

For replay and export, pass `--request-stdin` and send a JSON object containing `studio_plan` and/or `equipment_plan` on standard input. Do not place large mask layers on a Windows command line. Legacy small `--studio-plan` and `--equipment-plan` arguments remain available. Settings input is limited to 16 MB.

`--enhance --output NEW_PATH --dpi 300 --quality 96` exports original resolution. Add the saved studio plan through stdin to replay cleanup. `--approved` is required when the saved plan contains review reasons. Existing output files and originals are never overwritten. `--mask-output NEW.png` saves the preview-resolution repair-strength map (white=repair; black=unchanged). `--inspect-region X Y WIDTH HEIGHT` returns original-resolution before/after crops, maximum 1400 by 1400, using the full-resolution render and finishing pipeline.

Progress lines on stderr start with `@progress:` followed by a JSON stage and percentage. Other stderr text is an error. The Tauri bridge emits `photo-progress` events, drains stdout/stderr concurrently, and exposes `cancel_photo` to terminate the current child. Frontend batches call the worker sequentially; errors are attached to each photo. Cancellation stops the queue. No automatic retry loop is used.

`worker/service.py ACTION` reads JSON on stdin. Actions:

- `diagnostics`, `{}`: hardware, disk budget, runtime versions and local model-load self-tests.
- `save-project`, `{path, source, studioPlan, enhance, approved, history}`: creates a new `.jeanmedia` file, never overwriting.
- `open-project`, `{path}`: validates schema, file size and source SHA256 before returning edit settings.
- `mask`, `{studioPlan}`: returns a PNG data URL of the combined backdrop, floor and prop change map.

Modules separate model inference (`background`, `foreground`, `objects`, shared `runtime.session`), planning/rendering (`studio`, `floor`, `props`) and orchestration (`adaptive`). CPU is the default; `JEANMEDIA_DEVICE=gpu` selects CUDA only when the installed ONNX runtime advertises it. CUDA session-initialization failures fall back to CPU. This build has only CPU ONNX runtime installed. No model updater or network downloader is shipped.

Plans include background coefficients, floor treatment/repair masks, foreground-protection layers, optional furniture blemish masks, review state and model/runtime hashes. Replays validate geometry and repair masks against newly inferred protection. Projects bind these settings to a source hash; do not reuse a plan across unrelated photos. Raw input to the command API is a trusted local caller, not an internet API.

Run `python -m unittest discover -s worker` with the pinned environment. `scripts/evaluate_photos.py DIRECTORY` records automated preservation metrics only. `scripts/verify-workflow.cjs` exercises the React UI with a real Python worker through official Tauri IPC mocks; it does not substitute for a native installation test. Build using `pnpm tauri build --debug --no-bundle`. `scripts/package-portable.ps1` stages the executable, Python standard library, dependencies, workers and models, then self-tests model loading. That portable build is unsigned and still needs testing on a separate Windows computer.


## Local reconstruction backend

`worker/neural.py` verifies the pinned LaMa checksum, supplies RGB/float32 inputs with a single-item batch, and pads to the model's fixed 512-square input without changing source framing. Foreground colours are excluded from temporary donor context; final pixels can change only in the selected source-resolution repair mask. Large reconstructions add review state, and saved projects cache the generated patch for deterministic replay without running inference again. Tiny isolated marks on a smooth, unoccluded surround use local healing first.

New plans use `repairEngine: lama-v1`. Legacy colour-fit plans remain readable; Auto Edit is enabled to upgrade them from the original. Mask inspection now returns RGBA outlines; exported change masks remain grayscale. No network call occurs during editing. The optional installer script requires `--consent` and verifies an immutable download hash. CPU processing can take tens of seconds or longer; cancellation terminates the current worker.


## Sequential folder batch

The start page selects Solo or Batch without discarding either in-memory session. Solo retains its existing worker calls and approval guard. Batch lists source paths only, imports the current photo, invokes photo with autoEdit/enhance/output/approved/compactOutput, records the result, then opens the next. Its explicit automatic-export action permits saving review-flagged outputs and records those flags; it does not mark them as human-reviewed.

The local_tool allowlist includes batch-list, batch-output and batch-report. batch-list enumerates supported files at one folder level. batch-output creates edited beside the source, refuses linked destination folders and chooses a new numbered JPEG filename on collision. The existing atomic export remains the final no-overwrite guard. batch-report atomically writes a run-specific JSON journal under edited. Stop after current waits for export and journal completion; in-session resume skips saved entries. File failures continue, while disk-space/report failures stop the loop.

photo.py --compact-output returns only saved/backdropStatus/review for an export. The default Solo response is unchanged. This keeps per-photo masks and reconstruction plans out of the batch queue and avoids retaining 200 previews or replaying a separate export plan for every batch item.


## r26 - Intelligent edge crop

Auto Edit can crop measured unwanted edge strips when subjects, props and safety margins remain intact; otherwise it uses normal repair. Feet/supports reserve floor contact space. Discarded regions are excluded from repairs and donors. The saved smartCrop box uses original preview coordinates; internal layers and face masks stay on the original canvas. Delivered preview/export, native inspection and output dimensions share the same crop. Before/after comparison uses matched framing; Undo shows the full original. No manual crop marking is required. Originals remain unchanged. See SMART-CROP.md.

No photo pipeline tests were run; verification uses synthetic masks/geometry, code/build and runtime checks. User manual acceptance remains pending.
