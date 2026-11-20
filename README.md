# JeanMedia Portrait Enhancer

Project requirements and acceptance status: [CHECKLIST.md](CHECKLIST.md).

Development build for Windows. The React/Tauri desktop shell calls a local Python worker. OpenCV applies bounded channel and exposure curves. Pillow handles EXIF orientation, color conversion, previews and JPEG/PNG/TIFF export. Export always creates a separate file and refuses to overwrite an existing path.

## Run locally

Install Node.js and pnpm, Rust with the MSVC toolchain, Visual Studio Build Tools with Desktop development with C++, Microsoft Edge WebView2, and Python 3.11 or 3.12. Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r worker\requirements.txt
$env:JEANMEDIA_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
pnpm install
pnpm tauri dev
```

Dependency installation needs network access or a populated offline package cache. The photo workflow runs locally. Use the save dialog's format selector or type a JPEG, PNG or TIFF extension. JPEG exports composite transparency onto white.

Tested Python environment: 3.12.14; pinned runtime packages are in `worker/requirements-lock.txt`. The source worker has an 800 MB / 80 MP input limit.

Run the Stage 1 worker tests from the project root with `.\.venv\Scripts\python.exe -m unittest discover -s worker`. Build the frontend with `pnpm run build`.

Choose **Solo Edit** on the start page, then use **Auto Edit** after opening or dropping one or more photos. Batch processing is sequential, with per-photo errors, cancellation and retries. The built-in adaptive skill performs portrait correction, foreground/prop protection, backdrop repair and floor assessment together. Detected studio intrusions and backdrop gaps are repaired with local LaMa inpainting, retaining healthy backdrop pixels. Tiny marks on a smooth surround use local healing first. Cropped portraits without a detected floor do not receive an invented horizon. Only detected floor defects are repaired. Healthy floor pixels are retained.

Open the **⋯** menu for editable projects, batch saving, JPEG quality, print metadata and diagnostics. Subject masks run internally; there is no manual mask option. Use **100%** or double-click to inspect original-resolution pixels. Solo review results require approval before saving. **Batch Edit** automatically processes the selected folder one photo at a time and saves JPEGs in its **edited** subfolder before opening the next. Existing outputs receive numbered alternatives; review flags and errors are recorded in a local batch report. Supporting-seat ambiguity, patterned backdrops and large/textured gaps remain limitations. The [user guide](USER-GUIDE.md), [API](API.md), [acceptance gates](ACCEPTANCE.md) and [model decisions](MODEL-DECISIONS.md) record precise scope and outstanding work.

A portable development package can be staged with `scripts/package-portable.ps1` after building the native executable. It includes a local Python runtime, dependencies and models and self-tests their loading. It is unsigned; installation on another computer is not yet verified.

Every delivered app build gets a new GitHub release. After checks pass, commit
and push the changes, then run `scripts/publish-release.py --bundle <verified-package>
--tag <new-version> --title <release-title> --notes <notes-file>` with the local
Python runtime. This prepares and verifies a portable ZIP and SHA-256 checksum.
Run the same command with `--publish` to upload both to a draft, verify GitHub's
asset digests and publish the development prerelease. The script uses Git's
saved GitHub authentication without writing credentials to disk. Each release
uses a new tag; existing published releases are preserved. Browser profiles,
sessions, photos, caches and generated QA reports are excluded from the archive.

The bundled PP-HumanSeg ONNX weights are distributed under Apache 2.0 by [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg). The license accompanies the weights in `worker/models`.

Automatic backdrop cleanup reconstructs the detected plain studio backdrop to remove clutter and fill exposed wall, wallpaper and backdrop gaps. Floor patching changes only selected repair regions, retaining healthy floor texture and shadows. Detected mat edges can be reconstructed over exposed tiles/tape; large or uncertain fills are flagged for review. People and supporting surfaces are protected. Highly textured scenes, conflicting foreground masks or insufficient visible backdrop are skipped.

The cleanup uses CPU [YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md) to protect recognised furniture and contact objects. These official upstream model weights are Apache 2.0 licensed, with the license included in `worker/models/YOLOX-LICENSE`. Unknown supporting boxes are conservatively protected where their visible shape touches the lower body. Person and prop detection can miss boundaries, so inspect the preview around hair, seats, flowers, hands and shadows. Undo, redo and original-resolution export retain the backdrop settings for the current image.

New studio edits use a second lightweight [U²-Net foreground model](https://github.com/xuebinqin/U-2-Net) to protect footwear and foreground details without reserving the whole lower frame. The ONNX weights come from the [rembg release](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx), with the upstream Apache 2.0 license included in `worker/models/U2Net-LICENSE`. The original person mask and the new foreground mask must agree on the confident person area. Local support surfaces are protected separately using their boundaries and colour. Small contact shadows and uncertain boundary pixels can remain. Older saved edits retain their conservative protection. Protected pixels are rendered from the photograph entering studio cleanup.

Current edits extract the source foreground with local BiRefNet, clean a separate background plate, then composite protected subjects back. Dark neutral backdrops use a wider smoothing scale to remove broad fabric folds while retaining grain and gradual illumination. A low-contrast floor junction is assessed instead of always stopping cleanup at 80% of the frame. Older saved edits retain their previous settings; rerun **Auto Edit** to apply the update. See the [internal workflow](SEPARATED-WORKFLOW.md). Saved floor repair masks are validated against people and props before preview and export.

Semantic subject parsing, manual repair tools, broad prop restoration, custom resizing/upscaling, a signed Windows installer and full client-ready acceptance remain pending. Local LaMa reconstruction is implemented and tested on the two reported examples. Editable projects and cancellation are now implemented.

