# Model integration decisions — 2026-11-11

This is an integration and provenance review, not a completed legal or photographic quality certification. Runtime evaluation is limited to the bundled models. The initial review used only the small bundled models; the approved LaMa installation and evaluation are recorded in the 2026-11-12 update below.

| Candidate | Evidence and integration decision |
|---|---|
| SAM 2 | Upstream recommends a GPU setup and lists PyTorch/torchvision requirements. Code/checkpoint licensing and third-party files require independent review. The audited computer has no NVIDIA CUDA device. Defer installation and benchmark until a model size and CPU/GPU configuration are chosen. [Official repository](https://github.com/facebookresearch/sam2) |
| rembg / U2Netp | The app directly runs the existing small U2Netp ONNX checkpoint for rough foreground, without installing the rembg server or selecting its changing default model. MIT licensing of rembg does not license every weight. U2Netp source attribution and Apache license are recorded separately. Do not silently switch to BRIA's commercially restricted default. [rembg](https://github.com/danielgatis/rembg), [U2Net](https://github.com/xuebinqin/U-2-Net), [checkpoint release](https://github.com/danielgatis/rembg/releases/tag/v0.0.0) |
| LaMa / IOPaint | IOPaint's upstream repository is archived. The LaMa code license does not by itself resolve provenance and rights for every redistributed checkpoint. An open upstream weight-license clarification remains material. Defer checkpoint bundling until that is resolved; current isolated defects use OpenCV inpainting. [IOPaint](https://github.com/Sanster/IOPaint), [LaMa license](https://github.com/advimman/lama/blob/main/LICENSE), [upstream weight clarification](https://github.com/advimman/lama/issues/366) |
| ComfyUI | Official repository identifies GPL-3.0. Keep a potential future diffusion service replaceable and independently licensed rather than bundling it into this release. Model-specific weight terms and resource budgets remain to be evaluated. No ComfyUI service or integration is claimed implemented. [Official repository](https://github.com/Comfy-Org/ComfyUI) |
| Real-ESRGAN | Official code uses BSD-3-Clause. Optional image restoration would still need per-checkpoint provenance, resource testing and a user-visible risk control. No AI upscaling is included; PPI metadata is not upscaling. [Official code/license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) |

## Bundled weights

`MODEL-MANIFEST.json` records exact installed byte sizes and SHA256 hashes. The local license files accompany the weights. Sources: [OpenCV Zoo PP-HumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), and the U2Net/rembg release above. Original upstream commit/export provenance for all binary weights still needs final release review. SHA256 identifies the installed binary; it is not itself proof of licensing.

Python and JavaScript runtime dependency license inventory is generated locally. It does not establish license obligations for every bundled native codec, compiled library, transitive Rust dependency or asset. The commercial release audit therefore remains open. No signed installer or commercial legal clearance is claimed.

## Hardware fit

`HARDWARE-AUDIT.json` records the authorized Windows hardware read and current model-load checks. The original CPU-first build used about 14.4 MB of ONNX weights; adding the approved 208 MB LaMa model brings the current total to about 222.4 MB. The installed runtime advertises CPU and Azure execution providers, not CUDA. The app never uploads client portraits to Azure; only the CPU execution provider is selected by default. A separate low/high-VRAM test machine, signing certificate and broader quality dataset are still required for release acceptance.


## Approved local reconstruction update — 2026-11-12

The user approved the 208 MB Carve LaMa FP32 ONNX download. The exact 208,044,816-byte file was verified against publisher SHA256 `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`, pinned to revision `a3ee2fca54baebec351b8fa7786154ffa7555aa6`. The upstream code license and a separate weight-provenance record accompany it. Publisher-declared Apache-2.0 is recorded; this does not close the commercial provenance audit above.

Real local inference was evaluated on IMG_1083 (full softbox removal across the floor line) and IMG_0671 (mat extension over tiles/tape and frame-edge clutter). Preview and original-resolution protected pixels and unselected pixels passed containment checks. The reported ghost remnants were addressed by expanding panel coverage and excluding foreground/guard colours from temporary donor context. Original foreground is composited back unchanged. Broader image quality acceptance and the previously skipped ambiguous seats remain open.

Tiny marks on smooth surfaces use OpenCV healing; substantial missing regions use LaMa. The export's symbolic batch is fixed to one, matching the sequential photo queue. A measured startup test improved from roughly 50–70 seconds to about 12 seconds with that setting. Processing duration varies with CPU load and full-resolution export; it is not instantaneous. ORT-format caching was slower on this system and was discarded. No additional model or dependency was downloaded, and no client image was uploaded.
