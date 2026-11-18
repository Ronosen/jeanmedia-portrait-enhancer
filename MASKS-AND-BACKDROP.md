# Masks, backdrop colour and floor filling — revision 3

Auto Edit now lifts neutral light backdrops, deepens dark backdrops and increases the saturation of coloured backdrops within the protected backdrop region. It retains the existing colour rather than choosing a new background. In uncertain partial-cleanup scenes, the automatic colour boost is withheld to avoid a bright horizontal transition across the preserved lower scene.

Floor filling is enabled for detected missing mat edges. If neural continuation fails the floor-colour check, the app attempts adjacent-surface continuation instead of simply retaining the border. Contact shadows, reflections and texture still require review; this is not a perfect reconstruction guarantee.

## Editable masks

After Auto Edit, click **Masks**, then **Add mask**. Choose Subject, Background, Backdrop, Floor, Props, Brush, Linear, Radial, Colour or Luminance. **Show mask** displays the selection. Brush refinement adds or subtracts regions; brush size, feather and inversion refine the selection. Local exposure, contrast and saturation affect that layer only. **Apply masks** updates the preview and resets export approval. Layers can be named, disabled or deleted and are retained in projects and full-resolution exports.

Linear and radial masks currently use a fixed top-to-bottom gradient and centred oval respectively; refine their regions with the brush. Colour ranges use a colour picker and width; luminance ranges use brightness and width. Subject/prop masks use the app's current segmentation, so difficult hair or furniture edges may need brush correction. This is an initial local-mask editor, not full Lightroom feature parity: there are no automatic individual eyes, teeth, skin or hair selectors, arbitrary gradient handles or full per-mask history yet. A brush can select those areas manually.

The earlier 96-finding audit remains a historical baseline. Revision-3 validation and remaining issues are recorded separately in `QA-V3.json` and the full-folder comparison report.
