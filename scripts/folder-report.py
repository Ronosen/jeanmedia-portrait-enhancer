"""Write an evidence-linked result report without treating successful exports as perfect edits."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();rows=json.loads((root/'run.json').read_text())
notes={
 'IMG_0164':('Backdrop edges and frayed floor extension repaired; people retained.','Inspect thin cap fringes, residual floor scuffs and far-left stand tip.'),
 'IMG_0164-finished':('Already edited input processed separately.','Existing edits in this input cannot be reversed to recover missing source detail.'),
 'IMG_0203':('Backdrop extended around both graduates; held certificate, books and pedestal retained.','Inspect left floor for blue-cloth residue and cap/hair fringes.'),
 'IMG_0656':('Softbox removed and supporting white seat retained.','Fine floor scratches and reflected floor texture still need close inspection.'),
 'IMG_0671':('Complete seat top retained; studio clutter removed; floor reconstructed from closer context.','Inspect fine floor scuffs, repaired reflection continuity and contact edges.'),
 'IMG_0683':('Backdrop edge cleanup and disconnected frame-edge box detection applied.','Inspect black fabric folds near the floor and scattered scuffs.'),
 'IMG_0731':('Couple and connected furniture preserved while backdrop cleaned.','No floor reconstruction is applicable to this close crop.'),
 'IMG_0747':('White edge panel removed; chrome stool retained; mat lip repaired.','Chrome highlights and reflections are intentional; inspect the filled lower edge.'),
 'IMG_0747-finished':('Previously edited version processed separately, retaining stool.','A second tone pass can differ from the original single-pass edit.'),
 'IMG_0779':('Softbox removed; adult and baby retained.','Inspect baby fingers, clothing folds and fine hair boundaries.'),
 'IMG_0797':('White backdrop cleaned with cap, suit and held gown retained.','Source facial highlights and cap rim lighting are retained.'),
 'IMG_0881':('Mat edge filled while boots, sole and supporting white seat retained.','Inspect foreground sole contact/shadow and fine floor specks.'),
 'IMG_0957':('Edge softbox removed; red backdrop, hair, glasses and flowers retained.','Inspect hair/glasses edges and backdrop surface uniformity.'),
 'IMG_0966':('Uncertain-seat protection no longer freezes the backdrop or floor; balloons and paper edge removed; local repair transition blended.','Supporting seat remains a review case; inspect the dark floor/contact detail directly under the forward shoe.'),
 'IMG_1030':('Softbox, wallpaper, balloons, spare box/stand and coloured clutter reflection selected; pedestal protected.','Inspect pedestal side borders and faint floor stains.'),
 'IMG_1083':('Full softbox footprint removed, including the tip below the floor horizon.','Fine paper creases and floor scuffs may remain.'),
 'IMG_1097':('Softbox removed; phone, hands and clothing retained.','Inspect phone/hair contact edge and fine braids.'),
 'IMG_1167':('Backdrop cleaned while seated pose and supporting fabric retained.','Floor fabric folds are part of the support; inspect the hair/background edge.'),
 'IMG_1232':('Non-studio room intentionally preserved.','Studio object removal is withheld on this room scene; furniture is not treated as clutter.'),
 'IMG_1241':('Image processed with conservative highlight handling.','Source clipping is irreversible; missing white-shirt/facial highlight detail is not recovered.')}
def link(label,path):return '['+label+'](<'+str(path).replace('\\','/')+'>)'
errors=[r for r in rows if r.get('error')]
summary={'date':'2026-11-18','pipelineRevision':7,'imagesProcessed':len(rows),'exceptions':len(errors),
 'originalsUnchangedDuringProcessing':sum(bool(r.get('originalUnchanged')) for r in rows),
 'fullResolutionExports':sum(bool(r.get('fullResolutionExport')) for r in rows),
 'subjectCoreChangesBeforeTone':sum(r.get('subjectCoreChangesBeforeTone',0) for r in rows),
 'outsideCleanupMaskChanges':sum(r.get('outsideCleanupMaskChanges',0) for r in rows),
 'qualityAcceptance':'Review required. Processing success and protected-pixel checks do not certify every reconstructed pixel as perfect.',
 'coverage':'One full-folder pass, followed by focused full-resolution regression retests after fixes. Each artifact retains its exact plan and runtime hashes.',
 'inferenceCache':'Repeated planning uses exact float32 raw matte tensors keyed by source preview pixels and pinned model SHA256. Reconstruction, planning and full-resolution export run for each photo.'}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# JeanMedia folder review — 18 November 2026','',
 f"{len(rows)} images processed from `D:/test22`; {len(errors)} exceptions. Source files remained unchanged during processing. See "+link('machine-readable checks',root/'summary.json')+'.','',
 'The source folder also contained already edited `-finished` files; these were included as separate inputs. No filename-specific repair masks or hand-drawn masks were used. Each folder contains full-resolution PNG export, original/edited preview, comparison, plan and runtime hashes. Repeated raw model inference uses an exact source/model-keyed cache.','',
 '**Acceptance remains open.** The one-button workflow is improved, but retained source clipping, fine fringes and difficult floor texture must not be called fully resolved merely because exports pass. The table records outcomes and remaining inspection points, including small details.','',
 '| Photo | Changes / preservation | Remaining limitation or inspection | Evidence |','|---|---|---|---|']
for r in sorted(rows,key=lambda r:r['name']):
 name=Path(r['name']).stem;folder=Path(r['artifactDirectory']);fixed,remaining=notes.get(name,notes.get(name.replace('-finished',''),('Processed with automatic planning.','Requires visual inspection.')))
 if name.endswith('-finished'):fixed='Previously edited input processed separately. '+fixed;remaining='Existing edit artifacts may already be baked into this input. '+remaining
 if r.get('error'):fixed='Processing failed: '+r['error']
 evidence=link('Before / after',folder/(name+'-compare.jpg'))
 details=sorted(folder.glob(name+'-detail-*.jpg'))
 if details:evidence+='; '+link('Native crop',details[0])
 lines.append('| '+r['name']+' | '+fixed+' | '+remaining+' | '+evidence+' |')
lines += ['', 'Validation: protected opaque subject interiors were compared before the shared RGB tone/detail pass. An unchanged interior does not prove that the mask classified every hair, seat or background pixel correctly. Original hashes, ICC profile and export dimensions were checked per source. Review warnings remain available in the app.']
(root/'REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
