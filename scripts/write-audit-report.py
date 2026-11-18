"""Create a linked review checklist and CSV from reviewed observations."""
import json,csv,collections
from pathlib import Path
root=Path('qa/review-20261113').resolve()
issues=json.loads((root/'issues.json').read_text());runs=json.loads((root/'run.json').read_text())
assert len(runs)==16 and len({r['name'] for r in runs})==16
assert all(not r.get('error') and r['originalUnchanged'] and r['outsideCleanupMaskChanges']==0 for r in runs)
assert {r['name'] for r in runs}=={i['image'] for i in issues}
for i in issues:assert (root/i['evidence']).is_file(),i
types=collections.Counter(i['kind'] for i in issues);priority=collections.Counter(i['priority'] for i in issues)
summary={'date':'2026-11-13','imagesRun':16,'fullResolutionExports':16,'pipelineExceptions':0,
 'originalChecksumsUnchanged':16,'outsideCleanupMaskChanges':0,'visualFindings':len(issues),
 'findingsByType':dict(types),'findingsByPriority':dict(priority),
 'acceptance':'Unresolved; this is an audit, no pipeline fixes were made during this review.',
 'scope':'All full-frame pairs plus targeted native-pixel crops. Not an exhaustive every-pixel guarantee.'}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
with (root/'issues.csv').open('w',encoding='utf-8-sig',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(issues[0]));writer.writeheader();writer.writerows(issues)
def link(label,p):return '['+label+'](<'+str(p).replace('\\','/')+'>)'
lines=['# JeanMedia — per-image defect checklist','',
 'Review date: 13 November 2026. Input: `D:/test22`.', '',
 f'**16/16 images processed and exported at 4000 × 6000. {len(issues)} findings recorded. No pipeline exceptions. All 16 source SHA256 checksums remained unchanged.**','',
 'This records the current installed app, not an experimental source checkout. Runtime/model versions and worker hashes are pinned in '+link('runtime.json',root/'runtime.json')+'. The run used the app’s automatic cleanup, correction, sharpening and sRGB full-resolution export functions. Preflight warnings are included. Pipeline code was not changed during this audit.','',
 'All full-frame original/edited pairs were visually compared, followed by targeted native-pixel crops of repair edges, faces, props and floor. This is a detailed review of visible findings, not a guarantee that every possible pixel-level flaw has been found. Repeated scratches/specks in one region are grouped into a regional finding; natural skin texture, clothing design and real hair are not automatically defects.','',
 '## Reading the findings','',
 '- **P1:** damage to subject detail or intentional furniture; address first.',
 '- **P2:** obvious cleanup failure, conspicuous artifact, unsupported cleanup or major source limitation.',
 '- **P3:** minor residual marks or appearance decisions.',
 '- **introduced:** visible artifact/change caused by the edit.',
 '- **residual:** visible unwanted mark/object left in the output.',
 '- **withheld:** cleanup did not run because its safety gate preserved the scene.',
 '- **source-limit:** source clipping/focus problems remain; missing detail cannot simply be certified as recovered.',
 '- **review:** aesthetic/colour/prop decision, not a confirmed defect to remove blindly.','',
 'These are checklist entries for later fixing. None is marked fixed merely because an export succeeded.','',
 '## Counts','', '| Type | Count |','|---|---:|']
lines += [f'| {kind} | {count} |' for kind,count in sorted(types.items())]
lines += ['', '## Highest-priority evidence','']
for i in issues:
 if i['priority']=='P1':lines += [f"- **{i['id']} — {i['image']}**: {i['finding']} "+link('Native comparison',root/i['evidence'])+'.']
lines += ['', '## System problems to investigate next','',
 '1. Protection masks miss actual subject/seat boundaries. A zero-change result outside the computed mask does **not** prove that the mask excludes the real subject: the boot, seat and temple findings demonstrate this.',
 '2. Generated patches lose native grain/detail. Upscaled low-resolution repairs and hard mask boundaries produce blur islands, stair steps and visible seams.',
 '3. Floor repair is inconsistent: some areas are over-smoothed while tape, scuffs and dirt remain elsewhere.',
 '4. White/neutral colour and local highlight handling need review; global correction can shift colours without solving localized lighting problems.',
 '5. Seat uncertainty disables cleanup for IMG_0731, IMG_0966 and IMG_1030; non-studio preservation disables local room cleanup for IMG_1232.',
 '6. The current automated checks measure preservation relative to the planner’s masks, not semantic correctness or photographic quality.','',
 '## Image index','', '| Image | Findings | Cleanup engine | Comparison |','|---|---:|---|---|']
for r in runs:
 name=Path(r['name']).stem;items=[i for i in issues if i['image']==r['name']]
 lines.append(f"| {r['name']} | {len(items)} | {r['engine']} | "+link('Before / after',root/(name+'-compare.jpg'))+' |')
for r in runs:
 name=Path(r['name']).stem;items=[i for i in issues if i['image']==r['name']]
 lines+=['',f"## {r['name']}",'',
  link('Original / edited comparison',root/(name+'-compare.jpg'))+' · '+link('Full-resolution edited PNG',root/(name+'-edited.png'))+' · '+link('Cleanup mask',root/(name+'-mask.png')),'',
  '**Worker review:** '+('; '.join(r['review']) if r['review'] else 'No automatic warning — this is not a visual pass.')+'','']
 for i in items:
  lines.append(f"- [ ] **{i['id']} · {i['priority']} · {i['kind']}** — **{i['location']}**. {i['finding']} "+link('Evidence',root/i['evidence'])+'.')
 lines+=['','Native-pixel comparisons: '+ ' · '.join(link(p.stem.replace(name+'-detail-',''),p) for p in sorted(root.glob(name+'-detail-*.png')))]
lines+=['','## Supporting files','',
 link('Machine-readable findings',root/'issues.json')+' · '+link('CSV checklist',root/'issues.csv')+' · '+link('Run log/results',root/'run.json')+' · '+link('Summary',root/'summary.json'),'',
 'Original files remain in `D:/test22`. Audit exports are separate working outputs, not approved final photographs.']
Path('PIPELINE-DEFECTS-2026-11-13.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
