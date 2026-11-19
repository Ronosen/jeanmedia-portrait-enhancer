"""Create a local before/after review from isolated packaged-build audits.

Photo data stays in the supplied QA folder, which must not be committed.
Usage: python scripts/delivery-report.py qa/run-root --notes qa/notes.json
"""
import argparse
import html
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
parser.add_argument('--notes',type=Path)
args=parser.parse_args()
notes=json.loads(args.notes.read_text()) if args.notes else {}
rows=[];sections=[]
for report in sorted(args.root.glob('*/run.json')):
    for row in json.loads(report.read_text()):
        name=Path(row['name']).stem
        problems=notes.get(name,['Visual review pending'])
        technical=not row.get('error') and row.get('originalUnchanged') and row.get('fullResolutionExport') and all(row.get(k)==0 for k in ('subjectCoreChangesBeforeTone','anchoredSubjectCoreChangesBeforeTone','outsideCleanupMaskChanges'))
        row['technicalPass']=bool(technical);row['visualNotes']=problems
        rows.append(row)
        text=''.join('<li>'+html.escape(x)+'</li>' for x in problems)
        if row.get('error'):text+='<li>'+html.escape(row['error'])+'</li>'
        image=(report.parent/(name+'-compare.jpg')).relative_to(args.root).as_posix()
        detail=(report.parent/(name+'-native-details.jpg')).relative_to(args.root).as_posix()
        sections.append(f'<section><h2>{html.escape(row["name"])}</h2><p>Technical checks: {"pass" if technical else "needs attention"}</p><ul>{text}</ul><img src="{html.escape(image,quote=True)}" alt="Original and edited"><details><summary>Native-resolution details</summary><img src="{html.escape(detail,quote=True)}" alt="Native pixel comparisons"></details></section>')
summary={'images':len(rows),'technicalPasses':sum(x['technicalPass'] for x in rows),'visuallyPerfectClaimed':False,'results':rows}
(args.root/'delivery-review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(args.root/'delivery-review.html').write_text('<!doctype html><meta charset="utf-8"><title>JeanMedia output review</title><style>body{background:#181a1b;color:#eee;font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px}section{border-top:1px solid #555;margin:30px 0;padding-top:12px}img{max-width:100%;height:auto}summary{cursor:pointer;padding:12px}</style><h1>JeanMedia output review</h1><p>Originals are on the left; edits are on the right. Technical preservation checks do not certify photographic perfection.</p>'+''.join(sections),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
