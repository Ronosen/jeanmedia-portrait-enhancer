"""Build a source-indexed review ledger without modifying any photographs."""
import argparse,html,json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',action='append',default=[])
parser.add_argument('--candidate',action='append',default=[])
parser.add_argument('--notes',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
notes=json.loads(Path(args.notes).read_text(encoding='utf-8'))
baseline={};candidate={};scope=[]
for roots,target in ((args.baseline,baseline),(args.candidate,candidate)):
 for value in roots:
  root=Path(value)
  if target is baseline:
   scope.extend(json.loads((root/'scope.json').read_text())['sources'])
  if (root/'run.json').exists():
   for row in json.loads((root/'run.json').read_text()):target[row.get('source',row['name'])]=row
rows=[];cards=[]
def link(path):return Path(path).resolve().as_uri()
for source in dict.fromkeys(scope):
 name=Path(source).stem;old=baseline.get(source,{});new=candidate.get(source,{})
 findings=notes.get(name,['Visual review pending.'])
 checks={key:new.get(key) for key in ('fullResolutionExport','outputDimensions','originalUnchanged','subjectCoreChangesBeforeTone','anchoredSubjectCoreChangesBeforeTone','outsideCleanupMaskChanges','processingRevision','reconstructionCacheHit','reconstructionCacheReplayDifferences')}
 status='pending' if not new else 'processing error' if new.get('error') else 'rendered; visual review required'
 row={'source':source,'baselineError':old.get('error'),'candidateError':new.get('error'),'checks':checks,'findings':findings,'status':status,'baselineArtifacts':old.get('artifactDirectory'),'candidateArtifacts':new.get('artifactDirectory')}
 rows.append(row)
 pictures=[]
 for label,record,suffix in (('Original',old,'before'),('Current candidate',new,'after')):
  folder=record.get('artifactDirectory');path=Path(folder)/(name+'-'+suffix+'.png') if folder else None
  if path and path.exists():pictures.append('<figure><a href="'+link(record.get('output',path) if suffix=='after' else path)+'"><img loading="lazy" src="'+link(path)+'"></a><figcaption>'+label+'</figcaption></figure>')
 items=''.join('<li>'+html.escape(note)+'</li>' for note in findings)
 details=html.escape(json.dumps(checks,indent=2))
 cards.append('<article id="'+html.escape(name)+'"><h2>'+html.escape(name)+'</h2><p>'+html.escape(status)+'</p><div class="pair">'+''.join(pictures)+'</div><ul>'+items+'</ul><details><summary>Measured checks</summary><pre>'+details+'</pre></details></article>')
(output/'ledger.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
summary={'total':len(rows),'rendered':sum(bool(candidate.get(s)) and not candidate[s].get('error') for s in dict.fromkeys(scope)),'processingErrors':sum(bool(candidate.get(s,{}).get('error')) for s in dict.fromkeys(scope)),'visuallyPerfectClaimed':False}
(output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
page='''<!doctype html><meta charset="utf-8"><title>JeanMedia corpus review</title><style>
body{background:#17191b;color:#eee;font:16px system-ui;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}article{background:#232629;padding:18px;border-radius:12px}.pair{display:flex;gap:12px}figure{flex:1;margin:0}img{width:100%;height:430px;object-fit:contain;background:#101112}figcaption{color:#bbb}li{margin:8px 0}pre{white-space:pre-wrap}a{color:#9ed9ff}</style><h1>JeanMedia photo-by-photo review</h1><p>Processing checks do not certify visual perfection. Click an edited image to inspect its exported file. Originals are unchanged.</p><p>'''+html.escape(json.dumps(summary))+'</p><main>'+''.join(cards)+'</main>'
(output/'index.html').write_text(page,encoding='utf-8')
print(json.dumps(summary))
