"""Combine the full corpus run with successful focused regression retests."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();base=root/'base-run.json'
if not base.exists():base.write_text((root/'run.json').read_text())
rows=json.loads(base.read_text());by_name={r['name']:r for r in rows}
for target in sys.argv[2:]:
 folder=Path(target).resolve()
 for row in json.loads((folder/'run.json').read_text()):
  if row.get('error'):raise ValueError(row)
  previous=by_name.get(row['name'])
  if previous and previous.get('sourceSha256')!=row['sourceSha256']:raise ValueError('Source changed between retests: '+row['name'])
  row['artifactDirectory']=str(folder);by_name[row['name']]=row
rows=sorted(by_name.values(),key=lambda r:r['name'])
assert all(not r.get('error') and r.get('originalUnchanged') and r.get('fullResolutionExport') and r.get('subjectCoreChangesBeforeTone')==0 and r.get('outsideCleanupMaskChanges')==0 for r in rows)
(root/'run.json').write_text(json.dumps(rows,indent=2))
(root/'progress.json').write_text(json.dumps({'total':len(rows),'completed':len(rows),'active':None,'results':rows},indent=2))
