"""Run every source in its own worker process and retain failures for review."""
import json,os,subprocess,sys,time,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
source_root=Path(os.environ.get('JEANMEDIA_QA_SOURCE_DIR','D:/test22'))
sources=sorted(p for p in source_root.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff') and '-finished' not in p.stem.lower())
if os.environ.get('JEANMEDIA_QA_SOURCES_FILE'):
 sources=[Path(p) for p in json.loads(Path(os.environ['JEANMEDIA_QA_SOURCES_FILE']).read_text())['sources']]
priority=os.environ.get('JEANMEDIA_QA_PRIORITY','').split(',')
sources.sort(key=lambda p:(priority.index(p.stem) if p.stem in priority else len(priority),p.name))
env=dict(os.environ);env['JEANMEDIA_QA_MATTE_CACHE']=str(Path('qa/raw-matte-cache').resolve())
env['JEANMEDIA_QA_SOURCE_DIR']=str(source_root.resolve())
bundle=os.environ.get('JEANMEDIA_QA_BUNDLE','.')
(root/'scope.json').write_text(json.dumps({'sourceDirectory':str(source_root.resolve()),'bundle':str(Path(bundle).resolve()),'previewOnly':env.get('JEANMEDIA_QA_PREVIEW_ONLY')=='1','sources':[str(p.resolve()) for p in sources]},indent=2))
rows=json.loads((root/'run.json').read_text()) if (root/'run.json').exists() else []
retest=set(os.environ.get('JEANMEDIA_QA_RETEST','').split(','))
rows=[r for r in rows if not r.get('error') and Path(r['name']).stem not in retest]
done={r['name'] for r in rows};pending=[p for p in sources if p.name not in done]
def run_source(source):
 print('START',source.name,flush=True)
 folder=root/source.stem
 attempt=0
 while folder.exists():
  attempt+=1;folder=root/(source.stem+'-retry'+str(attempt))
 with (root/(folder.name+'.log')).open('w') as log:
  process=subprocess.run([sys.executable,'scripts/audit-review.py',bundle,str(folder),source.stem],env=env,stdout=log,stderr=subprocess.STDOUT)
 try:row=json.loads((folder/'run.json').read_text())[0]
 except Exception:row={'name':source.name,'error':'Worker did not complete','exitCode':process.returncode}
 row['artifactDirectory']=str(folder)
 print('DONE',source.name,row.get('error','OK'),flush=True)
 return row
workers=int(os.environ.get('JEANMEDIA_QA_WORKERS','1'))
with ThreadPoolExecutor(max_workers=workers) as pool:
 futures={pool.submit(run_source,p):p.name for p in pending}
 for future in as_completed(futures):
  rows.append(future.result())
  (root/'run.json').write_text(json.dumps(rows,indent=2))
  remaining=[name for f,name in futures.items() if not f.done()]
  (root/'progress.json').write_text(json.dumps({'total':len(sources),'completed':len(rows),'pending':remaining,'results':rows},indent=2))
(root/'progress.json').write_text(json.dumps({'total':len(sources),'completed':len(rows),'active':None,'results':rows},indent=2))
