"""QA-only reuse of exact reconstruction inputs, never filename-specific edits."""
import ast,hashlib,json
from pathlib import Path

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def dependency_key(bundle):
 worker=Path(bundle)/'worker'
 tree=ast.parse((worker/'separation.py').read_text(encoding='utf-8'))
 functions=[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('render','composite')]
 basis=ast.parse((worker/'studio.py').read_text(encoding='utf-8'))
 functions.extend(ast.dump(n,include_attributes=False) for n in basis.body if isinstance(n,ast.FunctionDef) and n.name=='background_basis')
 modules={name:hashlib.sha256((worker/name).read_bytes()).hexdigest() for name in ('neural.py','floor.py','floor_context.py','backdrop_finish.py','scene_evidence.py','props.py','photo.py','runtime.py','inpainting.py')}
 return digest({'functions':functions,'modules':modules})

def input_key(plan):
 if not plan.get('separation'):return None
 layers=plan['layers']
 fields={name:layers.get(name) for name in ('shape','source','foregroundAlpha','protected','backdrop','wallSurface','clutter','sourceContact','sourceContactBlend')}
 floor=plan.get('floorRepairs',{})
 return digest({'layers':fields,'floor':plan['floor'],'floorRepairs':{name:floor.get(name) for name in ('version','shape','defects','extension','color')},
                'coefficients':plan.get('coefficients'),'low':plan.get('low'),'high':plan.get('high'),
                'adaptiveRevision':plan.get('adaptiveRevision'),'separation':plan['separation'],'propRepairs':plan.get('propRepairs'),
                'runtime':plan.get('runtime',{})})

def build(roots,target):
 index={};dependencies={}
 for value in roots:
  root=Path(value)
  for row in json.loads((root/'run.json').read_text()):
   if row.get('error') or not row.get('fullResolutionExport') or not row.get('originalUnchanged'):continue
   folder=Path(row['artifactDirectory']);plan_path=folder/(Path(row['name']).stem+'-plan.json')
   provenance=json.loads((folder/'runtime.json').read_text());bundle=provenance['bundle']
   if bundle not in dependencies:dependencies[bundle]=dependency_key(bundle)
   plan=json.loads(plan_path.read_text());key=input_key(plan)
   if key:index[row['sourceSha256']]={'plan':str(plan_path.resolve()),'output':row['output'],'inputKey':key,'dependencyKey':dependencies[bundle]}
 Path(target).write_text(json.dumps(index,indent=2),encoding='utf-8')
 return len(index)

if __name__=='__main__':
 import sys
 print('Indexed',build(sys.argv[2:],sys.argv[1]),'verified exports')
