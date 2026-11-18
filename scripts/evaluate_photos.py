"""Run local preservation gates, without retaining portrait pixels in the report."""
import argparse
import base64
import ctypes
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'worker'))
import numpy as np
from PIL import Image
from adaptive import auto_edit
from photo import load_photo
from preflight import inspect
from service import change_mask


def evaluate(path):
    start = time.perf_counter()
    image, _ = load_photo(path)
    dimensions = image.size
    image.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    before = np.asarray(image)
    _, after, plan, status = auto_edit(image,image)
    changes = np.asarray(change_mask(plan)) > 0
    after = np.asarray(after)
    outside = int(np.count_nonzero(np.any(before[~changes] != after[~changes],axis=1)))
    with Image.open(io.BytesIO(base64.b64decode(plan['layers']['protected']))) as layer:
        protected = np.asarray(layer) > 0
    # Surface cleaning is explicitly permitted inside protected furniture, not people.
    from floor import decode_mask
    for section, names in [('propRepairs',['mask']),('floorRepairs',['defects','extension'])]:
        if plan.get(section):
            for name in names:
                protected &= decode_mask(plan[section][name],protected.shape) == 0
    protected_changes = int(np.count_nonzero(np.any(before[protected] != after[protected],axis=1)))
    return {'name':path.name,'dimensions':dimensions,'seconds':round(time.perf_counter()-start,2),
            'changedFraction':float(changes.mean()),'outsideMaskChanges':outside,
            'protectedPixelChanges':protected_changes,'floor':plan['assessment'],
            'propsRepaired':plan.get('propRepairs',{}).get('count',0),'review':plan.get('review',[]),
            'preflight':inspect(image),'gatesPassed':outside==0 and protected_changes==0}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,default=Path('QA-REPORT.json'))
    args=parser.parse_args()
    rows=[]
    for source in sorted(args.directory.glob('IMG_*.JPG')):
        if '-finished' in source.stem.lower():
            continue
        try:
            row=evaluate(source)
        except Exception as error:
            row={'name':source.name,'skipped':True,'reason':str(error),'gatesPassed':False}
        rows.append(row)
        args.output.write_text(json.dumps({'results':rows,'acceptance':'Automated preservation checks only; human patch-quality approval pending.'},indent=2))
        print(row['name'], 'SKIPPED' if row.get('skipped') else 'PASS' if row['gatesPassed'] else 'FAIL',flush=True)
