"""Folder-only batch file operations; originals are never overwritten."""
import json
import os
from pathlib import Path
import re
import tempfile

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def edited_folder(folder):
    parent = Path(folder).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError('Choose a photo folder.')
    destination = parent / 'edited'
    if destination.is_symlink() or getattr(destination, 'is_junction', lambda: False)():
        raise ValueError('The edited folder must not be a link.')
    destination.mkdir(exist_ok=True)
    if not destination.is_dir() or destination.resolve().parent != parent:
        raise ValueError('Invalid edited folder.')
    return destination


def dispatch(action, data):
    if action == 'batch-list':
        folder = Path(data['folder']).resolve(strict=True)
        if not folder.is_dir():
            raise ValueError('Choose a photo folder.')
        sources = [p for p in folder.iterdir() if p.is_file() and not p.is_symlink()
                   and p.suffix.lower() in EXTENSIONS and p.resolve().parent == folder]
        sources.sort(key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name.casefold())])
        return {'folder': str(folder), 'sources': [str(p) for p in sources]}
    if action == 'batch-output':
        source = Path(data['source']).resolve(strict=True)
        if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
            raise ValueError('Choose a supported photo.')
        destination = edited_folder(source.parent)
        output = destination / (source.stem + '.jpg')
        count = 2
        while output.exists():
            output = destination / (source.stem + '-' + str(count) + '.jpg')
            count += 1
        return {'output': str(output)}
    if action == 'batch-report':
        run_id = data['runId']
        if not isinstance(run_id, str) or not re.fullmatch(r'[a-f0-9-]{36}', run_id):
            raise ValueError('Invalid batch identifier.')
        rows = data['items']
        if not isinstance(rows, list) or len(rows) > 10000:
            raise ValueError('Invalid batch report.')
        destination = edited_folder(data['folder'])
        output = destination / ('batch-' + run_id + '.json')
        if output.is_symlink():
            raise ValueError('Invalid report path.')
        descriptor, temporary = tempfile.mkstemp(prefix='.batch-', suffix='.tmp', dir=destination)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump({'version': 1, 'automaticExport': True, 'items': rows}, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {'report': str(output)}
    raise ValueError('Unknown batch operation.')
