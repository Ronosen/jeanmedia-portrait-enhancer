"""Package a verified Windows build and publish it without bundling user state.

Prepare locally first; --publish creates a draft, uploads verified assets, then
publishes it. Authentication stays in memory using Git's credential helper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOCS = ('README.md', 'USER-GUIDE.md', 'MODEL-MANIFEST.json', 'EDGE-REFINEMENT.md', 'EDGE-DEPENDENCIES.json',
        'MODEL-DECISIONS.md', 'LICENSE-INVENTORY.json', 'SEPARATED-WORKFLOW.md', 'SCENE-INTENT.md')
SKIP = {'.cache', '__pycache__', '.runtime-cache', 'WebViewData', '.git', 'tests', 'test'}
PRIVATE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff', '.heic', '.dng', '.cr2',
                    '.cr3', '.jeanmedia', '.log', '.pyc', '.pem', '.key', '.pfx'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def package(bundle, tag, commit):
    required = ['jeanmedia.exe', 'python/python.exe', 'worker/photo.py',
                'worker/models/lama_fp32.onnx', *DOCS]
    for name in required:
        if not (bundle / name).is_file():
            raise ValueError('Missing package file: ' + name)
    checks = json.loads((bundle / 'SELF-TEST.json').read_text(encoding='utf-8-sig'))
    if not checks.get('selfTest') or not all(x['ok'] for x in checks['selfTest']):
        raise ValueError('Package self-tests have not passed')
    if not checks.get('reconstructionModel', {}).get('integrityVerified'):
        raise ValueError('Reconstruction model integrity was not verified')
    # The archive must represent this checkout, including its worker and UI.
    if sha256(bundle / 'jeanmedia.exe') != sha256(ROOT / 'src-tauri/target/debug/jeanmedia.exe'):
        raise ValueError('Package executable differs from the verified build')
    for source in (ROOT / 'worker').glob('*.py'):
        if sha256(source) != sha256(bundle / 'worker' / source.name):
            raise ValueError('Packaged worker differs: ' + source.name)
    target = ROOT / 'release' / ('JeanMedia-' + tag + '-windows-x64-portable.zip')
    if not target.exists():
        with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            count = 0
            for path in sorted(bundle.rglob('*')):
                relative = path.relative_to(bundle)
                if not path.is_file() or any(part in SKIP for part in relative.parts):
                    continue
                if relative.parts[0] not in {'python', '.worker-deps', 'worker', 'jeanmedia.exe', *DOCS}:
                    continue
                if path.suffix.lower() in PRIVATE_SUFFIXES or path.name.startswith('.env'):
                    continue
                if not path.resolve().is_relative_to(bundle):
                    raise ValueError('Package file escapes the build directory')
                archive.write(path, 'JeanMedia/' + relative.as_posix())
                count += 1
            archive.writestr('JeanMedia/RELEASE.json', json.dumps({'tag': tag, 'commit': commit}, indent=2))
        print('Packaged', count, 'files', flush=True)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed')
        provenance = json.loads(archive.read('JeanMedia/RELEASE.json'))
        if provenance != {'tag': tag, 'commit': commit}:
            raise ValueError('Existing archive belongs to another release')
    checksum = target.with_suffix('.zip.sha256')
    checksum.write_text(sha256(target) + '  ' + target.name + '\n', encoding='ascii')
    return target, checksum


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def publish(repo, tag, title, notes, commit, assets, stage_only=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    credentials = subprocess.run(['git', 'credential', 'fill'], cwd=ROOT,
        input='protocol=https\nhost=github.com\n\n', text=True,
        capture_output=True, env=env, timeout=60)
    fields = dict(line.split('=', 1) for line in credentials.stdout.splitlines() if '=' in line)
    token = fields.get('password')
    if credentials.returncode or not token:
        raise ValueError('Saved GitHub authentication is unavailable; no release was published')
    api = 'https://api.github.com/repos/' + repo
    opener = urllib.request.build_opener(NoRedirect)

    def request(method, url, data=None, file=None):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'api.github.com', 'uploads.github.com'} or not parsed.path.startswith('/repos/' + repo + '/'):
            raise ValueError('Unexpected GitHub API destination')
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                   'User-Agent': 'JeanMedia-release', 'X-GitHub-Api-Version': '2026-03-10'}
        if file:
            headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(file.stat().st_size)})
            with file.open('rb') as stream:
                with opener.open(urllib.request.Request(url, data=stream, headers=headers, method=method), timeout=600) as response:
                    return json.load(response)
        if data is not None:
            headers['Content-Type'] = 'application/json'
        body = json.dumps(data).encode() if data is not None else None
        with opener.open(urllib.request.Request(url, data=body, headers=headers, method=method), timeout=60) as response:
            return json.load(response)

    remote = request('GET', api + '/commits/' + commit)
    if remote['sha'] != commit:
        raise ValueError('Release commit is not present on GitHub')
    try:
        release = request('GET', api + '/releases/tags/' + urllib.parse.quote(tag, safe=''))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        # GitHub's tag endpoint can omit drafts. Resume a staged upload by
        # listing authenticated releases before creating another draft.
        candidates=[]
        page=1
        while True:
            batch=request('GET',api+'/releases?per_page=100&page='+str(page))
            candidates.extend(item for item in batch if item['tag_name']==tag)
            if len(batch)<100:break
            page+=1
        if candidates:
            release=max(candidates,key=lambda item:len(item.get('assets',[])))
        else:
            release = request('POST', api + '/releases', {'tag_name': tag, 'target_commitish': commit,
                'name': title, 'body': notes, 'draft': True, 'prerelease': True})
    if not release['draft'] or release['target_commitish'] != commit:
        raise ValueError('Release/tag already exists; choose a new version instead of replacing it')
    existing = {item['name']: item for item in request('GET', release['assets_url'] + '?per_page=100')}
    for asset in assets:
        print('Uploading', asset.name, asset.stat().st_size, 'bytes', flush=True)
        result = existing.get(asset.name)
        if result is None:
            result = request('POST', release['upload_url'].split('{', 1)[0] + '?' + urllib.parse.urlencode({'name': asset.name}), file=asset)
        if result.get('state') != 'uploaded' or result['size'] != asset.stat().st_size or result.get('digest') != 'sha256:' + sha256(asset):
            raise ValueError('Uploaded asset verification failed: ' + asset.name)
        print('Verified', asset.name, flush=True)
    if stage_only:
        print('Staged verified draft assets; publication is still pending.', flush=True)
        return
    published = request('PATCH', release['url'], {'draft': False, 'body': notes})
    verified = request('GET', published['url'])
    if verified['draft'] or len(verified['assets']) < len(assets):
        raise ValueError('Release publication could not be verified')
    print('Published', verified['html_url'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--notes', type=Path, required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--publish', action='store_true')
    mode.add_argument('--stage', action='store_true', help='Upload verified assets but keep the release draft for final QA')
    args = parser.parse_args()
    if not re.fullmatch(r'v[0-9][A-Za-z0-9._-]*', args.tag):
        raise ValueError('Use a version tag starting with v and a digit')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Commit completed changes before releasing')
    match = re.fullmatch(r'https://github.com/([\w.-]+/[\w.-]+?)(?:\.git)?', git('remote', 'get-url', 'origin'))
    if not match:
        raise ValueError('Expected an authorized HTTPS GitHub origin')
    commit = git('rev-parse', 'HEAD')
    assets = package(args.bundle.resolve(), args.tag, commit)
    print('Verified assets:', ', '.join(p.name for p in assets), flush=True)
    if args.publish or args.stage:
        publish(match[1], args.tag, args.title, args.notes.read_text(encoding='utf-8'), commit, assets, args.stage)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(str(error)) from None
