"""Download pinned LIBERO datasets and backbones to CPFS with scoped DNS repair."""
import argparse
import concurrent.futures
import functools
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import time
import threading
import urllib.request

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
_getaddrinfo = socket.getaddrinfo

@functools.lru_cache(maxsize=64)
def public_addresses(host):
    with urllib.request.urlopen('https://dns.google/resolve?name=' + host + '&type=A', timeout=15) as response:
        records = json.load(response)
    addresses = [r['data'] for r in records.get('Answer', []) if r['type'] == 1]
    if not addresses or not all(ipaddress.ip_address(ip).is_global for ip in addresses):
        raise RuntimeError('No public DNS answer for ' + host)
    return addresses

def resolve(host, port, *args, **kwargs):
    if isinstance(host, str) and (host == 'huggingface.co' or host.endswith(('.huggingface.co', '.hf.co', '.xethub.hf.co'))):
        records = []
        for ip in public_addresses(host):
            records.extend(_getaddrinfo(ip, port, *args, **kwargs))
        return records
    return _getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = resolve

from huggingface_hub import HfApi, get_token, hf_hub_url
import requests

_sessions = threading.local()

def file_session():
    if not hasattr(_sessions, 'session'):
        _sessions.session = requests.Session()
    return _sessions.session

def download_large(url, headers, path, entry):
    temporary = path.with_name(path.name + '.partial')
    # Resume the contiguous prefix left by a previous streaming download.
    prefix = temporary.stat().st_size if temporary.exists() else 0
    chunk_size = 64 * 1024 * 1024
    ranges = [(start, min(start+chunk_size, entry.size)-1) for start in range(prefix,entry.size,chunk_size)]
    part_dir = path.with_name(path.name + '.parts')
    part_dir.mkdir(exist_ok=True)
    def fetch(bounds):
        start,end = bounds
        part = part_dir / f'{start}-{end}'
        if part.exists() and part.stat().st_size == end-start+1:
            return part
        for attempt in range(12):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset == end-start+1:
                    return part
                resume_start = start + offset
                with requests.get(url, headers={**headers,'Range':f'bytes={resume_start}-{end}'},stream=True,timeout=(20,60)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {resume_start}-{end}/'):
                        raise ValueError('Server did not honor requested byte range')
                    with part.open('ab') as output:
                        for data in r.iter_content(1024*1024): output.write(data)
                assert part.stat().st_size == end-start+1
                return part
            except Exception as error:
                print(f'{path.name} range {start}-{end}: retry {attempt+1} ({type(error).__name__})',flush=True)
                if attempt == 11: raise
                time.sleep(min(2**attempt,15))
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for count,_ in enumerate(pool.map(fetch,ranges),1):
            if count%10 == 0: print(f'{path.name}: {count}/{len(ranges)} chunks',flush=True)
    with temporary.open('ab') as output:
        for start,end in ranges:
            with (part_dir / f'{start}-{end}').open('rb') as source:
                shutil.copyfileobj(source,output,4*1024*1024)
    assert temporary.stat().st_size == entry.size
    digest = hashlib.sha256()
    with temporary.open('rb') as source:
        while chunk := source.read(8*1024*1024): digest.update(chunk)
    assert digest.hexdigest() == entry.lfs.sha256, 'Model SHA256 mismatch'
    temporary.replace(path)

def download_file(repo, typ, sha, destination, entry):
    path = destination / entry.rfilename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == entry.size:
        return
    token = get_token()
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    url = hf_hub_url(repo, entry.rfilename, repo_type=typ, revision=sha)
    if entry.size > 512*1024*1024:
        if repo == 'Qwen/Qwen3-VL-4B-Instruct' and os.environ.get('QWEN_USE_MODELSCOPE') == '1':
            mirror = f'https://modelscope.cn/models/{repo}/resolve/master/{entry.rfilename}'
            # Never forward HF credentials. Check the official mirror's content
            # hash and length against the pinned HF revision before resuming.
            with requests.get(mirror, headers={'Range':'bytes=0-0'}, timeout=(20,60), stream=True) as check:
                check.raise_for_status()
                assert check.headers.get('x-linked-etag') == entry.lfs.sha256
                assert check.headers.get('Content-Range') == f'bytes 0-0/{entry.size}'
                url = check.url
            headers = {}
            print(f'{entry.rfilename}: verified identical SHA256 on official Qwen ModelScope mirror',flush=True)
        return download_large(url, headers, path, entry)
    for attempt in range(12):
        try:
            temporary = path.with_name(path.name + '.partial')
            with file_session().get(url, headers=headers, stream=True, timeout=(20,60)) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(4*1024*1024):
                        output.write(chunk)
                        digest.update(chunk)
            if temporary.stat().st_size != entry.size:
                raise ValueError(f'Size mismatch for {entry.rfilename}: {temporary.stat().st_size} != {entry.size}')
            if entry.lfs and digest.hexdigest() != entry.lfs.sha256:
                raise ValueError('SHA256 mismatch for ' + entry.rfilename)
            temporary.replace(path)
            return
        except requests.HTTPError as error:
            if error.response.status_code in (401,403):
                raise
            if attempt == 11:
                raise
            time.sleep(5 * (attempt+1))
        except Exception as error:
            print(f'{entry.rfilename}: retry {attempt+1} ({type(error).__name__})',flush=True)
            if attempt == 11:
                raise
            time.sleep(min(2 ** attempt, 15))

def download(item):
    repo, typ, revision, destination = item
    info = HfApi().repo_info(repo, repo_type=typ, revision=revision, files_metadata=True, timeout=40)
    size = sum(f.size or 0 for f in info.siblings)
    print(f'{repo}: sha={info.sha} files={len(info.siblings)} size={size/1e9:.2f}GB gated={info.gated}', flush=True)
    destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    entries = [f for f in info.siblings if not f.rfilename.endswith('.md') and f.rfilename != '.gitattributes']
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_file, repo, typ, info.sha, destination, entry) for entry in entries]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 100 == 0 or count == len(entries):
                print(f'{repo}: {count}/{len(entries)} verified files', flush=True)
    if typ == 'dataset':
        shutil.copy2(ROOT / 'starVLA/examples/LIBERO/train_files/modality.json', destination / 'meta/modality.json')
    manifest = {'repo': repo, 'revision': info.sha, 'requested_revision': revision,
                'bytes': size, 'files': [{'path': f.rfilename, 'size': f.size,
                                         'sha256': f.lfs.sha256 if f.lfs else None} for f in info.siblings]}
    (destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE ' + repo, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=['datasets', 'qwen', 'cosmos'])
    args = parser.parse_args()
    if args.target == 'datasets':
        items = [(f'IPEC-COMMUNITY/libero_{s}_no_noops_1.0.0_lerobot', 'dataset', 'main',
                  f'datasets/libero/libero_{s}_no_noops_1.0.0_lerobot') for s in ['spatial', 'object', 'goal', '10']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, items))
    elif args.target == 'qwen':
        download(('Qwen/Qwen3-VL-4B-Instruct', 'model', 'main', 'models/Qwen3-VL-4B-Instruct'))
    else:
        download(('nvidia/Cosmos-Predict2.5-2B', 'model', 'diffusers/base/post-trained', 'models/Cosmos-Predict2.5-2B'))
