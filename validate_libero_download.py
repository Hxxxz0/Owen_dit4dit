"""Check downloaded LIBERO files and exercise StarVLA's real data pipeline."""
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

ROOT = Path(__file__).resolve().parent
cfg = OmegaConf.load(ROOT / 'configs/starvla_libero_ppu16.yaml')
reports = {}
for suite in ('spatial', 'object', 'goal', '10'):
    folder = ROOT / f'datasets/libero/libero_{suite}_no_noops_1.0.0_lerobot'
    manifest = json.loads((folder / 'download_manifest.json').read_text())
    for entry in manifest['files']:
        if entry['path'].endswith('.md') or entry['path'] == '.gitattributes':
            continue
        path = folder / entry['path']
        assert path.is_file() and path.stat().st_size == entry['size'], path
        if entry.get('sha256'):
            with path.open('rb') as source:
                assert hashlib.file_digest(source, 'sha256').hexdigest() == entry['sha256'], path
    info = json.loads((folder / 'meta/info.json').read_text())
    parquets = sorted(folder.glob('data/**/*.parquet'))
    videos = list(folder.glob('videos/**/*.mp4'))
    frames = sum(pq.read_metadata(path).num_rows for path in parquets)
    assert len(parquets) == info['total_episodes']
    assert len(videos) == info['total_videos']
    assert frames == info['total_frames']
    reports[folder.name] = {'revision': manifest['revision'], 'episodes': len(parquets),
                            'frames': frames, 'videos': len(videos)}
    print(f'FILES_OK {suite}: {len(parquets)} episodes, {frames} frames, {len(videos)} videos', flush=True)

mixture = get_vla_dataset(cfg.datasets.vla_data)
for dataset in mixture.datasets:
    samples = []
    for index in (0, len(dataset)//2, len(dataset)-1):
        sample = dataset[index]
        action = np.asarray(sample['action'])
        assert action.shape == (8, 7), action.shape
        assert np.isfinite(action).all()
        assert len(sample['image']) == 2
        assert all(image.size == (224, 224) for image in sample['image'])
        assert isinstance(sample['lang'], str) and sample['lang']
        samples.append({'index': index, 'action_shape': list(action.shape),
                        'image_sizes': [list(image.size) for image in sample['image']],
                        'instruction': sample['lang']})
    reports[dataset.dataset_name]['decoded_samples'] = samples
    print(f'LOADER_OK {dataset.dataset_name}: 3 real samples, 2 cameras and 8x7 actions', flush=True)

output = ROOT / 'logs/libero_data_validation.json'
output.write_text(json.dumps(reports, indent=2))
print(f'PASS: four complete LIBERO suites; report: {output}', flush=True)
