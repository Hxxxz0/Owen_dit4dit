"""Lossless current-image sidecar; contains no model features or future frames."""
import argparse
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from .common import atomic_json, file_sha, fingerprint, load_config
from .data import CachedLibero, IndexedLibero


RECIPE = "nearest-source-timestamp/rgb24/bilinear224-align_corners_false/legacy-global-range/uint8-round/primary+wrist/v2"


def decode_current(job):
    torch.set_num_threads(1)
    path = Path(job["output"])
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (job["stop"]-job["start"], 224, 448, 3) or arr.dtype != np.uint8:
            raise ValueError(f"Invalid image sidecar: {path}")
        return {k: v for k, v in job.items() if k not in ("timestamps", "output")}
    views = []
    recovery = []
    for source in job["sources"]:
        with av.open(source) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            decoded, errors = [], 0
            for packet in container.demux(stream):
                try:
                    decoded.extend(packet.decode())
                except av.error.InvalidDataError:
                    errors += 1
            if not decoded or errors > 1:
                raise ValueError(f"Unrecoverable video {source}")
            distances = np.abs(np.asarray(job["timestamps"])[:, None] - np.asarray([f.time for f in decoded]))
            indices = distances.argmin(axis=1)
            if np.any(distances[np.arange(len(indices)), indices] > 1/float(stream.average_rate)+1e-5):
                raise ValueError(f"Source timestamp gap exceeds one frame: {source}")
            pixels = torch.from_numpy(np.stack([decoded[i].to_ndarray(format="rgb24") for i in indices]))
            resized = F.interpolate(pixels.permute(0, 3, 1, 2).float()/255,
                                    size=(224, 224), mode="bilinear", align_corners=False)
            views.append(resized.permute(0, 2, 3, 1).numpy())
            recovery.append(errors)
    array = np.concatenate(views, axis=2)
    # Reproduce current_views exactly, including its historical >1 range test
    # when FP32 bilinear interpolation overshoots 1 by a rounding epsilon.
    rescaled = array.max(axis=(1, 2, 3)) > 1
    array[rescaled] /= 255
    array = (np.clip(array, 0, 1)*255).round().astype(np.uint8)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.save(stream, array)
    path.with_suffix(".tmp").replace(path)
    return {**{k: v for k, v in job.items() if k not in ("timestamps", "output")},
            "invalid_packets": recovery, "legacy_range_rescaled_frames": int(rescaled.sum()),
            "bytes": path.stat().st_size}


class CurrentImageLibero(Dataset):
    """Reuse only frozen VAE targets and labels; Qwen always consumes pixels."""
    def __init__(self, cfg, allow_partial=False):
        self.cached = CachedLibero(cfg, allow_partial)
        self.shards = self.cached.shards
        self.index_root = self.cached.root
        self.root = Path(cfg.cache.current_images)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if (not manifest["complete"] or manifest["fingerprint"] != fingerprint(cfg)
                or manifest["recipe"] != RECIPE or manifest["samples"] != len(self.cached)):
            raise ValueError("Current-image cache provenance or coverage mismatch")
        self.entries = manifest["episodes"]
        self.starts = [r["start"] for r in self.entries]
        self.suite_ranges = self.cached.suite_ranges
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, index):
        sample = {k: v for k, v in self.cached[index].items() if k != "qwen"}
        entry_id = bisect_right(self.starts, index)-1
        entry = self.entries[entry_id]
        offset = index-entry["start"]
        if (index >= entry["stop"] or sample["dataset_index"] != entry["dataset_index"]
                or sample["episode"] != entry["episode"] or sample["frame"] != entry["frames"][offset]):
            raise ValueError("Image and VAE cache sample identities differ")
        if entry_id not in self.loaded:
            self.loaded[entry_id] = np.load(self.root / entry["file"], mmap_mode="r")
            while len(self.loaded) > 2:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(entry_id)
        sample["image"] = [np.array(self.loaded[entry_id][offset], copy=True)]
        return sample


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.cache.current_images)
    root.mkdir(parents=True, exist_ok=True)
    raw_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    raw_cfg.datasets.vla_data.video_delta_indices = [0]
    raw_cfg.datasets.vla_data.action_video_freq_ratio = 1
    raw = IndexedLibero(raw_cfg)
    cache = CachedLibero(cfg)
    if len(raw) != len(cache) or list(map(list, raw.suite_ranges)) != cache.suite_ranges:
        raise ValueError("Raw and frozen cache indexing differ")
    metadata = []
    index_files = []
    for shard in cache.shards:
        path = Path(cfg.cache.root) / Path(shard["file"]).with_suffix(".index.json")
        metadata.extend(json.loads(path.read_text()))
        index_files.append({"file": str(path), "sha256": file_sha(path)})
    jobs = []
    for i, (record, meta) in enumerate(zip(raw.records, metadata)):
        dsid, episode, frame = record
        if (meta["sample_index"], meta["dataset_index"], meta["episode"], meta["frame"]) != (i, dsid, episode, frame):
            raise ValueError(f"Cache index mismatch at {i}")
        if not jobs or (jobs[-1]["dataset_index"], jobs[-1]["episode"]) != (dsid, episode):
            ds = raw.mixture.datasets[dsid]
            timestamps = ds.get_trajectory_data(episode)["timestamp"].to_numpy()
            filename = f"dataset_{dsid}_episode_{episode:06d}.npy"
            jobs.append({"dataset_index": dsid, "episode": episode, "start": i, "stop": i,
                "frames": [], "timestamps": [], "output": str(root / filename), "file": filename,
                "sources": [str(ds.get_video_path(episode, key.removeprefix("video.")))
                            for key in ds.modality_keys["video"]]})
        jobs[-1]["frames"].append(frame)
        jobs[-1]["timestamps"].append(float(timestamps[frame]))
        jobs[-1]["stop"] = i+1
    atomic_json(root / "source_index.json", {"fingerprint": fingerprint(cfg), "recipe": RECIPE,
                "source_indices": index_files, "jobs": jobs})
    begin = time.monotonic()
    entries = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for entry in pool.map(decode_current, jobs):
            entries.append(entry)
            if len(entries) % 20 == 0:
                progress = {"episodes": len(entries), "total": len(jobs), "elapsed_seconds": time.monotonic()-begin}
                atomic_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    # Compare the fast decoder against the established training preprocessing,
    # including the source with the known isolated invalid packet.
    from .backbones import current_views
    probes = {a for a, _ in raw.suite_ranges} | {b-1 for _, b in raw.suite_ranges}
    probes |= {i for i, (d, e, f) in enumerate(raw.records) if e == 82 and f == 105}
    probes |= set(np.random.default_rng(42).integers(0, len(raw), size=16).tolist())
    starts = [r["start"] for r in entries]
    for i in sorted(probes):
        expected = np.concatenate(current_views(raw[i]), axis=1)
        entry = entries[bisect_right(starts, i)-1]
        actual = np.load(root / entry["file"], mmap_mode="r")[i-entry["start"]]
        if not np.array_equal(expected, actual):
            raise AssertionError(f"Current image pixels differ from established preprocessing at {i}")
    atomic_json(root / "manifest.json", {"complete": True, "fingerprint": fingerprint(cfg),
        "recipe": RECIPE, "samples": len(raw), "episodes": entries,
        "pixel_identity_probes": sorted(probes), "source_index_sha256": file_sha(root / "source_index.json")})
    print("CURRENT_IMAGES_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
