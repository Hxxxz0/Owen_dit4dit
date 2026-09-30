"""Stable sample indexing, sharded tensor cache, and equal-suite sampling."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .common import SUITES, fingerprint, seed_for


class IndexedLibero(Dataset):
    def __init__(self, cfg):
        from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset
        self.mixture = get_vla_dataset(cfg.datasets.vla_data)
        self.mixture.retry_same_index = True
        self.records = []
        self.suite_ranges = []
        self.episode_lengths = []
        for i, ds in enumerate(self.mixture.datasets):
            start = len(self.records)
            self.records.extend((i, int(ep), int(step)) for ep, step in ds.all_steps)
            self.suite_ranges.append((start, len(self.records)))
            self.episode_lengths.append(dict(zip(map(int, ds.trajectory_ids), map(int, ds.trajectory_lengths))))
        self.mixture.sample_step = self.sample_step

    def sample_step(self, index):
        ds, ep, step = self.records[index]
        return self.mixture.datasets[ds], ep, step

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        result = self.mixture[index]
        ds, ep, step = self.records[index]
        result.update(sample_index=index, dataset_index=ds, episode=ep, frame=step,
                      episode_length=self.episode_lengths[ds][ep])
        return result


class CachedLibero(Dataset):
    def __init__(self, cfg, allow_partial=False):
        self.root = Path(cfg.cache.root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["fingerprint"] != fingerprint(cfg):
            raise ValueError("Cache provenance does not match model/data/preprocessing")
        if not self.manifest["complete"] and not allow_partial:
            raise ValueError("Partial cache is only allowed for smoke tests")
        self.shards = self.manifest["shards"]
        self.starts = [s["start"] for s in self.shards]
        self.suite_ranges = self.manifest["suite_ranges"]
        self.loaded = OrderedDict()

    def __len__(self):
        return self.manifest["samples"]

    def __getitem__(self, index):
        shard_id = bisect_right(self.starts, index) - 1
        shard = self.shards[shard_id]
        if shard_id not in self.loaded:
            path = self.root / shard["file"]
            if path.stat().st_size != shard["bytes"]:
                raise ValueError(f"Cache shard has wrong size: {path}")
            self.loaded[shard_id] = torch.load(path, mmap=True, weights_only=True)
            while len(self.loaded) > 2:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(shard_id)
        return self.loaded[shard_id][index - shard["start"]]


class MixtureBatches(Sampler):
    """Each rank/microstep is reproducible without saving a prefetched iterator."""
    def __init__(self, ranges, batch_size, microsteps, seed, rank=0, start=0, chunk_size=None):
        self.ranges = [(a, b) for a, b in ranges if b > a]
        self.batch_size, self.microsteps, self.seed, self.rank, self.start = batch_size, microsteps, seed, rank, start
        # Keep the original random chunks across batch/accumulation changes.
        self.chunk_size = batch_size if chunk_size is None else int(chunk_size)
        if min(self.batch_size, self.chunk_size) < 1:
            raise ValueError("Batch size and sampling chunk size must be positive")

    def __iter__(self):
        for micro in range(self.start, self.microsteps):
            position, end = micro * self.batch_size, (micro + 1) * self.batch_size
            batch = []
            while position < end:
                chunk, offset = divmod(position, self.chunk_size)
                rng = np.random.default_rng(seed_for(self.seed, self.rank, chunk, "data"))
                indices = [int(rng.integers(*self.ranges[i]))
                           for i in rng.integers(len(self.ranges), size=self.chunk_size)]
                count = min(end - position, self.chunk_size - offset)
                batch.extend(indices[offset:offset + count])
                position += count
            yield batch

    def __len__(self):
        return self.microsteps - self.start


def collate(samples):
    return samples
