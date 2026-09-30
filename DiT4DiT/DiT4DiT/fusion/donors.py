"""Same-task, different-demonstration donors, selected independently of outcome."""
from collections import defaultdict
import json
from pathlib import Path

from .common import seed_for


class DonorIndex:
    def __init__(self, dataset):
        self.dataset = dataset
        groups = defaultdict(lambda: defaultdict(list))
        for shard in dataset.shards:
            path = getattr(dataset, "index_root", dataset.root) / Path(shard["file"]).with_suffix(".index.json")
            for r in json.loads(path.read_text()):
                groups[r["lang"]][(r["dataset_index"], r["episode"])].append(r)
        self.bank = {}
        self.diagnostic = {}
        for lang, episodes in groups.items():
            chosen = sorted(episodes)[:5]
            bank = {}
            for ep in chosen:
                records = episodes[ep]
                bank[ep] = [min(records, key=lambda r: abs(r["frame"]/max(1,r["episode_length"]-1) - b/9))
                            for b in range(10)]
            self.bank[lang] = bank
            all_records = [r for e in sorted(episodes) for r in episodes[e]]
            # Stable, evenly spread observations; includes multiple demonstrations.
            self.diagnostic[lang] = [all_records[min(len(all_records)-1, i*len(all_records)//100)] for i in range(min(100,len(all_records)))]

    def select(self, lang, episode_seed, progress, exclude_episode=None):
        bank = self.bank[lang]
        choices = [e for e in sorted(bank) if e != exclude_episode]
        if not choices:
            raise ValueError("Need a distinct demonstration for donor interventions")
        ep = choices[seed_for("donor", lang, episode_seed) % len(choices)]
        record = bank[ep][min(9, max(0, round(progress*9)))]
        return self.dataset[record["sample_index"]]
