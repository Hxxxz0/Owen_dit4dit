"""Paired offline interventions on a fixed training-distribution diagnostic set."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from DiT4DiT.model.framework.base_framework import baseframework
from .common import CONDITIONS, file_sha, seed_for
from .data import CachedLibero
from .donors import DonorIndex
from .protocol import unnormalize


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-task", type=int, default=100)
    args = p.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = baseframework.from_pretrained(args.checkpoint).cuda().eval()
    data = CachedLibero(model.config)
    donors = DonorIndex(data)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sha = file_sha(args.checkpoint)
    stats = next(iter(model.norm_stats.values()))["action"]
    jobs = [r for lang in sorted(donors.diagnostic) for r in donors.diagnostic[lang][:args.per_task]]
    path = output / f"offline_rank_{rank:02d}.jsonl"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for i in range(rank, len(jobs), world):
            r = jobs[i]
            sample = data[r["sample_index"]]
            donor = donors.select(sample["lang"], r["sample_index"], r["frame"]/max(1,r["episode_length"]-1),
                                  exclude_episode=(r["dataset_index"], r["episode"]))
            action_seed = seed_for("offline", r["sample_index"])
            outputs = {condition: model.predict_action([sample], intervention=condition,
                donor_examples=[donor] if condition.startswith("swap") else None,
                feature_seed=42, action_seed=action_seed)["normalized_actions"][0] for condition in CONDITIONS}
            baseline = unnormalize(outputs["full"], stats)
            for condition, prediction in outputs.items():
                processed = unnormalize(prediction, stats)
                delta = processed - baseline
                target = sample["action"].numpy()
                valid = sample["action_mask"].numpy()
                result = {"checkpoint_sha256": sha, "sample_index": r["sample_index"], "lang": sample["lang"],
                    "condition": condition, "donor_index": donor["sample_index"], "action_seed": action_seed,
                    "translation_delta_l2": float(np.linalg.norm(delta[:, :3], axis=-1).mean()),
                    "rotation_delta_l2": float(np.linalg.norm(delta[:, 3:6], axis=-1).mean()),
                    "gripper_disagreement": float((processed[:,6] != baseline[:,6]).mean()),
                    "normalized_action_mse": float((((prediction-target)**2)*valid).sum()/valid.sum())}
                stream.write(json.dumps(result)+"\n")
            stream.flush()
    temporary.replace(path)
    print(f"OFFLINE_COMPLETE rank={rank} observations={len(range(rank,len(jobs),world))}", flush=True)


if __name__ == "__main__":
    main()
