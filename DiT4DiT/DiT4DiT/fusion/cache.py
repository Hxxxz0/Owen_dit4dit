"""Usage: torchrun --nproc_per_node=16 -m DiT4DiT.fusion.cache --config ..."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .backbones import FrozenObservationEncoder, encode_texts
from .common import atomic_json, file_sha, fingerprint, load_config, provenance, require_ppu
from .data import IndexedLibero


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    require_ppu()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        # Cache workers do independent video decoding; their finish times may
        # differ by more than NCCL's default ten-minute collective timeout.
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    device = torch.device("cuda", local_rank)
    root = Path(cfg.cache.root)
    root.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(cfg)
    meta_path = root / "provenance.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["fingerprint"] != identity:
        raise ValueError("Refusing to reuse a cache with different provenance")
    if rank == 0:
        atomic_json(meta_path, {"fingerprint": identity, **provenance(cfg)})
        text_path = root / "text.pt"
        if not text_path.exists():
            instructions = []
            for path in sorted(Path(cfg.datasets.vla_data.data_root_dir).glob("*/meta/tasks.jsonl")):
                instructions.extend(json.loads(line)["task"] for line in path.read_text().splitlines())
            texts = encode_texts(cfg, instructions, device)
            torch.save(texts, root / "text.tmp")
            (root / "text.tmp").replace(text_path)
            atomic_json(root / "text.json", {"count": len(texts), "sha256": file_sha(text_path), "fingerprint": identity})
            del texts
        print("TEXT_CACHE_READY", flush=True)
    if world > 1:
        dist.barrier()
    if args.text_only:
        if world > 1:
            dist.destroy_process_group()
        return
    raw = IndexedLibero(cfg)
    if rank == 0:
        raw.mixture.save_dataset_statistics(root / "dataset_statistics.json")
    size = min(len(raw), args.limit or len(raw))
    shard_size = int(cfg.cache.shard_size)
    pending = []
    if rank == 0:
        for shard_id, start in enumerate(range(0, size, shard_size)):
            path = root / f"shard_{shard_id:06d}.pt"
            info_path = path.with_suffix(".json")
            end = min(start + shard_size, size)
            if info_path.exists() and path.exists() and path.with_suffix(".index.json").exists():
                info = json.loads(info_path.read_text())
                if info["fingerprint"] == identity and info["end"] == end and path.stat().st_size == info["bytes"]:
                    continue
            pending.append(shard_id)
    if world > 1:
        payload = [pending]
        dist.broadcast_object_list(payload, src=0)
        pending = payload[0]
    assigned = pending[rank::world]
    print(json.dumps({"rank": rank, "pending_shards": len(assigned)}), flush=True)
    encoder = FrozenObservationEncoder(cfg, device) if assigned else None
    started = time.monotonic()
    for shard_id in assigned:
        start = shard_id * shard_size
        path = root / f"shard_{shard_id:06d}.pt"
        info_path = path.with_suffix(".json")
        end = min(start + shard_size, size)
        if info_path.exists() and path.exists() and path.with_suffix(".index.json").exists():
            info = json.loads(info_path.read_text())
            if info["fingerprint"] == identity and info["end"] == end and path.stat().st_size == info["bytes"]:
                continue
        records = []
        for offset in range(start, end, int(cfg.cache.batch_size)):
            examples = [raw[i] for i in range(offset, min(offset + int(cfg.cache.batch_size), end))]
            encoded = encoder.encode(examples)
            for item, ex in zip(encoded, examples):
                item = {k: v.cpu().contiguous() if torch.is_tensor(v) else v for k, v in item.items()}
                if not all(torch.isfinite(v).all().item() for v in item.values() if torch.is_tensor(v)):
                    raise ValueError(f"Non-finite cache record {ex['sample_index']}")
                item.update({k: ex[k] for k in ("sample_index", "dataset_index", "episode", "frame", "episode_length")})
                records.append(item)
        tmp = path.with_suffix(".tmp")
        torch.save(records, tmp)
        tmp.replace(path)
        atomic_json(info_path, {"file": path.name, "start": start, "end": end, "bytes": path.stat().st_size,
                              "sha256": file_sha(path), "fingerprint": identity})
        atomic_json(path.with_suffix(".index.json"), [
            {k: r[k] for k in ("sample_index", "dataset_index", "episode", "frame", "episode_length", "lang")}
            for r in records])
        print(json.dumps({"rank": rank, "shard": shard_id, "end": end, "seconds": time.monotonic()-started}), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        shards = [json.loads((root / f"shard_{i:06d}.json").read_text()) for i in range((size+shard_size-1)//shard_size)]
        atomic_json(root / "manifest.json", {"fingerprint": identity, "samples": size, "complete": size == len(raw),
            "suite_ranges": [[a, min(b, size)] for a, b in raw.suite_ranges if a < size], "shards": shards})
        print(f"CACHE_COMPLETE samples={size} full_dataset={size == len(raw)}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
