"""Configuration, provenance, and deterministic random streams."""
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

CONDITIONS = ("full", "mask_cosmos", "mask_qwen", "swap_cosmos", "swap_qwen")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def load_config(path, overrides=()):
    return OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little") % (2**63 - 1)


def generator(device, *parts):
    return torch.Generator(device=device).manual_seed(seed_for(*parts))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(cfg):
    files = [Path(cfg.framework.cosmos25.base_model) / "download_manifest.json",
             Path(cfg.framework.qwenvl.base_vlm) / "download_manifest.json"]
    files += sorted(Path(cfg.datasets.vla_data.data_root_dir).glob("*/download_manifest.json"))
    return {"format": 1, "source_manifests": {str(p): file_sha(p) for p in files},
            "views": ["primary", "wrist"], "image_size": [224, 224],
            "video_offsets": [0, 2, 4, 6, 8], "qwen_layer": "last",
            "qwen_prompt": "two images, raw instruction, generation prefix",
            "condition_vae": "posterior sample; pixel zero padding",
            "target_vae": "posterior mean (upstream semantics)"}


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(provenance(cfg), sort_keys=True).encode()).hexdigest()


def require_ppu(count=1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise RuntimeError(f"Need {count} visible PPU devices")
    if "PPU" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This launch configuration requires the existing PPU PyTorch stack")
