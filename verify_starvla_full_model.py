"""Full pretrained StarVLA forward/backward on a real LIBERO Spatial sample."""
import json
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

ROOT = Path(__file__).resolve().parent
cfg = OmegaConf.load(ROOT / 'configs/starvla_libero_ppu16.yaml')
torch.cuda.set_device(0)
dataset = make_LeRobotSingleDataset(Path(cfg.datasets.vla_data.data_root_dir),
    'libero_spatial_no_noops_1.0.0_lerobot', 'libero_franka', data_cfg=cfg.datasets.vla_data)
sample = dataset[0]
model = build_framework(cfg).to(device='cuda:0', dtype=torch.bfloat16).train()
backbone = model.qwen_vl_interface.model
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
backbone.config.use_cache = False
print('FULL_MODEL_LOADED',sum(p.numel() for p in model.parameters()),flush=True)
start = time.perf_counter()
loss = model([sample])['action_loss']
print('FORWARD_LOSS',loss.item(),flush=True)
assert torch.isfinite(loss)
loss.backward()
gradients = {}
for name,param in model.named_parameters():
    group = 'backbone' if 'qwen_vl_interface' in name and 'q_proj.weight' in name else (
        'action_head' if 'action_model' in name and 'fc1.weight' in name else None)
    if group and group not in gradients:
        assert param.grad is not None and torch.isfinite(param.grad).all()
        gradients[group] = param.grad.float().abs().max().item()
        assert gradients[group] > 0
assert set(gradients) == {'backbone','action_head'}
report={'loss':loss.item(),'gradients':gradients,'seconds':time.perf_counter()-start,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
        'instruction':sample['lang'], 'note':'single real sample forward/backward; not 16-card training'}
(ROOT/'logs/starvla_full_model_preflight.json').write_text(json.dumps(report,indent=2))
print('PASS: full pretrained StarVLA real LIBERO sample forward/backward',report,flush=True)
