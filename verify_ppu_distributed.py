"""Preflight only: all-reduce and BF16 backward/AdamW across 16 actual PPUs."""
import json
import os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(minutes=5))
assert dist.get_world_size() == 16
assert 'PPU' in torch.cuda.get_device_name(rank)
value = torch.tensor([rank + 1.0], device='cuda')
dist.all_reduce(value)
assert value.item() == 136
model = DistributedDataParallel(torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.GELU(),torch.nn.Linear(128,7)).cuda(), device_ids=[rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
before = next(model.parameters()).detach().clone()
losses = []
for step in range(3):
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(torch.randn(4,64,device='cuda'))
        loss = output.float().square().mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()
    losses.append(loss.item())
changed = (next(model.parameters()).detach()-before).abs().max().item()
assert changed > 0
results = [None] * 16
dist.all_gather_object(results, dict(rank=rank, device=torch.cuda.get_device_name(rank), losses=losses, max_update=changed))
if rank == 0:
    Path('logs/ppu_distributed_preflight.json').write_text(json.dumps(results,indent=2))
    print('PASS: 16 PPUs all-reduce=136 BF16 backward fused AdamW updates on every rank',flush=True)
dist.destroy_process_group()
