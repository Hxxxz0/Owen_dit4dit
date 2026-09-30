"""Exercise the installed PPU DeepSpeed/Accelerate stack before loading 4B weights."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin

plugin = DeepSpeedPlugin(hf_ds_config={
    'train_micro_batch_size_per_gpu':1, 'gradient_accumulation_steps':1,
    'bf16':{'enabled':True}, 'zero_optimization':{'stage':2},
    'zero_allow_untested_optimizer':True,
})
acc = Accelerator(deepspeed_plugin=plugin, mixed_precision='bf16')
assert acc.num_processes == 16
model = torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.GELU(),torch.nn.Linear(128,7))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
model,opt = acc.prepare(model,opt)
before = next(model.parameters()).detach().clone()
for step in range(3):
    opt.zero_grad()
    output=model(torch.randn(1,64,device=acc.device,dtype=torch.bfloat16))
    loss=output.float().square().mean()
    assert torch.isfinite(loss).item()
    acc.backward(loss)
    opt.step()
changed=(next(model.parameters()).detach()-before).abs().max().item()
assert changed>0
results=[None]*16
dist.all_gather_object(results,dict(rank=acc.process_index,loss=loss.item(),max_update=changed))
if acc.is_main_process:
    Path('logs/ppu_deepspeed_preflight.json').write_text(json.dumps(results,indent=2))
    print('PASS: 16 PPU DeepSpeed ZeRO2 BF16 training updates',flush=True)
dist.destroy_process_group()
