"""Read the saved 16-PPU validation checkpoint and check rank evidence."""
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
run = ROOT / 'results/starvla_libero_ppu16'
records = json.loads((run/'ppu_rank_training_validation.json').read_text())
assert sorted(r['rank'] for r in records) == list(range(16))
for record in records:
    assert len(record['steps']) == 3
    assert all(math.isfinite(step['loss']) for step in record['steps'])
    assert set(record['parameter_updates']) == {'backbone','action_head'}
    assert all(math.isfinite(v['max_abs_change']) and v['max_abs_change'] > 0
               for v in record['parameter_updates'].values())

checkpoint = run/'final_model/pytorch_model.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
for name,tensor in state.items():
    assert torch.isfinite(tensor).all(), name
backbone = records[0]['parameter_updates']['backbone']
source_name = backbone['name'].removeprefix('qwen_vl_interface.model.')
index = json.loads((ROOT/'models/Qwen3-VL-4B-Instruct/model.safetensors.index.json').read_text())
with safe_open(ROOT/'models/Qwen3-VL-4B-Instruct'/index['weight_map'][source_name],framework='pt') as source:
    original = source.get_tensor(source_name)
difference = (state[backbone['name']].float()-original.float()).abs().max().item()
assert difference == backbone['max_abs_change']
report = {
    'passed': True,
    'ranks': 16,
    'steps_per_rank': 3,
    'parameters_in_state_dict': sum(t.numel() for t in state.values()),
    'tensors': len(state),
    'checkpoint_bytes': checkpoint.stat().st_size,
    'checkpoint_all_finite': True,
    'backbone_checkpoint_matches_rank_evidence': True,
    'mean_loss_per_step': [sum(r['steps'][i]['loss'] for r in records)/16 for i in range(3)],
    'peak_allocated_GiB_range': [min(r['peak_allocated_GiB'] for r in records),
                               max(r['peak_allocated_GiB'] for r in records)],
    'note': '3-step real-data training smoke test only; not convergence or LIBERO rollout evaluation',
}
(ROOT/'logs/starvla_ppu16_checkpoint_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
