"""Bounded local forward/backward probe alongside an active training run.

This checks candidate memory/compute feasibility; shared-device timings exclude
optimizer/communication and must not be reported as standalone 16-rank speed.
"""
import argparse
import gc
import statistics
import time

import torch

from .common import atomic_json, load_config, require_ppu
from .data import CachedLibero
from DiT4DiT.model.framework.QwenConcat import DiT4DiTQwenConcat


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=5)
    args = parser.parse_args()
    require_ppu()
    torch.cuda.set_device(0)
    # Preserve ample room for the active training process on this physical card.
    torch.cuda.set_per_process_memory_fraction(0.5)
    cfg = load_config(args.config)
    data = CachedLibero(cfg)
    samples = [data[a + (b-a)*i//4] for a,b in data.suite_ranges for i in range(4)]
    assert len(samples) == 16
    torch.manual_seed(42)
    model = DiT4DiTQwenConcat(cfg).cuda().bfloat16().train()
    variants = [("b8_a2_checkpoint", 8, 2, True), ("b16_a1_checkpoint", 16, 1, True),
                ("b16_a1_no_checkpoint", 16, 1, False), ("b8_a2_checkpoint_repeat", 8, 2, True)]
    report = {"scope":"local forward/backward; concurrent production job; no optimizer or communication",
              "samples_per_local_update":16,"variants":[]}
    for name, batch, accumulation, checkpoint in variants:
        model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()
        if checkpoint:
            model.cosmos.transformer.enable_gradient_checkpointing()
        else:
            model.cosmos.transformer.disable_gradient_checkpointing()
        model.action_model.model.gradient_checkpointing = checkpoint
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        try:
            for step in range(args.warmup+args.measure):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize();started=time.monotonic()
                for micro in range(accumulation):
                    losses=model(samples[micro*batch:(micro+1)*batch])
                    loss=(losses["action_loss"]+losses["future_video_loss"])/accumulation
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(name)
                    loss.backward()
                torch.cuda.synchronize()
                if step>=args.warmup:
                    timings.append(time.monotonic()-started)
            for module in (model.cosmos, model.fusion, model.action_model):
                grad=next(p.grad for p in module.parameters() if p.requires_grad and p.grad is not None)
                if not torch.isfinite(grad).all().item():
                    raise FloatingPointError(f"Non-finite gradient: {name}")
            row={"name":name,"batch":batch,"accumulation":accumulation,"gradient_checkpointing":checkpoint,
                 "median_local_seconds":statistics.median(timings),"seconds":timings,
                 "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
                 "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"passed":True}
        except torch.cuda.OutOfMemoryError:
            row={"name":name,"passed":False,"reason":"probe allocator limit"}
            model.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()
        report["variants"].append(row)
        atomic_json(args.output,report)
        print(row,flush=True)


if __name__ == "__main__":
    main()
