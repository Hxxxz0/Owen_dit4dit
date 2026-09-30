"""DeepSpeed owns backward, accumulation, optimizer, and scheduler stepping."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import deepspeed

from DiT4DiT.model.framework.QwenConcat import DiT4DiTQwenConcat
from .common import atomic_json, fingerprint, load_config, require_ppu
from .data import CachedLibero, MixtureBatches, collate
from .resume import validate_resume_config
from .joint_checks import optimizer_groups, update_anchors, FirstBackwardAudit
from .checkpoints import recovery_manifest, publish_recovery, prune_recovery, reconcile_metrics


def ds_config(cfg, world):
    batch = int(cfg.datasets.vla_data.per_device_batch_size)
    accum = int(cfg.trainer.gradient_accumulation_steps)
    return {"bf16": {"enabled": True}, "fp16": {"enabled": False},
        "train_micro_batch_size_per_gpu": batch, "gradient_accumulation_steps": accum,
        "train_batch_size": world * batch * accum, "gradient_clipping": float(cfg.trainer.max_grad_norm),
        "zero_optimization": {"stage": 2, "overlap_comm": False, "contiguous_gradients": True,
            "reduce_scatter": True, "reduce_bucket_size": 50_000_000, "allgather_bucket_size": 50_000_000},
        "steps_per_print": 1000000, "wall_clock_breakdown": False}


def make_scheduler(optimizer, steps, warmup, minimum):
    # A separate floor per group avoids Transformers deriving the floor from
    # AdamW's unrelated defaults['lr'] instead of the actual parameter-group LR.
    functions = []
    for group in optimizer.param_groups:
        floor = minimum / group["lr"]
        def multiplier(step, ratio=floor):
            if warmup and step < warmup:
                return step / warmup
            progress = min(1., max(0., (step-warmup)/max(1,steps-warmup)))
            return ratio + (1-ratio)*0.5*(1+math.cos(math.pi*progress))
        functions.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)


def save(engine, cfg, output, step, rank, sampler_chunk_size, publish_inference=True):
    tag = f"step_{step:06d}"
    state_dir = output / "training_state"
    engine.save_checkpoint(str(state_dir), tag=tag,
        client_state={"step": step, "fingerprint": fingerprint(cfg), "config": OmegaConf.to_container(cfg, resolve=True),
                      "sampler_chunk_size": sampler_chunk_size})
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state()}, state_dir / tag / f"rng_{rank}.pt")
    dist.barrier()  # Publish only after every rank's optimizer and RNG files exist.
    if rank == 0:
        publish_recovery(output, step, dist.get_world_size())
    if rank == 0 and publish_inference:
        path = output / "checkpoints" / f"steps_{step}_pytorch_model.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({k: v.detach().cpu() for k, v in engine.module.state_dict().items()
                    if not k.startswith("observation_encoder.")}, tmp)
        tmp.replace(path)
        atomic_json(output / "latest.json", {"step": step, "checkpoint": str(path), "training_state": str(state_dir / tag)})
    dist.barrier()
    # Keep two complete optimizer/RNG snapshots; inference snapshots are retained.
    if rank == 0:
        prune_recovery(output)
    dist.barrier()


def validate_updates(engine, anchors, output, step, start, rank, world):
    updates = {prefix: {"parameter": name, "max_update": float(
        (param.detach().flatten()[:len(before)].float()-before.float()).abs().max())}
        for prefix, (name, param, before) in anchors.items()}
    evidence = {"rank": rank, "steps": step-start, "updates": updates,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3}
    gathered = [None]*world
    dist.all_gather_object(gathered, evidence)
    if not all(math.isfinite(v["max_update"]) and v["max_update"]>0 for r in gathered for v in r["updates"].values()):
        raise AssertionError("Trainable components must update finitely on every rank")
    if rank == 0:
        atomic_json(output / "rank_validation.json", gathered)
        atomic_json(output / "health.json", {"step":step,"ranks":world,"passed":True,
            "max_reserved_gib":max(r["peak_reserved_gib"] for r in gathered),"updated_unix":time.time()})
    return gathered


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-batch-repartition", action="store_true",
                   help="Allow only microbatch/accumulation changes preserving the effective batch on resume")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-warmup", type=int, default=20)
    p.add_argument("--profile-measure", type=int, default=50)
    p.add_argument("--expected-world-size", type=int, default=16, choices=(16, 32, 64, 128, 256),
                   help="Explicit production topology; the effective global batch remains 256")
    p.add_argument("--pause-after-hours", type=float)
    p.add_argument("--pause-after-steps", type=int)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    if args.allow_batch_repartition and not args.resume:
        p.error("--allow-batch-repartition requires --resume")
    cfg = load_config(args.config, args.overrides)
    world = int(os.environ.get("WORLD_SIZE", 1))
    require_ppu()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", args.local_rank)))
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    batch = int(cfg.datasets.vla_data.per_device_batch_size)
    accum = int(cfg.trainer.gradient_accumulation_steps)
    if not args.smoke and world * batch * accum != 256:
        raise ValueError("Production experiment requires effective global batch 256")
    if not args.smoke and world != args.expected_world_size:
        raise ValueError(f"Production experiment requires all {args.expected_world_size} PPUs")
    if args.profile:
        cfg.trainer.max_train_steps = args.profile_warmup + args.profile_measure
    output = Path(cfg.run_root_dir) / cfg.run_id
    if rank == 0:
        if (output / "config.yaml").exists() and not args.resume:
            raise FileExistsError(f"Use a new run_id or --resume; existing run: {output}")
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
        if not args.resume:
            OmegaConf.save(cfg, output / "config.yaml")
        shutil.copy2(Path(cfg.cache.root) / "dataset_statistics.json", output / "dataset_statistics.json")
    dist.barrier()
    # Same initialization on all ranks. Training random streams split after initialization.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    joint = cfg.framework.name == "DiT4DiTQwenJoint"
    if joint:
        from DiT4DiT.model.framework.QwenJoint import DiT4DiTQwenJoint
        model = DiT4DiTQwenJoint(cfg)
    else:
        model = DiT4DiTQwenConcat(cfg)
    param_groups = optimizer_groups(model, cfg.trainer.learning_rate)
    if rank == 0:
        atomic_json(output / "trainable_scope.json", {"framework": cfg.framework.name,
            "fresh_initialization": not args.resume,
            "groups": {g["name"]: {"parameters": sum(p.numel() for p in g["params"]),
                "tensors": len(g["params"]), "learning_rate": g["lr"]} for g in param_groups},
            "frozen_components": ["Wan VAE", "Cosmos independent text encoder"],
            "action_gradients_to_backbones": joint})
        print("MODEL_INITIALIZED optimizer coverage verified", flush=True)
    opt_cfg = cfg.trainer.optimizer
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(opt_cfg.betas), eps=float(opt_cfg.eps), weight_decay=float(opt_cfg.weight_decay))
    warmup = math.ceil(int(cfg.trainer.max_train_steps) * float(cfg.trainer.warmup_fraction))
    scheduler = make_scheduler(optimizer, int(cfg.trainer.max_train_steps), warmup, float(cfg.trainer.min_lr))
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
        lr_scheduler=scheduler, config=ds_config(cfg, world), dist_init_required=False)
    engine.train()
    random.seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed(cfg.seed + rank)
    start = 0
    sampler_chunk_size = batch
    if args.resume:
        # DeepSpeed 0.14 predates PyTorch 2.6's weights_only default. This is
        # restricted to optimizer checkpoints produced in this experiment run.
        engine.checkpoint_engine.load = lambda path, map_location=None: torch.load(
            path, map_location=map_location, weights_only=False)
        latest = recovery_manifest(output)
        loaded, state = engine.load_checkpoint(str(output / "training_state"),
            tag=Path(latest["training_state"]).name)
        if loaded is None or state["fingerprint"] != fingerprint(cfg):
            raise ValueError("Missing or incompatible resume checkpoint")
        repartition = validate_resume_config(state["config"], OmegaConf.to_container(cfg, resolve=True),
                                             args.allow_batch_repartition)
        start = int(state["step"])
        sampler_chunk_size = int(state.get("sampler_chunk_size", repartition["old_batch"]))
        engine.micro_steps = start * accum
        # Verify actual optimizer/LR restoration before publishing the new config.
        optimizer_steps = sorted({int(s["step"].item()) if torch.is_tensor(s["step"]) else int(s["step"])
                                  for s in engine.optimizer.optimizer.state.values() if "step" in s})
        optimizer_lrs = [float(group["lr"]) for group in engine.optimizer.optimizer.param_groups]
        if (engine.global_steps != start or engine.global_samples != start * world * batch * accum
                or scheduler.last_epoch != start or optimizer_steps != [start]
                or optimizer_lrs != scheduler.get_last_lr() or engine.loaded_checkpoint_dp_world_size != world):
            raise AssertionError("Resume did not restore optimizer, scheduler, or update counters")
        evidence = {"rank": rank, "step": start, "global_samples": engine.global_samples,
                    "optimizer_steps": optimizer_steps, "scheduler_step": scheduler.last_epoch,
                    "learning_rates": scheduler.get_last_lr(), "optimizer_learning_rates": optimizer_lrs,
                    "sampler_chunk_size": sampler_chunk_size,
                    **repartition}
        gathered = [None] * world
        dist.all_gather_object(gathered, evidence)
        if rank == 0:
            reconcile_metrics(output, start)
            atomic_json(output / f"resume_validation_step_{start}.json", gathered)
            if repartition["changed"]:
                previous_config = output / f"config_before_step_{start}_repartition.yaml"
                if not previous_config.exists():
                    shutil.copy2(output / "config.yaml", previous_config)
                temporary = output / "config.yaml.tmp"
                OmegaConf.save(cfg, temporary)
                temporary.replace(output / "config.yaml")
            print(f"CONCAT_RESUME_VERIFIED step={start} batch={batch} accumulation={accum}", flush=True)
        rng = torch.load(Path(loaded).parent / f"rng_{rank}.pt", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
        if start == int(cfg.trainer.max_train_steps) and (output / "complete.json").exists():
            if rank == 0:
                print("CONCAT_TRAIN_ALREADY_COMPLETE", flush=True)
            dist.destroy_process_group()
            return
    if joint:
        from .current_images import CurrentImageLibero
        dataset = CurrentImageLibero(cfg, allow_partial=args.smoke)
    else:
        dataset = CachedLibero(cfg, allow_partial=args.smoke)
    sampler = MixtureBatches(dataset.suite_ranges, batch, int(cfg.trainer.max_train_steps) * accum,
                             cfg.seed, rank, start * accum, chunk_size=sampler_chunk_size)
    workers = int(cfg.datasets.vla_data.num_workers)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
        num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
        generator=torch.Generator().manual_seed(int(cfg.seed)+rank))
    anchors = update_anchors(engine.module)
    backward_audit = FirstBackwardAudit(engine.module) if joint else None
    records, timings = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    clock = time.monotonic()
    update_losses = torch.zeros(2, device=engine.device)
    step = start
    segment_started = time.monotonic()
    paused = False
    for samples in loader:
        losses = engine(samples)
        total = losses["action_loss"] + losses["future_video_loss"]
        if not torch.isfinite(total).item():
            raise FloatingPointError(f"Non-finite loss at update {step}, rank {rank}")
        update_losses += torch.stack([losses["action_loss"].detach(), losses["future_video_loss"].detach()]).float() / accum
        boundary = engine.is_gradient_accumulation_boundary()
        engine.backward(total)
        engine.step()
        if not boundary:
            continue
        step += 1
        if backward_audit is not None:
            evidence = {"rank": rank, **backward_audit.finish()}
            gathered = [None] * world
            dist.all_gather_object(gathered, evidence)
            if rank == 0:
                atomic_json(output / "first_backward_validation.json", gathered)
                print("FULL_PARAMETER_BACKWARD_VERIFIED", flush=True)
            backward_audit = None
        if engine.global_steps != step:
            raise AssertionError(f"DeepSpeed update count {engine.global_steps} != {step}")
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.monotonic() - clock, device=engine.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(update_losses)
        update_losses /= world
        grad_norm = float(engine.get_global_grad_norm())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at update {step}, rank {rank}")
        record = {"step": step, "action_loss": float(update_losses[0]), "video_loss": float(update_losses[1]),
            "gradient_norm": grad_norm,
            "seconds": float(elapsed), "samples_per_second": world*batch*accum/float(elapsed),
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
            "learning_rates": scheduler.get_last_lr()}
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if step % int(cfg.trainer.logging_frequency) == 0 or step <= 3:
                print(json.dumps(record), flush=True)
        if args.smoke:
            records.append(record)
        if args.profile and step > args.profile_warmup:
            timings.append(float(elapsed))
        paused = step < int(cfg.trainer.max_train_steps) and (
            (args.pause_after_steps is not None and step-start >= args.pause_after_steps) or
            (args.pause_after_hours is not None and time.monotonic()-segment_started >= args.pause_after_hours*3600))
        pause_tensor = torch.tensor(int(paused),device=engine.device)
        dist.all_reduce(pause_tensor,op=dist.ReduceOp.MAX)
        paused = bool(pause_tensor.item())
        initial_health_step = int(cfg.trainer.get("initial_health_step", 100))
        health_check = not args.profile and (step == initial_health_step or step % int(cfg.trainer.save_interval) == 0
                                             or (args.resume and step - start == initial_health_step))
        if health_check:
            validate_updates(engine, anchors, output, step, start, rank, world)
        checkpoint_due = step % int(cfg.trainer.save_interval) == 0
        initial_save = int(cfg.trainer.get("initial_checkpoint_step", 100))
        inference_due = checkpoint_due or (initial_save > 0 and step == initial_save) or step == int(cfg.trainer.max_train_steps)
        recovery_interval = int(cfg.trainer.get("recovery_interval", 0))
        recovery_due = (recovery_interval > 0 and step % recovery_interval == 0) or step == int(cfg.trainer.get("recovery_initial_step", -1))
        if not args.profile and (inference_due or recovery_due or paused):
            save_started = time.monotonic()
            save(engine, cfg, output, step, rank, sampler_chunk_size,
                 publish_inference=inference_due or (paused and recovery_interval == 0))
            if rank == 0:
                with (output / "checkpoint_metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "inference": inference_due,
                        "seconds": time.monotonic()-save_started}) + "\n")
        if paused:
            break
        update_losses.zero_()
        torch.cuda.synchronize()
        clock = time.monotonic()
    all_evidence = validate_updates(engine, anchors, output, step, start, rank, world)
    if rank == 0:
        atomic_json(output / "rank_validation.json", all_evidence)
        if args.profile:
            p95 = float(np.percentile(timings, 95))
            atomic_json(output / "profile.json", {"batch_size": batch, "accumulation": accum,
                "world_size": world, "gradient_checkpointing": bool(cfg.trainer.gradient_checkpointing),
                "mean_seconds": float(np.mean(timings)),
                "p50_seconds": float(np.median(timings)), "p95_seconds": p95,
                "samples_per_second": world*batch*accum/float(np.mean(timings)),
                "max_reserved_gib": max(r["peak_reserved_gib"] for r in all_evidence),
                "budget_steps": min(20000, int(float(cfg.trainer.training_hours)*3600/p95)//100*100)})
        atomic_json(output / ("paused.json" if paused else "complete.json"),
                    {"step": step, "profile": args.profile, "smoke": args.smoke, "ranks": world})
        print("CONCAT_TRAIN_VERIFIED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
