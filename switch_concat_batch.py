"""Switch an active pipeline only after its next complete optimizer checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "DiT4DiT"))
from DiT4DiT.fusion.common import atomic_json
from DiT4DiT.fusion.resume import validate_resume_config


def alive(process):
    return process.is_running() and process.status() != psutil.STATUS_ZOMBIE


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--expected-pipeline-pid", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    args = parser.parse_args()
    pipeline_dir = ROOT / "results" / (args.run_id + "_pipeline")
    training = ROOT / "results" / args.run_id
    state_path = pipeline_dir / "state.json"
    state = json.loads(state_path.read_text())
    if state["pid"] != args.expected_pipeline_pid or state["stage"] != "train_first":
        raise ValueError("Active pipeline identity or stage changed")
    current = OmegaConf.load(training / "config.yaml")
    target = OmegaConf.create(OmegaConf.to_container(current, resolve=True))
    target.datasets.vla_data.per_device_batch_size = args.batch
    target.trainer.gradient_accumulation_steps = args.accumulation
    validate_resume_config(OmegaConf.to_container(current, resolve=True),
                           OmegaConf.to_container(target, resolve=True), True)
    if args.batch * args.accumulation * 16 != 256:
        raise ValueError("Switch requires effective batch 256 on 16 devices")
    parent = psutil.Process(args.expected_pipeline_pid)
    if "run_concat_pipeline.py" not in " ".join(parent.cmdline()):
        raise ValueError("Unexpected parent process")
    launchers = [p for p in parent.children() if "torch.distributed.run" in p.cmdline()]
    if len(launchers) != 1:
        raise ValueError("Expected exactly one distributed launcher")
    launcher = launchers[0]
    ranks = [p for p in launcher.children() if "DiT4DiT.fusion.train" in p.cmdline()]
    if len(ranks) != 16 or any(os.getpgid(p.pid) != p.pid for p in ranks):
        raise ValueError("Expected 16 isolated training process groups")
    archive = pipeline_dir / "batch_switch"
    archive.mkdir(exist_ok=True)
    for src in (state_path, pipeline_dir / "selected.yaml", pipeline_dir / "selection.json", training / "config.yaml"):
        shutil.copy2(src, archive / ("before_" + src.name))
    OmegaConf.save(target, archive / "target.yaml")
    record = {"status": "waiting_for_checkpoint", "target_step": args.checkpoint_step,
              "old_pipeline_pid": parent.pid, "old_launcher_pid": launcher.pid,
              "old_rank_pids": [p.pid for p in ranks], "batch": args.batch,
              "accumulation": args.accumulation, "global_batch": 256, "started_unix": time.time()}
    record_path = archive / "switch.json"
    atomic_json(record_path, record)
    deadline, last_notice = time.monotonic() + 7200, 0.
    while True:
        if not all(alive(p) for p in [parent, launcher, *ranks]):
            raise RuntimeError("Original training stopped before the requested checkpoint")
        latest = json.loads((training / "latest.json").read_text())
        if int(latest["step"]) >= args.checkpoint_step:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for checkpoint; original training is untouched")
        if time.monotonic() - last_notice >= 30:
            print(f"WAIT checkpoint={latest['step']} target={args.checkpoint_step}", flush=True)
            last_notice = time.monotonic()
        time.sleep(.1)
    tag = Path(latest["training_state"])
    required = [tag / "mp_rank_00_model_states.pt", Path(latest["checkpoint"])]
    required += [tag / f"rng_{rank}.pt" for rank in range(16)]
    required += [tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(16)]
    if not all(p.is_file() and p.stat().st_size > 0 for p in required):
        raise RuntimeError("Published checkpoint is incomplete; original training is untouched")
    # Freeze only this job's known process groups before changing its on-disk state.
    groups = [parent.pid, *(p.pid for p in ranks)]
    if os.getpgid(parent.pid) != parent.pid:
        raise ValueError("Pipeline is not in an isolated process group")
    stopped = []
    try:
        for group in groups:
            os.killpg(group, signal.SIGSTOP)
            stopped.append(group)
        record.update(status="checkpoint_captured", checkpoint=latest, captured_unix=time.time())
        atomic_json(record_path, record)
        # Terminate stopped ranks first. SIGCONT delivers their pending SIGTERM.
        for group in reversed(groups):
            os.killpg(group, signal.SIGTERM)
            os.killpg(group, signal.SIGCONT)
    except BaseException:
        for group in stopped:
            try:
                os.killpg(group, signal.SIGCONT)
            except ProcessLookupError:
                pass
        raise
    _, remaining = psutil.wait_procs([parent, launcher, *ranks], timeout=45)
    remaining = [p for p in remaining if alive(p)]
    if remaining:
        raise RuntimeError(f"Old training processes have not exited: {[p.pid for p in remaining]}")
    metrics_path = training / "metrics.jsonl"
    lines = metrics_path.read_text().splitlines()
    extra = [line for line in lines if json.loads(line)["step"] > latest["step"]]
    if extra:
        (archive / "replayed_metrics.jsonl").write_text("\n".join(extra) + "\n")
        kept = [line for line in lines if json.loads(line)["step"] <= latest["step"]]
        temporary = metrics_path.with_suffix(".jsonl.tmp")
        temporary.write_text("\n".join(kept) + "\n")
        temporary.replace(metrics_path)
    state = json.loads(state_path.read_text())
    elapsed = float(state.get("train_first_elapsed_seconds", 0)) + record["captured_unix"] - state["stage_started"]
    state.update(status="restarting", train_first_elapsed_seconds=elapsed)
    state.setdefault("batch_switches", []).append({"step": latest["step"], "batch": args.batch,
                                                  "accumulation": args.accumulation, "replayed_updates": len(extra)})
    atomic_json(state_path, state)
    atomic_json(pipeline_dir / "batch_repartition.json", {"batch_size": args.batch, "accumulation": args.accumulation,
                "checkpoint_step": latest["step"], "reason": "User requested faster training; preserve global batch and optimizer state"})
    log_path = ROOT / "logs" / (args.run_id + "_batch16.log")
    command = [str(ROOT / ".venv-dit4dit/bin/python"), "-u", str(ROOT / "run_concat_pipeline.py"),
               "--run-id", args.run_id, "--train-now"]
    with log_path.open("a") as log:
        new = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(ROOT / "logs/concat_active_pipeline.json", {"pid": new.pid, "run_id": args.run_id,
        "started_unix": time.time(), "mode": "train-now; batch16 accumulation1 resume", "log": str(log_path), "state": str(state_path)})
    record.update(status="replacement_launched", new_pipeline_pid=new.pid, replayed_updates=len(extra),
                  remaining_to_intermediate_eval_hours=max(0, 18-elapsed/3600), launched_unix=time.time())
    atomic_json(record_path, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
