"""Run predeclared, complete LIBERO attribution rounds on independent devices."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

from .common import atomic_json, file_sha
from .interventions import ATTRIBUTION_CONDITIONS
from .protocol import SUITES, seed_for


def paired_jobs(first_trial, end_trial, conditions, workers, slots_per_device=1):
    """Keep each task/trial's conditions on one device, with fixed random order."""
    assignments = [[] for _ in range(workers * slots_per_device)]
    for suite_index, suite in enumerate(SUITES):
        for task in range(10):
            for trial in range(first_trial, end_trial):
                rank = (suite_index * 10 + task + trial * 7) % workers
                slot = ((suite_index * 10 + task) // workers + trial) % slots_per_device
                worker = rank * slots_per_device + slot
                rng = np.random.default_rng(seed_for(suite, task, trial, "condition_order"))
                for condition in rng.permutation(conditions):
                    assignments[worker].append({"suite": suite, "task": task, "trial": trial,
                                              "condition": str(condition)})
    return assignments


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 20
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-python", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--port-base", type=int, default=29800)
    parser.add_argument("--slots-per-device", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between one and sixteen")
    if not 1 <= args.slots_per_device <= 4:
        parser.error("slots-per-device must be between one and four")
    root = Path(args.output).resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    conditions = tuple(protocol["conditions"])
    if conditions != ATTRIBUTION_CONDITIONS:
        raise ValueError("The attribution protocol must contain all nine predeclared conditions")
    if protocol["observation_mode"] != "current_frame_with_single_branch_freeze":
        raise ValueError("Untargeted branches must retain the current frame")
    checkpoint = Path(protocol["checkpoint"])
    if file_sha(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the frozen protocol")
    rounds = protocol["trial_rounds"]
    if not rounds or rounds != sorted(set(rounds)) or not 1 <= rounds[0] <= rounds[-1] <= 50:
        raise ValueError("Invalid predeclared trial rounds")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    logs, servers, rollouts = [], [], []
    started = time.monotonic()
    protocol_hash = file_sha(protocol_path)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Evaluation received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    atomic_json(root / "runner.json", {"pid": os.getpid(), "status": "starting",
                                      "protocol_sha256": protocol_hash})
    try:
        for rank in range(args.workers):
            log = (root / f"server_{rank}.log").open("a")
            logs.append(log)
            command = [sys.executable, "-u", "-m", "DiT4DiT.fusion.server",
                       "--checkpoint", str(checkpoint), "--protocol", str(protocol_path),
                       "--port", str(args.port_base + rank)]
            servers.append(subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank)),
                                            stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 900
        ready = set()
        while len(ready) < args.workers:
            for rank, process in enumerate(servers):
                if process.poll() is not None:
                    raise RuntimeError(f"Policy server {rank} exited; inspect server_{rank}.log")
                if rank in ready:
                    continue
                try:
                    with opener.open(f"http://127.0.0.1:{args.port_base + rank}/health", timeout=2) as reply:
                        health = json.loads(reply.read())
                    if health.get("protocol_sha256") != protocol_hash:
                        raise ValueError("Policy server has the wrong protocol")
                    ready.add(rank)
                except OSError:
                    pass
            atomic_json(root / "runner.json", {"pid": os.getpid(), "status": "loading_servers",
                                              "ready_devices": sorted(ready)})
            if time.monotonic() > deadline:
                raise TimeoutError("Policy servers did not become ready within 15 minutes")
            if len(ready) < args.workers:
                time.sleep(2)
        atomic_json(root / "servers_ready.json", {"devices": sorted(ready), "protocol_sha256": protocol_hash})
        if protocol.get("runtime_probe"):
            from .evaluation_audit import runtime_preflight
            runtime_preflight(root, opener, f"http://127.0.0.1:{args.port_base}", protocol)
        first = 0
        for target in rounds:
            round_start = time.monotonic()
            rollouts = []
            assignments = paired_jobs(first, target, conditions, args.workers, args.slots_per_device)
            for worker, jobs in enumerate(assignments):
                if not jobs:
                    continue
                rank = worker // args.slots_per_device
                job_path = root / f"jobs_round_{target:02d}_worker_{worker:02d}.json"
                atomic_json(job_path, jobs)
                log = (root / f"rollout_{worker}.log").open("a")
                logs.append(log)
                command = [args.libero_python, "-u", "-m", "DiT4DiT.fusion.rollout",
                           "--jobs", str(job_path), "--output", str(root / "episodes"),
                           "--port", str(args.port_base + rank), "--video-trials", "1"]
                rollouts.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            while True:
                codes = [process.poll() for process in rollouts]
                errors = [(rank, code) for rank, code in enumerate(codes) if code not in (None, 0)]
                if errors:
                    raise RuntimeError(f"Rollout worker errors: {errors}; errors are not scored as failures")
                if any(process.poll() is not None for process in servers):
                    raise RuntimeError("A policy server exited during evaluation")
                atomic_json(root / "runner.json", {
                    "pid": os.getpid(), "status": "evaluating", "round_target": target,
                    "previous_complete_trials": first, "finished_workers": codes.count(0),
                    "completed_episodes": len(list((root / "episodes").glob("*.json"))),
                    "elapsed_seconds": time.monotonic() - started})
                if all(code == 0 for code in codes):
                    break
                time.sleep(5)
            with (root / "report_generation.log").open("a") as log:
                subprocess.run([sys.executable, "-m", "DiT4DiT.fusion.report", "--results", str(root)],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads((root / "report.json").read_text())
            if report["status"] != "complete" or report["trials_per_task"] != target:
                raise ValueError("A completed round must contain every paired task and condition")
            if protocol.get("reference_evaluation"):
                from .evaluation_audit import audit_round, write_comparison
                audit_round(root, target, protocol)
                write_comparison(root, target, protocol)
            atomic_json(root / f"report_trials_{target:02d}.json", report)
            elapsed = time.monotonic() - round_start
            atomic_json(root / "progress.json", {"complete_trials": target, "conditions": conditions,
                                                "elapsed_seconds": time.monotonic() - started})
            print(json.dumps({"complete_trials": target, "conditions": report["conditions"]}), flush=True)
            if target != rounds[-1]:
                next_target = rounds[rounds.index(target) + 1]
                estimate = elapsed * (next_target - target) / (target - first)
                if time.monotonic() - started + 1.25 * estimate > protocol["budget_hours"] * 3600:
                    break
            first = target
        atomic_json(root / "complete.json", {"trials_per_task": target, "tasks": 40,
                                            "conditions": conditions, "protocol_sha256": protocol_hash})
        atomic_json(root / "runner.json", {"pid": os.getpid(), "status": "complete", "trials_per_task": target})
    except BaseException as error:
        atomic_json(root / "runner.json", {"pid": os.getpid(), "status": "failed",
                                          "error": repr(error)})
        raise
    finally:
        stop_processes(rollouts)
        stop_processes(servers)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
