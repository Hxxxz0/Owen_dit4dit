"""Published recovery states are independent of retained inference milestones."""
import json
from pathlib import Path
import shutil

from .common import atomic_json


def validate_state_files(state, world):
    state = Path(state)
    files = [state / "mp_rank_00_model_states.pt"]
    files += [state / f"rng_{rank}.pt" for rank in range(world)]
    files += [state / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(world)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise ValueError("A recovery state is missing model/optimizer/RNG files")


def recovery_manifest(output):
    output = Path(output)
    path = output / "latest_training_state.json"
    if not path.exists():
        return json.loads((output / "latest.json").read_text())
    manifest = json.loads(path.read_text())
    state = Path(manifest["training_state"])
    complete = json.loads((state / "complete.json").read_text())
    if complete["step"] != manifest["step"]:
        raise ValueError("Recovery manifest and completion marker disagree")
    validate_state_files(state, complete["world_size"])
    return manifest


def publish_recovery(output, step, world):
    output = Path(output)
    state = output / "training_state" / f"step_{step:06d}"
    manifest = {"step": step, "training_state": str(state)}
    validate_state_files(state, world)
    atomic_json(state / "complete.json", {"step": step, "world_size": world})
    atomic_json(output / "latest_training_state.json", manifest)


def prune_recovery(output, keep=2):
    output = Path(output)
    complete = sorted(p.parent for p in (output / "training_state").glob("step_*/complete.json"))
    obsolete = complete[:-keep]
    inference_path = output / "latest.json"
    if inference_path.exists():
        inference = json.loads(inference_path.read_text())
        if inference.get("training_state") in {str(p) for p in obsolete}:
            inference.pop("training_state")
            inference["training_state_pruned"] = True
            atomic_json(inference_path, inference)
    for path in obsolete:
        shutil.rmtree(path)


def reconcile_metrics(output, restored_step):
    """Archive unsaved updates before replay so the active log stays unambiguous."""
    import time
    output = Path(output)
    path = output / "metrics.jsonl"
    if not path.exists():
        return
    retained, replayed = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            replayed.append(line)
            continue
        (retained if row["step"] <= restored_step else replayed).append(line)
    if replayed:
        (output / f"unsaved_metrics_after_{restored_step}_{time.time_ns()}.jsonl").write_text("".join(replayed))
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(retained))
        temporary.replace(path)
