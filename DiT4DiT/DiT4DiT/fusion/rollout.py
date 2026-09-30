"""Run in the isolated LIBERO Python environment, with CPU OSMesa rendering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LP_NUM_THREADS", "2")

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from deployment.model_server.tools import msgpack_numpy
from DiT4DiT.fusion.protocol import LIMITS, preprocess_observation, seed_for, unnormalize


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--jobs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, default=29700)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--video-trials", type=int, default=1)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{args.port}"
    metadata = {} if args.render_only else json.loads(opener.open(url+"/health", timeout=300).read())
    jobs = json.loads(Path(args.jobs).read_text())
    env = None
    initial_fingerprints = {}
    for job in jobs:
        suite, task_id, trial, condition = job["suite"], job["task"], job["trial"], job["condition"]
        pair_key = suite, task_id, trial
        result_path = output / f"{suite}_{task_id:02d}_{trial:02d}_{condition}.json"
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if (old.get("checkpoint_sha256") == metadata.get("checkpoint_sha256")
                    and old.get("protocol_sha256") == metadata.get("protocol_sha256")):
                if old.get("initial_state_sha256") and old.get("first_observation_sha256"):
                    previous = old["initial_state_sha256"], old["first_observation_sha256"]
                    if initial_fingerprints.setdefault(pair_key, previous) != previous:
                        raise ValueError(f"Stored initial observations are not paired: {pair_key}")
                continue
            raise ValueError(f"Result from a different checkpoint: {result_path}")
        task_suite = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        task = task_suite.get_task(task_id)
        episode_seed = seed_for(suite, task_id, trial, "episode") % (2**31-1)
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        # Controllers/renderers can retain state across env.reset(). Reconstruct
        # the environment after seeding, so every condition starts identically.
        if env is not None:
            env.close()
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
            camera_heights=256, camera_widths=256)
        env.seed(episode_seed)
        env.reset()
        initial_states = torch.load(Path(get_libero_path("init_states"))/task.problem_folder/task.init_states_file,
                                    map_location="cpu", weights_only=False)
        obs = env.set_init_state(initial_states[trial])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0]*6+[-1.0])
        first_example = preprocess_observation(obs, task.language)
        initial_state_sha256 = hashlib.sha256(np.asarray(initial_states[trial]).tobytes()).hexdigest()
        first_observation_sha256 = hashlib.sha256(
            first_example["image"][0].tobytes() + first_example["state"].tobytes()).hexdigest()
        observed = initial_state_sha256, first_observation_sha256
        if initial_fingerprints.setdefault(pair_key, observed) != observed:
            raise ValueError(f"Initial observation changed across conditions: {pair_key}, {condition}")
        frames, actions, input_trace, done = [], [], [], False
        started = time.monotonic()
        for step in range(20 if args.render_only else LIMITS[suite]):
            if step % 8 == 0:
                if args.render_only:
                    chunk = np.tile(np.array([0.]*6+[-1.]), (8,1))
                else:
                    example = preprocess_observation(obs, task.language)
                    query = {"example": example, "intervention": condition,
                             "initial_example": first_example,
                             "episode_seed": episode_seed, "progress": step/LIMITS[suite],
                             "action_seed": seed_for(suite, task_id, trial, step//8, "action"),
                             "feature_seed": seed_for(suite, task_id, trial, step//8, "features"),
                             "protocol_sha256": metadata.get("protocol_sha256")}
                    input_trace.append({"chunk": step//8, "action_seed": query["action_seed"],
                                        "feature_seed": query["feature_seed"],
                                        "current_image_sha256": hashlib.sha256(example["image"][0].tobytes()).hexdigest(),
                                        "initial_image_sha256": hashlib.sha256(first_example["image"][0].tobytes()).hexdigest()})
                    request = urllib.request.Request(url+"/predict", data=msgpack_numpy.packb(query), method="POST")
                    prediction = msgpack_numpy.unpackb(opener.open(request, timeout=300).read())
                    chunk = unnormalize(prediction["normalized_actions"][0], metadata["action_stats"])
            action = chunk[step % 8]
            if not np.isfinite(action).all():
                raise FloatingPointError("Policy returned non-finite action")
            actions.append(action)
            if trial < args.video_trials:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
        result = {**job, "success": bool(done), "steps": len(actions), "seconds": time.monotonic()-started,
                  "episode_seed": episode_seed, "render_only": args.render_only,
                  "initial_state_sha256": initial_state_sha256,
                  "first_observation_sha256": first_observation_sha256, **metadata}
        np.save(result_path.with_suffix(".actions.npy"), np.asarray(actions))
        result_path.with_suffix(".inputs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in input_trace))
        if frames:
            import imageio.v2 as imageio
            imageio.mimwrite(result_path.with_suffix(".mp4"), frames, fps=20)
        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2)); tmp.replace(result_path)
        print(json.dumps({k: v for k, v in result.items() if k != "action_stats"}), flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
