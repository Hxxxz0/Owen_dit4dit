"""Runtime and saved-trajectory controls for paired attribution evaluations."""
import json
from pathlib import Path
import urllib.request

import numpy as np

from .common import atomic_json, file_sha
from .protocol import SUITES, seed_for


def runtime_preflight(root, opener, url, protocol):
    from deployment.model_server.tools import msgpack_numpy

    example = msgpack_numpy.unpackb(Path(protocol["runtime_probe"]).read_bytes())
    query = {"example": example, "initial_example": example, "episode_seed": 42,
             "progress": 0., "action_seed": 42, "feature_seed": 42,
             "protocol_sha256": file_sha(root / "protocol.json")}

    def predict(condition):
        request = urllib.request.Request(url + "/predict", method="POST",
            data=msgpack_numpy.packb({**query, "intervention": condition}))
        with opener.open(request, timeout=300) as response:
            result = msgpack_numpy.unpackb(response.read())
        actions = np.asarray(result["normalized_actions"])
        if not actions.size or not np.isfinite(actions).all():
            raise ValueError(f"Invalid runtime probe actions: {condition}")
        return actions

    outputs = {condition: predict(condition) for condition in protocol["conditions"]}
    for condition in ("freeze_cosmos", "freeze_qwen"):
        if not np.array_equal(outputs["full"], outputs[condition]):
            raise ValueError(f"Initial=current must make {condition} identical to full")
    if not np.array_equal(outputs["full"], predict("full")):
        raise ValueError("Repeated seeded full predictions differ")
    atomic_json(root / "runtime_intervention_validation.json", {
        "passed": True, "all_nine_conditions_finite": True,
        "freeze_identity_when_initial_equals_current": True,
        "repeated_full_bitwise_equal": True,
        "protocol_sha256": query["protocol_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"]})


def audit_round(root, trials, protocol):
    root = Path(root)
    protocol_hash = file_sha(root / "protocol.json")
    baseline = Path(protocol["reference_evaluation"])
    successes = {condition: 0 for condition in protocol["conditions"]}
    for suite in SUITES:
        for task in range(10):
            for trial in range(trials):
                reference_path = baseline / "episodes" / f"{suite}_{task:02d}_{trial:02d}_full.json"
                reference = json.loads(reference_path.read_text())
                first_actions, initial_images = {}, set()
                for condition in protocol["conditions"]:
                    path = root / "episodes" / f"{suite}_{task:02d}_{trial:02d}_{condition}.json"
                    row = json.loads(path.read_text())
                    if (row["checkpoint_sha256"] != protocol["checkpoint_sha256"]
                            or row["protocol_sha256"] != protocol_hash):
                        raise ValueError(f"Mixed provenance: {path}")
                    for key in ("initial_state_sha256", "first_observation_sha256", "episode_seed"):
                        if row[key] != reference[key]:
                            raise ValueError(f"Pairing differs from the previous evaluation: {path}, {key}")
                    actions = np.load(path.with_suffix(".actions.npy"), allow_pickle=False)
                    if len(actions) != row["steps"] or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid saved actions: {path}")
                    first_actions[condition] = actions[:8]
                    traces = [json.loads(line) for line in path.with_suffix(".inputs.jsonl").read_text().splitlines()]
                    if len(traces) != (len(actions) + 7) // 8:
                        raise ValueError(f"Missing input trace: {path}")
                    for index, trace in enumerate(traces):
                        if (trace["chunk"] != index
                                or trace["action_seed"] != seed_for(suite, task, trial, index, "action")
                                or trace["feature_seed"] != seed_for(suite, task, trial, index, "features")):
                            raise ValueError(f"Incorrect paired random seeds: {path}")
                        initial_images.add(trace["initial_image_sha256"])
                    if traces[0]["current_image_sha256"] != traces[0]["initial_image_sha256"]:
                        raise ValueError(f"First current image differs from initial image: {path}")
                    successes[condition] += int(row["success"])
                if len(initial_images) != 1:
                    raise ValueError("Initial image changed within a paired task/trial")
                for condition in ("freeze_cosmos", "freeze_qwen"):
                    if not np.array_equal(first_actions["full"], first_actions[condition]):
                        raise ValueError(f"First action chunk differs: {suite}/{task}/{trial}/{condition}")
    audit = {"passed": True, "episodes": 40 * trials * len(successes),
             "paired_task_trials": 40 * trials, "successes": successes,
             "first_action_chunks_bitwise_equal": True, "all_actions_finite": True,
             "same_initial_states_images_and_seeds_as_reference": True,
             "reference_evaluation": str(baseline), "protocol_sha256": protocol_hash}
    atomic_json(root / f"fairness_validation_trials_{trials:02d}.json", audit)
    return audit


def write_comparison(root, trials, protocol):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    reference = json.loads((Path(protocol["reference_evaluation"]) / f"report_trials_{trials:02d}.json").read_text())
    labels = {"full": "完整输入", "mask_cosmos": "屏蔽 Cosmos", "mask_qwen": "屏蔽 Qwen",
              "freeze_cosmos": "Cosmos 固定任务首帧", "freeze_qwen": "Qwen 固定任务首帧",
              "noise_cosmos": "仅 Cosmos 图像加噪", "noise_qwen": "仅 Qwen 图像加噪",
              "swap_cosmos": "仅替换 Cosmos 观察", "swap_qwen": "仅替换 Qwen 观察"}
    lines = [f"# 联合训练 {protocol['checkpoint_step']:,} 步配对评测", "",
             f"40 个任务 × 每任务 {trials} 次 × 9 个条件，共 {40 * trials * 9} 个回合；公平性审计通过。", "",
             "| 条件 | 新联合训练模型成功率 | 旧 66,000 步模型成功率 | 新模型相对完整输入下降 |",
             "|---|---:|---:|---:|"]
    for condition in protocol["conditions"]:
        new = report["conditions"][condition]
        old = reference["conditions"][condition]
        lines.append(f"| {labels[condition]} | {new['success_rate']:.2%} | {old['success_rate']:.2%} | {100 * new.get('drop', 0):.2f} pp |")
    lines += ["", "两次评测使用相同任务、试次、初始化、观察预处理、随机种子、动作窗口和干预强度。",
              "冻结仅作用于选定支路的图像，另一支路使用当前帧；语言和本体状态保持原规则。",
              "新旧模型训练步数不同；新实验同时解冻 Qwen 并让动作梯度进入两条主干，因此变化不能仅归因于 Qwen 解冻。",
              "干预下降表示该检查点的条件依赖，不能直接换算成两条主干的贡献百分比。"]
    if trials == 1:
        lines += ["", "这是每任务一次的试跑，只有描述性结果，不报告推断置信区间。"]
    if report["conditions"]["full"]["success_rate"] < .5:
        lines += ["", "完整输入成功率低于 50%，存在地板效应，不据此强行排列两条支路贡献。"]
    path = root / f"report_trials_{trials:02d}_zh.md"
    path.write_text("\n".join(lines) + "\n")
