"""Equal-task-weight, paired success differences; no outcome-based selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .common import CONDITIONS, SUITES, atomic_json


OFFLINE_METRICS = ("translation_delta_l2", "rotation_delta_l2", "gripper_disagreement", "normalized_action_mse")


def summarize_offline(records, expected_tasks=40, expected_per_task=100):
    if not records:
        return {"status": "missing"}
    hashes = {r["checkpoint_sha256"] for r in records}
    if len(hashes) != 1:
        raise ValueError("Offline report requires exactly one checkpoint")
    groups = {}
    for row in records:
        key = (row["lang"], row["sample_index"])
        group = groups.setdefault(key, {})
        if row["condition"] in group:
            raise ValueError(f"Duplicate offline observation/condition: {key}")
        group[row["condition"]] = row
    languages = sorted({key[0] for key in groups})
    counts = {lang: sum(key[0] == lang and set(rows) == set(CONDITIONS)
                       for key, rows in groups.items()) for lang in languages}
    complete = (len(languages) == expected_tasks and all(n == expected_per_task for n in counts.values())
                and len(groups) == expected_tasks * expected_per_task)
    report = {"status": "complete" if complete else "incomplete", "checkpoint_sha256": next(iter(hashes)),
              "observations_per_task": counts, "distribution": "training; not held out"}
    if not complete:
        return report
    tasks = []
    for lang in languages:
        paired = [rows for (task, _), rows in groups.items() if task == lang]
        for condition in CONDITIONS:
            tasks.append({"instruction": lang, "condition": condition,
                          **{metric: float(np.mean([r[condition][metric] for r in paired]))
                             for metric in OFFLINE_METRICS}})
    report["tasks"] = tasks
    report["conditions"] = {c: {metric: float(np.mean([r[metric] for r in tasks if r["condition"] == c]))
                                   for metric in OFFLINE_METRICS} for c in CONDITIONS}
    return report


def offline_lines(report):
    if report["status"] != "complete":
        return ["", f"Offline diagnostic status: {report['status']}"]
    lines = ["", "## Offline action sensitivity", "",
             "100 observations per task from the training distribution, with equal task weights. These are not held-out results.", "",
             "| Condition | Translation delta L2 | Rotation delta L2 | Gripper disagreement | Normalized action MSE |",
             "|---|---:|---:|---:|---:|"]
    for c, row in report["conditions"].items():
        lines.append(f"| {c} | {row['translation_delta_l2']:.5f} | {row['rotation_delta_l2']:.5f} | "
                     f"{row['gripper_disagreement']:.1%} | {row['normalized_action_mse']:.5f} |")
    lines += ["", "Deltas are relative to full input after the official action postprocessing; MSE uses valid normalized target dimensions."]
    return lines


def summarize_full(records):
    records = [r for r in records if not r.get("render_only", False)]
    if not records:
        return {"status": "incomplete", "completed_episodes": 0}
    if len({r["checkpoint_sha256"] for r in records}) != 1 or any(r["condition"] != "full" for r in records):
        raise ValueError("Full-only report requires one checkpoint and only full-input episodes")
    groups = {(r["suite"], r["task"], r["trial"]): r for r in records}
    if len(groups) != len(records):
        raise ValueError("Duplicate full-input episodes")
    tasks = [(s, t) for s in SUITES for t in range(10)]
    trials = [i for i in range(50) if all((*task, i) in groups for task in tasks)]
    if not trials:
        return {"status": "incomplete", "completed_episodes": len(records)}
    values = np.array([[groups[(*task, i)]["success"] for i in trials] for task in tasks], dtype=float)
    return {"status": "complete", "checkpoint_sha256": records[0]["checkpoint_sha256"],
            "full_only": True, "tasks": 40, "trials_per_task": len(trials), "success_rate": float(values.mean()),
            "suites": {s: float(values[j*10:(j+1)*10].mean()) for j, s in enumerate(SUITES)}}


def summarize(records, bootstrap=10000, conditions=CONDITIONS):
    conditions = tuple(conditions)
    if not conditions or conditions[0] != "full" or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be unique, with full as the reference")
    records = [r for r in records if not r.get("render_only", False)]
    if not records:
        return {"status": "incomplete", "completed_episodes": 0}
    hashes = {r["checkpoint_sha256"] for r in records}
    if len(hashes) != 1:
        raise ValueError("Report requires exactly one checkpoint")
    protocols = {r.get("protocol_sha256") for r in records}
    if len(protocols) != 1:
        raise ValueError("Report cannot combine different evaluation protocols")
    groups = {}
    for r in records:
        if r["suite"] not in SUITES or r["task"] not in range(10) or r["trial"] not in range(50) or r["condition"] not in conditions:
            raise ValueError("Unexpected task, trial, or condition in evaluation results")
        key = (r["suite"], r["task"], r["trial"])
        entry = groups.setdefault(key, {})
        if r["condition"] in entry:
            raise ValueError(f"Duplicate evaluation result: {key}, {r['condition']}")
        entry[r["condition"]] = r
    for key, rows in groups.items():
        if len({r.get("episode_seed") for r in rows.values()}) != 1:
            raise ValueError(f"Conditions are not paired by episode seed: {key}")
        if next(iter(protocols)) is not None:
            for field in ("initial_state_sha256", "first_observation_sha256"):
                values = {r.get(field) for r in rows.values()}
                if None in values or len(values) != 1:
                    raise ValueError(f"Conditions have different {field}: {key}")
    tasks = [(s,t) for s in SUITES for t in range(10)]
    complete_trials = []
    for trial in range(50):
        if all(set(groups.get((*task,trial), {})) >= set(conditions) for task in tasks):
            complete_trials.append(trial)
    if not complete_trials:
        return {"status": "incomplete", "checkpoint_sha256": next(iter(hashes)),
                "completed_episodes": len(records), "message": "No complete paired trial across all 40 tasks and all conditions"}
    values = np.array([[[groups[(*task,trial)][c]["success"] for c in conditions]
                       for trial in complete_trials] for task in tasks], dtype=np.float32)
    rates = values.mean(axis=(0,1))
    differences = values[:, :, :1] - values[:, :, 1:]
    rng = np.random.default_rng(42)
    samples = []
    for start in range(0,bootstrap,250):
        indices = rng.integers(len(complete_trials),size=(min(250,bootstrap-start),40,len(complete_trials)))
        draws = differences[np.arange(40)[None,:,None],indices]
        samples.append(draws.mean(axis=(1,2)))
    samples = np.concatenate(samples)
    ci = np.percentile(samples,[2.5,97.5],axis=0)
    comparisons = {}
    for method in ("freeze", "mask", "swap", "noise"):
        cosmos, qwen = f"{method}_cosmos", f"{method}_qwen"
        if cosmos in conditions and qwen in conditions:
            c, q = conditions.index(cosmos) - 1, conditions.index(qwen) - 1
            comparisons[method] = {
                "cosmos_minus_qwen_drop": float(differences[:, :, c].mean() - differences[:, :, q].mean()),
                "paired_ci95": np.percentile(samples[:, c] - samples[:, q], [2.5, 97.5]).tolist()
                    if len(complete_trials) >= 2 else None}
    return {"status": "complete", "checkpoint_sha256": next(iter(hashes)),
        "protocol_sha256": next(iter(protocols)),
        "trials_per_task": len(complete_trials), "trial_indices": complete_trials,
        "training_seed_replicates": 1, "task_count": 40,
        "conditions": {c: {"success_rate":float(rates[i]), **({"drop":float(differences[:,:,i-1].mean()),
                            "paired_ci95":[float(ci[0,i-1]),float(ci[1,i-1])] if len(complete_trials) >= 2 else None} if i else {})}
                       for i,c in enumerate(conditions)},
        "branch_comparisons": comparisons,
        "cosmos_minus_qwen_drop_ci95": {method: result["paired_ci95"] for method, result in comparisons.items()},
        "suites": {s: {c:float(values[i*10:(i+1)*10,:,k].mean()) for k,c in enumerate(conditions)} for i,s in enumerate(SUITES)},
        "tasks": [{"suite":s,"task":t,**{c:float(values[i,:,k].mean()) for k,c in enumerate(conditions)}} for i,(s,t) in enumerate(tasks)],
        "interpretation": "exploratory_low_baseline_success" if rates[0] < .5 else "paired_checkpoint_interventions",
        "notes": ["Bootstrap resamples paired trials within each fixed task; task weights are equal.",
                  "The one-trial pilot has no inferential confidence interval; comparisons are descriptive and unadjusted for multiplicity.",
                  "Ablation effects are conditional on this training recipe and this checkpoint.",
                  "Masking, donor swaps and image noise are distribution shifts, not absolute backbone rankings."]}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--results",required=True)
    p.add_argument("--offline-only",action="store_true")
    p.add_argument("--full-only",action="store_true")
    args=p.parse_args()
    root=Path(args.results)
    offline = summarize_offline([json.loads(line) for path in sorted(root.glob("offline_rank_*.jsonl"))
                                for line in path.read_text().splitlines()])
    atomic_json(root/"offline_report.json",offline)
    if offline["status"] == "complete":
        with (root/"offline_task_metrics.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["instruction","condition",*OFFLINE_METRICS])
            writer.writeheader();writer.writerows(offline["tasks"])
        (root/"offline_report.md").write_text("\n".join(offline_lines(offline))+"\n")
    if args.offline_only:
        if offline["status"] != "complete":
            raise ValueError(f"Incomplete offline diagnostics: {offline}")
        return
    records=[]
    for path in root.glob("episodes/*.json"):
        r=json.loads(path.read_text())
        if "success" in r: records.append(r)
    if args.full_only:
        report=summarize_full(records)
        atomic_json(root/"report.json",report)
        if report["status"] != "complete":
            raise ValueError(f"Incomplete full-input evaluation: {report}")
        lines=["# Intermediate full-input LIBERO evaluation", "",
               f"40 tasks, {report['trials_per_task']} trials/task; success {report['success_rate']:.1%}.", "",
               f"Checkpoint: `{report['checkpoint_sha256']}`", "", "| Suite | Success |", "|---|---:|"]
        lines += [f"| {suite} | {rate:.1%} |" for suite,rate in report["suites"].items()]
        (root/"report.md").write_text("\n".join(lines)+"\n")
        print(json.dumps(report,indent=2))
        return
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    conditions = tuple(protocol.get("conditions", CONDITIONS))
    report=summarize(records, conditions=conditions)
    if offline.get("checkpoint_sha256") and report.get("checkpoint_sha256") != offline["checkpoint_sha256"]:
        raise ValueError("Offline and closed-loop diagnostics use different checkpoints")
    report["offline"]={k:v for k,v in offline.items() if k!="tasks"}
    atomic_json(root/"report.json",report)
    if report["status"] != "complete":
        print(report); return
    with (root/"task_success.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["suite","task",*conditions]); writer.writeheader(); writer.writerows(report["tasks"])
    lines=["# Qwen + DiT4DiT paired LIBERO evaluation", "",f"Checkpoint: `{report['checkpoint_sha256']}`", "",
           f"40 tasks, {report['trials_per_task']} paired trials/task, one training seed.", "",
           "| Condition | Success | Drop vs full | 95% paired interval |", "|---|---:|---:|---:|"]
    for c,row in report["conditions"].items():
        interval=row.get("paired_ci95")
        lines.append(f"| {c} | {row['success_rate']:.1%} | {row.get('drop',0):.1%} | " +
                     (f"[{interval[0]:.1%}, {interval[1]:.1%}]" if interval else "—")+" |")
    lines += ["",f"Interpretation: {report['interpretation']}","",*report["notes"]]
    lines += ["", "## Cosmos minus Qwen success drop", "", "Positive values mean a larger drop when intervening on Cosmos.", ""]
    for method, result in report["branch_comparisons"].items():
        lines.append(f"- {method}: {result['cosmos_minus_qwen_drop']:.1%}; paired 95% interval: {result['paired_ci95']}")
    lines += offline_lines(offline)
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},indent=2))


if __name__ == "__main__":
    main()
