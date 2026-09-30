"""Resumable cache -> profile -> train -> paired evaluation pipeline.

Run with .venv-dit4dit/bin/python. Only explicit profiling OOMs are skipped.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "DiT4DiT"))
from omegaconf import OmegaConf
from DiT4DiT.fusion.common import atomic_json
from DiT4DiT.fusion.resume import validate_resume_config


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id",required=True)
    p.add_argument("--config",default=str(ROOT/"configs/qwen_concat_ppu16.yaml"))
    p.add_argument("--train-now",action="store_true",help="Start the configured global-256 training run without preliminary profiling")
    args=p.parse_args()
    state_dir=ROOT/"results"/(args.run_id+"_pipeline");state_dir.mkdir(parents=True,exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    with (ROOT/"logs/concat_pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run_pipeline(args,state_dir)
        except BaseException as error:
            path=state_dir/"state.json"
            state=json.loads(path.read_text()) if path.exists() else {"run_id":args.run_id}
            state.update(status="failed",error_type=type(error).__name__,error=str(error))
            atomic_json(path,state)
            raise


def run_pipeline(args,state_dir):
    state_path=state_dir/"state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"run_id":args.run_id,"completed":[]}
    selection_path=state_dir/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()).get("selected",{}).get("selection","").startswith("configured;"):
        args.train_now=True
    for key in ("exit_code","error_type","error"):
        state.pop(key,None)
    state.update(pid=os.getpid(),status="running")
    os.chdir(ROOT)
    env=dict(os.environ,HF_HOME=str(ROOT/".cache/huggingface"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",TOKENIZERS_PARALLELISM="false",NO_ALBUMENTATIONS_UPDATE="1",OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="2",NCCL_DEBUG="WARN",LIBERO_CONFIG_PATH=str(ROOT/".cache/libero"),
        PYTHONPATH=str(ROOT/"DiT4DiT")+":"+str(ROOT/"LIBERO"),TRITON_CACHE_DIR=str(ROOT/".cache/triton"),
        TORCH_EXTENSIONS_DIR=str(ROOT/".cache/torch_extensions"),MUJOCO_GL="osmesa",PYOPENGL_PLATFORM="osmesa",LP_NUM_THREADS="2")
    python=str(ROOT/".venv-dit4dit/bin/python")
    libero_python=str(ROOT/".venv-libero/bin/python")
    def distributed(module,*tail):
        return [python,"-m","torch.distributed.run","--nproc_per_node=16","--master_port=29641","-m",module,*tail]
    def stage(name,command,allow_oom=False):
        if name in state["completed"]: return True
        if name in state.get("excluded",{}): return False
        state.update(stage=name,command=command,stage_started=time.time(),status="running")
        atomic_json(state_path,state)
        print(f"START {name}",flush=True)
        with (state_dir/f"{name}.log").open("a") as log:
            offset=log.tell()
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            with (state_dir/f"{name}.log").open() as log:
                log.seek(offset)
                message=log.read().lower()
            if allow_oom and ("out of memory" in message or "outofmemoryerror" in message):
                state.setdefault("excluded",{})[name]={"reason":"out_of_memory","exit_code":result.returncode}
                atomic_json(state_path,state)
                print(f"SKIP {name}: out of memory",flush=True)
                return False
            state.update(status="failed",exit_code=result.returncode)
            atomic_json(state_path,state)
            raise RuntimeError(f"Stage {name} failed; inspect {state_dir/name}.log")
        state["completed"].append(name);atomic_json(state_path,state)
        print(f"DONE {name}",flush=True)
        return True
    cfg=OmegaConf.load(args.config)
    snapshot=state_dir/"requested.yaml"
    if snapshot.exists() and OmegaConf.to_container(OmegaConf.load(snapshot),resolve=True)!=OmegaConf.to_container(cfg,resolve=True):
        raise ValueError("Pipeline configuration changed; use a new run ID")
    OmegaConf.save(cfg,snapshot)
    args.config=str(snapshot)
    render_jobs=state_dir/"render_jobs.json"
    atomic_json(render_jobs,[{"suite":s,"task":t,"trial":0,"condition":"full"}
        for s in ("libero_spatial","libero_object","libero_goal","libero_10") for t in range(10)])
    stage("verify",[python,"-m","DiT4DiT.fusion.verify","--config",args.config,"--full","--output",str(state_dir/"validation.json")])
    stage("render",[libero_python,"-m","DiT4DiT.fusion.rollout","--jobs",str(render_jobs),
        "--output",str(state_dir/"render_validation"),"--render-only"])
    stage("cache",distributed("DiT4DiT.fusion.cache","--config",args.config))
    profiles=[]
    def profile(batch,accum):
        name=f"profile_b{batch}"
        attempts=state.setdefault("profile_attempts",{})
        if name not in state["completed"] and name not in state.get("excluded",{}):
            attempts[name]=attempts.get(name,0)+1
            atomic_json(state_path,state)
        run_id=args.run_id+"_"+name+f"_attempt{attempts.get(name,1)}"
        if stage(name,distributed("DiT4DiT.fusion.train","--config",args.config,"--profile",
            f"run_id={run_id}",f"datasets.vla_data.per_device_batch_size={batch}",f"trainer.gradient_accumulation_steps={accum}"),allow_oom=True):
            profiles.append(json.loads((Path(cfg.run_root_dir)/run_id/"profile.json").read_text()))
    for batch,accum in (() if args.train_now else ((4,4),(8,2),(16,1))):
        profile(batch,accum)
    eligible=[r for r in profiles if r["max_reserved_gib"]<=float(cfg.trainer.max_reserved_gib)]
    if not eligible and not args.train_now:
        profile(2,8)
        eligible=[r for r in profiles if r["max_reserved_gib"]<=float(cfg.trainer.max_reserved_gib)]
    if args.train_now:
        best={"batch_size":int(cfg.datasets.vla_data.per_device_batch_size),
              "accumulation":int(cfg.trainer.gradient_accumulation_steps),
              "budget_steps":int(cfg.trainer.max_train_steps),"selection":"configured; validate during initial training"}
        if best["batch_size"]*best["accumulation"]*16!=256:
            raise ValueError("Direct training still requires effective global batch 256")
    else:
        if not eligible:raise RuntimeError("No profiled configuration fits the memory budget")
        best=max(eligible,key=lambda r:r["samples_per_second"])
    if best["budget_steps"]<100:raise RuntimeError("Measured throughput cannot support the experiment budget")
    repartition_path=state_dir/"batch_repartition.json"
    if repartition_path.exists():
        repartition=json.loads(repartition_path.read_text())
        candidate=OmegaConf.to_container(cfg,resolve=True)
        candidate["datasets"]["vla_data"]["per_device_batch_size"]=int(repartition["batch_size"])
        candidate["trainer"]["gradient_accumulation_steps"]=int(repartition["accumulation"])
        validate_resume_config(OmegaConf.to_container(cfg,resolve=True),candidate,True)
        best={**best,"batch_size":int(repartition["batch_size"]),"accumulation":int(repartition["accumulation"]),
              "selection":"configured; authorized batch repartition with preserved training state"}
    cfg.run_id=args.run_id
    cfg.datasets.vla_data.per_device_batch_size=best["batch_size"]
    cfg.trainer.gradient_accumulation_steps=best["accumulation"]
    cfg.trainer.max_train_steps=best["budget_steps"]
    selected=state_dir/"selected.yaml"
    OmegaConf.save(cfg,selected);atomic_json(state_dir/"selection.json",{"selected":best,"profiles":profiles,"excluded":state.get("excluded",{})})
    training=Path(cfg.run_root_dir)/cfg.run_id
    remaining_hours=max(0.,18-float(state.get("train_first_elapsed_seconds",0))/3600)
    command=distributed("DiT4DiT.fusion.train","--config",str(selected),"--pause-after-hours",str(remaining_hours))
    if (training/"latest.json").exists():
        command.append("--resume")
        if repartition_path.exists():command.append("--allow-batch-repartition")
    stage("train_first",command)
    checkpoint=json.loads((training/"latest.json").read_text())["checkpoint"]
    if not (training/"complete.json").exists():
        stage("intermediate_eval",[python,"-m","DiT4DiT.fusion.evaluate","--checkpoint",checkpoint,
            "--output",str(state_dir/"intermediate_eval"),"--libero-python",libero_python,
            "--full-only","--minimum-trials","5","--target-trials","5"])
        finish=distributed("DiT4DiT.fusion.train","--config",str(selected),"--resume")
        if repartition_path.exists():finish.append("--allow-batch-repartition")
        stage("train_finish",finish)
    checkpoint=json.loads((training/"latest.json").read_text())["checkpoint"]
    final=state_dir/"final_evaluation"
    stage("offline",distributed("DiT4DiT.fusion.diagnose","--checkpoint",checkpoint,"--output",str(final)))
    stage("offline_report",[python,"-m","DiT4DiT.fusion.report","--results",str(final),"--offline-only"])
    stage("closed_loop",[python,"-m","DiT4DiT.fusion.evaluate","--checkpoint",checkpoint,"--output",str(final),
        "--libero-python",libero_python,"--hours",str(cfg.evaluation.budget_hours)])
    state.update(status="complete",checkpoint=checkpoint,report=str(final/"report.md"))
    atomic_json(state_path,state)


if __name__=="__main__":
    main()
