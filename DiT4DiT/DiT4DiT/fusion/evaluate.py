"""Start 16 policy servers; complete whole paired rounds before extending trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from .common import CONDITIONS, SUITES, atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--libero-python",required=True)
    p.add_argument("--workers",type=int,default=16)
    p.add_argument("--minimum-trials",type=int,default=10)
    p.add_argument("--target-trials",type=int,default=50)
    p.add_argument("--hours",type=float,default=10)
    p.add_argument("--full-only",action="store_true")
    args=p.parse_args()
    if not 1 <= args.minimum_trials <= args.target_trials <= 50 or not 1 <= args.workers <= 16:
        raise ValueError("Use 1–50 paired trials and 1–16 workers")
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    servers=[]; logs=[]
    started=time.monotonic()
    try:
        for rank in range(args.workers):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank))
            log=(root/f"server_{rank}.log").open("a"); logs.append(log)
            servers.append(subprocess.Popen([sys.executable,"-m","DiT4DiT.fusion.server","--checkpoint",args.checkpoint,
                "--port",str(29700+rank)],env=env,stdout=log,stderr=subprocess.STDOUT))
        for rank,process in enumerate(servers):
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError(f"Policy server {rank} exited; see its log")
                try:
                    opener.open(f"http://127.0.0.1:{29700+rank}/health",timeout=5).close();break
                except OSError:time.sleep(1)
            else:raise TimeoutError(f"Policy server {rank} did not become ready")
        conditions=("full",) if args.full_only else CONDITIONS
        first=0
        target=args.minimum_trials
        while target<=args.target_trials:
            jobs=[{"suite":s,"task":t,"trial":i,"condition":c}
                  for s in SUITES for t in range(10) for i in range(first,target) for c in conditions]
            def worker(rank):
                job_path=root/f"jobs_{rank}.json"
                atomic_json(job_path,jobs[rank::args.workers])
                with (root/f"rollout_{rank}.log").open("a") as log:
                    subprocess.run([args.libero_python,"-m","DiT4DiT.fusion.rollout","--jobs",str(job_path),
                        "--output",str(root/"episodes"),"--port",str(29700+rank)],check=True,stdout=log,stderr=subprocess.STDOUT)
            round_start=time.monotonic()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(worker,range(args.workers)))
            elapsed=time.monotonic()-round_start
            atomic_json(root/"progress.json",{"complete_trials":target,"seconds":time.monotonic()-started,"full_only":args.full_only})
            subprocess.run([sys.executable,"-m","DiT4DiT.fusion.report","--results",str(root)] +
                (["--full-only"] if args.full_only else []),check=True)
            first=target;target=min(target+10,args.target_trials)
            if target==first:break
            expected=elapsed*(target-first)/(first-(0 if first==args.minimum_trials else first-10))
            if time.monotonic()-started+1.25*expected>args.hours*3600:break
        atomic_json(root/"complete.json",{"trials_per_task":first,"tasks":40,"conditions":conditions})
    finally:
        for process in servers:
            if process.poll() is None:process.terminate()
        for process in servers:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()


if __name__ == "__main__":
    main()
