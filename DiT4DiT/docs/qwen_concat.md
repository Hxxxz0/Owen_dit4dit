# Cosmos + frozen Qwen on 16 PPUs

This experiment trains a random Action DiT and two adapters with a pretrained
Cosmos video transformer, on all four LIBERO suites (40 tasks). Qwen3-VL-4B,
the Cosmos text encoder, and the VAE are frozen. No LIBERO policy initialization,
gates, branch dropout, or branch-balancing loss is used.

The current main/wrist views are resized to 224×224. Cosmos sees their 224×448
mosaic and future noise slots, producing 784 tokens at block 17. Qwen sees two
separate images and the instruction, producing 114–129 last-layer tokens for
these tasks. Independent LayerNorm/linear adapters map both streams to 2048
dimensions; source embeddings and token concatenation feed the existing DiT-B.

The action loss updates the adapters and Action DiT. The independent video
flow-matching loss updates Cosmos. The action feature pass has no gradient and
never reads real future frames. The supervised VAE target uses the posterior
**mean**, matching upstream; only the conditioning posterior is sampled.

## Run from the parent workspace

```bash
bash setup_concat_libero.sh
bash run_concat.sh verify --full
bash run_concat.sh cache
bash run_concat.sh train
```

The complete monitored pipeline is:

```bash
.venv-dit4dit/bin/python run_concat_pipeline.py --run-id qwen_concat_libero_all_RUN_ID
```

Add `--train-now` to use the configured microbatch/accumulation directly after
cache completion. This skips preliminary profiling, starts the full 20,000-step
schedule, and validates measured throughput and memory during real training.
The initial recovered run uses microbatch 8 × accumulation 2 × 16 devices.
The trainer records finite gradient norms and saves an initial checkpoint plus
all-rank parameter-update evidence at step 100, then at the normal save interval.

Use a unique run ID. Pipeline progress and per-stage logs are under
`results/<run-id>_pipeline/`. Re-running with the same run ID resumes completed
stages; training restarts from the latest complete optimizer/RNG checkpoint.
An exception stops the pipeline and marks its stage failed.

The base configuration is `../configs/qwen_concat_ppu16.yaml`. The pipeline
benchmarks effective global batch 256 with `(microbatch, accumulation)` equal
to `(4,4)`, `(8,2)`, and `(16,1)`. It selects the fastest measured configuration
within an 80 GiB reserved-memory limit. Explicit profiling OOMs are recorded and
skipped; if no candidate fits, `(2,8)` is tested. Other failures stop the pipeline.
Interrupted profiling attempts get separate run directories on restart.
The training target is capped at 20,000
updates and at 28 hours estimated from measured P95 update time. Warmup is
6.25%; Cosmos LR is 1e-5 and adapter/action LR 1e-4, each decaying to 5e-7.

The reference is `examples/LIBERO/train_files/run_libero.sh` plus
`DiT4DiT/config/libero/dit4dit_libero.yaml`: 64 devices × microbatch 4,
80,000 updates and 5,000 warmup updates, BF16 ZeRO-2, four repeated action
diffusion samples, layer 17 features, and the same two learning rates.
This preliminary run keeps its global batch and warmup fraction, shortens
the schedule, and adds frozen Qwen token conditioning.

DeepSpeed owns backward, accumulation, clipping, optimizer and scheduler steps.
PPU PyTorch/DeepSpeed installations must not be replaced by upstream CUDA wheels.
The isolated Python 3.10 LIBERO environment uses CPU PyTorch and OSMesa rendering.

## Cache and checkpoints

`datasets/libero_concat_cache/` contains versioned text embeddings and indexed
tensor shards. The cache stores frozen Qwen features, conditioning VAE posterior
parameters, deterministic target means, actions and states. It does not store
trainable Cosmos features or adapter outputs. Model/dataset manifests and the
preprocessing recipe identify cache compatibility. Each shard has a size/hash
record and a metadata index. Partial caches require the explicit `--smoke` flag.

Every 1,000 updates the trainer writes an inference snapshot, distributed
optimizer state, scheduler state and per-rank Python/NumPy/Torch/PPU RNG states.
It retains two optimizer snapshots and all inference snapshots. Frozen backbone
weights are reconstructed from the versioned local models rather than duplicated.
`latest.json` is written only after saving the checkpoint. `rank_validation.json`
records actual changes to Cosmos, fusion and action parameters on every rank.
Resume uses the published checkpoint tag after all ranks have saved their RNGs.
RNG streams, sample sequences and LR/update counters are restored; BF16 PPU
distributed updates are not guaranteed to be bitwise reproducible. Runtime tests
observed parameter differences below 5e-4 between short continuous runs and
between continuous/resumed runs, with exactly matching RNG states on all ranks.

An explicitly requested performance switch can repartition the same effective
batch using `train --resume --allow-batch-repartition`. Only microbatch and
accumulation may change; the model, data, seed, optimizer, and complete LR
schedule remain guarded. The original sampling chunk size is stored in each
checkpoint so merging two batches of 8 into one batch of 16 preserves the
per-rank sample sequence. Noise/dropout draws and floating-point reduction order
can still differ with the new batch shape. All ranks verify restored optimizer
steps, scheduler steps, and processed-sample counts before training resumes.
After 100 resumed updates, another checkpoint and all-rank update validation are
saved, in addition to the regular 1,000-step interval.

The parent workspace's `switch_concat_batch.py` waits for a published complete
checkpoint before replacing its specified active pipeline. It preserves the
remaining 18-hour intermediate-evaluation budget. Its `batch_repartition.json`
override persists across pipeline restarts without changing `requested.yaml`;
`batch_switch/` records the previous configuration and checkpoint handoff.

Cache workers use a six-hour synchronization timeout because independent decode
times can differ by more than ten minutes. On recovery only missing or explicitly
invalidated shards are assigned, evenly across workers. The source Goal episode
82 wrist video contains one undecodable AV1 frame (105), despite matching its
published SHA256. The decoder uses the nearest valid frame within one frame
interval for this isolated error; all tasks and demonstrations are retained.
Indexed caching always retries the same sample, preventing upstream random retry
from silently changing its identity. Shard 300 was rebuilt to remove ten such
substitutions; `datasets/libero_concat_cache/repairs.json` records the repair.

## Evaluation

Five conditions use the same checkpoint: full input, mask Cosmos, mask Qwen,
same-task donor Cosmos, and same-task donor Qwen. Masks exclude branch tokens
from attention; donors come from a different demonstration and a similar phase.
The policy reuses each observation's features for four action denoising steps.
Initial states and action random streams are paired across conditions.

The pipeline first evaluates all 40 tasks with 10 trials each, then extends in
complete 10-trial rounds toward 50 trials/task while time permits. Reports use
only trials complete across all 40 tasks and all five conditions. Paired bootstrap
intervals resample trials within each fixed task; tasks have equal weight.
Incomplete or mixed-checkpoint result sets cannot silently become a full report.
If training is still running after 18 hours, it pauses at an update boundary,
saves all optimizer states, evaluates full input on all 40 tasks with 5 trials,
then resumes to its original schedule. This intermediate result does not choose
the final checkpoint or change the remaining training schedule.

Offline diagnostics cover 100 observations/task from the training distribution.
They are not held-out generalization measurements. Results describe dependency
under frozen-Qwen training and one training seed, not an absolute backbone ranking.
Inspect full-input success before interpreting ablation effects.
The final directory contains `report.md`, `report.json`, `task_success.csv`,
`offline_report.md`, `offline_task_metrics.csv`, per-episode actions and videos.
Offline summaries include translation/rotation changes, gripper disagreement
and normalized target MSE, with equal weight per task.

## Local validation evidence

### Step-66,000 branch attribution

The dedicated `DiT4DiT.fusion.paired_evaluate` runner reads a frozen
`protocol.json` and evaluates nine conditions: full input, and separate Cosmos
or Qwen masks, donor substitutions, first-frame freezes, and image noise.
`freeze_cosmos` holds only the Cosmos image condition at the episode's initial
image; Qwen continues receiving the current image. `freeze_qwen` does the reverse.
Language and live proprioception retain the same interfaces in every condition.
No inference condition receives observed future images.

Gaussian corruption uses the same seeded pixel noise for either targeted branch
(standard deviation 0.1 of the pixel range). It changes only that branch's
encoded observations. First-image encodings are cached separately from live
proprioception; only the selected feature fields are substituted.

Use sixteen policy servers with up to three CPU simulators per device. Each
task/trial's conditions stay on the same device and use a predetermined random
condition order. Every episode reconstructs its environment after setting the
seeds: reusing an environment can retain controller state even after reset.
Exact initial observation/state hashes are checked before a paired episode runs,
and are checked again by the report. Reports also reject mixed checkpoint or
protocol hashes. Request traces preserve current/initial image hashes and both
action and feature seeds. Initial-frame and current-frame interventions must
produce identical first action chunks when those frames are identical.

The predeclared rounds contain 1, 5 and 10 initial states per task across all
40 tasks. The one-trial pilot is descriptive; subsequent reports use equal-task
paired bootstrap intervals. The primary comparison is the success drop from
freezing Cosmos minus the drop from freezing Qwen. Mask, noise and donor tests
are exploratory cross-checks, not independent claims about absolute backbone
quality. Runtime errors are not scored as task failures.

Tests are in `tests/test_fusion_attribution.py`. The step-66,000 experiment's
protocol, exact source snapshot, checkpoint reconstruction proof, reset audit,
episode actions/videos and reports are stored in the parent workspace under
`results/qwen_concat_ablation_step66000_20260918/`.

Runtime evidence is in the parent workspace `logs/`:

- `concat_validation.json`: native masks, gradients, frozen encoders, upstream
  Cosmos feature parity, and exclusion of ground-truth future images.
- `concat_resume_validation.json`: all-rank RNG equality, step/LR recovery and
  observed BF16 numerical variation.
- `concat_protocol_tests.log`: sampler resume, per-group LR floors, complete
  paired rounds, checkpoint/seed rejection, and offline aggregation checks.
- `libero_render_validation/`: real OSMesa rendering from all four suites.
- `concat_closed_loop_smoke/`: one real task under all five conditions using a
  three-update checkpoint. This tests runtime integration, not policy quality.
