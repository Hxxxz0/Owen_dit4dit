# DiT4DiT and StarVLA on PPU

Both repositories and their virtual environments live in this CPFS directory.
The environments inherit the image-provided PPU builds of PyTorch,
torchvision, torchaudio, Triton, and DeepSpeed. Do not install upstream CUDA
PyTorch wheels into these environments.

## Activate

```bash
cd /mnt/home/jiahaozhe/project/qwen_dit4dit
source .venv-dit4dit/bin/activate   # DiT4DiT
# or
source .venv-starvla/bin/activate  # StarVLA
```

## Verify PPU execution

```bash
./run_ppu_smoke_tests.sh
ppu-smi
```

The smoke tests execute a matrix multiplication plus a project-native action
module on `cuda:0`, which is the PPU SDK's CUDA-compatible device interface.
They do **not** establish that full-model training or LIBERO evaluation works.

## LIBERO / StarVLA 16-PPU validation

Verified on 2026-09-17: **PASS**, real pretrained QwenOFT training on all 16
PPU-ZW810E devices, 3 optimizer steps per rank, launcher exit status 0.
All ranks recorded finite losses and nonzero backbone/action-head updates.
The final checkpoint was read back with `weights_only=True`; all 730 saved
tensors were finite, and its backbone update matched the per-rank evidence.
See `logs/starvla_ppu16_checkpoint_validation.json` and
`results/starvla_libero_ppu16/ppu_rank_training_validation.json`.

LIBERO integrity and loader validation also passed: 1,693 episodes,
273,465 frames, 3,386 videos across Spatial, Object, Goal and LIBERO-10.
Downloaded LFS files were rehashed; three real samples per suite were decoded
through the training loader. See `logs/libero_data_validation.json`.

This establishes basic training compatibility only. Mean losses across the
three batches were 0.6590, 6.0965 and 1.6198; the brief no-warmup test is not
evidence of convergence, policy quality, sustained throughput or optimal
hyperparameters. All PPU processes exited after checkpoint verification.

All four datasets are stored under `datasets/libero/`. The downloader records
the resolved Hugging Face commit in each `download_manifest.json` and checks
sizes and LFS SHA256 hashes on downloaded files. It uses scoped DNS-over-HTTPS
resolution for Hugging Face hosts without modifying the machine DNS settings.

```bash
.venv-dit4dit/bin/python download_libero_assets.py datasets
.venv-dit4dit/bin/python download_libero_assets.py qwen
NO_ALBUMENTATIONS_UPDATE=1 .venv-starvla/bin/python validate_libero_download.py
bash train_starvla_ppu16.sh > logs/starvla_libero_ppu16.log 2>&1
```

For the completed download, Qwen's official ModelScope mirror was used for
the large shards after checking its content hashes against the resolved HF
revision; both complete local shards subsequently passed SHA256 validation.
Use `QWEN_USE_MODELSCOPE=1` on the Qwen downloader command to select that path.
No HF credentials are sent to the mirror. Partial/chunk files are retained to
support recovery and therefore the model directory uses more disk than the
final weights alone.

The training command runs the actual upstream `QwenOFT` trainer with the full
Qwen3-VL-4B backbone, LIBERO four-suite mixture, two camera views, action horizon
8, batch size 1 per PPU (global batch 16), BF16, gradient checkpointing and
DeepSpeed ZeRO-2. Its default run is only **3 optimizer steps**, not a convergence
or robotics evaluation benchmark. It does not require Cosmos. WandB is disabled.

`train_starvla_verified.py` checks finite losses and actual backbone/action-head
parameter updates on **every rank**, then saves the trainer's final checkpoint.
Successful runs produce `results/starvla_libero_ppu16/ppu_rank_training_validation.json`
and `results/starvla_libero_ppu16/final_model/pytorch_model.pt`. Do not infer success
from the presence of a run directory alone; check the exit status and the
`PPU_TRAIN_VERIFIED` message in the log.

```bash
.venv-starvla/bin/python verify_starvla_checkpoint.py
```

Independent 16-PPU communication/DDP and DeepSpeed preflight results are in
`logs/ppu_distributed_preflight.json` and `logs/ppu_deepspeed_preflight.json`.
These use tiny synthetic models and are not substitutes for the real training run.

For a new validation run, avoid reusing a run name with existing checkpoints:

```bash
bash train_starvla_ppu16.sh --run_id starvla_libero_ppu16_repeat
```

## Cosmos + Qwen full LIBERO experiment

Cosmos-Predict2.5-2B and Qwen3-VL-4B-Instruct are now downloaded and verified.
The 40-task concatenation experiment is implemented; see
[the experiment guide](DiT4DiT/docs/qwen_concat.md) and
[its configuration](configs/qwen_concat_ppu16.yaml).

The real Cosmos, fusion adapters and random Action DiT have completed 16-PPU
training and checkpoint restore tests. Frozen Qwen/VAE cache parity, separated
video/action gradients, attention masks and future-frame isolation passed.
The isolated LIBERO simulator rendered all four suites and executed a complete
real task under all five interventions using a three-update checkpoint.
These checks establish runtime correctness, not policy convergence.

```bash
bash setup_concat_libero.sh
.venv-dit4dit/bin/python run_concat_pipeline.py --run-id qwen_concat_libero_all_RUN_ID
```

The pipeline caches frozen features on all 16 PPUs, profiles global batch 256,
trains with BF16 ZeRO-2 within the measured budget, and generates paired
40-task evaluation reports. Read `results/<run-id>_pipeline/state.json` and
the corresponding stage logs for live progress.

## Rebuild the environments

```bash
./setup_ppu_envs.sh
```

The setup script deliberately filters upstream `torch`, `torchvision`,
`torchaudio`, `triton`, NVIDIA CUDA component, and DeepSpeed pins so they do
not replace the working PPU stack.
