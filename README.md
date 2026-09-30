# Owen_dit4dit

Qwen + Cosmos + DiT4DiT 的 LIBERO 机器人操作实验代码与结果归档。仓库包含训练、缓存、验证和配对归因评测脚本，以及两轮 40 任务 LIBERO 评测的可复核汇总。

> 这是实验工作区，不是预训练模型发布仓库。模型权重、LIBERO 数据、缓存、检查点、视频和运行日志体积较大，已排除在 Git 提交之外。

## 实验结果

两轮评测均覆盖 LIBERO 40 个任务、每任务 10 次试验、9 种输入条件，共 3,600 回合；公平性审计通过。成功率如下：

| 条件 | 冻结 Qwen，66k 步 | 联合训练，26k 步 | 联合训练相对完整输入下降 |
| --- | ---: | ---: | ---: |
| 完整输入 | **97.50%** | **94.75%** | 0.00 pp |
| 屏蔽 Cosmos | 0.00% | 0.00% | 94.75 pp |
| 屏蔽 Qwen | 90.00% | 88.75% | 6.00 pp |
| 替换 Cosmos 条件 | 0.00% | 0.25% | 94.50 pp |
| 替换 Qwen 条件 | 94.75% | 49.75% | 45.00 pp |
| Cosmos 固定任务首帧 | 0.00% | 0.00% | 94.75 pp |
| Qwen 固定任务首帧 | 91.25% | 40.50% | 54.25 pp |
| 仅 Cosmos 图像加噪 | 32.75% | 55.75% | 39.00 pp |
| 仅 Qwen 图像加噪 | 97.25% | 92.75% | 2.00 pp |

结果的正确解读是：当前模型对 Cosmos 当前观察依赖最强；联合训练模型对陈旧或错配的 Qwen 输入更敏感。两轮训练步数、动作梯度路径和 Qwen 是否解冻同时变化，因此不能把 26k 与 66k 的差异单独归因于 Qwen 解冻，也不能把干预后的成功率下降当作骨干网络的贡献占比。

完整表格、置信区间、协议限制和文件索引见 [`results/README.md`](results/README.md) 与 [`results/libero_ablation_summary.md`](results/libero_ablation_summary.md)。

## 目录

```text
.
├── DiT4DiT/                 # DiT4DiT 主体、Qwen/Cosmos 融合和测试
├── starVLA/                 # StarVLA 依赖及验证训练入口
├── LIBERO/                  # LIBERO 仿真环境源码
├── configs/                 # 训练与验证配置
├── results/                 # 只保留可读报告；原始运行目录在本地保存
├── run_concat.sh            # cache / train / verify 入口
├── run_concat_pipeline.py   # 可恢复的 cache → profile → train → eval 流程
├── train_starvla_verified.py
└── PPU_SETUP.md             # PPU 环境准备说明
```

## 环境与运行

建议使用 Python 3.10、CUDA/PPU 对应的 PyTorch 和独立虚拟环境。先准备上游依赖、LIBERO 数据和本地模型，再按 [`PPU_SETUP.md`](PPU_SETUP.md) 配置路径。

```bash
# 检查数据与模型接口
bash run_concat.sh verify --full

# 构建缓存
bash run_concat.sh cache

# 启动训练
bash run_concat.sh train

# 启动可恢复的完整流程（使用唯一 run id）
.venv-dit4dit/bin/python run_concat_pipeline.py \
  --run-id qwen_concat_libero_all
```

配置文件中的路径记录了本次实验机器的绝对路径。迁移到其他机器时，请修改 `configs/*.yaml` 中的模型、数据、缓存和结果目录；不要把这些大文件复制进 Git。

单元测试位于 `DiT4DiT/tests/`，可在依赖安装完成后运行：

```bash
python -m pytest DiT4DiT/tests
```

## 复现实验结果

报告对应的原始评测目录保留在实验机器的 `results/` 下：

- `qwen_concat_ablation_step66000_20260918/`：Qwen 冻结、66,000 步检查点。
- `qwen_joint_ablation_step26000_20260919/`：Qwen/Cosmos/融合层/动作头联合训练、26,000 步检查点。

每个目录的 `protocol.json`、`report_trials_10.json` 和 `fairness_validation_trials_10.json` 用于核对协议、成功率和公平性审计；episode 视频、动作数组与检查点因体积原因不提交。

## 上游项目

核心架构来自 [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT)，仿真环境来自 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，StarVLA 代码来自 [starVLA](https://github.com/starVLA/starVLA)。本仓库保留实验所需源码和本地改动，预训练权重仍需按各项目说明单独下载。
