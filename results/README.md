# 实验结果索引

本目录在 Git 中只保存报告和小型验证元数据；完整 checkpoint、episode 视频、动作数组、缓存和日志保留在实验机器的同名目录中。

## 主结果

- [`libero_ablation_summary.md`](libero_ablation_summary.md)：两轮评测的统一中文汇总、方法差异和限制。
- [`qwen_concat_ablation_step66000_20260918/report.md`](qwen_concat_ablation_step66000_20260918/report.md)：冻结 Qwen 的 66k 步报告。
- [`qwen_concat_ablation_step66000_20260918/report_trials_10_zh.md`](qwen_concat_ablation_step66000_20260918/report_trials_10_zh.md)：冻结 Qwen 的中文完整表。
- [`qwen_joint_ablation_step26000_20260919/report.md`](qwen_joint_ablation_step26000_20260919/report.md)：联合训练 26k 步报告。
- [`qwen_joint_ablation_step26000_20260919/report_trials_10_zh.md`](qwen_joint_ablation_step26000_20260919/report_trials_10_zh.md)：联合训练的中文完整表。

## 评测协议

每轮评测使用相同任务集合、初始状态、首帧、随机种子、动作窗口和干预强度，每个条件 400 次试验。配对区间在固定任务内按试次自举，任务等权；区间不包含训练种子不确定性。mask、首帧冻结、替换和噪声是条件干预，不是两条骨干能力的绝对排名。

两轮实验的关键区别：

| 实验 | 训练设置 |
| --- | --- |
| 66k | Qwen、Cosmos 主干固定；动作损失更新融合层和动作头，视频损失更新 Cosmos。 |
| 26k | Qwen 视觉/语言、Cosmos、融合层和动作头联合更新；动作梯度进入两条主干，VAE 与独立文本编码器固定。 |

## 进一步复核

在原始结果目录中检查 `protocol.json`、`report_trials_10.json`、`fairness_validation_trials_10.json` 和 `complete.json`。报告中的成功率均来自单个训练种子，不能替代多种子实验。
