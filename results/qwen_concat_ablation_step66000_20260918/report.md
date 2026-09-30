# Qwen + DiT4DiT paired LIBERO evaluation

Checkpoint: `9adc6205d10188f410ee3ea93ee7d515df2bfc1279a4600f4f23dc5d1eab05df`

40 tasks, 10 paired trials/task, one training seed.

| Condition | Success | Drop vs full | 95% paired interval |
|---|---:|---:|---:|
| full | 97.5% | 0.0% | — |
| mask_cosmos | 0.0% | 97.5% | [96.0%, 98.8%] |
| mask_qwen | 90.0% | 7.5% | [5.5%, 9.7%] |
| swap_cosmos | 0.0% | 97.5% | [96.0%, 98.8%] |
| swap_qwen | 94.7% | 2.7% | [0.5%, 5.0%] |
| freeze_cosmos | 0.0% | 97.5% | [96.0%, 98.8%] |
| freeze_qwen | 91.3% | 6.2% | [3.5%, 9.0%] |
| noise_cosmos | 32.7% | 64.7% | [61.5%, 68.0%] |
| noise_qwen | 97.3% | 0.2% | [-1.8%, 2.3%] |

Interpretation: paired_checkpoint_interventions

Bootstrap resamples paired trials within each fixed task; task weights are equal.
The one-trial pilot has no inferential confidence interval; comparisons are descriptive and unadjusted for multiplicity.
Ablation effects are conditional on this frozen-Qwen training recipe and this checkpoint.
Masking, donor swaps and image noise are distribution shifts, not absolute backbone rankings.

## Cosmos minus Qwen success drop

Positive values mean a larger drop when intervening on Cosmos.

- freeze: 91.3%; paired 95% interval: [0.8875000476837158, 0.9375]
- mask: 90.0%; paired 95% interval: [0.8799999952316284, 0.9175000190734863]
- swap: 94.8%; paired 95% interval: [0.9325000047683716, 0.9625000357627869]
- noise: 64.5%; paired 95% interval: [0.612500011920929, 0.6775000095367432]

Offline diagnostic status: missing
