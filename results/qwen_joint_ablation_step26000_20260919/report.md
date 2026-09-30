# Qwen + DiT4DiT paired LIBERO evaluation

Checkpoint: `9641e6361183c38e5e8940c3c22456862732feb95fbf56b19a5043e4601bf369`

40 tasks, 10 paired trials/task, one training seed.

| Condition | Success | Drop vs full | 95% paired interval |
|---|---:|---:|---:|
| full | 94.7% | 0.0% | — |
| mask_cosmos | 0.0% | 94.7% | [92.8%, 96.7%] |
| mask_qwen | 88.7% | 6.0% | [3.0%, 9.0%] |
| swap_cosmos | 0.2% | 94.5% | [92.5%, 96.5%] |
| swap_qwen | 49.8% | 45.0% | [40.7%, 49.3%] |
| freeze_cosmos | 0.0% | 94.7% | [92.8%, 96.7%] |
| freeze_qwen | 40.5% | 54.3% | [50.2%, 58.0%] |
| noise_cosmos | 55.8% | 39.0% | [35.5%, 42.5%] |
| noise_qwen | 92.8% | 2.0% | [-0.5%, 4.5%] |

Interpretation: paired_checkpoint_interventions

Bootstrap resamples paired trials within each fixed task; task weights are equal.
The one-trial pilot has no inferential confidence interval; comparisons are descriptive and unadjusted for multiplicity.
Ablation effects are conditional on this training recipe and this checkpoint.
Masking, donor swaps and image noise are distribution shifts, not absolute backbone rankings.

## Cosmos minus Qwen success drop

Positive values mean a larger drop when intervening on Cosmos.

- freeze: 40.5%; paired 95% interval: [0.375, 0.4350000023841858]
- mask: 88.7%; paired 95% interval: [0.8650000095367432, 0.9099999666213989]
- swap: 49.5%; paired 95% interval: [0.45500001311302185, 0.5349999666213989]
- noise: 37.0%; paired 95% interval: [0.33250001072883606, 0.4074999988079071]

Offline diagnostic status: missing
