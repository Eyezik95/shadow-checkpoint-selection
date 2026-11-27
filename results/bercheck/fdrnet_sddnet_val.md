# BER reporting check

Bootstrap: 10,000 paired resamples of 46 test units (540 images), rng seed 0. Selected epochs are held fixed in the bootstrap.

## Inputs

| Method | Runs | Epochs per run | Validation counts | Rule used for comparison |
|---|---|---|---|---|
| FDRNet | 5 | 10 | yes | val |
| SDDNet | 5 | 20 | yes | val |

## Checkpoint-selection rules (same runs, test BER %)

### FDRNet

| Rule | Mean | SD | Epochs chosen |
|---|---|---|---|
| last | 1.727 | 0.137 | 10, 10, 10, 10, 10 |
| val | 2.048 | 0.425 | 4, 3, 10, 10, 10 |
| test | 1.661 | 0.088 | 9, 9, 10, 9, 8 |

| Difference | Mean over seeds | 95% interval | Per seed |
|---|---|---|---|
| test-last | -0.066 | [-0.139, -0.012] | -0.062, -0.052, +0.000, -0.073, -0.144 |
| val-last | +0.321 | [+0.189, +0.479] | +0.640, +0.965, +0.000, +0.000, +0.000 |
| test-val | -0.387 | [-0.563, -0.241] | -0.702, -1.017, +0.000, -0.073, -0.144 |

Validation–test curve agreement (Spearman over epochs, per run): 0.12, 0.19, 0.90, 0.77, 0.87; mean 0.57.

### SDDNet

| Rule | Mean | SD | Epochs chosen |
|---|---|---|---|
| last | 1.792 | 0.254 | 20, 20, 20, 20, 20 |
| val | 1.959 | 0.378 | 3, 4, 10, 3, 6 |
| test | 1.726 | 0.254 | 6, 11, 13, 7, 17 |

| Difference | Mean over seeds | 95% interval | Per seed |
|---|---|---|---|
| test-last | -0.066 | [-0.130, -0.016] | -0.129, -0.078, -0.009, -0.059, -0.055 |
| val-last | +0.167 | [+0.023, +0.325] | +0.284, +0.370, +0.009, +0.117, +0.058 |
| test-val | -0.233 | [-0.389, -0.098] | -0.412, -0.448, -0.019, -0.176, -0.112 |

Validation–test curve agreement (Spearman over epochs, per run): 0.46, 0.45, 0.73, -0.12, 0.26; mean 0.36.

## Methods

| Method | Seeds | Mean BER | SD | SD 95% interval | Range |
|---|---|---|---|---|---|
| SDDNet | 5 | 1.959 | 0.378 | 0.226–1.086 | 0.957 |
| FDRNet | 5 | 2.048 | 0.425 | 0.255–1.222 | 1.065 |


| Pair | Difference of means | 95% interval | Excludes 0 | P(single runs keep this order) |
|---|---|---|---|---|
| SDDNet - FDRNet | -0.089 | [-0.493, +0.277] | no | 0.56 |

Rank stability: drawing one run per method (25 combinations), the order of the means is reproduced in 56% of draws.

## Minimum detectable difference (BER points)

Seed SD pooled over methods: 0.402 (95% interval 0.272–0.771). 

| Seeds per method | seed_only | interval | power80 |
|---|---|---|---|
| 1 | 1.12 | 1.23 | 1.75 |
| 2 | 0.79 | 0.90 | 1.29 |
| 3 | 0.64 | 0.76 | 1.09 |
| 5 | 0.50 | 0.63 | 0.90 |
| 10 | 0.35 | 0.51 | 0.73 |

seed_only: gap needed to exceed seed noise on this fixed test set. interval: observed gap whose 95% interval over seeds and test units excludes zero. power80: true gap detected with 80% power.

## Reporting checklist

| # | Item | Status from these files |
|---|---|---|
| 1 | Checkpoint rule stated | FDRNet: val, SDDNet: val. State it in the paper; 'test' means selection on the test set. |
| 2 | Validation set checked against test conditions | FDRNet: Spearman mean 0.57, lowest run 0.12; SDDNet: Spearman mean 0.36, lowest run -0.12. Low or negative agreement: prefer a fixed schedule. |
| 3 | Several seeds, mean and SD reported | SDDNet: 5 (SD 0.378), FDRNet: 5 (SD 0.425) |
| 4 | Claims sized against thresholds | SDDNet - FDRNet: within seed noise: not shown even on this test set |
| 5 | Both label sets reported where relabelled data exist | not checkable from counts; state it |
| 6 | Per-image, per-epoch counts released | these files; release them with the paper |
