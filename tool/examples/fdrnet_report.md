# BER reporting check

Bootstrap: 10,000 paired resamples of 46 test units (540 images), rng seed 0. Selected epochs are held fixed in the bootstrap.

## Inputs

| Method | Runs | Epochs per run | Validation counts | Rule used for comparison |
|---|---|---|---|---|
| FDRNet | 5 | 10 | yes | last |

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

## Methods

| Method | Seeds | Mean BER | SD | SD 95% interval | Range |
|---|---|---|---|---|---|
| FDRNet | 5 | 1.727 | 0.137 | 0.082–0.395 | 0.357 |


## Minimum detectable difference (BER points)

Seed SD pooled over methods: 0.137 (95% interval 0.082–0.395). one method: test-sampling variance of a difference needs two methods; seed-only thresholds only

| Seeds per method | seed_only |
|---|---|
| 1 | 0.38 |
| 2 | 0.27 |
| 3 | 0.22 |
| 5 | 0.17 |
| 10 | 0.12 |

seed_only: gap needed to exceed seed noise on this fixed test set. interval: observed gap whose 95% interval over seeds and test units excludes zero. power80: true gap detected with 80% power.

## Reporting checklist

| # | Item | Status from these files |
|---|---|---|
| 1 | Checkpoint rule stated | FDRNet: last. State it in the paper; 'test' means selection on the test set. |
| 2 | Validation set checked against test conditions | FDRNet: Spearman mean 0.57, lowest run 0.12. Low or negative agreement: prefer a fixed schedule. |
| 3 | Several seeds, mean and SD reported | FDRNet: 5 (SD 0.137) |
| 4 | Claims sized against thresholds | needs two or more methods with at least two runs each; the seed-only threshold above applies to any comparison on this test set |
| 5 | Both label sets reported where relabelled data exist | not checkable from counts; state it |
| 6 | Per-image, per-epoch counts released | these files; release them with the paper |
