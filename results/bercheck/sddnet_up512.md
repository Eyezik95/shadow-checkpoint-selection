# BER reporting check

Bootstrap: 10,000 paired resamples of 46 test units (540 images), rng seed 0. Selected epochs are held fixed in the bootstrap.

## Inputs

| Method | Runs | Epochs per run | Validation counts | Rule used for comparison |
|---|---|---|---|---|
| SDDNet-up512 | 5 | 20 | yes | last |

## Checkpoint-selection rules (same runs, test BER %)

### SDDNet-up512

| Rule | Mean | SD | Epochs chosen |
|---|---|---|---|
| last | 1.825 | 0.256 | 20, 20, 20, 20, 20 |
| val | 1.983 | 0.392 | 3, 4, 10, 3, 6 |
| test | 1.761 | 0.260 | 9, 11, 18, 7, 12 |

| Difference | Mean over seeds | 95% interval | Per seed |
|---|---|---|---|
| test-last | -0.064 | [-0.110, -0.027] | -0.110, -0.070, -0.012, -0.063, -0.065 |
| val-last | +0.158 | [+0.013, +0.318] | +0.249, +0.390, +0.010, +0.110, +0.032 |
| test-val | -0.222 | [-0.385, -0.080] | -0.360, -0.460, -0.023, -0.173, -0.097 |

Validation–test curve agreement (Spearman over epochs, per run): 0.44, 0.41, 0.70, -0.14, 0.11; mean 0.31.

## Methods

| Method | Seeds | Mean BER | SD | SD 95% interval | Range |
|---|---|---|---|---|---|
| SDDNet-up512 | 5 | 1.825 | 0.256 | 0.153–0.735 | 0.636 |


## Minimum detectable difference (BER points)

Seed SD pooled over methods: 0.256 (95% interval 0.153–0.735). one method: test-sampling variance of a difference needs two methods; seed-only thresholds only

| Seeds per method | seed_only |
|---|---|
| 1 | 0.71 |
| 2 | 0.50 |
| 3 | 0.41 |
| 5 | 0.32 |
| 10 | 0.22 |

seed_only: gap needed to exceed seed noise on this fixed test set. interval: observed gap whose 95% interval over seeds and test units excludes zero. power80: true gap detected with 80% power.

## Reporting checklist

| # | Item | Status from these files |
|---|---|---|
| 1 | Checkpoint rule stated | SDDNet-up512: last. State it in the paper; 'test' means selection on the test set. |
| 2 | Validation set checked against test conditions | SDDNet-up512: Spearman mean 0.31, lowest run -0.14. Low or negative agreement: prefer a fixed schedule. |
| 3 | Several seeds, mean and SD reported | SDDNet-up512: 5 (SD 0.256) |
| 4 | Claims sized against thresholds | needs two or more methods with at least two runs each; the seed-only threshold above applies to any comparison on this test set |
| 5 | Both label sets reported where relabelled data exist | not checkable from counts; state it |
| 6 | Per-image, per-epoch counts released | these files; release them with the paper |
