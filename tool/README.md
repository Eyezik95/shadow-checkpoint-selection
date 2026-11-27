# bercheck

How much of a balanced-error-rate (BER) difference comes from the checkpoint rule, the random seed and the sampling of
the test set? `bercheck` answers this from stored confusion counts, without retraining. It was written for
shadow-detection benchmarks (ISTD, SBU), but it works for any binary segmentation task scored by pooled BER.

From per-image, per-epoch counts of one or more methods (one file per seed) it reports:

- **Checkpoint rules on the same runs:** last epoch (fixed schedule), lowest validation BER and lowest test BER. You get their test BER and the paired differences, each with a 95% interval from a bootstrap over test scenes.
- **Validation–test agreement:** the Spearman correlation between each run's validation and test curves. Low values warn that the validation set does not behave like the test set.
- **Methods compared:** mean, SD (with a chi-square interval) and range over seeds, plus paired differences of the means with scene-bootstrap intervals. It also checks how often single runs keep the order of the means.
- **Minimum detectable difference** for n seeds per method: the gap needed to beat seed noise on a fixed test set, the gap whose interval over seeds and test sampling excludes zero, and the gap detected with 80% power.
- **A filled-in reporting checklist** (below), with a verdict for each pairwise claim.

## Install

```bash
pip install ./tool            # from the repository root; needs numpy and scipy
```

## Input

One `.npz` file per training run (seed):

| Array | Shape | Meaning |
|---|---|---|
| `test` | (E, I, 4) or (I, 4) | confusion counts per epoch and test image |
| `test_names` | (I,) | test image file names; the scene is the stem before the first `-` (ISTD convention) |
| `val` | (E, J, 4), optional | validation counts per epoch; enables the validation rule |
| `columns` | (4,), optional | `TP, TN, P, N` (default) or `TP, FN, TN, FP` |

P and N are the numbers of shadow and non-shadow pixels in the reference mask. Counts should be computed at the resolution you report, at a fixed threshold.
Saving these arrays after every epoch costs a few kilobytes per epoch; `code/fdrnet_dual_select.py` in this repository shows how.

## Use

```bash
bercheck report --method FDRNet results/fdrnet/*_counts.npz \
                --method SDDNet results/sddnet/*_counts.npz \
                --rule last --out report.md --json report.json
```

- `--rule` sets which checkpoint each run contributes to the method comparison: `last`, `val`, `test` or `epochN`.
- Use `--units image` for test sets without scene labels, such as SBU. Images are not independent, so those intervals are too narrow.
- `examples/fdrnet_report.md` is the report for the five FDRNet runs of the paper; `examples/fdrnet_sddnet_report.md` compares FDRNet and SDDNet (five runs each).

From Python:

```python
from bercheck import Method, load_run, analyse, markdown
m = Method("FDRNet", [load_run(p) for p in paths], rule="last")
res = analyse([m]); print(markdown(res))
```

## Reporting checklist

| # | Item | What to report |
|---|---|---|
| 1 | Checkpoint rule | How the reported checkpoint was chosen and on which data. A fixed schedule set in advance is legitimate. Selection on the test set must be stated as such. |
| 2 | Validation set | Whole scenes held out of training. Its per-epoch curve agrees with the test curve on a pilot run (`bercheck` reports the Spearman correlation). If it does not, use a fixed schedule. |
| 3 | Seeds | Number of seeds, mean and SD for every method compared. |
| 4 | Size of the claim | The gap compared with the thresholds. Below the seed-only threshold, the gap is within seed noise. Above it, the gap holds on this test set. A claim beyond the test set needs a paired interval over test scenes that excludes zero. |
| 5 | Labels | Results on both label sets where a relabelled test set exists (SBU). |
| 6 | Counts | Per-image, per-epoch confusion counts released with the paper. |

For ISTD (46 test scenes) the paper's thresholds are as follows. A single-run gap below 0.46 BER points is within seed noise on the fixed test set; with five seeds per method that falls to 0.21. A gap that should hold beyond the test set needs about 1.0 point (0.96–1.03), however many seeds are run. These come from six models, including FDRNet and SDDNet.

## Tests

```bash
pip install ./tool[test] && pytest tool/tests
```

The regression tests reproduce the FDRNet rule analysis and the ISTD minimum-detectable-difference table of the paper from the repository's result files. They are skipped when those files are absent.

## Licence

MIT.
