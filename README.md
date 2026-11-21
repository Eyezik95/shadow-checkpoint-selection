# Selected on the Test Set — code and results

Code, per-image confusion counts and result files for the paper *Selected on the Test Set: Checkpoint
Choice, Seed Variance and Label Noise in Shadow-Detection Benchmarks* (under double-blind review).
Every number in the paper is produced from the files in this repository by `code/make_numbers.py`.

## Layout

| Path | Contents |
|---|---|
| `code/fdrnet_dual_select.py` | Trains FDRNet (authors' code, imported unmodified) on the held-out ISTD split and scores every epoch on the validation and test images |
| `code/run_fdrnet_seeds.sh` | Runs seeds 0–4 one after another |
| `code/fdrnet_analysis.py` | Three selection rules (last epoch, lowest validation BER, lowest test BER) and paired scene-bootstrap intervals |
| `code/make_fdrnet_figure.py` | Per-epoch figure for FDRNet |
| `code/sbu_testnew_analysis.py` | Re-scores the development-sequence checkpoints on the relabelled SBU test set |
| `code/make_numbers.py` | Writes every number used in the paper, with its source file and field (`paper/numbers.tsv`) |
| `code/check_numbers.py` | Fails if a result-like number is typed into the manuscript instead of generated |
| `code/tools/make_istd_struct.py` | Builds the scene-disjoint held-out ISTD split (1,210 train / 120 validation, 9 scenes) |
| `code/setup_fdrnet.sh` | Clones FDRNet at commit `ea7e147` |
| `dev_models/` | Models, training, evaluation and analysis code for the four development-sequence models (BiFPN-RARM, BiFPN-RAN, BiFPN-SIRAN, UNet++) and their result files (`dev_models/results/obj2`) |
| `results/fdrnet/` | FDRNet runs: one JSON per seed (full configuration, per-epoch BER, selections, checkpoint checksums) and per-image counts for every epoch |
| `results/fdrnet_failed/` | Record of a first seed-0 attempt that ran out of GPU memory (not used) |
| `results/sbu_*` | SBU counts and analyses on the original and relabelled labels |
| `logs/` | Training and evaluation logs |
| `paper/audit_verified.md` | The checkpoint-selection audit of published detectors, with file and line references |

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
code/setup_fdrnet.sh
```

Datasets are not redistributed. Obtain ISTD and SBU from their authors, and the relabelled SBU test set
from the SILT release (`refined_sbu/relabeled_sbu_test.zip`). Then:

```bash
.venv/bin/python code/tools/make_istd_struct.py <ISTD root> data/istd_struct <a dev_models/results/obj2 retrain JSON>
code/run_fdrnet_seeds.sh                      # about 1 h per seed on an Apple M-series GPU
.venv/bin/python code/fdrnet_analysis.py
.venv/bin/python code/make_numbers.py && .venv/bin/python code/check_numbers.py
```

The analysis scripts run from the stored counts alone, so the numbers can be regenerated without any
training or data. Runs used Python 3.13 and PyTorch 2.15 on Apple MPS; see each result file's `config`.

## Licence

MIT (see `LICENSE`). FDRNet is the work of its authors and is under its own MIT licence.
