# Checkpoint-selection audit: verified entries (20 Nov 2026)

Verified by reading each repository's default branch with `gh api` (read-only) on 20 Nov 2026,
and the paper PDFs. Line numbers refer to that state. Wording for the paper must describe the
**released code** and the **paper text**, not infer how published numbers were produced.

| Method | Paper says | Released code does | Category |
|---|---|---|---|
| BDRAR (Zhu et al., ECCV 2018) | 3000 iterations | `train.py` L27 `'iter_num': 3000`; L144-145 saves `%d.pth` once `curr_iter > iter_num` | Fixed schedule |
| DSC (Hu et al., CVPR 2018) | "terminate the learning process after 12k iterations" | `examples/DSC/DSC_detection/solver.prototxt` L9 `max_iter: 12000`, L12 `snapshot: 2000` (Caffe) | Fixed schedule |
| MTMT-Net (Chen et al., CVPR 2020) | "10,000 iterations", poly learning-rate schedule | `train.py` L31 `--max_iterations` default 10000; L258 stops at `iter_num >= max_iterations` | Fixed schedule |
| FSDNet (Hu et al., TIP 2021) | official val split of CUHK-Shadow used "for tuning" | `train.py` L29 `'iter_num': 50000`; L179-182 snapshots by iteration | Fixed schedule (dataset has a val split) |
| FDRNet (Zhu et al., ICCV 2021) | holds out 10% of training data to grid-search the inference weight mu, then "put them back and retrain the model with the whole training set" for 10 epochs | evaluates `SBU_test+UCF_test` every epoch (`--eval_data` default) and saves every epoch; best-IoU tracking commented out; release ships `istd_epoch_010.pt` | Fixed schedule; test scored each epoch (selection not evidenced) |
| SDDNet (Cong et al., ACM MM 2023; FDRNet fork) | 20 epochs; selection not stated | `train.py` L496 `--eval_data` default `SBU_test+UCF_test`, every epoch saved; `test.py` L21 commented path `.../ckpt_7/ep_019.ckpt` | Test scored each epoch; rule not stated |
| SILT (Yang et al., ICCV 2023) | "we obtain the best result in round 6, 5, and 3 for models trained on SBU, UCF, and ISTD" | `data/dataset.py` L11-12: mode `'validate'` reads `test.txt`; `train.py` L135-151 saves a checkpoint whenever this BER improves | **Best on test set** |
| AdapterShadow (Jie & Zhang, arXiv 2023; journal version ShadowAdapter, ESWA 2025) | "trained by 40, 200 and 200 epochs for the SBU, ISTD and CUHK dataset"; selection rule not stated | `dataloader.py` `val_dataloader()` is built from `self.test_pairs`; `pl_train.py` L427-433 `ModelCheckpoint(monitor='ber', save_top_k=5, mode='min')` | **Best on test set** |
| SAM-Adapter (Chen et al., ICCVW 2023) | shadow results on ISTD | `train.py` L184, L219-225 save `'best'` on the val loader's BER when `eval_type == 'ber'`; the only released configs (camouflaged-object `demo.yaml`, `cod-sam-*.yaml`) set `val_dataset` to the Test images; no shadow config released | Indirect: best-on-"val", val = test in all released configs |
| DSD, RMLANet | – | code only as Drive zips, not inspected | Not checked |
| ST-CGAN, A+D Net | – | no training code released | Not available |

Corrections to the first-pass audit:
- FDRNet: the first pass said the paper used a 10% held-out validation and the code "has no such split". The paper's hold-out is only for mu; the final model is a fixed 10-epoch retrain on the full training set. Not a test-selection case.
- SAM-Adapter: the evidence is the generic selection code plus camouflaged-object configs; no released shadow config.
