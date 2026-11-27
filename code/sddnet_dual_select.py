"""SDDNet (Cong et al., ACM MM 2023) trained on ISTD with held-out-val and test BER recorded every epoch.

Same design as code/fdrnet_dual_select.py, for a second published detector. SDDNet's released train.py scores
the test set after every epoch and saves every epoch's checkpoint, so its released protocol leaves the
checkpoint choice to the reader; we record
  last - final epoch (the code's total_ep = 20),
  val  - lowest held-out-validation BER (9 ISTD training scenes, data/istd_struct/val_heldout),
  test - lowest test BER (selection on the 540-image ISTD test set).

Model-side code is SDDNet's own, imported unmodified from code/upstream/SDDNet (commit dddcdd4): network
(EfficientNet-B3 backbone, constructor arguments of its train.py), BBCE mask loss and the auxiliary losses with
their weights (0.2 L1 image reconstruction, 0.2 DiffLoss_2, 0.2 (0.5 cos + 0.5 cos + 0.01 ortho + 0.01 ortho)),
AdamW lr 5e-4, wd 1e-4, StepLR step 1 gamma 0.7, batch 4, 512 px, 20 epochs, threshold 0.5. No mu schedule
(commented out upstream). Ours: MPS device, data roots, held-out split, bookkeeping, and one workaround:
the released datasets/sbu_dataset_new.py assigns `self.noshad_dir = noshad_dir` although that argument is
commented out, so it raises NameError as shipped; we define the module-level name (None, unused) instead of
editing the file.

BER is recorded under two protocols, per image, every epoch:
  upstream512 - SDDNet's evaluate(): GT resized bilinearly to 512 and cast to int64 (only pixels that stay
                exactly 1 are shadow), prediction upsampled to 512;
  native      - FDRNet's protocol, as in our FDRNet runs: prediction upsampled to the native mask size,
                GT > 0.5 at native size.
Selections and the headline use `native` so that FDRNet and SDDNet are comparable; upstream512 is kept.

Output: results/sddnet/sddnet_istd_seed<k>_<stamp>.json and _counts.npz (keys val, test, val_up512, test_up512;
        (epoch, image, 4) with columns TP, TN, P, N), checkpoints/sddnet/..._{last,val,test}.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = HERE / "upstream" / "SDDNet"
sys.path.insert(0, str(UP))
from networks.sddnet import SDDNet  # noqa: E402  (upstream, unmodified)
from networks.loss import BBCEWithLogitLoss, DiffLoss_2, OrthoLoss  # noqa: E402
import datasets.sbu_dataset_new as sbu_mod  # noqa: E402

sbu_mod.noshad_dir = None  # workaround for the NameError in the released loader (attribute is never used)
SBUDataset = sbu_mod.SBUDataset


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def counts(pred, g):
    return [int((pred & g).sum()), int((~pred & ~g).sum()), int(g.sum()), int((~g).sum())]


@torch.no_grad()
def score(model, ds, loader, device, th=0.5):
    """Per-image (TP, TN, P, N): native protocol and SDDNet's upstream 512 protocol."""
    model.eval()
    nat, up, names = [], [], []
    for i, d in enumerate(loader):
        name = ds.img_names[i]
        x = d["ShadowImages_input"].to(device)
        logit = model(x)["logit"]
        g_nat = cv2.imread(str(Path(ds.root_dir) / "ShadowMasks" / (Path(name).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        g_nat = torch.from_numpy(g_nat > 125).to(device)
        p_nat = torch.sigmoid(F.interpolate(logit, size=g_nat.shape, mode="bilinear"))[0, 0]
        if not torch.isfinite(p_nat).all():
            raise FloatingPointError("non-finite prediction during evaluation")
        nat.append(counts(p_nat > th, g_nat))
        g_up = d["gt"].to(device)[0, 0].long() == 1          # upstream: int64 cast of the resized float mask
        p_up = torch.sigmoid(F.interpolate(logit, size=g_up.shape, mode="bilinear"))[0, 0]
        up.append(counts(p_up > th, g_up))
        names.append(name)
    return np.asarray(nat, np.int64), np.asarray(up, np.int64), names


def ber(c):
    tp, tn, p, n = c.sum(0)
    return float(100 * (1 - (tp / p + tn / n) / 2)), float(100 * (1 - tp / p)), float(100 * (1 - tn / n))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "istd_struct")
    ap.add_argument("--epochs", type=int, default=20)        # upstream total_ep
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr-step", type=int, default=1)
    ap.add_argument("--lr-gamma", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=4)          # upstream train_batch
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per batch (gradient accumulation)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test only: stop each epoch early")
    ap.add_argument("--max-eval", type=int, default=0, help="smoke test only: score only the first n images")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sddnet")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "sddnet")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.batch % args.accum:
        raise ValueError("--batch must be divisible by --accum")

    seed_all(args.seed)
    device = torch.device(args.device)
    split = json.loads((args.data / "split.json").read_text())
    kw = dict(img_dirs=["ShadowImages"], mask_dir="ShadowMasks", augmentation=False, im_size=args.size, normalize=False)
    train_ds = SBUDataset(str(args.data / "train_heldout"), phase="train", **kw)
    val_ds = SBUDataset(str(args.data / "val_heldout"), phase="test", **kw)
    test_ds = SBUDataset(str(args.data / "test"), phase="test", **kw)
    assert (len(train_ds), len(val_ds), len(test_ds)) == (1210, 120, 540), (len(train_ds), len(val_ds), len(test_ds))
    scene = lambda n: n.split("-")[0]  # noqa: E731
    assert not ({scene(n) for n in train_ds.img_names} & {scene(n) for n in val_ds.img_names}), "scene leak train/val"
    assert sorted({scene(n) for n in val_ds.img_names}) == sorted(split["val_scenes"])
    if args.max_eval:
        for ds in (val_ds, test_ds):
            ds.img_names = ds.img_names[:args.max_eval]; ds.size = len(ds.img_names)

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                                               num_workers=args.workers, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    # identical constructor arguments to SDDNet train.py create_model_and_optimizer()
    model = SDDNet(backbone="efficientnet-b3", proj_planes=16, pred_planes=32, use_pretrained=True,
                   fix_backbone=False, has_se=False, dropout_2d=0, normalize=True, mu_init=0.5,
                   reweight_mode="manual").to(device)
    opt = torch.optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    loss_fn = BBCEWithLogitLoss()
    loss_l1, loss_diff, loss_ortho, loss_cos = nn.L1Loss(), DiffLoss_2(), OrthoLoss(), nn.CosineEmbeddingLoss()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"sddnet_istd_seed{args.seed}{('_' + args.tag) if args.tag else ''}_{stamp}"
    args.out.mkdir(parents=True, exist_ok=True); args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    upstream_commit = subprocess.run(["git", "-C", str(UP), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import efficientnet_pytorch
    record = {"experiment": "SDDNet on ISTD, per-epoch held-out-val and test BER (checkpoint-selection study)",
              "run": run, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                         "upstream": "https://github.com/rmcong/SDDNet_ACMMM23", "upstream_commit": upstream_commit,
                         "upstream_licence": "none in repository",
                         "loader_workaround": "sbu_dataset_new.noshad_dir = None (NameError in released file)",
                         "losses": "bbce(mask) + 0.2 L1(shadimg, image) + 0.2 DiffLoss_2 + 0.2(0.5 cos5 + 0.5 cos6 + 0.01 ortho7 + 0.01 ortho8)",
                         "split": split, "train_images": len(train_ds), "val_images": len(val_ds), "test_images": len(test_ds),
                         "ber": "pooled 100*(1-(TP/P+TN/N)/2); 'native' (headline) and 'upstream512' recorded",
                         "torch": torch.__version__, "efficientnet_pytorch": efficientnet_pytorch.__version__},
              "epochs": []}
    print(f"{run}: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}; device {device}", flush=True)

    keep = {k: [] for k in ("val", "test", "val_up512", "test_up512")}
    best = {"val": (math.inf, -1), "test": (math.inf, -1)}
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for i, sample in enumerate(train_loader):  # upstream train() body, device-agnostic
            if args.max_batches and i >= args.max_batches:
                break
            opt.zero_grad()
            parts = []
            for image, label in zip(sample["ShadowImages_input"].chunk(args.accum), sample["gt"].chunk(args.accum)):
                image, label = image.to(device), label.to(device)
                (l_shadimg, l_mask, l_noshad, f_low_shad, f_high_shad, f_low_noshad, f_high_noshad,
                 f_low_feat, f_high_feat, _m, _s, _n) = model(image)
                ones = torch.ones(f_low_shad.size(0), device=device)
                loss1 = loss_l1(l_shadimg, image)
                loss2 = loss_fn(l_mask, label)
                loss3 = loss_diff(image, l_noshad, label, l_mask)
                loss5 = loss_cos(f_low_shad, f_low_feat, ones)
                loss6 = loss_cos(f_high_shad, f_high_feat, ones)
                loss7 = loss_ortho(f_low_noshad, f_low_shad)
                loss8 = loss_ortho(f_high_noshad, f_high_shad)
                loss_filter = 0.2 * (0.5 * loss5 + 0.5 * loss6 + 0.01 * loss7 + 0.01 * loss8)
                loss_total = loss2 + 0.2 * loss1 + 0.2 * loss3 + loss_filter
                if not torch.isfinite(loss_total):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {i}")
                (loss_total / args.accum).backward()
                parts.append([loss2.item(), loss_total.item()])
            opt.step()
            losses.append(np.mean(parts, 0).tolist())
            if args.max_batches or (i + 1) % 50 == 0:
                mem = torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0
                print(f"  batch {i + 1}/{len(train_loader)}  loss_mask {losses[-1][0]:.4f}  total {losses[-1][1]:.4f}  "
                      f"mps {mem:.1f} GiB  {time.time() - t0:.0f}s", flush=True)
        sched.step()
        t_train = time.time() - t0

        if device.type == "mps":
            torch.mps.empty_cache()
        vc, vu, vnames = score(model, val_ds, val_loader, device, args.th)
        tc, tu, tnames = score(model, test_ds, test_loader, device, args.th)
        for k, v in (("val", vc), ("test", tc), ("val_up512", vu), ("test_up512", tu)):
            keep[k].append(v)
        vb, vpe, vne = ber(vc); tb, tpe, tne = ber(tc)
        ep = {"epoch": epoch + 1, "lr_after": sched.get_last_lr()[0], "train_loss_mean": np.mean(losses, 0).tolist(),
              "val_ber": vb, "val_pos_err": vpe, "val_neg_err": vne,
              "test_ber": tb, "test_pos_err": tpe, "test_neg_err": tne,
              "val_ber_up512": ber(vu)[0], "test_ber_up512": ber(tu)[0],
              "seconds_train": round(t_train, 1), "seconds": round(time.time() - t0, 1)}
        record["epochs"].append(ep)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        for rule, value in (("val", vb), ("test", tb)):
            if value < best[rule][0]:
                best[rule] = (value, epoch + 1)
                torch.save({"model": state, "epoch": epoch + 1, "rule": rule, "run": run}, args.ckpt_dir / f"{run}_{rule}.pt")
        torch.save({"model": state, "epoch": epoch + 1, "rule": "last", "run": run}, args.ckpt_dir / f"{run}_last.pt")
        print(f"epoch {epoch + 1:2d}/{args.epochs}: val BER {vb:.2f}  test BER {tb:.2f}  (up512 {ep['test_ber_up512']:.2f})  "
              f"train {t_train:.0f}s total {ep['seconds']:.0f}s", flush=True)
        json.dump(record, open(args.out / f"{run}.json", "w"), indent=1)
        if device.type == "mps":
            torch.mps.empty_cache()

    E = record["epochs"]
    pick = {"last": len(E), "val": best["val"][1], "test": best["test"][1]}
    record["selection"] = {rule: {"epoch": e, "test_ber": E[e - 1]["test_ber"], "val_ber": E[e - 1]["val_ber"]}
                           for rule, e in pick.items()}
    record["selection"]["inflation_test_minus_val_rule"] = E[pick["val"] - 1]["test_ber"] - E[pick["test"] - 1]["test_ber"]
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["total_seconds"] = round(time.time() - t_start, 1)
    record["checkpoint_sha256"] = {r: sha256(args.ckpt_dir / f"{run}_{r}.pt") for r in ("last", "val", "test")}
    np.savez_compressed(args.out / f"{run}_counts.npz", **{k: np.stack(v) for k, v in keep.items()},
                        val_names=np.array(vnames), test_names=np.array(tnames),
                        columns=np.array(["TP", "TN", "P", "N"]))
    json.dump(record, open(args.out / f"{run}.json", "w"), indent=1)
    print(json.dumps(record["selection"], indent=1))


if __name__ == "__main__":
    main()
