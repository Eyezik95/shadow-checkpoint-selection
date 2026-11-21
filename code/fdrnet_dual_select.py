"""FDRNet (Zhu et al., ICCV 2021) trained on ISTD with BOTH selection rules recorded every epoch.

Purpose (TMLR revision of the checkpoint-selection paper): measure, on a published detector, how much
the reported BER depends on choosing the checkpoint on the ISTD test set. One training run gives
every rule at once, because each epoch is scored on
  * val  - the paper's held-out validation scenes carved from ISTD *train* (data/istd_struct/val_heldout,
           same 9 scenes as the fair protocol; built by tools/make_istd_struct.py), and
  * test - the full ISTD test partition (540 images),
and the rules are then read off the per-epoch record:
  last  - final epoch (FDRNet's own protocol: the paper trains a fixed 10 epochs on the full training set
           and its release ships istd_epoch_010.pt; its 10% hold-out only tunes the inference weight mu),
  val   - lowest held-out-validation BER (fair),
  test  - lowest test BER (test-set selection, as in the SILT / AdapterShadow / SAM-Adapter code).

Everything model-side is FDRNet's own code, imported unmodified from code/upstream/FDRNet
(commit ea7e147): network (EfficientNet-B3 backbone, FR unit, mu schedule), balanced BCE loss,
SBUDataset loader and transforms, the auxiliary invariance/variance losses and their weights, and
the optimiser and schedule defaults of its train.py (AdamW lr 5e-4, wd 1e-4, StepLR step 1 gamma 0.7,
batch 8, 512 px, threshold 0.5), with the paper's 10 epochs (Sec. 4.1) instead of the code default
of 15. Inference mu is the code's fixed mu_init = 0.5 (the paper grid-searches it on its hold-out). Only device handling (MPS instead of .cuda()),
the data roots, the held-out validation split and the bookkeeping are ours.

BER is pooled over the image set, 100 * (1 - (TP/P + TN/N) / 2), the same formula as FDRNet's
utils/evaluation.py and the paper's Eq 3.13; predictions are upsampled to the native mask size as in
FDRNet's evaluate(). Per-image confusion counts are saved for every epoch so that scene-bootstrap
intervals can be computed later without re-inference.

Output: results/fdrnet/fdrnet_istd_seed<k>_<stamp>.json (full config, per-epoch metrics, selections)
        results/fdrnet/fdrnet_istd_seed<k>_<stamp>_counts.npz (per-epoch per-image counts, val and test)
        checkpoints/fdrnet/..._{last,val,test}.pt (weights only)
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

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = HERE / "upstream" / "FDRNet"
sys.path.insert(0, str(UP))
from networks.fdrnet import FDRNet  # noqa: E402  (upstream, unmodified)
from networks.loss import BBCEWithLogitLoss  # noqa: E402
from datasets.sbu_dataset_new import SBUDataset  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def score(model, loader, device, th=0.5):
    """Per-image (TP, TN, P, N) at native mask resolution, threshold th; FDRNet's evaluate() protocol."""
    model.eval()
    rows, names = [], []
    for d in loader:
        x = d["ShadowImages_input"].to(device)
        gt = d["gt"].to(device)  # (1,1,h,w) in {0,1}
        logit = model(x)["logit"]
        prob = torch.sigmoid(F.interpolate(logit, size=gt.shape[-2:], mode="bilinear"))
        if not torch.isfinite(prob).all():
            raise FloatingPointError("non-finite prediction during evaluation")
        pred = prob > th
        g = gt > 0.5
        rows.append([int((pred & g).sum()), int((~pred & ~g).sum()), int(g.sum()), int((~g).sum())])
        names.append(d["im_name"][0])
    return np.asarray(rows, dtype=np.int64), names


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
    ap.add_argument("--epochs", type=int, default=10)        # FDRNet paper Sec. 4.1 (code default total_ep is 15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr-step", type=int, default=1)
    ap.add_argument("--lr-gamma", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per batch (gradient accumulation)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test only: stop each epoch early")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "fdrnet")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "fdrnet")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.batch % args.accum:
        raise ValueError("--batch must be divisible by --accum")

    seed_all(args.seed)
    device = torch.device(args.device)
    split = json.loads((args.data / "split.json").read_text())

    train_ds = SBUDataset(str(args.data / "train_heldout"), phase="train", augmentation=False, im_size=args.size, normalize=False)
    val_ds = SBUDataset(str(args.data / "val_heldout"), phase="test", augmentation=False, im_size=args.size, normalize=False)
    test_ds = SBUDataset(str(args.data / "test"), phase="test", augmentation=False, im_size=args.size, normalize=False)
    assert (len(train_ds), len(val_ds), len(test_ds)) == (1210, 120, 540), (len(train_ds), len(val_ds), len(test_ds))
    scene = lambda n: n.split("-")[0]  # noqa: E731
    assert not ({scene(n) for n in train_ds.img_names} & {scene(n) for n in val_ds.img_names}), "scene leak train/val"
    assert sorted({scene(n) for n in val_ds.img_names}) == sorted(split["val_scenes"])

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                                               num_workers=args.workers, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    # identical constructor arguments to FDRNet train.py create_model_and_optimizer()
    model = FDRNet(backbone="efficientnet-b3", proj_planes=16, pred_planes=32, use_pretrained=True,
                   fix_backbone=False, has_se=False, dropout_2d=0, normalize=True, mu_init=0.5,
                   reweight_mode="manual").to(device)
    opt = torch.optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    loss_fn = BBCEWithLogitLoss()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"fdrnet_istd_seed{args.seed}{('_' + args.tag) if args.tag else ''}_{stamp}"
    args.out.mkdir(parents=True, exist_ok=True); args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    upstream_commit = subprocess.run(["git", "-C", str(UP), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import efficientnet_pytorch
    record = {"experiment": "FDRNet on ISTD, per-epoch held-out-val and test BER (checkpoint-selection study)",
              "run": run, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                         "upstream": "https://github.com/rayleizhu/FDRNet", "upstream_commit": upstream_commit,
                         "model_kwargs": {"backbone": "efficientnet-b3", "proj_planes": 16, "pred_planes": 32,
                                          "normalize": True, "mu_init": 0.5, "reweight_mode": "manual"},
                         "aux_losses": "loss_det + 1.0*L1(f_inv(view2), f_inv) + 0.1*L1(delta_pred, delta), delta~U(-0.3,0.3)",
                         "mu_schedule": "after epoch e: mu = 1 - (e/total_ep)^2 (upstream)",
                         "split": split, "train_images": len(train_ds), "val_images": len(val_ds), "test_images": len(test_ds),
                         "ber": "pooled 100*(1-(TP/P+TN/N)/2), native mask resolution",
                         "torch": torch.__version__, "efficientnet_pytorch": efficientnet_pytorch.__version__},
              "epochs": []}
    print(f"{run}: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}; device {device}", flush=True)

    val_counts, test_counts, best = [], [], {"val": (math.inf, -1), "test": (math.inf, -1)}
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for i, sample in enumerate(train_loader):  # upstream train() body, device-agnostic
            if args.max_batches and i >= args.max_batches:
                break
            delta = np.random.uniform(-0.3, 0.3)  # one brightness shift per batch, as upstream
            opt.zero_grad()
            parts = []
            # the batch of args.batch images is processed in args.accum equal micro-batches whose gradients are
            # summed (losses scaled by 1/accum) before one optimiser step: same update as upstream except that
            # BatchNorm statistics and the balanced-BCE class ratio are computed per micro-batch (memory on MPS)
            for image, label in zip(sample["ShadowImages_input"].chunk(args.accum), sample["gt"].chunk(args.accum)):
                image, label = image.to(device), label.to(device)
                logit, f_inv, _ = model(image)
                loss_det = loss_fn(logit, label)
                view2 = torch.clamp(image + delta, 0., 1.)
                _, f_inv_view2, delta_pred = model(view2)
                loss_inv = F.l1_loss(f_inv_view2, f_inv.detach())
                loss_var = F.l1_loss(delta_pred, delta * torch.ones_like(delta_pred))
                loss_total = loss_det + 1.0 * loss_inv + 0.1 * loss_var
                if not torch.isfinite(loss_total):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {i}")
                (loss_total / args.accum).backward()
                parts.append([loss_det.item(), loss_inv.item(), loss_var.item()])
            opt.step()
            losses.append(np.mean(parts, 0).tolist())
            if args.max_batches or (i + 1) % 25 == 0:
                mem = torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0
                print(f"  batch {i + 1}/{len(train_loader)}  loss_det {losses[-1][0]:.4f}  mps {mem:.1f} GiB  "
                      f"{time.time() - t0:.0f}s", flush=True)
        sched.step()
        model.fr.set_mu(1 - (epoch / args.epochs) ** 2)  # upstream cumulative learning

        if device.type == "mps":
            torch.mps.empty_cache()  # the MPS cache grew ~0.1 GiB/epoch; first seed-0 attempt hit the cap in epoch 10
        vc, vnames = score(model, val_loader, device, args.th)
        tc, tnames = score(model, test_loader, device, args.th)
        val_counts.append(vc); test_counts.append(tc)
        vb, vpe, vne = ber(vc); tb, tpe, tne = ber(tc)
        ep = {"epoch": epoch + 1, "lr_after": sched.get_last_lr()[0], "train_loss_mean": np.mean(losses, 0).tolist(),
              "val_ber": vb, "val_pos_err": vpe, "val_neg_err": vne,
              "test_ber": tb, "test_pos_err": tpe, "test_neg_err": tne, "seconds": round(time.time() - t0, 1)}
        record["epochs"].append(ep)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        for rule, value in (("val", vb), ("test", tb)):
            if value < best[rule][0]:
                best[rule] = (value, epoch + 1)
                torch.save({"model": state, "epoch": epoch + 1, "rule": rule, "run": run}, args.ckpt_dir / f"{run}_{rule}.pt")
        torch.save({"model": state, "epoch": epoch + 1, "rule": "last", "run": run}, args.ckpt_dir / f"{run}_last.pt")
        print(f"epoch {epoch + 1:2d}/{args.epochs}: val BER {vb:.2f}  test BER {tb:.2f}  ({ep['seconds']:.0f}s)", flush=True)
        json.dump(record, open(args.out / f"{run}.json", "w"), indent=1)  # keep a partial record if interrupted
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
    np.savez_compressed(args.out / f"{run}_counts.npz", val=np.stack(val_counts), test=np.stack(test_counts),
                        val_names=np.array(vnames), test_names=np.array(tnames),
                        columns=np.array(["TP", "TN", "P", "N"]))
    json.dump(record, open(args.out / f"{run}.json", "w"), indent=1)
    print(json.dumps(record["selection"], indent=1))


if __name__ == "__main__":
    main()
