"""Redo BiFPN-SIRAN phase 2 with a held-out validation set (Objective 2; Section 3.3.7).

--arch ran / rarm applies the same redo to the BiFPN-RAN (EfficientNet-B5) and
BiFPN-RARM (custom backbone) comparison models of Table 4.2, whose original phase 2
also used ISTD test for validation (their checkpoints' val_ber equals their test BER at
tau 0.50). Their recipe is the same notebook script (train_shadow_detector_v3) with the
model classes, freezing and parameter groups of src.models.obj2_variants; the group
sizes are asserted against the original checkpoints' optimiser states:
  * ran:  EfficientNet-B5, nothing frozen; 501 / 40 / 14 tensors at 0.1x / 0.5x / 1x;
  * rarm: custom backbone, stem frozen; 27 / 45 / 40 / 34 tensors at 0.05x / 0.1x / 0.5x / 1x;
          batch size 8 (Section 4.4.1: the custom-backbone configuration used about eight);
  * unetpp: UNet++ with EfficientNet-B5 (src.models.unetpp_notebook), trained by the
          Untitled16 notebook copy of the same script: nothing frozen (its --freeze_modules
          stem names no UNet++ module), one parameter group of all 572 tensors at the base
          lr (script lines 496-498), batch size 8 (the run passed no --batch_size; script
          default 8). Its original phase 2 also validated on ISTD test (script line 488).

The original phase 2 (notebook train_shadow_detector_v3, lines 488-591) used the
ISTD *test* partition as its validation set: it drove the learning-rate schedule,
the early stopping, and the choice of the saved checkpoint. Every number reported
on ISTD test was therefore tuned on ISTD test.

This script repeats that phase 2 unchanged except for the validation set, which
here is a scene-disjoint subset of ISTD *train* (--val-frac of the scenes,
chosen with --split-seed). The test partition is never read. Everything else is
the original recipe, verified against the original checkpoint's saved optimiser
state:
  * start from sbu_pretrained_bifpn_siran_efficientnet_b5.pth (phase-1 output);
  * notebook SIRAN class and get_discriminative_param_groups (src.models.siran_notebook):
    groups of 501 / 40 / 48 tensors at 0.1x / 0.5x / 1x of base lr 1e-5;
  * AdamW, weight decay 1e-3; ReduceLROnPlateau(min, 0.5, patience 5) on validation loss;
  * DeepSupervisionLoss(0.5, 0.5) for training, BCEDiceLoss(logits=False) for validation;
  * notebook ISTDDataset with its augmentation, 256 x 256, gradient clipping 1.0;
  * 30 epochs, early stopping after 10 epochs without validation-IoU gain, or after
    5 consecutive epochs with a train-validation IoU gap above 0.10;
  * checkpoint kept at the best validation IoU.
Not recoverable from the checkpoint and therefore assumptions, recorded in the
output: batch size 4 (Section 4.4.1 states EfficientNet runs used 4) and
dropout 0.1 (the script default passed by build_model).

Usage:
    python scripts/obj2_retrain_siran.py --epochs 1 --tag timing     # timing trial
    python scripts/obj2_retrain_siran.py
    python scripts/obj2_retrain_siran.py --arch ran --seed 1 --tag seed1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco import file_sha256  # noqa: E402
from src.models import obj2_variants as variants  # noqa: E402
from src.models.siran_notebook import BiFPN_SIRAN_ShadowDetector, get_discriminative_param_groups  # noqa: E402
from src.training import obj2_train_v3 as nb  # noqa: E402


def scene_of(p: Path) -> str:
    return p.stem.split("-")[0]


def split_scenes(paths, val_frac, seed):
    scenes = sorted({scene_of(p) for p in paths}, key=lambda s: (len(s), s))
    rng = random.Random(seed)
    val = set(rng.sample(scenes, max(1, round(val_frac * len(scenes)))))
    return [p for p in paths if scene_of(p) not in val], [p for p in paths if scene_of(p) in val], sorted(val)


# per architecture: phase-1 checkpoint, batch size, parameter-group sizes of the original phase-2 optimiser
ARCH = {"siran": {"sbu": "sbu_pretrained_bifpn_siran_efficientnet_b5.pth", "batch": 4, "groups": [501, 40, 48],
                  "batch_note": "4 (Section 4.4.1, EfficientNet runs)"},
        "ran": {"sbu": "sbu_pretrained_bifpn_ran_efficientnet_b5.pth", "batch": 4, "groups": [501, 40, 14],
                "batch_note": "4 (Section 4.4.1, EfficientNet runs)"},
        "rarm": {"sbu": "sbu_pretrained_bifpn_rarm.pth", "batch": 8, "groups": [27, 45, 40, 34],
                 "batch_note": "8 (Section 4.4.1, custom-backbone configuration 'approximately eight')"},
        "unetpp": {"sbu": "sbu_pretrained_unetplusplus.pth", "batch": 8, "groups": [572],
                   "batch_note": "8 (Untitled16 notebook: run without --batch_size, script default 8)"}}


def build(arch, dropout_p):
    """Model, frozen module names; constructor as notebook build_model / scripts/obj2_eval.load_model."""
    if arch == "unetpp":
        from src.models.unetpp_notebook import UNetPlusPlus_EfficientNetB5
        return UNetPlusPlus_EfficientNetB5(in_channels=3, out_channels=1, encoder_weights=None), []
    if arch == "siran":
        return BiFPN_SIRAN_ShadowDetector(backbone="efficientnet_b5", pretrained_backbone=False,
                                          dropout_p=dropout_p), []
    backbone = "efficientnet_b5" if arch == "ran" else "custom"
    kw = dict(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024], dropout_p=dropout_p,
              share_weights=True, use_distraction_head=False, backbone=backbone, pretrained_backbone=False)
    model = (variants.BiFPN_RAN_ShadowDetector(ra_channels=64, **kw) if arch == "ran"
             else variants.BiFPN_RARM_ShadowDetector(rarm_iterations=2, **kw))
    return model, (["stem"] if backbone == "custom" else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCH), default="siran")
    ap.add_argument("--istd", type=Path, default=Path("data/raw/istd/ISTD_Dataset"))
    ap.add_argument("--sbu-ckpt", type=Path, default=None, help="default: the architecture's phase-1 checkpoint")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=None, help="default: per architecture (ARCH)")
    ap.add_argument("--lr", type=float, default=1e-4, help="phase-2 base lr is lr/10, as in the notebook")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout-p", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--overfit-gap", type=float, default=0.10)
    ap.add_argument("--overfit-patience", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/obj2"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/obj2"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    spec = ARCH[args.arch]
    args.sbu_ckpt = args.sbu_ckpt or Path("data/raw/checkpoints") / spec["sbu"]
    args.batch_size = args.batch_size or spec["batch"]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run = f"{args.arch}_phase2_heldout{'_' + args.tag if args.tag else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train_ds = nb.ISTDDataset(str(args.istd), split="train", img_size=256, augment=True)
    val_ds = nb.ISTDDataset(str(args.istd), split="train", img_size=256, augment=False)
    tr, va, val_scenes = split_scenes(list(train_ds.images), args.val_frac, args.split_seed)
    train_ds.images, val_ds.images = tr, va
    if {scene_of(p) for p in tr} & {scene_of(p) for p in va}:
        raise AssertionError("a scene appears in both training and validation")
    test_dir = (args.istd / "test" / "test_A").resolve()
    if any(p.resolve().parent == test_dir for p in tr + va):
        raise AssertionError("a test image entered training or validation")

    model, frozen = build(args.arch, args.dropout_p)
    model = model.to(device)
    ck = torch.load(args.sbu_ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    if args.arch != "unetpp":                     # UNet++ wraps smp in .model, so its keys keep the prefix
        sd = {k[6:] if k.startswith("model.") else k: x for k, x in sd.items()}
    model.load_state_dict(sd, strict=True)
    if args.arch == "siran":
        groups = get_discriminative_param_groups(model, args.lr / 10)
    elif args.arch == "unetpp":                   # script lines 496-498: one group at the base lr
        groups = [{"params": list(model.parameters()), "lr": args.lr / 10}]
    else:
        frozen = variants.freeze_for_finetuning(model, frozen_modules=tuple(frozen)) if frozen else []
        groups = variants.get_discriminative_param_groups(model, args.lr / 10, frozen)
    sizes = [len(g["params"]) for g in groups]
    if sizes != spec["groups"]:
        raise AssertionError(f"parameter groups {sizes} differ from the original run's {spec['groups']}")
    opt = optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    train_crit = nb.DeepSupervisionLoss(bce_weight=0.5, dice_weight=0.5)
    val_crit = nb.BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, logits=False)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.out.mkdir(parents=True, exist_ok=True); args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_path, ckpt_path = args.out / f"{run}.json", args.ckpt_dir / f"{run}.pth"
    result = {"experiment": f"obj2_{args.arch}_phase2_heldout_validation", "run": run,
              "thesis_refs": ["Section 3.3.7", "Algorithm 3.2", "Table 4.2"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                         "sbu_ckpt_sha256": file_sha256(args.sbu_ckpt), "base_lr_phase2": args.lr / 10,
                         "param_group_sizes": sizes, "param_group_lrs": [x["lr"] for x in groups],
                         "frozen_modules": frozen,
                         "train_images": len(tr), "val_images": len(va), "val_scenes": val_scenes,
                         "assumptions": {"batch_size": spec["batch_note"], "dropout_p": ("not used (UNet++ has no dropout)" if args.arch == "unetpp"
                                                                   else "0.1 (script default)")},
                         "torch": str(torch.__version__)},
              "epochs": [], "best": None}
    print(f"{run}: train {len(tr)} images, val {len(va)} images ({len(val_scenes)} scenes); device {device}", flush=True)

    best_iou, patience, high_gap = 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tl, ti, tb, _, _ = nb.train_epoch(model, train_loader, train_crit, opt, device, epoch, args.epochs,
                                          frozen_module_names=frozen, grad_clip_norm=args.grad_clip)
        vl, vi, vb, vs, vn = nb.validate_epoch(model, val_loader, val_crit, device, epoch, args.epochs)
        for k, x in (("train_loss", tl), ("val_loss", vl), ("val_iou", vi)):
            if not np.isfinite(x):
                raise FloatingPointError(f"epoch {epoch}: {k} is not finite")
        sched.step(vl)
        gap = ti - vi
        rec = {"epoch": epoch, "train_loss": tl, "train_iou": ti, "train_ber": tb, "val_loss": vl, "val_iou": vi,
               "val_ber": vb, "val_shadow_err": vs, "val_nonshadow_err": vn, "gap": gap,
               "lrs": [x["lr"] for x in opt.param_groups], "time_s": time.perf_counter() - t0}
        result["epochs"].append(rec)
        if device.type == "mps":
            torch.mps.empty_cache()
        if vi > best_iou:
            best_iou, patience = vi, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_iou": vi, "val_loss": vl,
                        "val_ber": vb, "val_scenes": val_scenes}, ckpt_path)
            result["best"] = {"epoch": epoch, "val_iou": vi, "val_ber": vb, "checkpoint": str(ckpt_path),
                              "checkpoint_sha256": file_sha256(ckpt_path)}
        else:
            patience += 1
        high_gap = high_gap + 1 if gap > args.overfit_gap else 0
        res_path.write_text(json.dumps(result, indent=2))
        print(f"epoch {epoch:2d} train loss {tl:.4f} IoU {ti:.4f} | val loss {vl:.4f} IoU {vi:.4f} BER {vb:.2f}% "
              f"| {rec['time_s']:.0f}s", flush=True)
        if patience >= args.patience:
            result["stopped"] = f"no validation-IoU gain for {args.patience} epochs"; break
        if high_gap >= args.overfit_patience:
            result["stopped"] = f"train-validation IoU gap above {args.overfit_gap} for {args.overfit_patience} epochs"; break
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    res_path.write_text(json.dumps(result, indent=2))
    print("best:", result["best"]); print("written:", res_path)


if __name__ == "__main__":
    main()
