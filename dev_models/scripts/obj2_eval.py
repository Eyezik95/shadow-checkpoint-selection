"""Shadow-detector evaluation on the ISTD or SBU test partition (Objective 2; Section 4.4, Eq 3.13).

For every checkpoint given, per-image confusion counts are computed at every
threshold 0.05 ... 0.95 under two protocols:

  * thesis256 - image and reference mask resized to 256 x 256 exactly as the
    notebook's ISTDDataset(split='test', augment=False) does; this is the
    protocol behind Tables 4.2 and 4.3;
  * native    - the 256 x 256 probability map is resized (bilinear) back to the
    image's own resolution and compared with the original mask, the usual
    convention in the shadow-detection literature.

Reported per checkpoint and protocol:
  * native BER at tau = 0.50;
  * test-selected BER: the minimum over the sweep, as in Table 4.2. The threshold
    is chosen on the test images themselves, so this figure is biased and is
    reported only for comparison with the thesis;
  * cross-fitted BER: test scenes split into two halves, the threshold chosen on
    one half is applied to the other and vice versa (src.eval.shadow_metrics.cross_fit);
  * boundary-band error (k = 2, 5, 10 px) at tau = 0.50 and at the cross-fitted thresholds;
  * IoU at tau = 0.50 (pooled over pixels, and mean of per-image IoU).

Per-image counts are saved next to the JSON so later analyses need no re-inference.

--dataset sbu evaluates on the 638 SBU test images (Vicente et al. 2016) with the same two protocols.
SBU has no scene labels, so each image is its own unit: cross-fitting halves and the scene bootstrap of
the later analyses then resample images. Output files carry the prefix sbu_.

Usage:
    python scripts/obj2_eval.py --model siran=data/raw/checkpoints/best_model_bifpn_siran_efficientnet_b5.pth ...
    (name=path; architecture is read from the name: siran, ran, rarm, rarm_ran, ran_iter1, unetpp)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hashing import file_sha256  # noqa: E402
from src.eval.shadow_metrics import BAND_WIDTHS, THRESHOLDS, ber_from_counts, cross_fit, image_counts  # noqa: E402
from src.models.shadow import load_siran  # noqa: E402
from src.training.obj2_train_v3 import ISTDDataset  # noqa: E402


def scene_of(path: Path) -> str:
    """ISTD names are <scene>-<index>.png; images of one scene share a background."""
    return path.stem.split("-")[0]


def scene_halves(scenes):
    """Deterministic split: sorted unique scenes, alternately assigned to halves 0 and 1."""
    order = sorted(set(scenes), key=lambda s: (len(s), s))
    half = {s: i % 2 for i, s in enumerate(order)}
    return np.array([half[s] for s in scenes])


def load_model(name: str, path: Path):
    """Returns (model producing probabilities in eval mode, metadata)."""
    arch = name.split("@")[0]
    if arch == "ran_iter1":                       # the code that trained best_model_bifpn_ran.pth
        from src.models.ran_iter1_notebook import BiFPN_RAN_ShadowDetector as RanIter1
        model = RanIter1(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024])
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        meta = {k: float(x) for k, x in ck.items() if k in ("epoch", "val_iou", "val_loss", "val_ber")}
        return model.eval(), {"class": "src.models.ran_iter1_notebook.BiFPN_RAN_ShadowDetector", **meta}
    if arch == "unetpp":                          # Table 4.2 internal baseline (Untitled16 notebook)
        from src.models.unetpp_notebook import UNetPlusPlus_EfficientNetB5
        model = UNetPlusPlus_EfficientNetB5(in_channels=3, out_channels=1, encoder_weights=None)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
        meta = {k: float(x) for k, x in ck.items() if k in ("epoch", "val_iou", "val_loss", "val_ber")}
        return model.eval(), {"class": "src.models.unetpp_notebook.UNetPlusPlus_EfficientNetB5", **meta}
    if arch == "siran":
        model, meta = load_siran(path)
        return _Sigmoid(model), {"class": "src.models.shadow.BiFPN_SIRAN_ShadowDetector", **meta}
    from src.models import obj2_variants as v
    cls = {"ran": v.BiFPN_RAN_ShadowDetector, "rarm": v.BiFPN_RARM_ShadowDetector,
           "rarm_ran": v.BiFPN_RARM_RAN_ShadowDetector}[arch]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k[6:] if k.startswith("model.") else k: x for k, x in sd.items()}
    backbone = "efficientnet_b5" if any(k.startswith("timm_backbone.") for k in sd) else "custom"
    kwargs = dict(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024], dropout_p=0.1,
                  share_weights=True, use_distraction_head=any(k.startswith("distraction_head.") for k in sd),
                  backbone=backbone, pretrained_backbone=False)
    if arch == "ran":
        kwargs["ra_channels"] = 64
    elif arch == "rarm":
        kwargs["rarm_iterations"] = 2
    else:
        kwargs.update(rarm_iterations=2, ra_channels=64)
    model = cls(**kwargs)
    model.load_state_dict(sd, strict=True)
    meta = {k: (float(x) if isinstance(x, (int, float)) else x) for k, x in ck.items()
            if k in ("epoch", "val_iou", "val_loss", "val_ber")}
    return model.eval(), {"class": f"src.models.obj2_variants.{cls.__name__}", "backbone": backbone,
                          "constructor": {k: x for k, x in kwargs.items() if k != "features"}, **meta}


class _Sigmoid(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def summarise(counts, band_err, band_px, halves):
    i50 = THRESHOLDS.index(0.5)
    pooled = counts.sum(0)                                            # (T, 4)
    sweep = ber_from_counts(pooled)
    best = int(np.argmin(sweep["ber"]))
    cf = cross_fit(counts, halves)
    t_for = {0: cf["thresholds_chosen"]["on_group_1_applied_to_0"], 1: cf["thresholds_chosen"]["on_group_0_applied_to_1"]}
    idx_img = np.array([THRESHOLDS.index(t_for[h]) for h in halves])
    band_cf = band_err[np.arange(len(halves)), idx_img].sum(0) / band_px.sum(0)
    c50 = counts[:, i50]
    per_img_iou = c50[:, 0] / np.maximum(c50[:, 0] + c50[:, 1] + c50[:, 3], 1)
    return {
        "native_tau_0.50": {k: float(x[i50]) for k, x in sweep.items()},
        "test_selected_BIASED": {"tau": THRESHOLDS[best], **{k: float(x[best]) for k, x in sweep.items()}},
        "cross_fitted": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cf.items()},
        "iou_tau_0.50": {"pooled": float(sweep["iou_pooled"][i50]), "mean_per_image": float(per_img_iou.mean())},
        "boundary_band_error_pct": {
            f"k={k}": {"tau_0.50": float(band_err[:, i50, j].sum() / band_px[:, j].sum() * 100),
                       "cross_fitted": float(band_cf[j] * 100)} for j, k in enumerate(BAND_WIDTHS)},
        "sweep": {"tau": list(THRESHOLDS), **{k: x.tolist() for k, x in sweep.items()}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path, name in siran|ran|rarm|rarm_ran|ran_iter1[@tag]")
    ap.add_argument("--istd", type=Path, default=Path("data/raw/istd/ISTD_Dataset"))
    ap.add_argument("--dataset", default="istd", choices=["istd", "sbu"])
    ap.add_argument("--sbu", type=Path, default=Path("data/raw/sbu/SBU-shadow/SBU-Test"))
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/obj2"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    ds = ISTDDataset(str(args.istd), split="test", img_size=256, augment=False)
    expected, prefix = 540, ""
    if args.dataset == "sbu":
        # same loader and transforms; only the folders change (images .jpg, masks .png of the same stem)
        ds.image_dir, ds.mask_dir = args.sbu / "ShadowImages", args.sbu / "ShadowMasks"
        ds.images = sorted(ds.image_dir.glob("*.jpg"))
        expected, prefix = 638, "sbu_"
        missing = [p.name for p in ds.images if not (ds.mask_dir / (p.stem + ".png")).exists()]
        if missing:
            raise AssertionError(f"SBU masks missing for {len(missing)} images, e.g. {missing[:3]}")
    if len(ds) != expected:
        raise AssertionError(f"{args.dataset} test should have {expected} images, found {len(ds)}")
    paths = list(ds.images)
    scenes = [scene_of(p) for p in paths] if args.dataset == "istd" else [p.stem for p in paths]
    halves = scene_halves(scenes)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {"experiment": f"obj2_{args.dataset}_test_evaluation", "thesis_refs": ["Section 4.4", "Table 4.2", "Table 4.3", "Eq 3.13"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": {"dataset": args.dataset, "istd": str(args.istd),
                         "sbu": str(args.sbu) if args.dataset == "sbu" else None,
                         "unit": "scene" if args.dataset == "istd" else "image (SBU has no scene labels)",
                         "images": len(ds), "scenes": len(set(scenes)),
                         "scene_halves": {"0": int((halves == 0).sum()), "1": int((halves == 1).sum())},
                         "halving_rule": "sorted unique scene ids alternately assigned",
                         "thresholds": list(THRESHOLDS), "band_widths_px": list(BAND_WIDTHS),
                         "ber": "pooled pixel counts over the image set (Eq 3.13)",
                         "device": args.device, "torch": str(torch.__version__)},
              "models": {}}
    device = torch.device(args.device)
    for spec in args.model:
        name, path = spec.split("=", 1); path = Path(path)
        model, meta = load_model(name, path)
        model = model.to(device).eval()
        n = len(ds); T, B = len(THRESHOLDS), len(BAND_WIDTHS)
        res = {p: {"counts": np.zeros((n, T, 4), np.int64), "band_err": np.zeros((n, T, B), np.int64),
                   "band_px": np.zeros((n, B), np.int64)} for p in ("thesis256", "native")}
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, n, args.batch):
                idx = list(range(s, min(n, s + args.batch)))
                imgs, masks = zip(*(ds[i] for i in idx))
                prob = model(torch.stack(imgs).to(device)).float().cpu()
                if prob.dim() != 4 or prob.shape[1] != 1:
                    raise AssertionError(f"{name}: unexpected output shape {tuple(prob.shape)}")
                for j, i in enumerate(idx):
                    p256 = prob[j, 0].numpy()
                    m256 = masks[j].numpy().reshape(256, 256) > 0.5
                    c = image_counts(p256, m256)
                    for k in c:
                        res["thesis256"][k][i] = c[k]
                    mpath = ds.mask_dir / paths[i].name
                    if not mpath.exists():
                        mpath = ds.mask_dir / (paths[i].stem + ".png")
                    orig = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 127
                    pn = F.interpolate(prob[j:j + 1], size=orig.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
                    c = image_counts(pn, orig)
                    for k in c:
                        res["native"][k][i] = c[k]
        wall = time.perf_counter() - t0
        np.savez_compressed(args.out / f"{prefix}obj2_counts_{name}_{stamp}.npz", images=np.array([p.name for p in paths]),
                            scenes=np.array(scenes), halves=halves,
                            **{f"{p}_{k}": v for p, d in res.items() for k, v in d.items()})
        result["models"][name] = {"checkpoint": str(path), "checkpoint_sha256": file_sha256(path), "meta": meta,
                                  "inference_s": wall,
                                  **{p: summarise(d["counts"], d["band_err"], d["band_px"], halves) for p, d in res.items()}}
        r = result["models"][name]["thesis256"]
        print(f"{name:22s} thesis256: native BER {r['native_tau_0.50']['ber']:.2f}%  test-selected {r['test_selected_BIASED']['ber']:.2f}% "
              f"(tau {r['test_selected_BIASED']['tau']})  cross-fitted {r['cross_fitted']['ber']:.2f}%  IoU {r['iou_tau_0.50']['mean_per_image']:.4f} "
              f"| native-res BER {result['models'][name]['native']['native_tau_0.50']['ber']:.2f}%  ({wall:.0f}s)", flush=True)
        dest = args.out / f"{prefix}obj2_eval_{stamp}.json"
        dest.write_text(json.dumps(result, indent=2))
    print("written:", dest)


if __name__ == "__main__":
    main()
