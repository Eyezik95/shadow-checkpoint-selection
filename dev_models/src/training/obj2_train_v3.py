"""Verbatim copy of reference/notebooks/obj2_shadow_detector.ipynb cell 2 (train_shadow_detector_v3.py; notebook sha256 5f16e738d422e492...),
minus the Colab mount and the notebook runner, with the model import pointed at
src.models.obj2_variants. Provides ISTDDataset, losses, train_epoch, validate_epoch.
"""

"""
Two-Phase Training Script (v3): BiFPN-RAN / BiFPN-RARM / BiFPN-RARM-RAN
==========================================================================
Phase 1: Pre-train on SBU shadow dataset
Phase 2: Fine-tune on ISTD shadow dataset

WHAT'S NEW IN v3:
  - --architecture {bifpn_ran, bifpn_rarm, bifpn_rarm_ran}: pick which of
    the three architectures to train, from the same script/schedule/
    regularization stack, so ablation comparisons are actually fair.
  - --use_distraction_head (Option A): the DSDNet-inspired distraction-
    aware auxiliary branch, now available on ALL THREE architectures via
    the same flag. Uses DistractionAwareLoss instead of DeepSupervisionLoss
    when enabled.
  - --distraction_warmup_epochs: the distraction pseudo-labels are
    generated from the model's OWN current predictions each batch, which
    are close to noise in the first few epochs of Phase 2 before the
    network has learned anything reliable. The distraction loss term is
    linearly ramped from 0 to --distraction_weight over this many epochs
    instead of applied at full strength from epoch 1, so Phase 2 doesn't
    spend its early, most-important epochs chasing garbage targets.
  - BER (Balanced Error Rate) now computed every epoch alongside IoU, for
    both train and val, plus its two components (shadow-region error
    rate, non-shadow-region error rate) logged separately. This matters
    specifically for Option A: distraction suppression targets false
    positives, which are a small fraction of total mask pixels and can
    move BER's non-shadow-error component clearly while barely
    registering on whole-mask IoU. If you only watch IoU you can
    conclude "Option A didn't work" when it actually did.

Requires bifpn_rarm_ran_model.py in the same directory (or on sys.path).

Usage (Colab):
    python train_shadow_detector_v3.py \
        --architecture bifpn_rarm_ran \
        --use_distraction_head \
        --sbu_image_dir "/content/drive/MyDrive/Datasets/sbu/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages" \
        --sbu_mask_dir "/content/drive/MyDrive/Datasets/sbu/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks" \
        --istd_root "/content/drive/MyDrive/Datasets/ISTD/ISTD_Dataset" \
        --output_dir "/content/drive/MyDrive/Results"
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
#     from google.colab import drive    # Colab only; removed
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
sys.path.insert(0, script_dir)

from src.models.obj2_variants import (
    BiFPN_RAN_ShadowDetector,
    BiFPN_RARM_ShadowDetector,
    BiFPN_RARM_RAN_ShadowDetector,
    DeepSupervisionLoss,
    BCEDiceLoss,
    DistractionAwareLoss,
    freeze_for_finetuning,
    reassert_frozen_eval,
    get_discriminative_param_groups,
)

ARCHITECTURES = {
    'bifpn_ran': BiFPN_RAN_ShadowDetector,
    'bifpn_rarm': BiFPN_RARM_ShadowDetector,
    'bifpn_rarm_ran': BiFPN_RARM_RAN_ShadowDetector,
}


# ==============================================================================
# 1. DATASETS (unchanged from v2)
# ==============================================================================

class SBUDataset(Dataset):
    """SBU shadow dataset loader. Flat image_dir / mask_dir pair."""
    def __init__(self, image_dir, mask_dir, img_size=256, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment

        self.images = sorted(
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.jpg')) +
            list(self.image_dir.glob('*.jpeg'))
        )
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        print(f"[SBU] Loaded {len(self.images)} images")

        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def _find_mask_path(self, img_path):
        candidate = self.mask_dir / img_path.name
        if candidate.exists():
            return candidate
        candidate = self.mask_dir / (img_path.stem + '.png')
        if candidate.exists():
            return candidate
        return self.mask_dir / (img_path.stem + '.jpg')

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self._find_mask_path(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found for {img_path.name} (looked at {mask_path})")
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float()


class ISTDDataset(Dataset):
    """
    ISTD_Dataset/
        train/{train_A, train_B}
        test/{test_A, test_B}
    """
    def __init__(self, root_dir, split='train', img_size=256, augment=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        if split == 'train':
            self.image_dir = self.root_dir / 'train' / 'train_A'
            self.mask_dir = self.root_dir / 'train' / 'train_B'
        else:
            self.image_dir = self.root_dir / 'test' / 'test_A'
            self.mask_dir = self.root_dir / 'test' / 'test_B'

        self.images = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        print(f"[ISTD] Loaded {len(self.images)} images from {split} set")

        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.GaussNoise(p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32),
                                hole_width_range=(8, 32), fill=0, fill_mask=None, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

        self.mask_transform = A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + '.png')

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = self.transform(image=image)['image']
            mask = self.mask_transform(image=mask)['image']

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float()


# ==============================================================================
# 2. METRICS: IoU + BER (with confusion-matrix accumulation across an epoch)
# ==============================================================================

def calculate_iou(pred_prob, target, threshold=0.5):
    pred = (pred_prob > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()


class ConfusionAccumulator:
    """
    Accumulates TP/FP/TN/FN counts across an entire epoch (not per-batch
    averages) so BER is computed correctly at epoch granularity -- BER
    is a ratio of ratios (0.5*(TPR+TNR)), and averaging per-batch BER
    values is not equivalent to computing it from pooled counts,
    especially with small/uneven-composition batches like this dataset's.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0

    def update(self, pred_prob, target):
        pred = (pred_prob > self.threshold).float()
        target = (target > 0.5).float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()

    def compute(self):
        """Returns (BER, shadow_error_rate, non_shadow_error_rate), all as
        percentages, matching the convention used in shadow-detection
        papers (BDRAR/DSDNet/SDDNet-style tables)."""
        eps = 1e-6
        tpr = self.tp / (self.tp + self.fn + eps)   # shadow recall
        tnr = self.tn / (self.tn + self.fp + eps)   # non-shadow recall
        shadow_error_rate = 100.0 * (1.0 - tpr)      # missed shadow pixels
        non_shadow_error_rate = 100.0 * (1.0 - tnr)  # false-positive shadow pixels
        ber = 100.0 * (1.0 - 0.5 * (tpr + tnr))
        return ber, shadow_error_rate, non_shadow_error_rate


# ==============================================================================
# 3. TRAIN / VALIDATE LOOPS
# ==============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs,
                 frozen_module_names=(), grad_clip_norm=None):
    model.train()
    reassert_frozen_eval(model, frozen_module_names)

    running_loss = 0.0
    running_iou = 0.0
    confusion = ConfusionAccumulator()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip_norm
            )

        optimizer.step()

        running_loss += loss.item()
        final_pred_prob = torch.sigmoid(outputs[0])
        running_iou += calculate_iou(final_pred_prob, masks)
        confusion.update(final_pred_prob.detach(), masks)

        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'iou': f'{running_iou / (pbar.n + 1):.4f}'
        })

    ber, shadow_err, non_shadow_err = confusion.compute()
    return running_loss / len(dataloader), running_iou / len(dataloader), ber, shadow_err, non_shadow_err


def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    confusion = ConfusionAccumulator()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Val]')

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            pred_prob = model(images)
            loss = criterion(pred_prob, masks)
            running_loss += loss.item()
            running_iou += calculate_iou(pred_prob, masks)
            confusion.update(pred_prob, masks)
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'iou': f'{running_iou / (pbar.n + 1):.4f}'
            })

    ber, shadow_err, non_shadow_err = confusion.compute()
    return running_loss / len(dataloader), running_iou / len(dataloader), ber, shadow_err, non_shadow_err


# ==============================================================================
# 4. TWO-PHASE TRAINING PIPELINE
# ==============================================================================

def build_model(args):
    cls = ARCHITECTURES[args.architecture]
    kwargs = dict(
        in_channels=3, out_channels=1,
        features=[64, 128, 256, 512, 1024],
        dropout_p=args.dropout_p,
        share_weights=args.share_weights,
        use_distraction_head=args.use_distraction_head,
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
    )
    if args.architecture == 'bifpn_ran':
        kwargs['ra_channels'] = 64
    elif args.architecture == 'bifpn_rarm':
        kwargs['rarm_iterations'] = 2
    elif args.architecture == 'bifpn_rarm_ran':
        kwargs['rarm_iterations'] = 2
        kwargs['ra_channels'] = 64
    return cls(**kwargs)


def train_two_phase(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Architecture: {args.architecture} | Backbone: {args.backbone} "
          f"(pretrained={args.pretrained_backbone}) | Option A (distraction head): {args.use_distraction_head}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_tag = args.backbone if args.backbone != 'custom' else 'scratch'
    run_tag = f"{args.architecture}_{backbone_tag}{'_distraction' if args.use_distraction_head else ''}"

    model = build_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (share_weights={args.share_weights})")

    plain_criterion = DeepSupervisionLoss(bce_weight=0.5, dice_weight=0.5)
    distraction_criterion = DistractionAwareLoss(
        bce_weight=0.5, dice_weight=0.5,
        distraction_weight=args.distraction_weight,
        hard_negative_thresh=args.hard_negative_thresh,
    ) if args.use_distraction_head else None
    val_criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, logits=False)

    def get_train_criterion(epoch_in_phase2=None):
        """Applies distraction-loss warmup: ramps distraction_weight linearly
        from 0 to args.distraction_weight over distraction_warmup_epochs,
        since the pseudo-labels are near-noise before the model has learned
        anything reliable. Only relevant in Phase 2 / when Option A is on."""
        if not args.use_distraction_head or epoch_in_phase2 is None:
            return plain_criterion if not args.use_distraction_head else distraction_criterion
        if args.distraction_warmup_epochs <= 0:
            return distraction_criterion
        warmup_frac = min(1.0, epoch_in_phase2 / args.distraction_warmup_epochs)
        distraction_criterion.distraction_weight = args.distraction_weight * warmup_frac
        return distraction_criterion

    # ==========================================
    # PHASE 1: PRE-TRAIN ON SBU
    # (Option A intentionally NOT used here even if requested for Phase 2:
    #  the distraction pseudo-labels need a model that already produces
    #  reasonably calibrated predictions, which Phase 1 -- starting from
    #  random init -- does not have for most of its schedule. Phase 1
    #  always trains with plain DeepSupervisionLoss regardless of
    #  --use_distraction_head; the flag only affects Phase 2.)
    # ==========================================
    if args.skip_phase1:
        print("\n" + "=" * 80)
        print("PHASE 1 SKIPPED (--skip_phase1): proceeding directly to Phase 2 with "
              f"{'ImageNet-pretrained' if args.backbone != 'custom' and args.pretrained_backbone else 'randomly-initialized'} "
              "weights.")
        print("=" * 80 + "\n")
        torch.save({'epoch': 0, 'model_state_dict': model.state_dict()},
                   output_dir / f'sbu_pretrained_{run_tag}.pth')
    else:
        print("\n" + "=" * 80)
        print("PHASE 1: PRE-TRAINING ON SBU DATASET")
        print("=" * 80 + "\n")

        sbu_train_dataset = SBUDataset(args.sbu_image_dir, args.sbu_mask_dir, args.img_size, augment=True)
        sbu_train_loader = DataLoader(sbu_train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_phase1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history_phase1 = {'train_loss': [], 'train_iou': [], 'train_ber': [],
                           'train_shadow_err': [], 'train_non_shadow_err': [], 'lr': []}
        best_loss = float('inf')

        for epoch in range(1, args.phase1_epochs + 1):
            train_loss, train_iou, train_ber, train_shadow_err, train_non_shadow_err = train_epoch(
                model, sbu_train_loader, plain_criterion, optimizer, device, epoch, args.phase1_epochs,
                grad_clip_norm=args.grad_clip_norm
            )
            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history_phase1['train_loss'].append(train_loss)
            history_phase1['train_iou'].append(train_iou)
            history_phase1['train_ber'].append(train_ber)
            history_phase1['train_shadow_err'].append(train_shadow_err)
            history_phase1['train_non_shadow_err'].append(train_non_shadow_err)
            history_phase1['lr'].append(current_lr)

            print(f"\nPhase 1 - Epoch {epoch}/{args.phase1_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | LR: {current_lr:.6f}")
            print(f"  Train BER: {train_ber:.2f}% (shadow err: {train_shadow_err:.2f}%, "
                  f"non-shadow err: {train_non_shadow_err:.2f}%)")

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss, 'train_iou': train_iou},
                           output_dir / f'sbu_pretrained_{run_tag}.pth')
                print("  \u2713 Best SBU model saved!")
            print("=" * 80)

        with open(output_dir / f'phase1_history_{run_tag}.json', 'w') as f:
            json.dump(history_phase1, f, indent=2)

        print("\nPHASE 1 COMPLETE\n")

    # ==========================================
    # PHASE 2: FINE-TUNE ON ISTD
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNING ON ISTD DATASET")
    print("=" * 80 + "\n")

    checkpoint = torch.load(output_dir / f'sbu_pretrained_{run_tag}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\u2713 Loaded SBU pre-trained weights")

    # Resolve the freeze-module default: 'stem' only exists for
    # backbone='custom'; for a pretrained (timm) backbone the equivalent
    # gentle-start module is 'timm_backbone' itself, handled at 0.1x LR
    # by get_discriminative_param_groups rather than a hard freeze, so
    # the sensible default there is "freeze nothing" unless requested.
    freeze_modules = args.freeze_modules
    if freeze_modules is None:
        freeze_modules = ['stem'] if args.backbone == 'custom' else []

    frozen_module_names = []
    if freeze_modules:
        print(f"\nFreezing for Phase 2: {freeze_modules}")
        frozen_module_names = freeze_for_finetuning(model, frozen_modules=tuple(freeze_modules))

    istd_train_dataset = ISTDDataset(args.istd_root, split='train', img_size=args.img_size, augment=True)
    istd_val_dataset = ISTDDataset(args.istd_root, split='test', img_size=args.img_size, augment=False)

    istd_train_loader = DataLoader(istd_train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
    istd_val_loader = DataLoader(istd_val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    phase2_base_lr = args.lr / 10
    param_groups = get_discriminative_param_groups(model, phase2_base_lr, frozen_module_names)
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay_phase2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history_phase2 = {'train_loss': [], 'train_iou': [], 'train_ber': [],
                       'val_loss': [], 'val_iou': [], 'val_ber': [],
                       'val_shadow_err': [], 'val_non_shadow_err': [],
                       'lr': [], 'overfit_gap': [], 'distraction_weight': []}
    best_iou = 0.0
    patience_counter = 0
    consecutive_high_gap = 0

    for epoch in range(1, args.phase2_epochs + 1):
        criterion = get_train_criterion(epoch_in_phase2=epoch)
        current_distraction_weight = getattr(criterion, 'distraction_weight', 0.0)

        train_loss, train_iou, train_ber, train_shadow_err, train_non_shadow_err = train_epoch(
            model, istd_train_loader, criterion, optimizer, device, epoch, args.phase2_epochs,
            frozen_module_names=frozen_module_names, grad_clip_norm=args.grad_clip_norm
        )
        val_loss, val_iou, val_ber, val_shadow_err, val_non_shadow_err = validate_epoch(
            model, istd_val_loader, val_criterion, device, epoch, args.phase2_epochs
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[-1]['lr']
        overfit_gap = train_iou - val_iou

        history_phase2['train_loss'].append(train_loss)
        history_phase2['train_iou'].append(train_iou)
        history_phase2['train_ber'].append(train_ber)
        history_phase2['val_loss'].append(val_loss)
        history_phase2['val_iou'].append(val_iou)
        history_phase2['val_ber'].append(val_ber)
        history_phase2['val_shadow_err'].append(val_shadow_err)
        history_phase2['val_non_shadow_err'].append(val_non_shadow_err)
        history_phase2['lr'].append(current_lr)
        history_phase2['overfit_gap'].append(overfit_gap)
        history_phase2['distraction_weight'].append(current_distraction_weight)

        print(f"\nPhase 2 - Epoch {epoch}/{args.phase2_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train BER: {train_ber:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f} | Val BER:   {val_ber:.2f}%")
        print(f"  Val shadow err: {val_shadow_err:.2f}% | Val non-shadow (false-positive) err: {val_non_shadow_err:.2f}%")
        print(f"  Train-Val IoU gap: {overfit_gap:+.4f} | Decoder LR: {current_lr:.6f}", end="")
        if args.use_distraction_head:
            print(f" | Distraction weight: {current_distraction_weight:.3f}")
        else:
            print()

        if overfit_gap > args.overfit_gap_threshold:
            consecutive_high_gap += 1
            print(f"  \u26a0 Overfitting warning: gap above {args.overfit_gap_threshold} "
                  f"for {consecutive_high_gap} consecutive epoch(s)")
        else:
            consecutive_high_gap = 0

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_iou': val_iou, 'val_loss': val_loss, 'val_ber': val_ber},
                       output_dir / f'best_model_{run_tag}.pth')
            print(f"  \u2713 Best model saved! (IoU: {best_iou:.4f}, BER: {val_ber:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping! No val IoU improvement for {args.patience} epochs.")
            break

        if consecutive_high_gap >= args.overfit_patience:
            print(f"\nStopping: train-val IoU gap exceeded {args.overfit_gap_threshold} "
                  f"for {args.overfit_patience} consecutive epochs.")
            break

        print("=" * 80)

    with open(output_dir / f'phase2_history_{run_tag}.json', 'w') as f:
        json.dump(history_phase2, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Architecture: {args.architecture} | Backbone: {args.backbone} | Option A: {args.use_distraction_head}")
    print(f"Best validation IoU: {best_iou:.4f}")
    if history_phase2['val_ber']:
        best_idx = history_phase2['val_iou'].index(best_iou)
        print(f"BER at best-IoU epoch: {history_phase2['val_ber'][best_idx]:.2f}% "
              f"(shadow err: {history_phase2['val_shadow_err'][best_idx]:.2f}%, "
              f"non-shadow err: {history_phase2['val_non_shadow_err'][best_idx]:.2f}%)")
    print(f"Models saved to: {output_dir}")
    print("=" * 80 + "\n")


# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Two-phase SBU -> ISTD shadow detector training (v3)")

    parser.add_argument('--architecture', type=str, default='bifpn_rarm_ran',
                         choices=list(ARCHITECTURES.keys()))
    parser.add_argument('--backbone', type=str, default='custom',
                         help="'custom' for the from-scratch encoder (original behavior), "
                              "or any timm model name (e.g. 'resnet50', 'efficientnet_b4') "
                              "for an ImageNet-pretrained encoder feeding the same BiFPN/"
                              "RARM/RAN decoder. Must produce exactly 5 feature levels via "
                              "features_only=True, out_indices=(0,1,2,3,4) -- verified for "
                              "resnet50 (channels [64,256,512,1024,2048]); other backbones "
                              "aren't guaranteed to have 5 usable stages.")
    parser.add_argument('--pretrained_backbone', action='store_true', default=True,
                         help='Load ImageNet weights for --backbone (ignored if backbone=custom).')
    parser.add_argument('--no_pretrained_backbone', dest='pretrained_backbone', action='store_false',
                         help='Use --backbone architecture with random init instead of ImageNet weights '
                              '(useful for isolating architecture-shape effects from pretraining effects).')

    parser.add_argument('--sbu_image_dir', type=str,
                         default="/content/drive/MyDrive/Datasets/sbu/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages")
    parser.add_argument('--sbu_mask_dir', type=str,
                         default="/content/drive/MyDrive/Datasets/sbu/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks")
    parser.add_argument('--istd_root', type=str, default="/content/drive/MyDrive/Datasets/ISTD/ISTD_Dataset")
    parser.add_argument('--output_dir', type=str, default="/content/drive/MyDrive/Results")

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay_phase1', type=float, default=1e-4)
    parser.add_argument('--weight_decay_phase2', type=float, default=1e-3)
    parser.add_argument('--phase1_epochs', type=int, default=50)
    parser.add_argument('--phase2_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--skip_phase1', action='store_true', default=False,
                         help='Skip SBU pretraining and fine-tune on ISTD directly. Only sensible '
                              'with a pretrained --backbone (e.g. resnet50) -- with backbone=custom '
                              'this trains a randomly-initialized encoder on just 1,330 images.')

    # Model regularization
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--share_weights', action='store_true', default=True)
    parser.add_argument('--no_share_weights', dest='share_weights', action='store_false')

    # Phase 2 freezing. Default is None (resolved dynamically in
    # train_two_phase): ['stem'] for backbone='custom', [] for a
    # pretrained backbone (which instead gets a gentle 0.1x LR via
    # get_discriminative_param_groups rather than a hard freeze).
    parser.add_argument('--freeze_modules', nargs='*', default=None,
                         help="Submodule names to freeze during Phase 2. Pass an empty list "
                              "to disable freezing entirely. Omit this flag to use the "
                              "backbone-appropriate default described above.")

    # Overfitting monitor
    parser.add_argument('--overfit_gap_threshold', type=float, default=0.10)
    parser.add_argument('--overfit_patience', type=int, default=5)

    # Option A: distraction-aware auxiliary branch
    parser.add_argument('--use_distraction_head', action='store_true', default=False,
                         help='Enable the DSDNet-inspired distraction-aware branch (Option A). '
                              'Works with any --architecture.')
    parser.add_argument('--distraction_weight', type=float, default=0.3,
                         help='Final (post-warmup) weight of the distraction loss term.')
    parser.add_argument('--hard_negative_thresh', type=float, default=0.3,
                         help='Prediction probability above which a non-shadow pixel is '
                              'treated as a distraction pseudo-label.')
    parser.add_argument('--distraction_warmup_epochs', type=int, default=5,
                         help='Linearly ramp distraction_weight from 0 over this many Phase-2 '
                              'epochs, since pseudo-labels are unreliable before the model '
                              'has learned anything. Set 0 to disable (full weight from epoch 1).')

    return parser.parse_args(args=[]) if _running_in_notebook() else parser.parse_args()

