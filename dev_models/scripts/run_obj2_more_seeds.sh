#!/bin/zsh
# Phase 2 with held-out validation, seeds 3 and 4, for all four shadow detectors (paper experiment B),
# then each new checkpoint evaluated on the ISTD and SBU test sets. One MPS job at a time.
# Order: fastest first (UNet++ ~0.9 h/run, SIRAN ~1.6 h, RAN 1.5-2.8 h, RARM 1.4-3.4 h).
# Each training needs >= 3 GB free disk; the script stops rather than fill the disk.
set -u
cd "${0:A:h}/.."
need_disk() { (( $(df -k . | awk 'NR==2{print $4}') >= 3*1024*1024 )) || { echo "less than 3 GB free disk; stopping"; exit 1; } }
for arch in unetpp siran ran rarm; do
  for s in 3 4; do
    need_disk
    .venv/bin/python scripts/obj2_retrain_siran.py --arch $arch --seed $s --tag seed$s
    echo "$arch seed $s exit $?"
  done
done
models=()
for arch in unetpp siran ran rarm; do
  for s in 3 4; do
    f=$(ls -t checkpoints/obj2/${arch}_phase2_heldout_seed${s}_*.pth 2>/dev/null | head -1)
    [[ -n "$f" ]] && models+=(--model "$arch@heldout_seed$s=$f")
  done
done
.venv/bin/python scripts/obj2_eval.py --batch 4 "${models[@]}"
echo "istd eval exit $?"
.venv/bin/python scripts/obj2_eval.py --dataset sbu --batch 4 "${models[@]}"
echo "sbu eval exit $?"
