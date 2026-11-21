#!/bin/zsh
# UNet++ phase 2 with held-out validation, seeds 0-2, then their ISTD-test evaluation (Table 4.2).
# Called by run_obj2_unetpp.sh; on its own it redoes only these steps. Each training needs >= 3 GB free disk.
set -u
cd "${0:A:h}/.."
need_disk() { (( $(df -k . | awk 'NR==2{print $4}') >= 3*1024*1024 )) || { echo "less than 3 GB free disk; stopping"; exit 1; } }
for s in 0 1 2; do
  need_disk
  .venv/bin/python scripts/obj2_retrain_siran.py --arch unetpp --seed $s --tag seed$s
  echo "unetpp seed $s exit $?"
done
models=()
for f in checkpoints/obj2/unetpp_phase2_heldout_seed?_*.pth; do
  s=${${f:t}#*heldout_seed}; s=${s%%_*}
  models+=(--model "unetpp@heldout_seed$s=$f")
done
.venv/bin/python scripts/obj2_eval.py --batch 4 "${models[@]}"
echo "unetpp heldout eval exit $?"
