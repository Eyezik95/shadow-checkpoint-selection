#!/bin/zsh
# Objective 2 fair baselines (Table 4.2): RAN iteration-1 evaluation with its original code, then
# BiFPN-RAN and BiFPN-RARM phase 2 redone with held-out validation, seeds 0-2, then their test evaluation.
# One MPS job at a time. Logs: logs/obj2_baseline_retrains.log
set -u
cd "${0:A:h}/.."
while pgrep -f "obj2_retrain_siran.py --arch rarm --epochs 1" >/dev/null; do sleep 30; done   # timing trial
.venv/bin/python scripts/obj2_eval.py --batch 4 --model ran_iter1=data/raw/checkpoints/best_model_bifpn_ran.pth
echo "ran_iter1 eval exit $?"
for arch in ran rarm; do
  for s in 0 1 2; do
    .venv/bin/python scripts/obj2_retrain_siran.py --arch $arch --seed $s --tag seed$s
    echo "$arch seed $s exit $?"
  done
done
models=()
for f in checkpoints/obj2/ran_phase2_heldout_seed?_*.pth checkpoints/obj2/rarm_phase2_heldout_seed?_*.pth; do
  a=${${f:t}%%_phase2*}; s=${${f:t}#*heldout_seed}; s=${s%%_*}
  models+=(--model "$a@heldout_seed$s=$f")
done
.venv/bin/python scripts/obj2_eval.py --batch 4 "${models[@]}"
echo "baseline eval exit $?"
