#!/bin/zsh
# Objective 2 fair baseline (Table 4.2): UNet++ (EfficientNet-B5). Waits for the RAN/RARM queue
# (one MPS job at a time) and for the two checkpoints from Drive, evaluates the original
# best_model_unetplusplus.pth on ISTD test, redoes phase 2 with held-out validation for seeds 0-2,
# then evaluates those on ISTD test. Log: logs/obj2_unetpp.log
#
# First redoes BiFPN-RARM seed 2, whose run in run_obj2_baseline_retrains.sh crashed while saving
# (results/obj2/failed/README.txt), and evaluates it. Each training needs >= 3 GB free disk.
set -u
cd "${0:A:h}/.."
ck=data/raw/checkpoints
need_disk() { (( $(df -k . | awk 'NR==2{print $4}') >= 3*1024*1024 )) || { echo "less than 3 GB free disk; stopping"; exit 1; } }
while pgrep -f run_obj2_baseline_retrains.sh >/dev/null; do sleep 60; done
need_disk
.venv/bin/python scripts/obj2_retrain_siran.py --arch rarm --seed 2 --tag seed2
echo "rarm seed 2 rerun exit $?"
for f in checkpoints/obj2/rarm_phase2_heldout_seed2_*.pth; do
  .venv/bin/python scripts/obj2_eval.py --batch 4 --model "rarm@heldout_seed2=$f"
done
echo "rarm seed 2 eval exit $?"
typeset -A size=(best_model_unetplusplus.pth 376098818 sbu_pretrained_unetplusplus.pth 376109614)   # Drive sizes
for n in ${(k)size}; do
  until [[ -f $ck/$n ]] && (( $(stat -f %z $ck/$n) == ${size[$n]} )); do sleep 60; done
done
shasum -a 256 $ck/best_model_unetplusplus.pth $ck/sbu_pretrained_unetplusplus.pth | tee -a data/raw/obj2_sha256.txt
.venv/bin/python scripts/obj2_eval.py --batch 4 --model unetpp=$ck/best_model_unetplusplus.pth
echo "unetpp original eval exit $?"
scripts/run_obj2_unetpp_retrains.sh
