#!/bin/zsh
# Five SDDNet seeds (0-4), one MPS job at a time, under caffeinate so the Mac does not idle-sleep.
# Memory caps make MPS fail with an error instead of swapping (17 GB Mac; see code/sddnet_dual_select.py).
set -u
cd "${0:A:h}/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
for s in 0 1 2 3 4; do
  free_kb=$(df -k . | awk 'NR==2{print $4}')
  (( free_kb >= 3*1024*1024 )) || { echo "QUEUE STOP: less than 3 GB free disk before seed $s"; exit 1; }
  echo "QUEUE seed $s start $(date '+%F %T')"
  .venv/bin/python code/sddnet_dual_select.py --seed $s --accum 2 > logs/sddnet_seed$s.log 2>&1
  rc=$?
  echo "QUEUE seed $s exit $rc $(date '+%F %T')"
  (( rc == 0 )) || { echo "QUEUE STOP: seed $s failed"; tail -5 logs/sddnet_seed$s.log; exit $rc; }
done
echo "QUEUE ALL DONE $(date '+%F %T')"
