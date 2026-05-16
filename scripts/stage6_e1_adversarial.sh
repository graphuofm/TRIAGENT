#!/bin/bash
# Stage 6: E1 adversarial-perturbation detector experiment.
# Re-runs V+F+Qwen-7B on the 2500 perturbed sentences and analyses ΔSDI.
# Fires after Stage 5's "ALL DONE" marker.
set -u
cd /home/jding/triagent
source venv/bin/activate

LOG=/tmp/overnight_stage6.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 6 waiting for Stage 5 marker at $(date) ==="
while true; do
    if grep -q "^=== STAGE 5 ALL DONE" /tmp/overnight_stage5.log 2>/dev/null; then
        break
    fi
    sleep 60
done
echo "=== Stage 5 done at $(date), starting Stage 6 (E1 adversarial) ==="

echo
echo "--- Run committee on 2500 perturbed sentences at $(date) ---"
python3 experiments/L5p5_e1_adversarial.py --run-committee

echo
echo "--- Analyse ΔSDI + AUC at $(date) ---"
python3 experiments/L5p5_e1_adversarial.py --analyse

echo
echo "=== STAGE 6 ALL DONE at $(date) ==="
