#!/bin/bash
# Stage 4: cross-family validation with Mistral-7B-Instruct.
# Fires after Stage 3's "ALL DONE" marker.
set -u
cd /home/jding/triagent
source venv/bin/activate

LOG=/tmp/overnight_stage4.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 4 waiting for Stage 3 marker at $(date) ==="
echo "    Watching: /tmp/overnight_stage3.log for '=== STAGE 3 ALL DONE'"
while true; do
    if grep -q "^=== STAGE 3 ALL DONE" /tmp/overnight_stage3.log 2>/dev/null; then
        break
    fi
    sleep 60
done
echo "=== Stage 3 done detected at $(date), starting Stage 4 ==="

echo
echo "--- Mistral-7B single-agent on FPB at $(date) ---"
python3 experiments/L1p5_size_sweep.py --sizes Mistral-7B --resume
MISTRAL_RC=$?
echo "Mistral L1.5 exit code: $MISTRAL_RC"

if [ $MISTRAL_RC -eq 0 ]; then
    echo
    echo "--- Refreshing sdi_data.csv to include Mistral column ---"
    python3 experiments/L2_sdi_analysis.py
fi

echo
echo "--- critic@Mistral-7B on FPB at $(date) ---"
python3 experiments/L2p5_interaction.py --protocols critic --llm-size Mistral-7B \
    --critic-trigger-col sdi_le --critic-threshold 0.4

echo
echo "--- Refresh L3.5 + per-class with Mistral data ---"
python3 experiments/L3p5_scaling.py
python3 experiments/L2p5_per_class_breakdown.py

echo
echo "=== STAGE 4 ALL DONE at $(date) ==="
