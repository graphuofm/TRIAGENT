#!/bin/bash
# Stage 7: Chinese language expansion.
#  (a) Translate 1500 FPB sentences (5x current pilot)
#  (b) Multi-size Qwen sweep on the translated set
#  (c) Cross-lingual critic experiment: use ENGLISH V+F outputs +
#      Chinese sentence sent to LLM → does the plateau hold?
#  (d) Optional: run on FinChinaSentiment real Chinese (Qwen only,
#      single-agent, since V+F don't handle Chinese)
# Fires after the current debate sweep finishes.
set -u
cd /home/jding/triagent
source venv/bin/activate

LOG=/tmp/overnight_stage7.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 7 (Chinese expansion) waiting for debate sweep done at $(date) ==="
while true; do
    if grep -q "^=== DEBATE SWEEP ALL DONE" /tmp/overnight_debate_sweep.log 2>/dev/null; then
        break
    fi
    sleep 60
done
echo "=== Debate sweep done at $(date), starting Stage 7 ==="

echo
echo "--- Translate 1500 FPB sentences to Chinese (Qwen-7B) at $(date) ---"
python3 experiments/L7_chinese_pilot.py --translate --n 1500

echo
echo "--- Multi-size Qwen sweep on translated FPB (1500 zh) at $(date) ---"
python3 experiments/L7_chinese_pilot.py --sweep --sizes 0.5B,1.5B,3B,7B

echo
echo "--- FinChinaSentiment single-agent on real Chinese (Qwen-7B only) ---"
python3 experiments/L1_data_collection.py --dataset finchina --skip-llm --yes
echo "(Note: FinChina has no positive class; binary neg/neu only; V+F skipped — they don't do Chinese)"

echo
echo "=== STAGE 7 ALL DONE at $(date) ==="
