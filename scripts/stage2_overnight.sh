#!/bin/bash
# Stage 2 overnight chain — fires after Stage 1's "ALL DONE" marker appears.
set -u  # treat unset vars as errors; do NOT use -e (we want best-effort)

cd /home/jding/triagent
source venv/bin/activate

LOG=/tmp/overnight_stage2.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 2 waiting for Stage 1 marker at $(date) ==="
echo "    Watching: /tmp/overnight.log for '=== ALL DONE'"
while true; do
    if grep -q "^=== ALL DONE" /tmp/overnight.log 2>/dev/null; then
        break
    fi
    sleep 60
done
echo "=== Stage 1 done detected at $(date), starting Stage 2 ==="

echo
echo "--- TFNS L1 with LLM (Qwen-7B) at $(date) ---"
python3 experiments/L1_data_collection.py --dataset tfns --yes
TFNS_L1_RC=$?
echo "TFNS L1 exit code: $TFNS_L1_RC"

if [ $TFNS_L1_RC -eq 0 ]; then
    echo
    echo "--- Building TFNS sdi_data shim at $(date) ---"
    python3 - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv('results/data/committee_data_tfns.csv')
df['sdi_le'] = (df['vader_score']  - df['finbert_score']).abs()
df['sdi_lr'] = (df['vader_score']  - df['llm_score']    ).abs()
df['sdi_er'] = (df['finbert_score']- df['llm_score']    ).abs()
df['sdi_max'] = df[['sdi_le','sdi_lr','sdi_er']].max(axis=1)
df['sdi_mean']= df[['sdi_le','sdi_lr','sdi_er']].mean(axis=1)
out = Path('results/data/sdi_data_tfns.csv')
df.to_csv(out, index=False)
print(f"saved {out}, n={len(df)}")
PY

    echo
    echo "--- TFNS critic@1.5B at $(date) ---"
    cp results/data/sdi_data.csv results/data/sdi_data_fpb_backup.csv
    cp results/data/sdi_data_tfns.csv results/data/sdi_data.csv
    python3 experiments/L2p5_interaction.py --protocols critic --llm-size 1.5B \
         --critic-trigger-col sdi_le --critic-threshold 0.4
    if [ -f results/data/interaction_results_critic_1p5b.csv ]; then
        mv results/data/interaction_results_critic_1p5b.csv results/data/interaction_results_critic_1p5b_tfns.csv
    fi
    if [ -f results/data/interaction_summary_critic_1p5b.csv ]; then
        mv results/data/interaction_summary_critic_1p5b.csv results/data/interaction_summary_critic_1p5b_tfns.csv
    fi
    cp results/data/sdi_data_fpb_backup.csv results/data/sdi_data.csv
fi

echo
echo "--- Chinese translate (300 FPB sentences) at $(date) ---"
python3 experiments/L7_chinese_pilot.py --translate --n 300

echo
echo "--- Chinese sweep Qwen-1.5B/3B/7B at $(date) ---"
python3 experiments/L7_chinese_pilot.py --sweep --sizes 1.5B,3B,7B

echo
echo "--- Refresh L3.5 + per-class breakdown at $(date) ---"
python3 experiments/L3p5_scaling.py
python3 experiments/L2p5_per_class_breakdown.py

echo
echo "=== STAGE 2 ALL DONE at $(date) ==="
