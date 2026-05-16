#!/bin/bash
# Stage 5: TFNS critic@1.5B fixup (the inline-python bug in stage 3 prevented this)
# + full L5 backtest + L5.5 E1 perturbation experiment.
# Fires after Stage 4's "ALL DONE" marker.
set -u
cd /home/jding/triagent
source venv/bin/activate

LOG=/tmp/overnight_stage5.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage 5 waiting for Stage 4 marker at $(date) ==="
echo "    Watching: /tmp/overnight_stage4.log for '=== STAGE 4 ALL DONE'"
while true; do
    if grep -q "^=== STAGE 4 ALL DONE" /tmp/overnight_stage4.log 2>/dev/null; then
        break
    fi
    sleep 60
done
echo "=== Stage 4 done detected at $(date), starting Stage 5 ==="

echo
echo "--- Rebuild sdi_data_tfns.csv with multi-size LLM columns at $(date) ---"
python3 -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('results/data/committee_data_tfns.csv')
df['sdi_le'] = (df['vader_score']  - df['finbert_score']).abs()
df['sdi_lr'] = (df['vader_score']  - df['llm_score']    ).abs()
df['sdi_er'] = (df['finbert_score']- df['llm_score']    ).abs()
df['sdi_max']  = df[['sdi_le','sdi_lr','sdi_er']].max(axis=1)
df['sdi_mean'] = df[['sdi_le','sdi_lr','sdi_er']].mean(axis=1)
out = Path('results/data/sdi_data_tfns.csv')
df.to_csv(out, index=False)
has_1p5b = 'llm_1p5b_label' in df.columns
has_3b   = 'llm_3b_label'   in df.columns
print(f'rebuilt {out}, n={len(df)}, has 1p5b={has_1p5b}, has 3b={has_3b}')
"

echo
echo "--- TFNS critic@1.5B at $(date) ---"
cp results/data/sdi_data.csv results/data/sdi_data_fpb_backup3.csv
cp results/data/sdi_data_tfns.csv results/data/sdi_data.csv
python3 experiments/L2p5_interaction.py --protocols critic --llm-size 1.5B \
    --critic-trigger-col sdi_le --critic-threshold 0.4
[ -f results/data/interaction_results_critic_1p5b.csv ] && \
    mv results/data/interaction_results_critic_1p5b.csv results/data/interaction_results_critic_1p5b_tfns.csv
[ -f results/data/interaction_summary_critic_1p5b.csv ] && \
    mv results/data/interaction_summary_critic_1p5b.csv results/data/interaction_summary_critic_1p5b_tfns.csv
cp results/data/sdi_data_fpb_backup3.csv results/data/sdi_data.csv

echo
echo "--- Full L5 backtest on all 20 tickers (FPB-mode) at $(date) ---"
python3 experiments/L5_backtest.py --input results/data/sdi_data_fpb_backup2.csv

echo
echo "--- Refresh L3.5 + per-class breakdown ---"
python3 experiments/L3p5_scaling.py
python3 experiments/L2p5_per_class_breakdown.py
python3 experiments/L5p5_e2_hallucination_detector.py
python3 experiments/L5p5_e3_bias_diversity_detail.py

echo
echo "=== STAGE 5 ALL DONE at $(date) ==="
