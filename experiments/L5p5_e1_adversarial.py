"""L5.5 Experiment E1: SDI as adversarial-perturbation detector.

For 500 held-out FPB sentences, generate four kinds of perturbations:
  - synonym_swap: replace one sentiment word with a WordNet synonym
  - negation_insert: insert "did not" / "no" before a sentiment word
  - numeric_perturb: scale up a number by 10-100x
  - char_drop: drop random non-space chars (typo simulation)

Re-run VADER + FinBERT + Qwen-7B on the perturbed sentences. Compute
ΔSDI per perturbation, then AUC of "ΔSDI > τ" as binary "is this
perturbed" detector.

Headline target: AUC ≥ 0.75 for at least one SDI variant on at least
one perturbation type.

Outputs:
    data/raw/fpb_perturbed.csv
    results/data/e1_adversarial_committee.csv
    results/data/e1_delta_sdi.csv
    results/figures/fig_e1_delta_sdi.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, RAW_DIR
from src.viz.style import apply_style


# ---------------------------------------------------------------------------
# Perturbation operators (CPU-only)
# ---------------------------------------------------------------------------

# A small in-house "synonym" map for finance-relevant sentiment words.
# WordNet would technically be more rigorous but adds nltk dependency; for
# a 500-sample probe, this hand-curated set is enough.
SYNONYMS = {
    'profit':    ['gain', 'earnings'],
    'loss':      ['deficit', 'shortfall'],
    'increased': ['rose', 'climbed'],
    'decreased': ['fell', 'declined'],
    'rose':      ['climbed', 'grew'],
    'fell':      ['dropped', 'declined'],
    'declined':  ['fell', 'dropped'],
    'growth':    ['expansion', 'rise'],
    'gained':    ['rose', 'increased'],
    'higher':    ['greater', 'larger'],
    'lower':     ['smaller', 'reduced'],
    'good':      ['solid', 'positive'],
    'bad':       ['poor', 'weak'],
    'strong':    ['robust', 'firm'],
    'weak':      ['soft', 'sluggish'],
}


def synonym_swap(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    targets = [(i, w) for i, w in enumerate(words)
               if w.lower().strip('.,;:!?') in SYNONYMS]
    if not targets:
        return text
    i, w = targets[rng.integers(0, len(targets))]
    base = w.lower().strip('.,;:!?')
    syn = rng.choice(SYNONYMS[base])
    # preserve trailing punctuation
    suf = w[len(base):] if w.lower().startswith(base) else ''
    words[i] = syn + suf
    return ' '.join(words)


def negation_insert(text: str, rng: np.random.Generator) -> str:
    """Insert 'did not' before the first verb-y sentiment word."""
    targets = ['rose', 'fell', 'increased', 'decreased', 'declined',
               'gained', 'dropped', 'grew', 'expanded']
    words = text.split()
    for i, w in enumerate(words):
        if w.lower().strip('.,;:') in targets:
            words[i] = 'did not ' + w
            return ' '.join(words)
    return text + ' but this did not actually happen.'


def numeric_perturb(text: str, rng: np.random.Generator) -> str:
    """Scale the first numeric literal by 10-100x."""
    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if not m:
        return text
    val = float(m.group(1))
    factor = float(rng.choice([10, 100, 0.1, 0.01]))
    new_val = val * factor
    new_str = str(int(new_val)) if new_val.is_integer() else f"{new_val:.2f}"
    return text[:m.start()] + new_str + text[m.end():]


def char_drop(text: str, rng: np.random.Generator, drop_prob: float = 0.05) -> str:
    chars = list(text)
    out = []
    for c in chars:
        if c != ' ' and rng.random() < drop_prob:
            continue
        out.append(c)
    return ''.join(out)


PERTURBATIONS = {
    'synonym':  synonym_swap,
    'negation': negation_insert,
    'numeric':  numeric_perturb,
    'chardrop': char_drop,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def cmd_generate(args):
    """Generate perturbed sentences and save to data/raw/fpb_perturbed.csv."""
    src = RESULTS_DATA_DIR / 'sdi_data_fpb_backup2.csv'
    if not src.exists():
        src = RESULTS_DATA_DIR / 'sdi_data.csv'
    sdi = pd.read_csv(src)
    sample = sdi.sample(args.n, random_state=42).reset_index(drop=True)
    print(f"Generating perturbations for {len(sample)} held-out FPB sentences...")
    rng = np.random.default_rng(42)
    rows = []
    for _, r in sample.iterrows():
        clean = str(r['sentence'])
        rows.append({'sentence_id': int(r['sentence_id']),
                     'gold_label':  r['label_text'],
                     'perturbation':'clean',
                     'sentence':    clean})
        for pname, fn in PERTURBATIONS.items():
            rows.append({'sentence_id': int(r['sentence_id']),
                         'gold_label':  r['label_text'],
                         'perturbation': pname,
                         'sentence':    fn(clean, rng)})
    out = pd.DataFrame(rows)
    out_path = RAW_DIR / 'fpb_perturbed.csv'
    out.to_csv(out_path, index=False)
    print(f"  ✓ saved {out_path} ({len(out)} rows = {args.n} samples × 5 variants)")
    return out_path


def cmd_run_committee(args):
    """Run VADER + FinBERT + Qwen-7B on the perturbed sentences."""
    src = RAW_DIR / 'fpb_perturbed.csv'
    if not src.exists():
        raise FileNotFoundError(f"Run --generate first to produce {src}")
    df = pd.read_csv(src)
    print(f"Running committee on {len(df)} perturbation variants...")

    from src.agents.vader_agent import VADERAgent
    from src.agents.finbert_agent import FinBERTAgent
    from src.agents.llm_agent import LLMAgent

    sentences = df['sentence'].fillna('').tolist()

    print("\n=== VADER ===")
    v = VADERAgent()
    vres = v.predict_batch(sentences)
    df['vader_score']   = [r.score for r in vres]
    df['vader_label']   = [r.label for r in vres]

    print("\n=== FinBERT ===")
    f = FinBERTAgent()
    fres = f.predict_batch(sentences, batch_size=32)
    df['finbert_score'] = [r.score for r in fres]
    df['finbert_label'] = [r.label for r in fres]

    print("\n=== Qwen-7B ===")
    llm = LLMAgent(model_name='Qwen/Qwen2.5-7B-Instruct')
    lres = llm.predict_batch(sentences, batch_size=8)
    df['llm_score']     = [r.score for r in lres]
    df['llm_label']     = [r.label for r in lres]

    df['sdi_le'] = (df['vader_score']   - df['finbert_score']).abs()
    df['sdi_lr'] = (df['vader_score']   - df['llm_score']).abs()
    df['sdi_er'] = (df['finbert_score'] - df['llm_score']).abs()
    df['sdi_max']  = df[['sdi_le','sdi_lr','sdi_er']].max(axis=1)
    df['sdi_mean'] = df[['sdi_le','sdi_lr','sdi_er']].mean(axis=1)

    out = RESULTS_DATA_DIR / 'e1_adversarial_committee.csv'
    df.to_csv(out, index=False)
    print(f"  ✓ saved {out}")


def cmd_analyse(args):
    apply_style()
    src = RESULTS_DATA_DIR / 'e1_adversarial_committee.csv'
    if not src.exists():
        raise FileNotFoundError(f"Run --run-committee first to produce {src}")
    df = pd.read_csv(src)

    # ΔSDI per (sentence_id, perturbation) vs the clean variant
    clean = df[df['perturbation'] == 'clean'].set_index('sentence_id')
    rows = []
    for ptype in ['synonym', 'negation', 'numeric', 'chardrop']:
        sub = df[df['perturbation'] == ptype].set_index('sentence_id')
        common = clean.index.intersection(sub.index)
        for sdi_col in ['sdi_le', 'sdi_lr', 'sdi_er', 'sdi_max']:
            delta = sub.loc[common, sdi_col].to_numpy() - clean.loc[common, sdi_col].to_numpy()
            rows.append({
                'perturbation': ptype, 'sdi_col': sdi_col,
                'mean_delta':   float(delta.mean()),
                'median_delta': float(np.median(delta)),
                'pct_increased': float((delta > 0).mean()),
            })
    summary = pd.DataFrame(rows)
    print("\nΔSDI under perturbations:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(RESULTS_DATA_DIR / 'e1_delta_sdi.csv', index=False)

    # AUC of "is this perturbed" given an SDI value
    auc_rows = []
    for sdi_col in ['sdi_le', 'sdi_lr', 'sdi_er', 'sdi_max']:
        for ptype in ['synonym', 'negation', 'numeric', 'chardrop', 'any']:
            if ptype == 'any':
                pos = df[df['perturbation'] != 'clean'][sdi_col].to_numpy()
                neg = df[df['perturbation'] == 'clean'][sdi_col].to_numpy()
            else:
                pos = df[df['perturbation'] == ptype][sdi_col].to_numpy()
                neg = df[df['perturbation'] == 'clean'][sdi_col].to_numpy()
            if len(pos) == 0 or len(neg) == 0:
                continue
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            x = np.concatenate([pos, neg])
            auc = roc_auc_score(y, x)
            auc_rows.append({'sdi_col': sdi_col, 'perturbation': ptype, 'auc': float(auc)})
    auc_df = pd.DataFrame(auc_rows)
    print("\nAUC of SDI as adversarial detector:")
    print(auc_df.pivot(index='sdi_col', columns='perturbation', values='auc')
          .round(3).to_string())
    auc_df.to_csv(RESULTS_DATA_DIR / 'e1_auc.csv', index=False)

    # ΔSDI distribution figure
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, sdi_col in zip(axes.ravel(), ['sdi_le', 'sdi_lr', 'sdi_er', 'sdi_max']):
        for ptype, color in [('clean', '#888'),
                              ('synonym', '#1f77b4'),
                              ('negation', '#ff7f0e'),
                              ('numeric', '#2ca02c'),
                              ('chardrop', '#d62728')]:
            vals = df[df['perturbation'] == ptype][sdi_col].to_numpy()
            ax.hist(vals, bins=30, alpha=0.45, label=ptype, color=color)
        ax.set_title(sdi_col)
        ax.set_xlabel('SDI value')
        ax.set_ylabel('count')
        ax.legend(fontsize=7)
    fig.suptitle('SDI distributions: clean vs. perturbed')
    fig.tight_layout()
    out = FIGURES_DIR / 'fig_e1_delta_sdi.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ saved {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true',
                        help='Generate perturbed FPB sentences (CPU only)')
    parser.add_argument('--run-committee', action='store_true',
                        help='Run V+F+L on perturbed sentences (GPU)')
    parser.add_argument('--analyse', action='store_true',
                        help='Compute ΔSDI + AUC + figure (CPU only)')
    parser.add_argument('--n', type=int, default=500,
                        help='Number of held-out FPB sentences to perturb')
    args = parser.parse_args()
    if not (args.generate or args.run_committee or args.analyse):
        parser.error('pass at least one of --generate / --run-committee / --analyse')
    if args.generate:
        cmd_generate(args)
    if args.run_committee:
        cmd_run_committee(args)
    if args.analyse:
        cmd_analyse(args)
