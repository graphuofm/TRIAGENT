"""L2.5: Interaction protocols.

Runs the three protocols `vote`, `critic`, `debate` against the SDI-augmented
committee data, then reports F1 and cost vs the relevant baselines:

    - Always-L1 (VADER alone)
    - Always-L2 (FinBERT alone)
    - Always-L3 (the chosen LLM alone)
    - Vote / Critic / Debate (built on top of the three agents)

This is the empirical heart of the agentic-pivot story: do small models
*interacting* outperform larger models alone at equal compute?

Usage:
    # vote-only (no GPU)
    python experiments/L2p5_interaction.py --protocols vote

    # vote + critic at the default LLM size (Qwen-7B)
    python experiments/L2p5_interaction.py --protocols vote,critic

    # full sweep at a specific size
    python experiments/L2p5_interaction.py --protocols vote,critic,debate --llm-size 3B

    # smoke test
    python experiments/L2p5_interaction.py --sample 200 --protocols vote,critic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR
from src.interaction.vote import apply_vote


# Maps the friendly key the user passes on the CLI to the real Qwen HF id
# and to the suffix used in committee_data.csv columns.
SIZE_REGISTRY = {
    '0.5B':       {'hf': 'Qwen/Qwen2.5-0.5B-Instruct',         'suffix': '0p5b'},
    '1.5B':       {'hf': 'Qwen/Qwen2.5-1.5B-Instruct',         'suffix': '1p5b'},
    '3B':         {'hf': 'Qwen/Qwen2.5-3B-Instruct',           'suffix': '3b'},
    # 7B is the canonical "llm_*" columns — no suffix needed.
    '7B':         {'hf': 'Qwen/Qwen2.5-7B-Instruct',           'suffix': None},
    '14B':        {'hf': 'Qwen/Qwen2.5-14B-Instruct',          'suffix': '14b'},
    'Mistral-7B': {'hf': 'mistralai/Mistral-7B-Instruct-v0.3', 'suffix': 'mistral7b'},
}


def _llm_cols_for(size: str | None) -> dict:
    """Pick the set of LLM columns for a given size key.

    Returns the canonical `llm_*` columns when size is None or maps to
    the canonical (suffix=None) entry; otherwise uses the `llm_<suffix>_*` form.
    """
    suffix = None if size is None else SIZE_REGISTRY[size]['suffix']
    if suffix is None:
        return dict(
            llm_label_col='llm_label', llm_score_col='llm_score',
            llm_conf_col='llm_confidence', llm_reasoning_col='llm_reasoning',
        )
    return dict(
        llm_label_col=f'llm_{suffix}_label',
        llm_score_col=f'llm_{suffix}_score',
        llm_conf_col=f'llm_{suffix}_confidence',
        llm_reasoning_col=f'llm_{suffix}_reasoning',
    )


def _summarize(name: str, df: pd.DataFrame, label_col: str,
               extra_cost_col: str | None = None) -> dict:
    acc = accuracy_score(df['label_text'], df[label_col])
    f1 = f1_score(df['label_text'], df[label_col], average='macro')
    cost = float(df[extra_cost_col].sum()) if extra_cost_col else 0.0
    return {'protocol': name, 'acc': acc, 'f1_macro': f1, 'extra_cost_usd': cost}


def main(args):
    src_csv = RESULTS_DATA_DIR / 'sdi_data.csv'
    if not src_csv.exists():
        raise FileNotFoundError(f"Run L2 first to produce {src_csv}")
    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} rows from {src_csv}")

    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"⚠  Smoke test on {len(df)} samples")

    # If a specific size is requested, swap in its columns as the "L3 reasoner"
    llm_cols = _llm_cols_for(args.llm_size)
    for k, v in llm_cols.items():
        if v not in df.columns:
            raise KeyError(f"Column {v!r} not found — did you run L1.5 with that size?")
    print(f"Using LLM columns: {llm_cols}")

    protocols = [p.strip().lower() for p in args.protocols.split(',') if p.strip()]
    print(f"Running protocols: {protocols}\n")

    rows = []

    # Always-baselines (free)
    rows.append(_summarize('always_L1_vader',   df, 'vader_label'))
    rows.append(_summarize('always_L2_finbert', df, 'finbert_label'))
    rows.append(_summarize('always_L3_llm',     df, llm_cols['llm_label_col']))

    # ---- vote: no GPU ----
    if 'vote' in protocols:
        print("=== vote (confidence-weighted majority) ===")
        df = apply_vote(df,
                        llm_label_col=llm_cols['llm_label_col'],
                        llm_conf_col=llm_cols['llm_conf_col'],
                        llm_score_col=llm_cols['llm_score_col'])
        rows.append(_summarize('vote', df, 'vote_label'))

    # ---- critic: needs LLM ----
    if 'critic' in protocols:
        from src.agents.llm_agent import LLMAgent
        from src.interaction.critic import apply_critic
        hf = SIZE_REGISTRY[args.llm_size or '7B']['hf']
        print(f"\n=== critic (LLM-as-judge using {hf}) ===")
        llm = LLMAgent(model_name=hf)
        critic_df = apply_critic(
            df, critic=__import__('src.interaction.critic',
                                  fromlist=['CriticAgent']).CriticAgent(llm),
            trigger_col=args.critic_trigger_col,
            trigger_threshold=args.critic_threshold,
            fallback_label_col='finbert_label',
            fallback_score_col='finbert_score',
            fallback_conf_col='finbert_confidence',
        )
        for c in ['critic_label', 'critic_score', 'critic_confidence',
                  'critic_triggered', 'critic_extra_cost_usd',
                  'critic_extra_latency_ms', 'critic_rationale']:
            df[c] = critic_df[c]
        rows.append(_summarize('critic', df, 'critic_label', 'critic_extra_cost_usd'))
        del llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- debate: needs LLM ----
    if 'debate' in protocols:
        from src.agents.llm_agent import LLMAgent
        from src.interaction.debate import apply_debate, DebateAgent
        hf = SIZE_REGISTRY[args.llm_size or '7B']['hf']
        print(f"\n=== debate (round-2 reconciliation using {hf}) ===")
        llm = LLMAgent(model_name=hf)
        debate_df = apply_debate(
            df, debater=DebateAgent(llm),
            trigger_col=args.debate_trigger_col,
            trigger_threshold=args.debate_threshold,
            **llm_cols,
            fallback_label_col=llm_cols['llm_label_col'],
            fallback_score_col=llm_cols['llm_score_col'],
            fallback_conf_col=llm_cols['llm_conf_col'],
        )
        for c in ['debate_label', 'debate_score', 'debate_confidence',
                  'debate_triggered', 'debate_extra_cost_usd',
                  'debate_extra_latency_ms', 'debate_rationale']:
            df[c] = debate_df[c]
        rows.append(_summarize('debate', df, 'debate_label', 'debate_extra_cost_usd'))

    # Persist — name files per (protocol-set, size) so reruns don't clobber each other.
    suffix = (args.llm_size or '7B').lower().replace('.', 'p')
    proto_tag = '_'.join(p for p in protocols if p in ('vote', 'critic', 'debate')) or 'baseline'
    if args.sample:
        out_csv = RESULTS_DATA_DIR / f"interaction_smoke_{proto_tag}_{suffix}_n{len(df)}.csv"
    else:
        out_csv = RESULTS_DATA_DIR / f"interaction_results_{proto_tag}_{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved {out_csv}")

    summary = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_path = RESULTS_DATA_DIR / f"interaction_summary_{proto_tag}_{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✓ saved {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocols', default='vote',
                        help='Comma-separated subset of {vote,critic,debate}')
    parser.add_argument('--llm-size', choices=list(SIZE_REGISTRY.keys()),
                        default=None,
                        help='Which LLM size to use as L3 / critic / debater')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--critic-trigger-col', default='sdi_le')
    parser.add_argument('--critic-threshold', type=float, default=0.3)
    parser.add_argument('--debate-trigger-col', default='sdi_max')
    parser.add_argument('--debate-threshold', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
