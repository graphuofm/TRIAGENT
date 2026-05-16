"""L9: Same-size multi-agent committee + consistency.

Three instances of the SAME small model (Qwen-1.5B-Instruct) instantiated
with different *persona* system prompts and disagreement-rooted prompts,
then majority-voted. Tests whether persona diversification within one
size adds the same lift as cross-tier (V/F/L) diversification.

This complements our cross-tier critic plateau finding: if same-size
multi-persona vote also reaches ~0.87, then "interaction substitutes
for parameters" is a TRULY size-orthogonal phenomenon. If it doesn't,
the cross-tier diversity (lexicon vs specialist vs reasoner) is what's
doing the work, not just multi-agent voting per se.

Also reports inter-persona AGREEMENT RATE — the consistency metric the
user asked about: do same-size multi-agent committees produce stable
output if you re-roll personas?

Run: python experiments/L9_same_size_multiagent.py
     python experiments/L9_same_size_multiagent.py --sample 500   # quick
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR
from src.agents.llm_agent import LLMAgent, _parse_json


PERSONAS = {
    'bull': (
        "You are a sell-side equity analyst with a long bias. "
        "You instinctively look for upside catalysts and prefer to err "
        "toward 'positive' when there is reasonable evidence."
    ),
    'bear': (
        "You are a short-only credit analyst. "
        "You instinctively look for downside risk and prefer to err "
        "toward 'negative' when there is reasonable evidence."
    ),
    'neutral': (
        "You are a conservative compliance analyst. "
        "You instinctively prefer to flag a sentence as 'neutral' "
        "unless the evidence for positive or negative is unambiguous."
    ),
}


PROMPT = """{persona_intro}

Classify the sentiment of the following financial sentence as exactly
one of: positive, negative, or neutral.

DOMAIN RULES (apply to all personas):
- Consider financial impact, not surface-level word sentiment.
- "Loss narrowed" / "loss decreased" → positive.
- "Profit declined" / "earnings missed" → negative.
- "Liability" / "debt" / "short" are often neutral accounting terms.

Respond with ONLY a JSON object, no markdown:
{{"sentiment": "positive|negative|neutral", "score": -1.0 to 1.0,
  "confidence": 0.0 to 1.0, "reasoning": "brief"}}

Sentence: "{sentence}"
"""


@torch.no_grad()
def run_persona(llm: LLMAgent, sentences: list[str], persona_intro: str,
                 batch_size: int = 8) -> list[tuple[str, float, float, float]]:
    """Returns list of (label, score, confidence, latency_ms) per sentence."""
    tok = llm.tokenizer
    model = llm.model
    out = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='persona'):
        chunk = sentences[i:i+batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user",
                   "content": PROMPT.format(persona_intro=persona_intro, sentence=s)}],
                tokenize=False, add_generation_prompt=True
            ) for s in chunk
        ]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                  max_length=1024).to(model.device)
        padded = enc['input_ids'].shape[1]
        t0 = time.perf_counter()
        outs = model.generate(**enc, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        per_lat = (time.perf_counter() - t0) * 1000 / len(chunk)
        for j in range(len(chunk)):
            new_ids = outs[j, padded:]
            eos = tok.eos_token_id; pad = tok.pad_token_id
            clean = []
            for tid in new_ids.tolist():
                if tid == eos or tid == pad:
                    break
                clean.append(tid)
            text = tok.decode(clean, skip_special_tokens=True)
            try:
                p = _parse_json(text)
                lbl = str(p.get('sentiment','neutral')).lower()
                if lbl not in ('positive','negative','neutral'):
                    lbl = 'neutral'
                sc = float(p.get('score', 0.0))
                cf = float(p.get('confidence', 0.7))
            except Exception:
                lbl, sc, cf = 'neutral', 0.0, 0.0
            out.append((lbl, sc, cf, per_lat))
    return out


def majority_vote(labels: list[str]) -> str:
    from collections import Counter
    cnt = Counter(labels)
    top = cnt.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return 'neutral'  # tie-break to conservative
    return top[0][0]


def main(args):
    src = RESULTS_DATA_DIR / 'sdi_data_fpb_backup2.csv'
    if not src.exists():
        src = RESULTS_DATA_DIR / 'sdi_data.csv'
    sdi = pd.read_csv(src)
    if args.sample:
        sdi = sdi.sample(args.sample, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(sdi)} sentences from {src}")

    print(f"\nLoading {args.model}...")
    llm = LLMAgent(model_name=args.model)

    sentences = sdi['sentence'].fillna('').astype(str).tolist()
    persona_results: dict[str, list] = {}
    for name, intro in PERSONAS.items():
        print(f"\n=== Running persona: {name} ===")
        persona_results[name] = run_persona(llm, sentences, intro,
                                            batch_size=args.batch_size)

    # Build dataframe of per-persona predictions
    rows = []
    for i in range(len(sdi)):
        rec = {'gold': sdi['label_text'].iloc[i]}
        for name in PERSONAS:
            lbl, sc, cf, lat = persona_results[name][i]
            rec[f'{name}_label'] = lbl
            rec[f'{name}_score'] = sc
            rec[f'{name}_conf']  = cf
        rec['vote_label'] = majority_vote([rec[f'{name}_label'] for name in PERSONAS])
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS_DATA_DIR / 'same_size_multiagent.csv', index=False)

    # ---- F1 per persona, vote, vs single-instance baseline ----
    print("\n=== Same-size multi-agent results ===")
    summary = []
    for name in PERSONAS:
        f1 = f1_score(out_df['gold'], out_df[f'{name}_label'], average='macro')
        acc = accuracy_score(out_df['gold'], out_df[f'{name}_label'])
        summary.append({'agent': f'persona-{name}', 'acc': acc, 'f1_macro': f1})
    f1_vote = f1_score(out_df['gold'], out_df['vote_label'], average='macro')
    acc_vote = accuracy_score(out_df['gold'], out_df['vote_label'])
    summary.append({'agent': '3-persona-vote', 'acc': acc_vote, 'f1_macro': f1_vote})

    # Reference baseline: original Qwen-1.5B alone (from L1.5)
    if 'llm_1p5b_label' in sdi.columns:
        f1_baseline = f1_score(sdi['label_text'], sdi['llm_1p5b_label'], average='macro')
        summary.append({'agent': 'Qwen-1.5B (single, no persona)',
                         'acc': accuracy_score(sdi['label_text'], sdi['llm_1p5b_label']),
                         'f1_macro': f1_baseline})
    if 'finbert_label' in sdi.columns:
        f1_finbert = f1_score(sdi['label_text'], sdi['finbert_label'], average='macro')
        summary.append({'agent': 'FinBERT (cross-tier reference)',
                         'acc': accuracy_score(sdi['label_text'], sdi['finbert_label']),
                         'f1_macro': f1_finbert})
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sdf.to_csv(RESULTS_DATA_DIR / 'same_size_multiagent_summary.csv', index=False)

    # ---- Pairwise CONSISTENCY across personas ----
    print("\n=== Pairwise inter-persona agreement ===")
    cons = []
    for a in PERSONAS:
        for b in PERSONAS:
            if a >= b: continue
            agree = (out_df[f'{a}_label'] == out_df[f'{b}_label']).mean()
            kappa = cohen_kappa_score(out_df[f'{a}_label'], out_df[f'{b}_label'])
            cons.append({'pair': f'{a}↔{b}', 'agreement_rate': float(agree),
                          'cohen_kappa': float(kappa)})
    cdf = pd.DataFrame(cons)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    cdf.to_csv(RESULTS_DATA_DIR / 'same_size_multiagent_consistency.csv', index=False)

    # ---- All-three-agree rate ----
    all_agree = (
        (out_df['bull_label']    == out_df['bear_label']) &
        (out_df['bull_label']    == out_df['neutral_label'])
    ).mean()
    print(f"\nAll three personas agree on the SAME label: {all_agree:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--sample', type=int, default=None,
                        help='Run on a random sample of N sentences (default: full FPB)')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    main(args)
